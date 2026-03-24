"""Model-agnostic wrapper for latent multi-agent inference.

Works with **any** ``AutoModelForCausalLM``-compatible HuggingFace model.
Provides three core operations:

1. ``prepare_chat_batch`` -- tokenise chat messages using the model's own
   chat template.
2. ``generate_text_batch`` -- standard autoregressive text generation (with
   optional KV-cache seeding from prior latent agents).
3. ``generate_latent_batch`` -- iterative latent-space "thinking" that grows
   the KV-cache without producing any tokens.
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoModelForCausalLM, AutoTokenizer

from .realigner import LatentRealigner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    """Make sure the tokenizer has a pad token (required for batched inputs)."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def past_kv_length(past_key_values) -> int:
    """Return the sequence length stored in a KV-cache (tuple or Cache object)."""
    if not past_key_values:
        return 0
    # Modern transformers Cache objects (DynamicCache, etc.)
    if hasattr(past_key_values, 'get_seq_length'):
        return past_key_values.get_seq_length()
    # Legacy tuple format: ((key, value), ...) per layer
    k = past_key_values[0][0]
    return k.shape[-2]


# ---------------------------------------------------------------------------
# LatentModel
# ---------------------------------------------------------------------------

class LatentModel:
    """Thin, model-agnostic wrapper around a HuggingFace causal LM.

    Parameters
    ----------
    model_name_or_path : str
        Any HuggingFace model identifier or local path.
    device : Union[str, torch.device]
        Target device (e.g. ``"cuda"``, ``"cpu"``).
    realign : bool
        Whether to build the full latent realignment projection.  When
        ``False``, an identity matrix is used but norm-matching is still
        applied (useful as an ablation).
    torch_dtype : Optional[torch.dtype]
        Dtype for model weights.  Defaults to ``bfloat16`` on CUDA, else
        ``float32``.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Union[str, torch.device] = "cuda",
        *,
        realign: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.device = torch.device(device)

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, use_fast=True,
        )
        _ensure_pad_token(self.tokenizer)
        self.tokenizer.padding_side = "left"

        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, torch_dtype=torch_dtype,
            )

        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device).eval()

        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        self.realigner = LatentRealigner(self.model, self.device, enabled=realign)

    # ------------------------------------------------------------------
    # Chat template helpers
    # ------------------------------------------------------------------

    def render_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Render chat messages into a prompt string."""
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        segments: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Tokenise a batch of chat conversations.

        Returns ``(prompts, input_ids, attention_mask)``.
        """
        prompts: List[str] = [
            self.render_chat(msgs, add_generation_prompt=add_generation_prompt)
            for msgs in batch_messages
        ]
        encoded = self.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        return prompts, input_ids, attention_mask

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        """Standard text generation, optionally seeded with a KV-cache."""
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)

        prompt_lengths = attention_mask.sum(dim=1).tolist()

        cache_position = None
        if past_key_values is not None:
            past_len = past_kv_length(past_key_values)
            cache_position = torch.arange(
                past_len, past_len + input_ids.shape[-1],
                dtype=torch.long, device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            generated_ids = sequences[idx, int(length):]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)

        return generations, outputs.past_key_values

    # ------------------------------------------------------------------
    # Latent generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        convergence_threshold: Optional[float] = None,
    ) -> Tuple[Tuple, int]:
        """Run latent-space thinking -- grows the KV-cache without decoding.

        Returns ``(past_key_values, actual_steps)`` where *actual_steps* may
        be less than *latent_steps* if early stopping via *convergence_threshold*
        triggered.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            plen = past_kv_length(past_key_values)
            if plen > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], plen),
                    dtype=attention_mask.dtype, device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        prev_hidden = last_hidden.clone()

        actual_steps = 0
        for step in range(latent_steps):
            latent_vec = self.realigner.apply(last_hidden)
            latent_embed = latent_vec.unsqueeze(1)

            plen = past_kv_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], plen + 1),
                dtype=torch.long, device=self.device,
            )

            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            actual_steps += 1

            if convergence_threshold is not None and step > 0:
                delta = (last_hidden - prev_hidden).norm() / last_hidden.norm().clamp_min(1e-8)
                if delta.item() < convergence_threshold:
                    break

            prev_hidden = last_hidden.clone()

        return past, actual_steps
