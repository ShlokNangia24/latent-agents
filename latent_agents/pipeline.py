"""Latent multi-agent pipeline.

Orchestrates a sequence of ``Agent`` objects:
* Non-final agents run latent-space "thinking" (no tokens produced).
* The final agent generates text seeded with the accumulated KV-cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

from .agent import Agent
from .model import LatentModel, past_kv_length


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Output of a single pipeline run for one question."""

    text: str
    """Generated text from the final agent."""

    agent_traces: List[Dict[str, Any]] = field(default_factory=list)
    """Per-agent metadata (prompt, role, latent_steps, output text, etc.)."""


# ---------------------------------------------------------------------------
# KV-cache truncation helpers
# ---------------------------------------------------------------------------

def _slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
    if tokens_to_keep <= 0:
        return tensor[..., 0:0, :].contiguous()
    keep = min(tokens_to_keep, tensor.shape[-2])
    start = tensor.shape[-2] - keep
    return tensor[..., start:, :].contiguous()


def truncate_kv_cache(
    past_kv: Optional[Tuple],
    tokens_to_keep: int,
) -> Optional[Tuple]:
    """Trim a KV-cache to the last *tokens_to_keep* entries."""
    if past_kv is None or tokens_to_keep <= 0:
        return None

    if Cache is not None and isinstance(past_kv, Cache):
        legacy = past_kv.to_legacy_cache()
        trimmed = tuple(
            tuple(_slice_tensor(t, tokens_to_keep) for t in layer)
            for layer in legacy
        )
        return past_kv.__class__.from_legacy_cache(trimmed)

    trimmed_layers: list = []
    for layer in past_kv:
        if isinstance(layer, tuple):
            trimmed_layers.append(
                tuple(_slice_tensor(t, tokens_to_keep) for t in layer)
            )
        elif torch.is_tensor(layer):
            trimmed_layers.append(_slice_tensor(layer, tokens_to_keep))
        else:
            trimmed_layers.append(layer)
    return tuple(trimmed_layers)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class LatentPipeline:
    """Run a sequence of agents, communicating through latent space.

    Parameters
    ----------
    model : LatentModel
        The wrapped language model.
    agents : List[Agent]
        Ordered list of agents.  Exactly one must have ``is_final=True``.
    latent_steps : int
        Number of latent recurrence steps per non-final agent.
    max_new_tokens : int
        Maximum tokens for the final agent's text generation.
    temperature : float
        Sampling temperature for the final agent.
    top_p : float
        Nucleus sampling threshold for the final agent.
    keep_only_latent : bool
        When ``True``, only the latent-step KV entries (not the prompt tokens)
        are kept between agents.  Reduces memory at the cost of discarding
        the textual prompt context from earlier agents.
    """

    def __init__(
        self,
        model: LatentModel,
        agents: List[Agent],
        *,
        latent_steps: int = 20,
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.95,
        keep_only_latent: bool = False,
    ) -> None:
        final_agents = [a for a in agents if a.is_final]
        if len(final_agents) != 1:
            raise ValueError(
                f"Exactly one agent must have is_final=True, found {len(final_agents)}."
            )

        self.model = model
        self.agents = agents
        self.latent_steps = latent_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.keep_only_latent = keep_only_latent

    def run(self, question: str, *, context: str = "") -> PipelineResult:
        """Run the full agent pipeline on a single question."""
        return self.run_batch([question], context=context)[0]

    @torch.no_grad()
    def run_batch(
        self,
        questions: List[str],
        *,
        context: str = "",
    ) -> List[PipelineResult]:
        """Run the pipeline on a batch of questions."""
        batch_size = len(questions)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
        final_texts: List[str] = [""] * batch_size

        for agent in self.agents:
            batch_messages = [agent.prompt_fn(q, context) for q in questions]
            prompts, input_ids, attention_mask = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True,
            )

            if not agent.is_final:
                prev_len = past_kv_length(past_kv)

                past_kv = self.model.generate_latent_batch(
                    input_ids,
                    attention_mask=attention_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )

                if self.keep_only_latent:
                    new_len = past_kv_length(past_kv)
                    tokens_added = new_len - prev_len
                    past_kv = truncate_kv_cache(past_kv, tokens_added)

                for idx in range(batch_size):
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "prompt": prompts[idx],
                        "latent_steps": self.latent_steps,
                        "output": "",
                    })

            else:
                past_for_decoding = past_kv if self.latent_steps > 0 else None

                generated_batch, _ = self.model.generate_text_batch(
                    input_ids,
                    attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )

                for idx in range(batch_size):
                    text = generated_batch[idx].strip()
                    final_texts[idx] = text
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "prompt": prompts[idx],
                        "output": text,
                    })

        return [
            PipelineResult(text=final_texts[idx], agent_traces=agent_traces[idx])
            for idx in range(batch_size)
        ]
