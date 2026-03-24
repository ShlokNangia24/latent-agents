"""Latent multi-agent pipeline.

Orchestrates a sequence of ``Agent`` objects:
* Non-final agents run latent-space "thinking" (no tokens produced).
* The final agent generates text seeded with the accumulated KV-cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

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
# Self-consistency voting helpers
# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> str:
    """Extract a canonical answer string from a generation."""
    # Priority 1: \boxed{...}
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    # Priority 2: #### <answer>
    m = re.search(r'####\s*(.+)', text)
    if m:
        return m.group(1).strip()
    # Priority 3: "the answer is <X>"
    m = re.search(r'the answer is\s+(.+?)[\.\n]', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Priority 4: "FINAL.*: <answer>"
    m = re.search(r'FINAL[^:]*:\s*(.+)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: last non-empty line
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def _majority_vote(candidates: List[str]) -> str:
    """Select the most common answer from a list of candidates."""
    if len(candidates) == 1:
        return candidates[0]

    extracted = [_extract_answer(c) for c in candidates]
    counter = Counter(extracted)
    most_common_answer = counter.most_common(1)[0][0]

    # Return the full candidate text that corresponds to the winning answer
    for c, e in zip(candidates, extracted):
        if e == most_common_answer:
            return c

    return candidates[0]


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
        Default number of latent recurrence steps per non-final agent.
        Individual agents can override this via ``Agent.latent_steps``.
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
    convergence_threshold : float or None
        Default early-stopping threshold for latent generation.  Individual
        agents can override this via ``Agent.convergence_threshold``.
    n_samples : int
        Number of independent text generations for the final agent.
        When > 1, self-consistency voting selects the best answer.
    vote_fn : callable or None
        Custom voting function ``(candidates: List[str]) -> str``.
        Defaults to majority voting via answer extraction.
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
        convergence_threshold: Optional[float] = None,
        n_samples: int = 1,
        vote_fn: Optional[Callable] = None,
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
        self.convergence_threshold = convergence_threshold
        self.n_samples = n_samples
        self.vote_fn = vote_fn

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

        last_latent_steps = self.latent_steps  # track for final-agent gate

        for agent in self.agents:
            batch_messages = [agent.prompt_fn(q, context) for q in questions]
            prompts, input_ids, attention_mask = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True,
            )

            if not agent.is_final:
                # Resolve per-agent overrides
                steps = agent.latent_steps if agent.latent_steps is not None else self.latent_steps
                threshold = (
                    agent.convergence_threshold
                    if agent.convergence_threshold is not None
                    else self.convergence_threshold
                )
                last_latent_steps = steps

                past_kv, actual_steps = self.model.generate_latent_batch(
                    input_ids,
                    attention_mask=attention_mask,
                    latent_steps=steps,
                    past_key_values=past_kv,
                    convergence_threshold=threshold,
                )

                if self.keep_only_latent:
                    past_kv = truncate_kv_cache(past_kv, actual_steps)

                for idx in range(batch_size):
                    agent_traces[idx].append({
                        "name": agent.name,
                        "role": agent.role,
                        "prompt": prompts[idx],
                        "latent_steps": actual_steps,
                        "latent_steps_configured": steps,
                        "output": "",
                    })

            else:
                past_for_decoding = past_kv if last_latent_steps > 0 else None

                if self.n_samples <= 1:
                    # Single generation (default behavior)
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
                else:
                    # Self-consistency voting: generate n_samples candidates
                    all_candidates: List[List[str]] = [[] for _ in range(batch_size)]

                    for sample_idx in range(self.n_samples):
                        # Deep-copy KV-cache because generate() mutates it in-place
                        kv = (
                            copy.deepcopy(past_for_decoding)
                            if sample_idx > 0 and past_for_decoding is not None
                            else past_for_decoding
                        )
                        generated_batch, _ = self.model.generate_text_batch(
                            input_ids,
                            attention_mask,
                            max_new_tokens=self.max_new_tokens,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            past_key_values=kv,
                        )
                        for idx in range(batch_size):
                            all_candidates[idx].append(generated_batch[idx].strip())

                    vote = self.vote_fn or _majority_vote
                    for idx in range(batch_size):
                        best = vote(all_candidates[idx])
                        final_texts[idx] = best
                        agent_traces[idx].append({
                            "name": agent.name,
                            "role": agent.role,
                            "prompt": prompts[idx],
                            "output": best,
                            "candidates": all_candidates[idx],
                            "n_samples": self.n_samples,
                        })

        return [
            PipelineResult(text=final_texts[idx], agent_traces=agent_traces[idx])
            for idx in range(batch_size)
        ]
