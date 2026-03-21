"""Agent definition for the latent multi-agent pipeline."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


# Type alias for prompt functions.
# A prompt function takes (question: str, context: str) and returns a list of
# chat messages suitable for ``tokenizer.apply_chat_template``.
PromptFn = Callable[[str, str], List[Dict[str, str]]]


@dataclass
class Agent:
    """A single agent in the latent pipeline.

    Parameters
    ----------
    name : str
        Human-readable name for logging (e.g. "Planner", "Critic").
    role : str
        Short identifier used internally (e.g. "planner", "critic").
    prompt_fn : PromptFn
        Callable ``(question, context) -> List[Dict]`` that builds the chat
        messages for this agent.  *context* is an opaque string that the
        pipeline may pass (empty for latent-only pipelines).
    is_final : bool
        If ``True`` this agent generates text output.  All preceding agents
        operate in latent space only.  Exactly one agent in the pipeline
        should have ``is_final=True`` (typically the last one).
    latent_steps : int or None
        Per-agent override for the number of latent thinking steps.
        If ``None`` (default), the pipeline's global ``latent_steps`` is used.
    convergence_threshold : float or None
        If set, latent generation stops early when the relative change in
        hidden-state norm falls below this value.  ``None`` means no early
        stopping (or uses the pipeline-level default if one is set).
    """

    name: str
    role: str
    prompt_fn: PromptFn
    is_final: bool = False
    latent_steps: Optional[int] = None
    convergence_threshold: Optional[float] = None
