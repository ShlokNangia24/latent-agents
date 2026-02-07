"""latent_agents -- model-agnostic latent multi-agent communication library.

Quick start::

    from latent_agents import LatentModel, LatentPipeline, Agent

    model = LatentModel("meta-llama/Llama-3-8B-Instruct", device="cuda")

    agents = [
        Agent(name="Planner", role="planner",
              prompt_fn=lambda q, c: [{"role": "user", "content": f"Plan: {q}"}]),
        Agent(name="Solver", role="solver", is_final=True,
              prompt_fn=lambda q, c: [{"role": "user", "content": f"Solve: {q}"}]),
    ]

    pipeline = LatentPipeline(model, agents, latent_steps=20)
    result = pipeline.run("What is 2+2?")
    print(result.text)
"""

from .agent import Agent
from .model import LatentModel
from .pipeline import LatentPipeline, PipelineResult
from .realigner import LatentRealigner
from .utils import set_seed, auto_device

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "LatentModel",
    "LatentPipeline",
    "PipelineResult",
    "LatentRealigner",
    "set_seed",
    "auto_device",
]
