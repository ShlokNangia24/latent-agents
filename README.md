# latent-agents

**Model-agnostic latent multi-agent communication for HuggingFace language models.**

Instead of agents passing text to each other (slow, token-heavy), agents think silently in latent space and pass their neural activations through a shared KV-cache. Only the final agent speaks.

Based on the [LatentMAS paper](https://arxiv.org/abs/2511.20639) (arXiv:2511.20639). This library extracts the core mechanism into a clean, reusable package that works with **any** `AutoModelForCausalLM`-compatible model.

---

## Install

```bash
pip install git+https://github.com/YOUR_USERNAME/latent-agents.git
```

Or for local development:

```bash
git clone https://github.com/YOUR_USERNAME/latent-agents.git
cd latent-agents
pip install -e .
```

**Requirements:** Python >= 3.9, PyTorch >= 2.0, Transformers >= 4.36

---

## Quick Start

```python
from latent_agents import Agent, LatentModel, LatentPipeline, set_seed

set_seed(42)

# 1. Load any HuggingFace model
model = LatentModel("meta-llama/Llama-3-8B-Instruct", device="cuda")

# 2. Define your agents
agents = [
    Agent(name="Planner", role="planner",
          prompt_fn=lambda q, c: [
              {"role": "user", "content": f"Plan how to solve: {q}"}
          ]),
    Agent(name="Solver", role="solver", is_final=True,
          prompt_fn=lambda q, c: [
              {"role": "user", "content": f"Solve step by step: {q}"}
          ]),
]

# 3. Run
pipeline = LatentPipeline(model, agents, latent_steps=20)
result = pipeline.run("What is 24 * 37?")
print(result.text)
```

---

## How It Works

In a traditional multi-agent system, each agent generates full text responses that the next agent reads. This is slow -- agents 1 through N each produce hundreds of tokens.

**latent-agents** does it differently:

1. Each non-final agent reads its prompt, runs a forward pass, then feeds the output hidden state back into the model repeatedly (`latent_steps` times) -- growing the KV-cache without producing any tokens.
2. A **realignment matrix** (computed once from the model's own weights, no training) projects output hidden states back into input-embedding space so they can be fed back in without drift.
3. The final agent receives the accumulated KV-cache and generates text normally.

The result: the same collaborative reasoning, with 50-80% fewer tokens and 3-7x wall-clock speedups.

---

## Compatible Models

Any `AutoModelForCausalLM` model that has input embeddings, output embeddings, and supports `inputs_embeds`:

```python
# Llama
model = LatentModel("meta-llama/Llama-3.2-1B-Instruct", device="cuda")

# Mistral
model = LatentModel("mistralai/Mistral-7B-Instruct-v0.3", device="cuda")

# Gemma
model = LatentModel("google/gemma-2-2b-it", device="cuda")

# Phi
model = LatentModel("microsoft/Phi-3-mini-4k-instruct", device="cuda")

# Qwen
model = LatentModel("Qwen/Qwen3-4B", device="cuda")

# Local fine-tuned model
model = LatentModel("/path/to/my-model", device="cuda")
```

---

## Defining Agents

An agent is a name, a role, a prompt function, and a flag:

```python
from latent_agents import Agent

def my_prompt(question, context):
    return [
        {"role": "system", "content": "You are a careful analyst."},
        {"role": "user", "content": f"Analyze: {question}"},
    ]

agent = Agent(
    name="Analyst",
    role="analyst",
    prompt_fn=my_prompt,
    is_final=False,   # Thinks silently in latent space
)
```

**One rule:** exactly one agent must have `is_final=True`. That agent generates text. Everyone before it operates silently.

---

## Examples

### Two-agent pipeline

```python
agents = [
    Agent(name="Thinker", role="thinker",
          prompt_fn=lambda q, c: [{"role": "user", "content": f"Think about: {q}"}]),
    Agent(name="Speaker", role="speaker", is_final=True,
          prompt_fn=lambda q, c: [{"role": "user", "content": f"Answer: {q}"}]),
]

pipeline = LatentPipeline(model, agents, latent_steps=15)
result = pipeline.run("Why is the sky blue?")
```

### Four-agent reasoning chain

```python
agents = [
    Agent(name="Planner", role="planner",
          prompt_fn=lambda q, c: [{"role": "user", "content": f"Plan: {q}"}]),
    Agent(name="Critic", role="critic",
          prompt_fn=lambda q, c: [{"role": "user", "content": f"Critique the plan: {q}"}]),
    Agent(name="Refiner", role="refiner",
          prompt_fn=lambda q, c: [{"role": "user", "content": f"Refine the plan: {q}"}]),
    Agent(name="Solver", role="solver", is_final=True,
          prompt_fn=lambda q, c: [{"role": "user", "content": f"Solve: {q}"}]),
]

pipeline = LatentPipeline(model, agents, latent_steps=25, max_new_tokens=2048)
```

### Batch processing

```python
questions = ["What is 2+2?", "What is 3*7?", "What is 100/4?"]
results = pipeline.run_batch(questions)
for q, r in zip(questions, results):
    print(f"{q} -> {r.text}")
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_steps` | 20 | Thinking iterations per silent agent (10-40 typical) |
| `max_new_tokens` | 2048 | Token budget for the final agent |
| `temperature` | 0.6 | Sampling temperature (final agent only) |
| `top_p` | 0.95 | Nucleus sampling (final agent only) |
| `keep_only_latent` | False | Drop prompt tokens from KV-cache between agents (saves memory) |
| `realign` | True | Enable latent-space realignment projection |

---

## Strengths

- **Fast** -- 50-80% fewer tokens, 3-7x wall-clock speedups vs text-based multi-agent systems
- **No training required** -- realignment matrix is computed from existing model weights
- **Model-agnostic** -- works with any HuggingFace causal LM
- **Fully customisable** -- you define agents, prompts, and pipeline structure
- **Composable** -- use `LatentModel`, `LatentRealigner`, or `LatentPipeline` independently

## Constraints

- **Single model** -- all agents share the same model (latent space must match)
- **Lossy communication** -- agents pass neural activations, not words; the final agent can't quote earlier agents
- **GPU memory** -- KV-cache grows with agents and latent steps; use `keep_only_latent=True` if memory is tight
- **Instruct models work best** -- base models don't reliably follow chat-formatted prompts
- **No streaming** -- final agent generates its complete response in one call

---

## API Reference

```
LatentModel(model_name, device, realign=True, torch_dtype=None)
    .generate_text_batch(...)   -> (texts, kv_cache)
    .generate_latent_batch(...) -> kv_cache
    .prepare_chat_batch(...)    -> (prompts, input_ids, attention_mask)
    .tokenizer                  -- HuggingFace tokenizer
    .model                      -- raw HuggingFace model
    .realigner                  -- LatentRealigner instance

LatentPipeline(model, agents, latent_steps, max_new_tokens, temperature, top_p, keep_only_latent)
    .run(question)              -> PipelineResult
    .run_batch(questions)       -> List[PipelineResult]

PipelineResult
    .text                       -- final agent's generated text
    .agent_traces               -- per-agent metadata

Agent(name, role, prompt_fn, is_final=False)

LatentRealigner(model, device, enabled=True, reg=1e-5)
    .apply(hidden_state)        -> realigned embedding

set_seed(seed)
auto_device(preference=None)    -> torch.device
```

---

## Citation

This library is based on the LatentMAS paper. If you use it in research, please cite:

```bibtex
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

## License

Apache 2.0
