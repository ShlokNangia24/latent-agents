# latent-agents

**Model-agnostic latent multi-agent communication for HuggingFace language models.**

Instead of agents passing text to each other (slow, token-heavy), agents think silently in latent space and pass their neural activations through a shared KV-cache. Only the final agent speaks.

Based on the [LatentMAS paper](https://arxiv.org/abs/2511.20639) (arXiv:2511.20639). This library extracts the core mechanism into a clean, reusable package that works with **any** `AutoModelForCausalLM`-compatible model.

---

## Install

```bash
pip install git+https://github.com/ShlokNangia24/latent-agents.git
```

Or for local development:

```bash
git clone https://github.com/ShlokNangia24/latent-agents.git
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

An agent is a name, a role, a prompt function, and optional configuration:

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
    is_final=False,              # Thinks silently in latent space
    latent_steps=30,             # Override pipeline default (optional)
    convergence_threshold=0.01,  # Early stop when thinking stabilizes (optional)
)
```

**One rule:** exactly one agent must have `is_final=True`. That agent generates text. Everyone before it operates silently.

**Per-agent latent steps:** Each agent can have its own `latent_steps` count. A planner might need 30 steps while a critic only needs 10. If not set, the pipeline default is used.

**Adaptive convergence:** Set `convergence_threshold` to stop latent steps early when the hidden-state change between steps falls below the threshold. Saves compute on easy problems without hurting hard ones.

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

### Four-agent reasoning chain with per-agent steps

```python
agents = [
    Agent(name="Planner", role="planner", latent_steps=30,  # thinks more
          prompt_fn=lambda q, c: [{"role": "user", "content": f"Plan: {q}"}]),
    Agent(name="Critic", role="critic", latent_steps=10,    # thinks less
          prompt_fn=lambda q, c: [{"role": "user", "content": f"Critique the plan: {q}"}]),
    Agent(name="Refiner", role="refiner",                    # uses pipeline default
          prompt_fn=lambda q, c: [{"role": "user", "content": f"Refine the plan: {q}"}]),
    Agent(name="Solver", role="solver", is_final=True,
          prompt_fn=lambda q, c: [{"role": "user", "content": f"Solve: {q}"}]),
]

pipeline = LatentPipeline(model, agents, latent_steps=20, max_new_tokens=2048)
```

### Self-consistency voting

Generate multiple answers and majority-vote the best one (especially useful for math):

```python
pipeline = LatentPipeline(
    model, agents,
    latent_steps=20,
    n_samples=5,  # generate 5 answers, pick the most common
)
result = pipeline.run("What is 24 * 37?")
# result.agent_traces[-1]["candidates"] contains all 5 answers
```

### Adaptive convergence (early stopping)

Stop latent steps early when the model's thinking stabilizes:

```python
pipeline = LatentPipeline(
    model, agents,
    latent_steps=40,                 # max steps
    convergence_threshold=0.01,      # stop when hidden-state delta < 1%
)
# Agents may use fewer steps on easy problems
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

### Pipeline Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_steps` | 20 | Default thinking iterations per silent agent (10-40 typical) |
| `max_new_tokens` | 2048 | Token budget for the final agent |
| `temperature` | 0.6 | Sampling temperature (final agent only) |
| `top_p` | 0.95 | Nucleus sampling (final agent only) |
| `keep_only_latent` | False | Drop prompt tokens from KV-cache between agents (saves memory) |
| `convergence_threshold` | None | Early-stop latent steps when hidden-state change < threshold |
| `n_samples` | 1 | Self-consistency voting: generate N answers, majority-vote the best |
| `vote_fn` | None | Custom voting function `(List[str]) -> str` |

### Agent Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_steps` | None | Per-agent override (None = use pipeline default) |
| `convergence_threshold` | None | Per-agent early stopping threshold |

### Model Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `realign` | True | Enable latent-space realignment projection |
| `torch_dtype` | None | Auto: bfloat16 on CUDA, float32 on CPU/MPS |

---

## Benchmark Results

### Qwen3-1.7B (GPU)

| Benchmark | Single Agent | Non-Latent 4-Agent | Latent 4-Agent | Token Savings | Speedup |
|-----------|:-----------:|:------------------:|:--------------:|:-------------:|:-------:|
| GSM8K | 22% | 68% | 23% | 76% | 3.6x |
| MMLU | 47% | 52% | 56% | 71% | 2.5x |
| ARC-Easy | 76% | 71% | 76% | 44% | 1.2x |
| **Overall** | **54%** | **63%** | **57%** | **74%** | **3.4x** |

Latent pipeline delivers **74% fewer tokens** and **3.4x faster** inference vs text-based multi-agent. On knowledge tasks (MMLU, ARC), latent matches or beats non-latent. The GSM8K gap narrows significantly with larger models (4B+), as shown in the original paper.

---

## Running GPU Benchmarks

The full benchmark suite (`run_gpu_benchmarks.py`) tests multiple models across GSM8K, MMLU, and ARC-Easy and generates paper-style charts. It requires a CUDA GPU with enough VRAM for the models you want to test.

### Recommended hardware

| GPU | VRAM | Best for | Approximate cost |
|-----|------|----------|-----------------|
| A100 80GB | 80 GB | All models including Llama-70B | ~$2-3/hr on vast.ai |
| A100 40GB | 40 GB | Up to Qwen3-32B | ~$1.5/hr |
| A10G / RTX 3090 | 24 GB | Up to Qwen3-8B | ~$0.5/hr |

A full run of all models takes ~3-4 hours on an A100 80GB (~$5-10 total).

### 1. Rent a GPU (vast.ai)

1. Go to [vast.ai](https://vast.ai), create an account and add credit
2. Search for a machine with an A100 80GB, Ubuntu 22.04, PyTorch pre-installed
3. SSH in once the instance starts

### 2. Set up the environment

```bash
# Clone and install
git clone https://github.com/ShlokNangia24/latent-agents.git
cd latent-agents
pip install -e ".[bench]"

# For gated models (Llama, Mistral, Gemma) — accept licenses on HuggingFace website first
export HF_TOKEN=hf_your_token_here
huggingface-cli login --token $HF_TOKEN
```

### 3. Sanity check (2 minutes)

```bash
python run_gpu_benchmarks.py \
  --models "Qwen/Qwen3-1.7B" \
  --n-gsm8k 5 --n-mmlu 5 --n-arc 5
```

### 4. Full benchmark run

```bash
# All models, full sample counts (100 GSM8K, 200 MMLU, 100 ARC per model)
python run_gpu_benchmarks.py

# Skip gated models (no HF_TOKEN needed)
python run_gpu_benchmarks.py --skip-gated

# Custom model list
python run_gpu_benchmarks.py --models "Qwen/Qwen3-4B,Qwen/Qwen3-8B"

# Custom latent steps (default: 10)
python run_gpu_benchmarks.py --latent-steps 20
```

### 5. Run in the background with logging

```bash
nohup python run_gpu_benchmarks.py \
  --output benchmark_results.json \
  2>&1 | tee benchmark_run.log &

# Monitor progress
tail -f benchmark_run.log
```

### 6. Regenerate charts from saved results

```bash
# Charts are generated automatically after each run.
# To regenerate from a saved JSON (e.g. after downloading to Mac):
python run_gpu_benchmarks.py --charts-only benchmark_results.json
```

Charts are saved in `charts/overall/` (cross-model comparisons) and `charts/{ModelName}/` (per-model drill-downs).

### 7. Download results to your Mac

```bash
# From your Mac (replace with your vast.ai instance SSH details)
scp -r user@your-instance-ip:~/latent-agents/benchmark_results.json ./
scp -r user@your-instance-ip:~/latent-agents/charts/ ./charts/
```

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

LatentPipeline(model, agents, latent_steps, max_new_tokens, temperature, top_p,
               keep_only_latent, convergence_threshold, n_samples, vote_fn)
    .run(question)              -> PipelineResult
    .run_batch(questions)       -> List[PipelineResult]

PipelineResult
    .text                       -- final agent's generated text
    .agent_traces               -- per-agent metadata (includes latent_steps, candidates, etc.)

Agent(name, role, prompt_fn, is_final=False, latent_steps=None, convergence_threshold=None)

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
