# CLAUDE.md — latent-agents (LatentMAS)

> **Maintainer note:** Update this file whenever you fix a bug, resolve a common error, or change an interface. Future agents and contributors depend on it being accurate.

---

## What This Project Is

This is the Python implementation of **LatentMAS** ("Latent Collaboration in Multi-Agent Systems"), a training-free multi-agent reasoning framework from Princeton/UIUC/Stanford (arXiv:2511.20639). The upstream paper repo is at `https://github.com/Gen-Verse/LatentMAS`. The local repo is a fork/re-implementation at `github.com/ShlokNangia24/latent-agents`.

**Core idea:** Instead of having LLM agents communicate by generating text that the next agent reads, LatentMAS agents "think" by feeding their own hidden states back through the model repeatedly (latent steps), accumulating a KV-cache. Only the final agent decodes text. This yields 70-83% fewer tokens and 4-7x faster inference vs text-based multi-agent, with higher accuracy.

**The two MAS topologies from the paper:**
- **Sequential**: planner → critic → refiner → solver (last agent generates text)
- **Hierarchical**: domain experts (math, science, code) run in parallel → summarizer generates text

---

## Repository Layout

```
latent_agents/          # Core library (install with: pip install -e .)
  __init__.py           # Public API: Agent, LatentModel, LatentPipeline, PipelineResult, LatentRealigner, set_seed, auto_device
  agent.py              # Agent dataclass (pure config)
  model.py              # LatentModel: wraps HuggingFace causal LM
  pipeline.py           # LatentPipeline: orchestrates agents + KV-cache flow
  realigner.py          # LatentRealigner: training-free projection matrix
  utils.py              # set_seed(), auto_device()
example.py              # 4-agent demo (planner→critic→refiner→solver)
tests/
  conftest.py           # pytest plugin — adds --run-integration flag
  test_latent_agents.py # Unit + integration tests
  test_benchmark.py     # Custom accuracy/speed benchmarks
  test_standard_benchmarks.py # GSM8K / MMLU / ARC benchmarks
run_quick_benchmark.py  # Quick CPU benchmark (5 samples, M4/laptop-friendly)
run_gpu_benchmarks.py   # Full GPU benchmark suite (Qwen3, Llama, Mistral, etc.)
GPU_SETUP.md            # Vast.ai / A100 setup guide
pyproject.toml          # Package metadata, deps
```

---

## How the System Works (READ THIS FIRST)

### Data Flow Through the Pipeline

```
User question(s)
     │
     ▼
LatentPipeline.run_batch(questions)
     │
     │  ┌─────────────────────────────────────────────┐
     │  │  For each NON-FINAL agent (latent thinkers): │
     │  │  1. agent.prompt_fn(question, context)       │
     │  │     → List[Dict] (chat messages)             │
     │  │  2. model.prepare_chat_batch(messages)       │
     │  │     → input_ids, attention_mask              │
     │  │  3. model.generate_latent_batch(             │
     │  │        input_ids, past_key_values=kv_cache)  │
     │  │     → updated KV-cache (NO TEXT PRODUCED)    │
     │  │  4. Optionally: truncate_kv_cache()          │
     │  └─────────────────────────────────────────────┘
     │
     │  ┌─────────────────────────────────────────────┐
     │  │  For the FINAL agent (generates text):       │
     │  │  1. agent.prompt_fn(question, context)       │
     │  │  2. model.prepare_chat_batch(messages)       │
     │  │     → input_ids, attention_mask              │
     │  │  3. model.generate_text_batch(               │
     │  │        input_ids, past_key_values=kv_cache)  │
     │  │     → list of text strings                   │
     │  └─────────────────────────────────────────────┘
     │
     ▼
List[PipelineResult]  →  .text (str), .agent_traces (list of dicts)
```

### What Latent Generation Actually Does (`generate_latent_batch`)

Each non-final agent runs `latent_steps` iterations:
1. Forward pass on `input_ids` → get last-layer hidden state `h` (shape: `[B, D]`)
2. **Realign**: project `h` back to input-embedding space via `W_a = (W_out^T W_out + λI)^{-1} W_out^T W_in`
3. Unsqueeze to `[B, 1, D]` — treat as a synthetic "token" embedding
4. Feed as `inputs_embeds` (not `input_ids`) to model with current `past_key_values`
5. Extends KV-cache by 1 token per step, no decoding

After `latent_steps` iterations, the KV-cache holds the agent's "thought" as accumulated key-value pairs across all transformer layers. This cache is passed to the next agent.

### What the Realigner Does

The realigner is a single matrix `W_a ∈ ℝ^{d_h × d_h}` computed once from model weights:

```
W_a = (W_out^T W_out + λI)^{-1} W_out^T W_in
```

It projects output hidden states back into input-embedding space so latent tokens don't cause out-of-distribution activations. It is **training-free** and computed in `__init__`. The paper shows this alignment is critical: removing it causes 2-5% accuracy drops.

---

## Key Classes and Their Contracts

### `Agent` (agent.py)

```python
@dataclass
class Agent:
    name: str          # Display name, e.g. "Planner"
    role: str          # Role identifier, e.g. "planner"
    prompt_fn: Callable[[str, str], List[Dict[str, str]]]
                       # (question, context) -> chat messages
    is_final: bool = False                        # True for exactly ONE agent per pipeline
    latent_steps: Optional[int] = None            # Per-agent override (None = use pipeline default)
    convergence_threshold: Optional[float] = None # Early stopping threshold (None = no early stop)
```

**Contract: exactly one agent must have `is_final=True`.** The pipeline enforces this at init time with a `ValueError`. Non-final agents produce no text — their `agent_traces[i]["output"]` will be `""`.

**Important:** The `context` parameter in `prompt_fn` is the literal string passed to `pipeline.run(question, context=...)`. For latent agents it is always `""` unless the caller passes something. It does NOT auto-inject prior agent outputs — that communication happens through the KV-cache.

### `LatentModel` (model.py)

```python
model = LatentModel(
    model_name_or_path="Qwen/Qwen3-8B",
    device="cuda",        # or "cpu", "mps" (MPS needs torch_dtype=torch.float32)
    realign=True,         # Set False to ablate realignment
    torch_dtype=None,     # Auto: bfloat16 on CUDA, float32 on CPU
)
```

Key methods:
- `render_chat(messages)` → `str` — applies chat template; falls back to `<|role|>...<|/role|>` if model has no template (WARNING: this fallback format was not in any model's training data)
- `prepare_chat_batch(batch_messages)` → `(prompts, input_ids, attention_mask)`
- `generate_text_batch(input_ids, ..., past_key_values=kv)` → `(List[str], kv_cache)`
- `generate_latent_batch(input_ids, ..., past_key_values=kv, convergence_threshold=None)` → `(kv_cache, actual_steps)` — returns actual steps taken (may be less than `latent_steps` if convergence triggered)

### `LatentPipeline` (pipeline.py)

```python
pipeline = LatentPipeline(
    model=model,
    agents=agents,
    latent_steps=20,                    # Default per non-final agent (agents can override)
    max_new_tokens=2048,
    temperature=0.6,
    top_p=0.95,
    keep_only_latent=False,             # True: drop prompt tokens from KV-cache
    convergence_threshold=None,         # Early stopping: stop when hidden-state delta < threshold
    n_samples=1,                        # Self-consistency voting: generate N answers, majority-vote
    vote_fn=None,                       # Custom voting function (default: _majority_vote)
)
result = pipeline.run("What is 24 * 37?")
results = pipeline.run_batch(["Q1", "Q2"])
```

### `LatentRealigner` (realigner.py)

Created automatically inside `LatentModel`. Rarely used directly. If you do:

```python
realigner = LatentRealigner(model=hf_model, device=device, enabled=True, reg=1e-5)
realigned = realigner.apply(hidden_state)  # [B, D] or [B, L, D] → same shape
```

`apply()` always converts to float32 for stability, then back to the original dtype.

---

## Things That Will Trip You Up (COMMON MISTAKES)

### 1. Final agent ordering is NOT enforced

The pipeline validates that exactly one agent has `is_final=True`, but **does not enforce it must be the last agent**. If you put `is_final=True` on agent index 1 of 4, agents 2 and 3 will run latent steps whose KV-cache results are silently discarded. Always put the final agent last.

### 2. Temperature defaults are inconsistent

- `LatentPipeline` default: `temperature=0.6`
- `LatentModel.generate_text_batch` default: `temperature=0.7`

If you call `model.generate_text_batch()` directly (bypassing the pipeline), you get 0.7. Through the pipeline you get 0.6. Always pass `temperature` explicitly when calling the model directly.

### 3. The KV-cache supplements, not replaces, the final prompt

When the final agent generates text, the model sees: `[accumulated KV-cache from all latent agents] + [final agent's prompt tokens]`. The final agent's prompt is tokenized and appended *after* the latent cache. Do not write final-agent prompts that assume the agent starts from a blank slate.

### 4. Chat template fallback is dangerous

If a model has no `chat_template`, `render_chat` falls back to `<|role|>content<|/role|>` format. No model was trained on this. If you load a base model (not an instruct model), outputs will be garbage and there is **no warning**. Only use instruct variants.

### 5. `bfloat16` on old GPUs

The code automatically uses `bfloat16` on CUDA without checking `torch.cuda.is_bf16_supported()`. On V100, T4, and RTX 20-series GPUs, bfloat16 ops silently fall back to float32, meaning no memory savings. Check hardware before assuming dtype.

### 6. MPS (Apple Silicon) is unsupported by default

`auto_device()` returns `"cpu"` when CUDA is unavailable, even on M1/M2/M3 Macs. To use MPS:
```python
model = LatentModel("...", device="mps", torch_dtype=torch.float32)
```
MPS requires float32 (bfloat16 has limited MPS support). This is not documented in the README.

### 7. `transformers>=4.36` minimum is too low

The code passes `cache_position` to `model.generate()` (model.py). This argument was added in transformers ~4.38-4.40. With 4.36, some models may raise unexpected keyword argument errors or silently ignore it. If you see cache-related errors, upgrade to `transformers>=4.40`.

### 8. KV-cache memory blows up silently

With 3 latent agents × 20 steps = 60 latent tokens + all prompt tokens per layer. For a 70B model with 80 layers, this can be gigabytes. The code has no memory guard. Use `keep_only_latent=True` if you hit OOM. Note that `keep_only_latent=True` strips the prompt context from the KV-cache — the final agent only receives the latent thoughts, not the prompts that generated them.

### 9. Benchmark code is massively duplicated

`RunStats`, `count_tokens`, `print_table`, `run_text_pipeline`, `run_latent_pipeline`, and all prompt functions (planner, critic, refiner, verifier) are copy-pasted across `test_benchmark.py`, `test_standard_benchmarks.py`, and `run_gpu_benchmarks.py`. Changes in one do NOT propagate. If you fix a bug in a prompt or runner, update all three files.

### 10. `keep_only_latent` keeps only latent steps

When `keep_only_latent=True`, the KV-cache is truncated to keep only the `actual_steps` latent tokens from each agent, discarding all prompt tokens. If `actual_steps` is small (e.g., due to early convergence), the final agent may receive very little context. The final agent still receives its own prompt tokens normally.

---

## How to Add Things

### Adding a New Agent Role

Define a `prompt_fn` and create an `Agent`. No base class to extend:

```python
def my_agent_prompt(question: str, context: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a ..."},
        {"role": "user", "content": f"Question: {question}\n\nThink carefully about..."},
    ]

agent = Agent(name="MyAgent", role="my_agent", prompt_fn=my_agent_prompt, is_final=False)
```

Put the final agent **last** in the list. Only the final agent gets `is_final=True`.

### Adding a New Model

Works automatically if the model is `AutoModelForCausalLM`-compatible with:
- `get_input_embeddings()` method
- `get_output_embeddings()` method
- `inputs_embeds` parameter support in `forward()`
- `use_cache=True` support

Things to verify:
- Use an **instruct** variant, not a base model
- Check the model's chat template exists (`tokenizer.chat_template is not None`)
- For models with tied input/output embeddings (W_in == W_out), realignment degenerates to near-identity — this is correct but realignment adds no value; run with `realign=False` for ablation

### Adding a New Benchmark

Follow the pattern in `test_standard_benchmarks.py`:
1. Write `run_single_agent_pipeline()`, `run_text_pipeline()`, `run_latent_pipeline()`
2. Define answer extraction (be careful with `check_answer` substring matching — "180" matches "1800")
3. Use `@pytest.fixture(scope="module")` to load the dataset once
4. Mark with `@pytest.mark.integration`
5. **Add to `run_gpu_benchmarks.py` as well** (and `test_benchmark.py` if relevant)

### Per-Agent Latent Steps

Supported via `Agent(latent_steps=N)`. Each agent can override the pipeline default. The pipeline resolves: `agent.latent_steps if agent.latent_steps is not None else self.latent_steps`. Same pattern applies to `convergence_threshold`.

### Self-Consistency Voting

Set `n_samples > 1` on `LatentPipeline` to generate multiple answers from the same KV-cache and majority-vote the best. The default `_majority_vote` extracts answers via `\boxed{}`, `####`, "the answer is", or last-line fallback. Provide a custom `vote_fn: (List[str]) -> str` if needed.

**Critical:** HuggingFace `model.generate()` mutates `past_key_values` in-place. The pipeline handles this with `copy.deepcopy()` for samples after the first.

---

## Running Tests

```bash
# Fast unit tests (no model download, runs in seconds)
pytest tests/test_latent_agents.py -v

# Integration tests (downloads SmolLM2-135M ~270MB, takes 5-10 min on CPU)
pytest tests/test_latent_agents.py -v --run-integration

# Standard benchmarks (GSM8K / MMLU / ARC, CPU, slow)
pytest tests/test_standard_benchmarks.py -v -s --run-integration

# Custom benchmark comparison (3 pipeline types)
pytest tests/test_benchmark.py -v -s --run-integration

# Specific benchmark
pytest tests/test_standard_benchmarks.py -v -s --run-integration -k gsm8k

# Quick benchmark on CPU/laptop (5 samples, ~3 min on M4 Pro)
conda run -n latent-agents python run_quick_benchmark.py
```

### GPU Benchmark Suite

```bash
# Sanity check first (~2 min)
python run_gpu_benchmarks.py --models "Qwen/Qwen3-1.7B" --n-gsm8k 5 --n-mmlu 5 --n-arc 5

# Full run (3-4 hours on A100/B200)
python run_gpu_benchmarks.py

# Skip gated models (no HF_TOKEN needed — runs Qwen3-1.7B, Qwen3-8B, Qwen3-32B)
python run_gpu_benchmarks.py --skip-gated

# Custom latent steps (default: 10)
python run_gpu_benchmarks.py --latent-steps 20

# Regenerate charts from saved results
python run_gpu_benchmarks.py --charts-only benchmark_results.json
```

For gated models (Llama, Mistral, Gemma):
```bash
export HF_TOKEN=hf_your_token_here
huggingface-cli login --token $HF_TOKEN
# Accept model licenses on the HuggingFace website first
```

### Recommended GPUs (vast.ai / Lambda / RunPod)

| GPU | VRAM | What it runs | Rough cost | Verdict |
|-----|------|-------------|-----------|---------|
| **A100 80GB** | 80 GB | All models up to Qwen3-32B | ~$2-3/hr | **Best value** |
| **H100 80GB** | 80 GB | Same as A100, ~2x faster | ~$4-5/hr | Worth it for full suite |
| **B200 180GB** | 180 GB | Everything including Llama-70B | ~$8-10/hr | Only if you need 70B |
| RTX 4090 | 24 GB | Qwen3-1.7B, Qwen3-8B only | ~$0.5/hr | Budget option, limited |
| A10G | 24 GB | Same as 4090 | ~$0.6/hr | Skip it |

**Recommendation:** Rent an **A100 80GB** on vast.ai. Gets you all non-gated models (1.7B, 8B, 32B) for the full 500 questions each. The full suite takes ~2 hours. Cost: ~$5 total.

Setup on the rented machine:
```bash
git clone https://github.com/ShlokNangia24/latent-agents.git
cd latent-agents
pip install -e ".[dev,bench]"

# Sanity check
python run_gpu_benchmarks.py --models "Qwen/Qwen3-1.7B" --n-gsm8k 5 --n-mmlu 5 --n-arc 5

# Full non-gated run in background
nohup python run_gpu_benchmarks.py --skip-gated > run.log 2>&1 &
tail -f run.log

# Copy results back to Mac when done
scp -P <PORT> root@<HOST>:~/latent-agents/benchmark_results.json .
scp -rP <PORT> root@<HOST>:~/latent-agents/charts/ ./charts/
```

---

## Test Coverage Gaps (Things That Break Silently)

These paths have no test coverage. Be careful when modifying them:

1. **`cache_position` computation** (model.py ~line 170-176) — critical for correct generation with pre-filled KV caches
2. **Attention mask extension in `generate_latent_batch`** (model.py ~line 229-236)
3. **`truncate_kv_cache` with `Cache` objects** — the `transformers.Cache` branch is conditional import, completely untested
4. **Multi-agent KV-cache correctness** — mocks verify methods are called but don't verify the cache actually improves output
5. **`realign=False` through full pipeline** — only tested in isolation
6. **Batch size > 1 KV-cache shape compatibility** — integration test checks output is non-empty, not that shapes are correct

### Recently Fixed Bugs (for context)

- **`past_kv_length` now supports `DynamicCache`** — uses `get_seq_length()` when available, falls back to legacy tuple `[0][0].shape[-2]`. Tested with real `DynamicCache` objects.
- **Left-padding enabled** — `tokenizer.padding_side = "left"` set in `LatentModel.__init__`. Fixes batch > 1 inference where hidden states were extracted from pad tokens.
- **`keep_only_latent` now keeps only latent steps** — previously kept `prompt_len + actual_steps` tokens; now keeps only `actual_steps` as documented.
- **`check_answer` in `run_quick_benchmark.py`** — removed bidirectional substring match (`gen in exp`) that caused false positives (e.g., `"4"` matching `"40"`).
- **`--latent-steps` CLI arg in `run_gpu_benchmarks.py`** — was parsed but never passed to `benchmark_one_model`, so the pipeline always used `latent_steps=10` regardless of the flag. Now correctly wired through.

---

## Configuration Reference

| Parameter | Default | Where | Notes |
|-----------|---------|-------|-------|
| `latent_steps` | 20 | Pipeline / Agent | Pipeline default; agents can override via `Agent(latent_steps=N)` |
| `convergence_threshold` | None | Pipeline / Agent | Early stop when hidden-state delta < threshold; agents can override |
| `n_samples` | 1 | Pipeline | Self-consistency voting: generate N answers, majority-vote the best |
| `vote_fn` | None | Pipeline | Custom `(List[str]) -> str`; defaults to `_majority_vote` |
| `max_new_tokens` | 2048 | Pipeline | For final agent only |
| `temperature` | 0.6 | Pipeline | ⚠️ Model default is 0.7 — different |
| `top_p` | 0.95 | Pipeline | |
| `keep_only_latent` | False | Pipeline | Saves memory; drops textual context from cache |
| `realign` | True | LatentModel | Set False to ablate; expect 2-5% accuracy drop |
| `reg` | 1e-5 | Realigner | Tikhonov regularization for W_a; tune if realignment is unstable |
| `torch_dtype` | None (auto) | LatentModel | bfloat16 on CUDA, float32 on CPU/MPS |

---

## Dependencies

```toml
python = ">=3.9"
torch = ">=2.0"
transformers = ">=4.36"   # NOTE: >=4.40 recommended for cache_position support
numpy = "*"
accelerate = "*"

# Dev
pytest = "*"
ruff = "*"

# Benchmarks
datasets = "*"
matplotlib = "*"
```

Install:
```bash
pip install -e ".[dev,bench]"
```

---

## Paper Reference

**"Latent Collaboration in Multi-Agent Systems"**
Zou et al., arXiv:2511.20639v2 (Dec 2025)
Princeton / UIUC / Stanford

Key empirical results:
- +14.6% avg accuracy over single model (sequential MAS)
- 70.8-83.7% fewer output tokens vs text-based MAS
- 4-7x faster end-to-end inference
- Tested on Qwen3-4B/8B/14B across 9 benchmarks
- Optimal latent steps: 20-40 (accuracy peaks around 40-80, then plateaus/degrades)

```bibtex
@article{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Zou, Jiaru and Yang, Xiyuan and Qiu, Ruizhong and Li, Gaotang and Tieu, Katherine and Lu, Pan and Shen, Ke and Tong, Hanghang and Choi, Yejin and He, Jingrui and Zou, James and Wang, Mengdi and Yang, Ling},
  journal={arXiv preprint arXiv:2511.20639},
  year={2025}
}
```

---

## Updating This File

**When to update CLAUDE.md:**
- After fixing a bug: add it to the relevant "Things That Will Trip You Up" section
- After adding a feature: update configuration reference and how-to sections
- After resolving a common error (import errors, CUDA errors, tokenizer errors): add the error + fix
- After modifying a public API signature: update the class contracts section
- After discovering a test coverage gap: add to the coverage gaps section

**Do not** use CLAUDE.md to track in-progress work, PR lists, or session-specific notes. Use git history and commit messages for that.
