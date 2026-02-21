# Running Benchmarks on vast.ai B200

## Step 1: SSH into your machine

```bash
ssh -p <PORT> root@<HOST_IP>
# e.g. ssh -p 54587 root@host68137
```

## Step 2: Clone the repo and install

```bash
git clone https://github.com/ShlokNangia24/latent-agents.git
cd latent-agents
pip install -e ".[dev,bench]"
```

## Step 3: Set HuggingFace token (needed for Llama, Mistral, Gemma)

Go to https://huggingface.co/settings/tokens and create a token, then:

```bash
export HF_TOKEN=hf_your_token_here
huggingface-cli login --token $HF_TOKEN
```

You also need to accept the license for each gated model:
- https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- https://huggingface.co/google/gemma-2-9b-it

## Step 4: Run the full benchmark

```bash
# Run ALL models (takes ~3-4 hours on B200)
python run_gpu_benchmarks.py

# Or skip gated models if you don't want to set up HF token
python run_gpu_benchmarks.py --skip-gated

# Or run specific models only
python run_gpu_benchmarks.py --models "Qwen/Qwen3-8B,Qwen/Qwen3-32B"

# Quick test run first (fewer samples, ~30 min total)
python run_gpu_benchmarks.py --n-gsm8k 20 --n-mmlu 30 --n-arc 30

# Run in background with logging
nohup python run_gpu_benchmarks.py > benchmark_log.txt 2>&1 &
tail -f benchmark_log.txt
```

## Step 5: Get results

Results are saved to `benchmark_results.json` and charts are auto-generated in the `charts/` directory.

```bash
# On your local Mac -- copy everything:
scp -P <PORT> root@<HOST_IP>:~/latent-agents/benchmark_results.json .
scp -rP <PORT> root@<HOST_IP>:~/latent-agents/charts/ ./charts/

# If you want to regenerate charts from an existing results file:
python run_gpu_benchmarks.py --charts-only benchmark_results.json
# Optionally change the output directory:
python run_gpu_benchmarks.py --charts-only benchmark_results.json --charts-dir my_charts/
```

### Charts generated automatically

| Chart | File | What it shows |
|-------|------|---------------|
| Accuracy by benchmark | `accuracy_gsm8k.png`, `accuracy_mmlu.png`, `accuracy_arc.png` | Grouped bar: all 3 pipelines per model |
| Overall accuracy | `accuracy_overall.png` | Grouped bar: combined accuracy across benchmarks |
| Token generation | `tokens_generated.png` | Total tokens produced per pipeline per model |
| Token savings | `token_savings.png` | % fewer tokens Latent uses vs Non-Latent |
| Speedup | `speedup.png` | Wall-clock speedup (Latent vs Non-Latent) |
| Wall time | `wall_time.png` | Total inference time per pipeline |
| Scaling curve | `accuracy_vs_size.png` | Accuracy vs model parameters (log scale) |
| Radar charts | `radar_*.png` | Per-pipeline accuracy profile across benchmarks |
| Accuracy delta heatmap | `accuracy_delta_heatmap.png` | Latent minus Non-Latent accuracy per model x benchmark |

## Models tested

| Model | Size | VRAM | Gated? | Notes |
|-------|------|------|--------|-------|
| Qwen3-1.7B | 1.7B | ~3.4 GB | No | Small baseline |
| Llama-3.1-8B-Instruct | 8B | ~16 GB | Yes | Standard reference |
| Qwen3-8B | 8B | ~16 GB | No | Top OSS model |
| Mistral-7B-Instruct-v0.3 | 7B | ~14 GB | Yes | Sliding window attention |
| Gemma-2-9b-it | 9B | ~18 GB | Yes | Google family |
| Qwen3-32B | 32B | ~64 GB | No | Large OSS |
| Llama-3.1-70B-Instruct | 70B | ~140 GB | Yes | Largest test |

## Benchmarks

| Benchmark | Questions | What it tests |
|-----------|-----------|---------------|
| GSM8K | 100 | Grade school math reasoning |
| MMLU | 200 | General knowledge (57 subjects) |
| ARC-Easy | 200 | Science reasoning (grade school) |

## Expected runtime on B200 (179GB)

| Model | Approx time |
|-------|------------|
| Qwen3-1.7B | ~10 min |
| 8B models (x3) | ~20 min each |
| Qwen3-32B | ~40 min |
| Llama-3.1-70B | ~90 min |
| **Total** | **~3-4 hours** |

## Quick test first

Before running everything, do a quick sanity check:

```bash
python run_gpu_benchmarks.py --models "Qwen/Qwen3-1.7B" --n-gsm8k 5 --n-mmlu 5 --n-arc 5
```

This should finish in ~2 minutes and verify everything works.
