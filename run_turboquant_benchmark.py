#!/usr/bin/env python3
"""TurboQuant memory and speed benchmark.

Measures the before/after impact of TurboQuant KV-cache quantization:
  - Peak GPU/CPU memory usage
  - Compression ratio (actual bytes)
  - Compress + decompress latency
  - Round-trip cosine similarity (fidelity)

Usage::

    # CPU benchmark (no GPU required)
    python run_turboquant_benchmark.py

    # GPU benchmark with a real model
    python run_turboquant_benchmark.py --model Qwen/Qwen3-8B --device cuda

    # Vary latent steps to see scaling
    python run_turboquant_benchmark.py --latent-steps 40

    # Vary bits
    python run_turboquant_benchmark.py --bits 4 --qjl-dim 128
"""

from __future__ import annotations

import argparse
import time
import sys
from typing import Optional

import torch
import torch.nn.functional as F

from latent_agents.turboquant import TurboQuant, TurboQuantConfig


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _mem_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    return 0  # CPU memory tracking via psutil would need extra dep


def _tensor_bytes(kv_cache: tuple) -> int:
    total = 0
    for layer in kv_cache:
        for t in layer:
            total += t.nelement() * t.element_size()
    return total


def _compressed_bytes(compressed) -> int:
    total = 0
    for layer in compressed.layers:
        for field in ("k_quant", "k_radius", "k_qjl", "k_qjl_scale",
                      "v_quant", "v_radius", "v_qjl", "v_qjl_scale"):
            t = getattr(layer, field)
            total += t.nelement() * t.element_size()
    return total


def _cosine_similarity_mean(orig_kv: tuple, recon_kv: tuple) -> float:
    cos_sims = []
    for (ok, ov), (rk, rv) in zip(orig_kv, recon_kv):
        for o, r in [(ok, rk), (ov, rv)]:
            o_flat = o.float().reshape(-1, o.shape[-1])
            r_flat = r.float().reshape(-1, r.shape[-1])
            cos = F.cosine_similarity(o_flat, r_flat, dim=-1)
            cos_sims.append(cos.mean().item())
    return sum(cos_sims) / len(cos_sims)


def make_fake_kv_cache(
    n_layers: int,
    batch: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple:
    layers = []
    for _ in range(n_layers):
        k = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
        layers.append((k, v))
    return tuple(layers)


# ---------------------------------------------------------------------------
# Synthetic benchmark (no model needed)
# ---------------------------------------------------------------------------

def run_synthetic_benchmark(
    bits: int,
    qjl_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    latent_steps: int,
    n_agents: int,
    warmup: int = 3,
    repeats: int = 20,
) -> None:
    print(f"\n{'='*65}")
    print(f"  TurboQuant Synthetic Benchmark")
    print(f"  device={device}  dtype={dtype}  bits={bits}  qjl_dim={qjl_dim}")
    print(f"{'='*65}\n")

    # --- Model configurations to benchmark ---------------------------------
    configs = [
        # (label, n_layers, n_heads, head_dim)  — representative of real models
        ("Qwen3-1.7B (24L,16H,D=128)", 24, 16, 128),
        ("Qwen3-8B  (32L,32H,D=128)", 32, 32, 128),
        ("Qwen3-32B (64L,64H,D=128)", 64, 64, 128),
    ]

    tq = TurboQuant(TurboQuantConfig(bits=bits, qjl_dim=qjl_dim), device, dtype)

    total_latent_tokens = n_agents * latent_steps  # total KV tokens accumulated

    for label, n_layers, n_heads, head_dim in configs:
        kv = make_fake_kv_cache(
            n_layers=n_layers,
            batch=1,
            n_heads=n_heads,
            seq_len=total_latent_tokens,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        # Warmup
        for _ in range(warmup):
            compressed = tq.compress_kv(kv)
            _ = tq.decompress_kv(compressed)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Compress timing
        t0 = time.perf_counter()
        for _ in range(repeats):
            compressed = tq.compress_kv(kv)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compress_ms = (time.perf_counter() - t0) / repeats * 1000

        # Decompress timing
        t0 = time.perf_counter()
        for _ in range(repeats):
            reconstructed = tq.decompress_kv(compressed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decompress_ms = (time.perf_counter() - t0) / repeats * 1000

        # Memory and fidelity
        orig_bytes = _tensor_bytes(kv)
        comp_bytes = _compressed_bytes(compressed)
        ratio = orig_bytes / max(comp_bytes, 1)
        cos_sim = _cosine_similarity_mean(kv, reconstructed)

        orig_mb = orig_bytes / 1e6
        comp_mb = comp_bytes / 1e6
        saved_mb = orig_mb - comp_mb

        print(f"  Model : {label}")
        print(f"    Latent tokens : {total_latent_tokens} ({n_agents} agents × {latent_steps} steps)")
        print(f"    KV original   : {orig_mb:7.1f} MB")
        print(f"    KV compressed : {comp_mb:7.1f} MB  (saves {saved_mb:.1f} MB)")
        print(f"    Ratio         : {ratio:.2f}x")
        print(f"    Compress      : {compress_ms:.2f} ms")
        print(f"    Decompress    : {decompress_ms:.2f} ms")
        print(f"    Cosine sim    : {cos_sim:.4f}")
        print()


# ---------------------------------------------------------------------------
# Real-model pipeline benchmark (optional)
# ---------------------------------------------------------------------------

def run_pipeline_benchmark(
    model_name: str,
    device: torch.device,
    bits: int,
    qjl_dim: int,
    latent_steps: int,
) -> None:
    print(f"\n{'='*65}")
    print(f"  TurboQuant Pipeline Benchmark  (model={model_name})")
    print(f"{'='*65}\n")

    try:
        from latent_agents import LatentModel, LatentPipeline, Agent, TurboQuant, TurboQuantConfig
    except ImportError as e:
        print(f"  ERROR: {e}")
        return

    print(f"  Loading {model_name} ...")
    model = LatentModel(model_name, device=str(device))
    model_dtype = next(model.model.parameters()).dtype
    print(f"  Loaded. dtype={model_dtype}\n")

    tq = TurboQuant(TurboQuantConfig(bits=bits, qjl_dim=qjl_dim), device, model_dtype)

    def planner_fn(q, c):
        return [{"role": "system", "content": "You are a careful planner."},
                {"role": "user", "content": f"Make a plan for: {q}"}]

    def critic_fn(q, c):
        return [{"role": "system", "content": "You review plans critically."},
                {"role": "user", "content": f"Critique the plan for: {q}"}]

    def solver_fn(q, c):
        return [{"role": "system", "content": "You are a concise solver."},
                {"role": "user", "content": f"Solve: {q}"}]

    agents = [
        Agent("Planner", "planner", planner_fn),
        Agent("Critic",  "critic",  critic_fn),
        Agent("Solver",  "solver",  solver_fn, is_final=True),
    ]

    q = "What is 347 * 28?"

    def mem_mb() -> float:
        return torch.cuda.memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0

    # --- Without TurboQuant ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    pipe = LatentPipeline(model, agents, latent_steps=latent_steps, max_new_tokens=128)
    t0 = time.perf_counter()
    result = pipe.run(q)
    elapsed_no_tq = time.perf_counter() - t0
    peak_no_tq = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0

    print(f"  Without TurboQuant:")
    print(f"    Answer  : {result.text[:80].strip()}")
    print(f"    Time    : {elapsed_no_tq:.2f}s")
    if device.type == "cuda":
        print(f"    Peak GPU: {peak_no_tq:.1f} MB")
    print()

    # --- With TurboQuant ---
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    pipe_tq = LatentPipeline(model, agents, latent_steps=latent_steps, max_new_tokens=128,
                              turbo_quant=tq)
    t0 = time.perf_counter()
    result_tq = pipe_tq.run(q)
    elapsed_tq = time.perf_counter() - t0
    peak_tq = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0

    print(f"  With TurboQuant (bits={bits}, qjl_dim={qjl_dim}):")
    print(f"    Answer  : {result_tq.text[:80].strip()}")
    print(f"    Time    : {elapsed_tq:.2f}s")
    if device.type == "cuda":
        print(f"    Peak GPU: {peak_tq:.1f} MB")
    print()

    if device.type == "cuda":
        mem_savings = peak_no_tq - peak_tq
        print(f"  Memory saved by TurboQuant: {mem_savings:.1f} MB")
    print(f"  Overhead from compress/decompress: {(elapsed_tq - elapsed_no_tq)*1000:.0f} ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant benchmark")
    parser.add_argument("--model", default=None,
                        help="HuggingFace model name for pipeline benchmark (optional)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--qjl-dim", type=int, default=64)
    parser.add_argument("--latent-steps", type=int, default=20)
    parser.add_argument("--n-agents", type=int, default=3,
                        help="Number of non-final latent agents (synthetic benchmark)")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    run_synthetic_benchmark(
        bits=args.bits,
        qjl_dim=args.qjl_dim,
        device=device,
        dtype=dtype,
        latent_steps=args.latent_steps,
        n_agents=args.n_agents,
    )

    if args.model:
        run_pipeline_benchmark(
            model_name=args.model,
            device=device,
            bits=args.bits,
            qjl_dim=args.qjl_dim,
            latent_steps=args.latent_steps,
        )


if __name__ == "__main__":
    main()
