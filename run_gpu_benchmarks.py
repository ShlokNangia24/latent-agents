#!/usr/bin/env python3
"""
Full GPU benchmark: Non-Latent vs Latent multi-agent pipelines
across multiple models and standard LLM benchmarks.

4-Agent pipeline:
  1. PLANNER  (Plan-and-Solve)    – plan + acceptance checks
  2. SOLVER   (Self-consistency)  – 3 candidate solutions, pick best
  3. CRITIC   (Self-Refine)       – tighten / fix the chosen solution
  4. VERIFIER (Chain-of-Verification) – verify claims, output final

Models tested (ordered by size):
  - Qwen/Qwen3-1.7B                    (~3.4 GB)
  - meta-llama/Llama-3.1-8B-Instruct   (~16 GB, gated)
  - Qwen/Qwen3-8B                      (~16 GB)
  - mistralai/Mistral-7B-Instruct-v0.3 (~14 GB)
  - google/gemma-2-9b-it               (~18 GB)
  - Qwen/Qwen3-32B                     (~64 GB)
  - meta-llama/Llama-3.1-70B-Instruct  (~140 GB, gated)

Benchmarks:
  - GSM8K   (100 questions) -- math reasoning
  - MMLU    (200 questions) -- multitask knowledge
  - ARC-Easy(200 questions) -- science reasoning

Usage:
  # Run everything:
  python run_gpu_benchmarks.py

  # Run a single model:
  python run_gpu_benchmarks.py --models "Qwen/Qwen3-8B"

  # Run fewer samples (faster):
  python run_gpu_benchmarks.py --n-gsm8k 30 --n-mmlu 50 --n-arc 50

  # Skip gated models (no HF_TOKEN needed):
  python run_gpu_benchmarks.py --skip-gated

  # Regenerate charts from previous results:
  python run_gpu_benchmarks.py --charts-only benchmark_results.json
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import matplotlib
matplotlib.use("Agg")  # headless rendering for servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from datasets import load_dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
from latent_agents import Agent, LatentModel, LatentPipeline, set_seed


# ===========================================================================
# Configuration
# ===========================================================================

ALL_MODELS = [
    # (hf_id, display_name, is_gated, approx_gb)
    ("Qwen/Qwen3-1.7B",                    "Qwen3-1.7B",       False, 3.4),
    ("meta-llama/Llama-3.1-8B-Instruct",   "Llama-3.1-8B",     True,  16),
    ("Qwen/Qwen3-8B",                      "Qwen3-8B",         False, 16),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B",       True,  14),
    ("google/gemma-2-9b-it",               "Gemma-2-9b",       True,  18),
    ("Qwen/Qwen3-32B",                     "Qwen3-32B",        False, 64),
    ("meta-llama/Llama-3.1-70B-Instruct",  "Llama-3.1-70B",    True,  140),
]

DEFAULT_N_GSM8K = 100
DEFAULT_N_MMLU = 200
DEFAULT_N_ARC = 200


# ===========================================================================
# Stats
# ===========================================================================

@dataclass
class RunStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    intermediate_tokens: int = 0
    wall_time_s: float = 0.0


def count_tokens(tokenizer, text: str) -> int:
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


# ===========================================================================
# Answer checking
# ===========================================================================

def extract_gsm8k_gt(answer_str: str) -> str:
    match = re.search(r"####\s*(.+)", answer_str)
    return match.group(1).strip().replace(",", "") if match else ""


def extract_number(text: str) -> str:
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip().replace(",", "")
    ans = re.search(r"(?:answer|result|total|equals?)\s*(?:is|=|:)\s*\$?\s*([-\d,]+(?:\.\d+)?)", text, re.I)
    if ans:
        return ans.group(1).strip().replace(",", "")
    nums = re.findall(r"([-]?\d[\d,]*(?:\.\d+)?)", text)
    return nums[-1].replace(",", "") if nums else ""


def check_gsm8k(output: str, gt: str) -> bool:
    expected = extract_gsm8k_gt(gt)
    extracted = extract_number(output)
    if expected and extracted:
        try:
            return float(extracted) == float(expected)
        except ValueError:
            pass
    return expected in output if expected else False


def check_mcq(output: str, correct_idx: int, choices: List[str]) -> bool:
    labels = ["A", "B", "C", "D"]
    correct_letter = labels[correct_idx] if correct_idx < len(labels) else ""
    text = output.strip().upper()
    if text and text[0] == correct_letter:
        return True
    pat = re.search(r"(?:answer|correct)\s*(?:is|:)\s*\(?([A-D])\)?", output, re.I)
    if pat and pat.group(1).upper() == correct_letter:
        return True
    if correct_idx < len(choices) and choices[correct_idx].lower() in output.lower():
        return True
    return False


def check_arc(output: str, correct_key: str, choices: Dict) -> bool:
    labels = choices["label"]
    texts = choices["text"]
    text = output.strip().upper()
    if text and text[0] == correct_key.upper():
        return True
    pat = re.search(r"(?:answer|correct)\s*(?:is|:)\s*\(?([A-D])\)?", output, re.I)
    if pat and pat.group(1).upper() == correct_key.upper():
        return True
    idx = labels.index(correct_key) if correct_key in labels else -1
    if idx >= 0 and texts[idx].lower() in output.lower():
        return True
    return False


# ===========================================================================
# Prompts  –  4-agent pipeline: Planner → Solver → Critic → Verifier
#
#   1. PLANNER  (Plan-and-Solve):   plan + acceptance checks
#   2. SOLVER   (Self-consistency): 3 candidate solutions, pick best
#   3. CRITIC   (Self-Refine):      tighten / fix the chosen solution
#   4. VERIFIER (CoVe):             verify claims, output final answer
# ===========================================================================

# ---- Single-agent baselines (unchanged) ----------------------------------

def single_gsm8k(q, ctx):
    return [
        {"role": "system", "content": (
            "You are an expert math tutor. Solve the problem step by step, showing "
            "all arithmetic clearly. After your reasoning, write the final numeric "
            "answer on the last line in the format: The answer is <NUMBER>."
        )},
        {"role": "user", "content": q},
    ]

def single_mcq(q, ctx):
    return [
        {"role": "system", "content": (
            "You are a knowledgeable assistant. Read the question and all options "
            "carefully. Briefly reason about each option, then clearly state the "
            "correct answer as a single letter (A, B, C, or D) on the last line."
        )},
        {"role": "user", "content": q},
    ]

# ---- Non-latent (text-based) 4-agent pipeline ----------------------------
# Each agent generates text that is passed explicitly to the next agent.

def planner_text(q, ctx):
    """Agent 1 – PLANNER (Plan-and-Solve). Generates plan + success checks."""
    return [
        {"role": "system", "content": (
            "You are PLANNER.\n"
            "Create a plan before solving.\n\n"
            "Output only:\n"
            "PLAN: 3-7 numbered steps, each with a concrete deliverable.\n"
            "CHECKS: bullet list of acceptance criteria and how to validate correctness."
        )},
        {"role": "user", "content": f"Task: {q}"},
    ]

def solver_text(q, prior):
    """Agent 2 – SOLVER (Self-consistency, N=3). 3 attempts then pick best."""
    return [
        {"role": "system", "content": (
            "You are SOLVER.\n"
            "Use the PLAN. Generate 3 independent solution attempts (different approaches).\n"
            "For each attempt output: Final answer + key steps only.\n"
            "Then select the best using consistency and the CHECKS.\n\n"
            "Output only:\n"
            "ATTEMPT 1:\n"
            "ATTEMPT 2:\n"
            "ATTEMPT 3:\n"
            "CHOSEN ANSWER: (with brief why)"
        )},
        {"role": "user", "content": (
            f"--- PLANNER output ---\n{prior}\n--- end ---\n\n"
            f"Task: {q}"
        )},
    ]

def critic_text(q, prior):
    """Agent 3 – CRITIC (Self-Refine). Fix / tighten the chosen answer."""
    return [
        {"role": "system", "content": (
            "You are CRITIC.\n"
            "Improve the CHOSEN ANSWER with minimal edits.\n\n"
            "Output only:\n"
            "TOP ISSUES (max 5): specific problems or gaps\n"
            "PATCH: the revised answer (clean, final-draft quality)"
        )},
        {"role": "user", "content": (
            f"--- Previous agents ---\n{prior}\n--- end ---\n\n"
            f"Task: {q}"
        )},
    ]

def verifier_text_gsm8k(q, prior):
    """Agent 4 – VERIFIER (CoVe) for math. Final agent, generates output."""
    return [
        {"role": "system", "content": (
            "You are VERIFIER using Chain-of-Verification. Treat PATCH as a draft.\n\n"
            "Do these steps strictly and keep them short:\n"
            "1. List claims that might be wrong.\n"
            "2. Write verification questions for those claims.\n"
            "3. Answer them independently (mark unknown if unsure).\n"
            "4. Produce a corrected final response.\n\n"
            "Output only:\n"
            "CLAIMS TO CHECK:\n"
            "VERIFICATION QUESTIONS:\n"
            "INDEPENDENT ANSWERS:\n"
            "FINAL VERIFIED OUTPUT:\n"
            "(The last line of FINAL VERIFIED OUTPUT must be: The answer is <NUMBER>)"
        )},
        {"role": "user", "content": (
            f"--- Previous agents ---\n{prior}\n--- end ---\n\n"
            f"Task: {q}"
        )},
    ]

def verifier_text_mcq(q, prior):
    """Agent 4 – VERIFIER (CoVe) for MCQ. Final agent, generates output."""
    return [
        {"role": "system", "content": (
            "You are VERIFIER using Chain-of-Verification. Treat PATCH as a draft.\n\n"
            "Do these steps strictly and keep them short:\n"
            "1. List claims that might be wrong.\n"
            "2. Write verification questions for those claims.\n"
            "3. Answer them independently (mark unknown if unsure).\n"
            "4. Produce a corrected final response.\n\n"
            "Output only:\n"
            "CLAIMS TO CHECK:\n"
            "VERIFICATION QUESTIONS:\n"
            "INDEPENDENT ANSWERS:\n"
            "FINAL VERIFIED OUTPUT:\n"
            "(The last line of FINAL VERIFIED OUTPUT must state the correct letter: A, B, C, or D)"
        )},
        {"role": "user", "content": (
            f"--- Previous agents ---\n{prior}\n--- end ---\n\n"
            f"Task: {q}"
        )},
    ]

# ---- Latent 4-agent pipeline (KV-cache communication) --------------------
# Only the Verifier (agent 4) produces visible text. Agents 1-3 think
# through the shared KV-cache.

def latent_planner(q, ctx):
    """Latent Agent 1 – PLANNER. Reasons internally via KV-cache."""
    return [
        {"role": "system", "content": (
            "You are PLANNER.\n"
            "Create a plan before solving.\n\n"
            "Think about:\n"
            "PLAN: 3-7 numbered steps, each with a concrete deliverable.\n"
            "CHECKS: acceptance criteria and how to validate correctness."
        )},
        {"role": "user", "content": f"Task: {q}"},
    ]

def latent_solver(q, ctx):
    """Latent Agent 2 – SOLVER. Reasons internally via KV-cache."""
    return [
        {"role": "system", "content": (
            "You are SOLVER. You have received internal reasoning context from PLANNER.\n"
            "Use the plan to consider 3 independent solution approaches.\n"
            "Think about which attempt is most consistent with the acceptance checks.\n"
            "Settle on the best candidate answer."
        )},
        {"role": "user", "content": f"Task: {q}"},
    ]

def latent_critic(q, ctx):
    """Latent Agent 3 – CRITIC. Reasons internally via KV-cache."""
    return [
        {"role": "system", "content": (
            "You are CRITIC. You have received internal reasoning context from "
            "PLANNER and SOLVER.\n"
            "Identify up to 5 specific problems or gaps in the chosen answer.\n"
            "Think about how to patch and improve it to final-draft quality."
        )},
        {"role": "user", "content": f"Task: {q}"},
    ]

def latent_verifier_gsm8k(q, ctx):
    """Latent Agent 4 – VERIFIER (final) for math. Generates visible text."""
    return [
        {"role": "system", "content": (
            "You are VERIFIER using Chain-of-Verification. You have received "
            "internal reasoning context from PLANNER, SOLVER, and CRITIC.\n\n"
            "Do these steps strictly and keep them short:\n"
            "1. List claims that might be wrong.\n"
            "2. Write verification questions for those claims.\n"
            "3. Answer them independently (mark unknown if unsure).\n"
            "4. Produce a corrected final response.\n\n"
            "Output only:\n"
            "CLAIMS TO CHECK:\n"
            "VERIFICATION QUESTIONS:\n"
            "INDEPENDENT ANSWERS:\n"
            "FINAL VERIFIED OUTPUT:\n"
            "(The last line of FINAL VERIFIED OUTPUT must be: The answer is <NUMBER>)"
        )},
        {"role": "user", "content": f"Task: {q}"},
    ]

def latent_verifier_mcq(q, ctx):
    """Latent Agent 4 – VERIFIER (final) for MCQ. Generates visible text."""
    return [
        {"role": "system", "content": (
            "You are VERIFIER using Chain-of-Verification. You have received "
            "internal reasoning context from PLANNER, SOLVER, and CRITIC.\n\n"
            "Do these steps strictly and keep them short:\n"
            "1. List claims that might be wrong.\n"
            "2. Write verification questions for those claims.\n"
            "3. Answer them independently (mark unknown if unsure).\n"
            "4. Produce a corrected final response.\n\n"
            "Output only:\n"
            "CLAIMS TO CHECK:\n"
            "VERIFICATION QUESTIONS:\n"
            "INDEPENDENT ANSWERS:\n"
            "FINAL VERIFIED OUTPUT:\n"
            "(The last line of FINAL VERIFIED OUTPUT must state the correct letter: A, B, C, or D)"
        )},
        {"role": "user", "content": f"Task: {q}"},
    ]


# ===========================================================================
# Pipeline runners
# ===========================================================================

@torch.no_grad()
def run_text_pipeline(model, question, specs, max_inter=300, max_final=400):
    stats = RunStats()
    accumulated = ""
    output = ""
    for spec in specs:
        is_final = spec["is_final"]
        msgs = spec["prompt_fn"](question, accumulated)
        _, ids, mask = model.prepare_chat_batch([msgs], add_generation_prompt=True)
        stats.prompt_tokens += int(mask.sum().item())
        t0 = time.perf_counter()
        texts, _ = model.generate_text_batch(
            ids, mask, max_new_tokens=max_final if is_final else max_inter,
            temperature=0.6, top_p=0.95,
        )
        stats.wall_time_s += time.perf_counter() - t0
        output = texts[0].strip()
        gen = count_tokens(model.tokenizer, output)
        stats.generated_tokens += gen
        if not is_final:
            stats.intermediate_tokens += gen
            accumulated += output + "\n"
    return output, stats


def run_latent_pipeline(model, pipeline, question):
    stats = RunStats()
    for agent in pipeline.agents:
        msgs = agent.prompt_fn(question, "")
        rendered = model.render_chat(msgs, add_generation_prompt=True)
        stats.prompt_tokens += count_tokens(model.tokenizer, rendered)
    t0 = time.perf_counter()
    result = pipeline.run(question)
    stats.wall_time_s = time.perf_counter() - t0
    stats.generated_tokens = count_tokens(model.tokenizer, result.text)
    return result.text, stats


# ===========================================================================
# Formatting
# ===========================================================================

def fmt_mcq(question, choices):
    labels = ["A", "B", "C", "D"]
    opts = "\n".join(f"  {labels[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{opts}"


def fmt_arc(question, choices):
    opts = "\n".join(f"  {l}. {t}" for l, t in zip(choices["label"], choices["text"]))
    return f"{question}\n\n{opts}"


def print_table(title, rows, cols):
    widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            widths[c] = max(widths[c], len(str(r.get(c, ""))))
    w = sum(widths.values()) + 3 * (len(cols) - 1)
    hdr = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")
    print(hdr)
    print(sep)
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print()


# ===========================================================================
# Core benchmark
# ===========================================================================

def benchmark_one_model(
    model_id: str,
    display_name: str,
    gsm8k_data: list,
    mmlu_data: list,
    arc_data: list,
) -> Dict[str, Any]:
    """Run all benchmarks for one model. Returns results dict."""

    print(f"\n{'#' * 70}")
    print(f"#  MODEL: {model_id}")
    print(f"#  Loading...")
    print(f"{'#' * 70}\n")

    set_seed(42)
    t_load = time.perf_counter()
    model = LatentModel(model_id, device="cuda", realign=True)
    load_time = time.perf_counter() - t_load
    print(f"  Loaded in {load_time:.1f}s\n")

    # Pipeline configs – 4-agent: Planner → Solver → Critic → Verifier
    def make_configs(bench_type):
        is_math = bench_type == "gsm8k"
        return {
            "Single Agent": {
                "type": "text",
                "specs": [{"prompt_fn": single_gsm8k if is_math else single_mcq, "is_final": True}],
            },
            "Non-Latent 4-Agent": {
                "type": "text",
                "specs": [
                    {"prompt_fn": planner_text, "is_final": False},
                    {"prompt_fn": solver_text, "is_final": False},
                    {"prompt_fn": critic_text, "is_final": False},
                    {"prompt_fn": verifier_text_gsm8k if is_math else verifier_text_mcq, "is_final": True},
                ],
            },
            "Latent 4-Agent": {
                "type": "latent",
                "agents": [
                    Agent(name="Planner", role="planner", prompt_fn=latent_planner),
                    Agent(name="Solver", role="solver", prompt_fn=latent_solver),
                    Agent(name="Critic", role="critic", prompt_fn=latent_critic),
                    Agent(name="Verifier", role="verifier",
                          prompt_fn=latent_verifier_gsm8k if is_math else latent_verifier_mcq,
                          is_final=True),
                ],
            },
        }

    benchmarks = {
        "gsm8k": ("GSM8K", gsm8k_data),
        "mmlu":  ("MMLU", mmlu_data),
        "arc":   ("ARC-Easy", arc_data),
    }

    model_results = {
        "model_id": model_id,
        "display_name": display_name,
        "load_time_s": round(load_time, 1),
        "benchmarks": {},
    }

    grand_summary = []

    for bench_key, (bench_name, data) in benchmarks.items():
        configs = make_configs(bench_key)
        n = len(data)
        max_tok = 400 if bench_key == "gsm8k" else 256

        bench_results = {}

        for cfg_name, cfg in configs.items():
            set_seed(42)
            correct = 0
            total = RunStats()

            if cfg["type"] == "latent":
                pipeline = LatentPipeline(
                    model, cfg["agents"], latent_steps=10,
                    max_new_tokens=max_tok, temperature=0.6,
                )

            for i, sample in enumerate(data):
                if bench_key == "gsm8k":
                    q = sample["question"]
                elif bench_key == "mmlu":
                    q = fmt_mcq(sample["question"], sample["choices"])
                else:
                    q = fmt_arc(sample["question"], sample["choices"])

                if cfg["type"] == "text":
                    answer, stats = run_text_pipeline(model, q, cfg["specs"], max_final=max_tok)
                else:
                    answer, stats = run_latent_pipeline(model, pipeline, q)

                if bench_key == "gsm8k":
                    hit = check_gsm8k(answer, sample["answer"])
                elif bench_key == "mmlu":
                    hit = check_mcq(answer, sample["answer"], sample["choices"])
                else:
                    hit = check_arc(answer, sample["answerKey"], sample["choices"])

                if hit:
                    correct += 1
                total.prompt_tokens += stats.prompt_tokens
                total.generated_tokens += stats.generated_tokens
                total.intermediate_tokens += stats.intermediate_tokens
                total.wall_time_s += stats.wall_time_s

                # Progress
                if (i + 1) % 25 == 0:
                    print(f"    {bench_name} | {cfg_name}: {i+1}/{n} done ({correct} correct)")

            acc = correct / n * 100
            bench_results[cfg_name] = {
                "correct": correct,
                "total": n,
                "accuracy_pct": round(acc, 1),
                "prompt_tokens": total.prompt_tokens,
                "generated_tokens": total.generated_tokens,
                "intermediate_tokens": total.intermediate_tokens,
                "wall_time_s": round(total.wall_time_s, 1),
            }

        model_results["benchmarks"][bench_key] = bench_results

        # Print per-benchmark table
        rows = []
        for cfg_name, r in bench_results.items():
            rows.append({
                "Pipeline": cfg_name,
                "Accuracy": f"{r['correct']}/{r['total']} ({r['accuracy_pct']}%)",
                "Prompt Tok": r["prompt_tokens"],
                "Intermed": r["intermediate_tokens"],
                "Gen Tok": r["generated_tokens"],
                "All Tok": r["prompt_tokens"] + r["generated_tokens"],
                "Time": f"{r['wall_time_s']:.0f}s",
            })
        print_table(
            f"{display_name} -- {bench_name} ({n} questions)",
            rows,
            ["Pipeline", "Accuracy", "Prompt Tok", "Intermed", "Gen Tok", "All Tok", "Time"],
        )

    # Grand summary for this model
    print(f"\n  --- {display_name} Summary ---")
    for cfg_name in ["Single Agent", "Non-Latent 4-Agent", "Latent 4-Agent"]:
        total_c = sum(model_results["benchmarks"][b][cfg_name]["correct"] for b in benchmarks)
        total_n = sum(model_results["benchmarks"][b][cfg_name]["total"] for b in benchmarks)
        total_gen = sum(model_results["benchmarks"][b][cfg_name]["generated_tokens"] for b in benchmarks)
        total_time = sum(model_results["benchmarks"][b][cfg_name]["wall_time_s"] for b in benchmarks)
        print(f"  {cfg_name:25s}: {total_c}/{total_n} ({total_c/total_n*100:.1f}%) | gen={total_gen:,} tok | {total_time:.0f}s")

    nl_gen = sum(model_results["benchmarks"][b]["Non-Latent 4-Agent"]["generated_tokens"] for b in benchmarks)
    la_gen = sum(model_results["benchmarks"][b]["Latent 4-Agent"]["generated_tokens"] for b in benchmarks)
    nl_time = sum(model_results["benchmarks"][b]["Non-Latent 4-Agent"]["wall_time_s"] for b in benchmarks)
    la_time = sum(model_results["benchmarks"][b]["Latent 4-Agent"]["wall_time_s"] for b in benchmarks)
    if nl_gen > 0:
        print(f"  Token savings (latent vs non-latent): {(1 - la_gen/nl_gen)*100:.0f}%")
    if la_time > 0:
        print(f"  Speedup: {nl_time/la_time:.1f}x")
    print()

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return model_results


# ===========================================================================
# Charts & Visualisations
# ===========================================================================

# Colour palette – consistent across every chart
PIPE_COLORS = {
    "Single Agent":       "#4C72B0",
    "Non-Latent 4-Agent": "#DD8452",
    "Latent 4-Agent":     "#55A868",
}
BENCH_LABELS = {"gsm8k": "GSM8K", "mmlu": "MMLU", "arc": "ARC-Easy"}
PIPELINES = ["Single Agent", "Non-Latent 4-Agent", "Latent 4-Agent"]


def _ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def _save(fig, path: str):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    -> saved {path}")


def generate_charts(all_results: Dict[str, Any], out_dir: str = "charts"):
    """Generate all charts from a completed results dict."""
    _ensure_dir(out_dir)

    successful = [m for m in all_results["models"] if "error" not in m]
    if not successful:
        print("  No successful model runs – skipping charts.")
        return

    model_names = [m["display_name"] for m in successful]
    bench_keys = list(BENCH_LABELS.keys())

    # ------------------------------------------------------------------
    # 1. Accuracy grouped bar chart – one chart per benchmark
    # ------------------------------------------------------------------
    for bk in bench_keys:
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(model_names)), 5))
        x = np.arange(len(model_names))
        width = 0.25

        for i, pipe in enumerate(PIPELINES):
            accs = []
            for m in successful:
                accs.append(m["benchmarks"][bk][pipe]["accuracy_pct"])
            bars = ax.bar(x + i * width, accs, width, label=pipe,
                          color=PIPE_COLORS[pipe], edgecolor="white", linewidth=0.5)
            # labels on bars
            for bar, v in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{BENCH_LABELS[bk]} Accuracy by Pipeline", fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.15))
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        _save(fig, f"{out_dir}/accuracy_{bk}.png")

    # ------------------------------------------------------------------
    # 2. Overall accuracy across all benchmarks
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(model_names)), 5))
    x = np.arange(len(model_names))
    width = 0.25

    for i, pipe in enumerate(PIPELINES):
        overall = []
        for m in successful:
            total_c = sum(m["benchmarks"][bk][pipe]["correct"] for bk in bench_keys)
            total_n = sum(m["benchmarks"][bk][pipe]["total"] for bk in bench_keys)
            overall.append(total_c / total_n * 100 if total_n else 0)
        bars = ax.bar(x + i * width, overall, width, label=pipe,
                      color=PIPE_COLORS[pipe], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, overall):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy (All Benchmarks Combined)", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, min(100, ax.get_ylim()[1] * 1.15))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, f"{out_dir}/accuracy_overall.png")

    # ------------------------------------------------------------------
    # 3. Token generation comparison (bar chart)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(model_names)), 5))
    x = np.arange(len(model_names))
    width = 0.25

    for i, pipe in enumerate(PIPELINES):
        gen_tok = []
        for m in successful:
            total_gen = sum(m["benchmarks"][bk][pipe]["generated_tokens"] for bk in bench_keys)
            gen_tok.append(total_gen)
        bars = ax.bar(x + i * width, gen_tok, width, label=pipe,
                      color=PIPE_COLORS[pipe], edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, gen_tok):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:,.0f}", ha="center", va="bottom", fontsize=6, rotation=45)

    ax.set_ylabel("Total Generated Tokens")
    ax.set_title("Total Tokens Generated (All Benchmarks)", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.grid(axis="y", alpha=0.3)
    _save(fig, f"{out_dir}/tokens_generated.png")

    # ------------------------------------------------------------------
    # 4. Token savings % (Latent vs Non-Latent) – horizontal bar
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, max(3, 0.7 * len(model_names))))
    savings = []
    for m in successful:
        nl = sum(m["benchmarks"][bk]["Non-Latent 4-Agent"]["generated_tokens"] for bk in bench_keys)
        la = sum(m["benchmarks"][bk]["Latent 4-Agent"]["generated_tokens"] for bk in bench_keys)
        savings.append((1 - la / nl) * 100 if nl > 0 else 0)

    y_pos = np.arange(len(model_names))
    colors = ["#55A868" if s > 0 else "#C44E52" for s in savings]
    bars = ax.barh(y_pos, savings, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, savings):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}%", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xlabel("Token Savings (%)")
    ax.set_title("Token Savings: Latent vs Non-Latent 4-Agent", fontweight="bold")
    ax.axvline(x=0, color="grey", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, f"{out_dir}/token_savings.png")

    # ------------------------------------------------------------------
    # 5. Speedup (wall-time) – horizontal bar
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, max(3, 0.7 * len(model_names))))
    speedups = []
    for m in successful:
        nl_t = sum(m["benchmarks"][bk]["Non-Latent 4-Agent"]["wall_time_s"] for bk in bench_keys)
        la_t = sum(m["benchmarks"][bk]["Latent 4-Agent"]["wall_time_s"] for bk in bench_keys)
        speedups.append(nl_t / la_t if la_t > 0 else 0)

    bars = ax.barh(y_pos, speedups, color="#4C72B0", edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, speedups):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}x", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xlabel("Speedup (x)")
    ax.set_title("Wall-Clock Speedup: Latent vs Non-Latent 4-Agent", fontweight="bold")
    ax.axvline(x=1.0, color="red", linewidth=0.8, linestyle="--", label="Break-even (1x)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    _save(fig, f"{out_dir}/speedup.png")

    # ------------------------------------------------------------------
    # 6. Wall-time comparison (grouped bar)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(model_names)), 5))
    x = np.arange(len(model_names))
    width = 0.25

    for i, pipe in enumerate(PIPELINES):
        times = []
        for m in successful:
            total_time = sum(m["benchmarks"][bk][pipe]["wall_time_s"] for bk in bench_keys)
            times.append(total_time)
        bars = ax.bar(x + i * width, times, width, label=pipe,
                      color=PIPE_COLORS[pipe], edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Total Wall Time (s)")
    ax.set_title("Total Inference Time (All Benchmarks)", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, f"{out_dir}/wall_time.png")

    # ------------------------------------------------------------------
    # 7. Accuracy vs Model Size scatter (one line per pipeline)
    # ------------------------------------------------------------------
    # Try to estimate param count from display name for x-axis ordering
    param_estimates = {
        "Qwen3-1.7B": 1.7, "Llama-3.1-8B": 8, "Qwen3-8B": 8,
        "Mistral-7B": 7, "Gemma-2-9b": 9, "Qwen3-32B": 32, "Llama-3.1-70B": 70,
    }
    # Collect (param_count, model_name, {pipeline: accuracy})
    scatter_data = []
    for m in successful:
        p = param_estimates.get(m["display_name"], 0)
        accs = {}
        for pipe in PIPELINES:
            total_c = sum(m["benchmarks"][bk][pipe]["correct"] for bk in bench_keys)
            total_n = sum(m["benchmarks"][bk][pipe]["total"] for bk in bench_keys)
            accs[pipe] = total_c / total_n * 100 if total_n else 0
        scatter_data.append((p, m["display_name"], accs))
    scatter_data.sort(key=lambda x: x[0])

    if scatter_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        for pipe in PIPELINES:
            xs = [d[0] for d in scatter_data]
            ys = [d[2][pipe] for d in scatter_data]
            ax.plot(xs, ys, "o-", label=pipe, color=PIPE_COLORS[pipe], markersize=7)
            for xv, yv, d in zip(xs, ys, scatter_data):
                ax.annotate(d[1], (xv, yv), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=6, alpha=0.7)

        ax.set_xlabel("Model Size (B params)")
        ax.set_ylabel("Overall Accuracy (%)")
        ax.set_title("Accuracy Scaling with Model Size", fontweight="bold")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}B"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        _save(fig, f"{out_dir}/accuracy_vs_size.png")

    # ------------------------------------------------------------------
    # 8. Radar / spider chart per benchmark  (accuracy comparison)
    # ------------------------------------------------------------------
    if len(successful) >= 3:
        bench_list = list(BENCH_LABELS.values())
        num_vars = len(bench_list)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        for pipe in PIPELINES:
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            for m in successful:
                vals = [m["benchmarks"][bk][pipe]["accuracy_pct"] for bk in bench_keys]
                vals += vals[:1]
                ax.plot(angles, vals, "o-", label=m["display_name"], markersize=4)
                ax.fill(angles, vals, alpha=0.05)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(bench_list, fontsize=9)
            ax.set_ylim(0, 100)
            ax.set_title(f"{pipe}: Accuracy Radar", fontweight="bold", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
            pipe_slug = pipe.lower().replace(" ", "_").replace("-", "")
            _save(fig, f"{out_dir}/radar_{pipe_slug}.png")

    # ------------------------------------------------------------------
    # 9. Accuracy delta heatmap (Latent minus Non-Latent)
    # ------------------------------------------------------------------
    if len(successful) >= 1:
        delta_matrix = []
        for m in successful:
            row = []
            for bk in bench_keys:
                la = m["benchmarks"][bk]["Latent 4-Agent"]["accuracy_pct"]
                nl = m["benchmarks"][bk]["Non-Latent 4-Agent"]["accuracy_pct"]
                row.append(la - nl)
            delta_matrix.append(row)

        delta_arr = np.array(delta_matrix)
        fig, ax = plt.subplots(figsize=(max(5, len(bench_keys) * 1.5),
                                        max(3, len(model_names) * 0.6)))
        vabs = max(abs(delta_arr.min()), abs(delta_arr.max()), 5)
        im = ax.imshow(delta_arr, cmap="RdYlGn", aspect="auto",
                       vmin=-vabs, vmax=vabs)
        ax.set_xticks(range(len(bench_keys)))
        ax.set_xticklabels([BENCH_LABELS[bk] for bk in bench_keys], fontsize=9)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=9)
        for i in range(len(model_names)):
            for j in range(len(bench_keys)):
                ax.text(j, i, f"{delta_arr[i, j]:+.1f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if abs(delta_arr[i, j]) > vabs * 0.6 else "black")
        ax.set_title("Accuracy Delta: Latent - Non-Latent 4-Agent (pp)", fontweight="bold")
        fig.colorbar(im, ax=ax, label="Accuracy Difference (pp)")
        _save(fig, f"{out_dir}/accuracy_delta_heatmap.png")

    print(f"\n  All charts saved to {out_dir}/\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="GPU benchmarks for latent-agents")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model IDs to test (default: all)")
    parser.add_argument("--skip-gated", action="store_true",
                        help="Skip models that require HF token")
    parser.add_argument("--n-gsm8k", type=int, default=DEFAULT_N_GSM8K)
    parser.add_argument("--n-mmlu", type=int, default=DEFAULT_N_MMLU)
    parser.add_argument("--n-arc", type=int, default=DEFAULT_N_ARC)
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--latent-steps", type=int, default=10)
    parser.add_argument("--charts-dir", type=str, default="charts",
                        help="Directory to save generated charts (default: charts/)")
    parser.add_argument("--charts-only", type=str, default=None,
                        help="Skip benchmarks; generate charts from an existing JSON file")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Charts-only mode: regenerate from a previous results JSON
    # ------------------------------------------------------------------
    if args.charts_only:
        print(f"  Loading results from {args.charts_only}...")
        with open(args.charts_only) as f:
            existing = json.load(f)
        generate_charts(existing, out_dir=args.charts_dir)
        return

    # Select models
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",")]
        models_to_test = [(mid, mid.split("/")[-1], False, 0) for mid in model_ids]
    elif args.skip_gated:
        models_to_test = [m for m in ALL_MODELS if not m[2]]
    else:
        models_to_test = ALL_MODELS

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU found. This script requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"\n  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"  Models to test: {len(models_to_test)}")
    print(f"  Samples: GSM8K={args.n_gsm8k}, MMLU={args.n_mmlu}, ARC={args.n_arc}")
    print()

    # Load datasets once
    print("  Loading datasets...")
    gsm8k = list(load_dataset("openai/gsm8k", "main", split="test")
                 .select(range(min(args.n_gsm8k, 1319))))
    mmlu = list(load_dataset("cais/mmlu", "all", split="test")
                .shuffle(seed=42).select(range(min(args.n_mmlu, 14042))))
    arc = list(load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
               .shuffle(seed=42).select(range(min(args.n_arc, 2376))))
    print(f"  Loaded: GSM8K={len(gsm8k)}, MMLU={len(mmlu)}, ARC={len(arc)}\n")

    # Run benchmarks
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "gpu": gpu_name,
        "gpu_mem_gb": round(gpu_mem, 1),
        "config": {
            "n_gsm8k": len(gsm8k),
            "n_mmlu": len(mmlu),
            "n_arc": len(arc),
            "latent_steps": args.latent_steps,
        },
        "models": [],
    }

    for model_id, display, is_gated, approx_gb in models_to_test:
        if approx_gb > gpu_mem * 0.95:
            print(f"\n  SKIPPING {display}: needs ~{approx_gb}GB, GPU has {gpu_mem:.0f}GB")
            continue

        try:
            result = benchmark_one_model(model_id, display, gsm8k, mmlu, arc)
            all_results["models"].append(result)
        except Exception as e:
            print(f"\n  ERROR on {display}: {e}")
            all_results["models"].append({
                "model_id": model_id,
                "display_name": display,
                "error": str(e),
            })
            gc.collect()
            torch.cuda.empty_cache()

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {args.output}")

    # -----------------------------------------------------------------------
    # Final cross-model summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 120}")
    print(f"  FINAL CROSS-MODEL SUMMARY")
    print(f"{'=' * 120}\n")

    successful = [m for m in all_results["models"] if "error" not in m]
    if not successful:
        print("  No models completed successfully.")
        return

    summary_rows = []
    for m in successful:
        benches = m["benchmarks"]
        for cfg_name in ["Single Agent", "Non-Latent 4-Agent", "Latent 4-Agent"]:
            gsm_acc = benches["gsm8k"][cfg_name]["accuracy_pct"]
            mmlu_acc = benches["mmlu"][cfg_name]["accuracy_pct"]
            arc_acc = benches["arc"][cfg_name]["accuracy_pct"]
            total_c = sum(benches[b][cfg_name]["correct"] for b in benches)
            total_n = sum(benches[b][cfg_name]["total"] for b in benches)
            total_gen = sum(benches[b][cfg_name]["generated_tokens"] for b in benches)
            total_time = sum(benches[b][cfg_name]["wall_time_s"] for b in benches)
            summary_rows.append({
                "Model": m["display_name"],
                "Pipeline": cfg_name,
                "GSM8K": f"{gsm_acc}%",
                "MMLU": f"{mmlu_acc}%",
                "ARC": f"{arc_acc}%",
                "Overall": f"{total_c}/{total_n} ({total_c/total_n*100:.0f}%)",
                "Gen Tok": f"{total_gen:,}",
                "Time": f"{total_time:.0f}s",
            })

    print_table(
        "All Models x All Pipelines",
        summary_rows,
        ["Model", "Pipeline", "GSM8K", "MMLU", "ARC", "Overall", "Gen Tok", "Time"],
    )

    # Token savings per model
    print("  --- Token Savings & Speedup (Latent vs Non-Latent) ---")
    for m in successful:
        benches = m["benchmarks"]
        nl_gen = sum(benches[b]["Non-Latent 4-Agent"]["generated_tokens"] for b in benches)
        la_gen = sum(benches[b]["Latent 4-Agent"]["generated_tokens"] for b in benches)
        nl_time = sum(benches[b]["Non-Latent 4-Agent"]["wall_time_s"] for b in benches)
        la_time = sum(benches[b]["Latent 4-Agent"]["wall_time_s"] for b in benches)
        savings = (1 - la_gen / nl_gen) * 100 if nl_gen > 0 else 0
        speedup = nl_time / la_time if la_time > 0 else 0
        nl_acc = sum(benches[b]["Non-Latent 4-Agent"]["correct"] for b in benches)
        la_acc = sum(benches[b]["Latent 4-Agent"]["correct"] for b in benches)
        total_n = sum(benches[b]["Latent 4-Agent"]["total"] for b in benches)
        print(f"  {m['display_name']:20s}: {savings:.0f}% fewer tokens, {speedup:.1f}x faster | "
              f"Non-Latent={nl_acc}/{total_n}, Latent={la_acc}/{total_n}")

    print(f"\n  Full results: {args.output}")

    # -----------------------------------------------------------------------
    # Generate all charts
    # -----------------------------------------------------------------------
    print(f"\n  Generating charts...")
    generate_charts(all_results, out_dir=args.charts_dir)

    print("  Done!\n")


if __name__ == "__main__":
    main()
