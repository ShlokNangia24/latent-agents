#!/usr/bin/env python3
"""Quick benchmark for M4 Pro Mac: 5 samples per benchmark, 256 max tokens.

Usage:
    conda run -n latent-agents python run_quick_benchmark.py
"""

import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import torch
from datasets import load_dataset

from latent_agents import Agent, LatentModel, LatentPipeline, set_seed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEVICE = "cpu"
N_SAMPLES = 5
MAX_NEW_TOKENS = 256
LATENT_STEPS = 10

set_seed(42)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class RunStats:
    correct: int = 0
    total: int = 0
    wall_time_s: float = 0.0
    generated_tokens: int = 0

    @property
    def accuracy(self):
        return self.correct / self.total * 100 if self.total else 0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def single_prompt(question, context):
    return [
        {"role": "system", "content": "You are a helpful assistant. Think step by step and give the final answer inside \\boxed{YOUR_ANSWER}."},
        {"role": "user", "content": question},
    ]

def planner_prompt(question, context):
    return [
        {"role": "system", "content": "You are a planning assistant."},
        {"role": "user", "content": f"Design a step-by-step plan to solve:\n\n{question}"},
    ]

def critic_prompt(question, context):
    return [
        {"role": "system", "content": "You are a critical reviewer."},
        {"role": "user", "content": f"Review and critique the approach to:\n\n{question}"},
    ]

def refiner_prompt(question, context):
    return [
        {"role": "system", "content": "You are a plan refiner."},
        {"role": "user", "content": f"Refine the solution approach for:\n\n{question}"},
    ]

def solver_prompt(question, context):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Solve step by step and put the final answer inside \\boxed{{YOUR_ANSWER}}.\n\n{question}"},
    ]

def text_agent_prompt(question, context):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{context}\n\nNow solve: {question}\n\nPut the final answer inside \\boxed{{YOUR_ANSWER}}."},
    ]


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def extract_answer(text):
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'####\s*(.+)', text)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def normalize(s):
    return re.sub(r'[^a-z0-9.]', '', s.lower().strip())


def check_answer(generated, expected):
    gen = normalize(extract_answer(generated))
    exp = normalize(expected)
    return gen == exp or exp in gen


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def load_gsm8k(n):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        q = row["question"]
        m = re.search(r'####\s*(.+)', row["answer"])
        a = m.group(1).strip().replace(",", "") if m else ""
        items.append((q, a))
    return items


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

def load_mmlu(n):
    ds = load_dataset("cais/mmlu", "all", split="test")
    items = []
    choices_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    for row in ds.select(range(min(n, len(ds)))):
        q = row["question"]
        opts = row["choices"]
        q_full = f"{q}\nA) {opts[0]}\nB) {opts[1]}\nC) {opts[2]}\nD) {opts[3]}\nAnswer with just the letter."
        a = choices_map[row["answer"]]
        items.append((q_full, a))
    return items


# ---------------------------------------------------------------------------
# ARC-Easy
# ---------------------------------------------------------------------------

def load_arc(n):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    items = []
    for row in ds.select(range(min(n, len(ds)))):
        q = row["question"]
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        opts = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        q_full = f"{q}\n{opts}\nAnswer with just the letter."
        a = row["answerKey"]
        items.append((q_full, a))
    return items


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

def run_single_agent(model, questions, expected_answers):
    stats = RunStats(total=len(questions))
    agents = [Agent(name="Solver", role="solver", prompt_fn=single_prompt, is_final=True)]
    pipeline = LatentPipeline(model, agents, latent_steps=0, max_new_tokens=MAX_NEW_TOKENS)

    t0 = time.time()
    for q, exp in zip(questions, expected_answers):
        result = pipeline.run(q)
        if check_answer(result.text, exp):
            stats.correct += 1
        stats.generated_tokens += len(model.tokenizer.encode(result.text, add_special_tokens=False))
    stats.wall_time_s = time.time() - t0
    return stats


def run_text_pipeline(model, questions, expected_answers):
    stats = RunStats(total=len(questions))

    t0 = time.time()
    for q, exp in zip(questions, expected_answers):
        context = ""
        for role_name, role_id, pfn in [
            ("Planner", "planner", planner_prompt),
            ("Critic", "critic", critic_prompt),
            ("Refiner", "refiner", refiner_prompt),
        ]:
            agents_step = [Agent(name=role_name, role=role_id, prompt_fn=pfn, is_final=True)]
            pipe = LatentPipeline(model, agents_step, latent_steps=0, max_new_tokens=MAX_NEW_TOKENS)
            r = pipe.run(q, context=context)
            context += f"\n\n[{role_name}]: {r.text}"
            stats.generated_tokens += len(model.tokenizer.encode(r.text, add_special_tokens=False))

        final_agents = [Agent(name="Solver", role="solver",
                              prompt_fn=lambda q, c: text_agent_prompt(q, c), is_final=True)]
        pipe = LatentPipeline(model, final_agents, latent_steps=0, max_new_tokens=MAX_NEW_TOKENS)
        # pass context through prompt_fn
        final_agents[0].prompt_fn = lambda question, ctx, _c=context: [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{_c}\n\nNow solve: {question}\n\nPut the final answer inside \\boxed{{YOUR_ANSWER}}."},
        ]
        r = pipe.run(q)
        if check_answer(r.text, exp):
            stats.correct += 1
        stats.generated_tokens += len(model.tokenizer.encode(r.text, add_special_tokens=False))

    stats.wall_time_s = time.time() - t0
    return stats


def run_latent_pipeline(model, questions, expected_answers):
    stats = RunStats(total=len(questions))
    agents = [
        Agent(name="Planner", role="planner", prompt_fn=planner_prompt,
              latent_steps=LATENT_STEPS),
        Agent(name="Critic", role="critic", prompt_fn=critic_prompt,
              latent_steps=max(1, LATENT_STEPS // 2)),
        Agent(name="Refiner", role="refiner", prompt_fn=refiner_prompt),
        Agent(name="Solver", role="solver", prompt_fn=solver_prompt, is_final=True),
    ]
    pipeline = LatentPipeline(
        model, agents,
        latent_steps=LATENT_STEPS,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    t0 = time.time()
    for q, exp in zip(questions, expected_answers):
        result = pipeline.run(q)
        if check_answer(result.text, exp):
            stats.correct += 1
        stats.generated_tokens += len(model.tokenizer.encode(result.text, add_special_tokens=False))
    stats.wall_time_s = time.time() - t0
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(benchmarks):
    header = f"{'Benchmark':<12} {'Pipeline':<22} {'Accuracy':>10} {'Tokens':>10} {'Time (s)':>10}"
    print(header)
    print("-" * len(header))
    for name, results in benchmarks:
        for pname, stats in results:
            print(f"{name:<12} {pname:<22} {stats.accuracy:>9.1f}% {stats.generated_tokens:>10} {stats.wall_time_s:>10.1f}")
        print()


def main():
    print(f"Loading {MODEL} on {DEVICE} ...")
    model = LatentModel(MODEL, device=DEVICE, realign=True)
    print(f"Model loaded.\n")

    benchmarks = []

    # --- GSM8K ---
    print("=" * 60)
    print(f"GSM8K ({N_SAMPLES} samples)")
    print("=" * 60)
    gsm8k = load_gsm8k(N_SAMPLES)
    questions, answers = zip(*gsm8k)
    questions, answers = list(questions), list(answers)

    print("  Running single agent ...")
    s1 = run_single_agent(model, questions, answers)
    print(f"    -> {s1.accuracy:.0f}% ({s1.correct}/{s1.total}) in {s1.wall_time_s:.1f}s")

    print("  Running non-latent 4-agent ...")
    s2 = run_text_pipeline(model, questions, answers)
    print(f"    -> {s2.accuracy:.0f}% ({s2.correct}/{s2.total}) in {s2.wall_time_s:.1f}s")

    print("  Running latent 4-agent ...")
    s3 = run_latent_pipeline(model, questions, answers)
    print(f"    -> {s3.accuracy:.0f}% ({s3.correct}/{s3.total}) in {s3.wall_time_s:.1f}s")

    benchmarks.append(("GSM8K", [("Single Agent", s1), ("Non-Latent 4-Agent", s2), ("Latent 4-Agent", s3)]))

    # --- MMLU ---
    print("\n" + "=" * 60)
    print(f"MMLU ({N_SAMPLES} samples)")
    print("=" * 60)
    mmlu = load_mmlu(N_SAMPLES)
    questions, answers = zip(*mmlu)
    questions, answers = list(questions), list(answers)

    print("  Running single agent ...")
    s1 = run_single_agent(model, questions, answers)
    print(f"    -> {s1.accuracy:.0f}% ({s1.correct}/{s1.total}) in {s1.wall_time_s:.1f}s")

    print("  Running non-latent 4-agent ...")
    s2 = run_text_pipeline(model, questions, answers)
    print(f"    -> {s2.accuracy:.0f}% ({s2.correct}/{s2.total}) in {s2.wall_time_s:.1f}s")

    print("  Running latent 4-agent ...")
    s3 = run_latent_pipeline(model, questions, answers)
    print(f"    -> {s3.accuracy:.0f}% ({s3.correct}/{s3.total}) in {s3.wall_time_s:.1f}s")

    benchmarks.append(("MMLU", [("Single Agent", s1), ("Non-Latent 4-Agent", s2), ("Latent 4-Agent", s3)]))

    # --- ARC ---
    print("\n" + "=" * 60)
    print(f"ARC-Easy ({N_SAMPLES} samples)")
    print("=" * 60)
    arc = load_arc(N_SAMPLES)
    questions, answers = zip(*arc)
    questions, answers = list(questions), list(answers)

    print("  Running single agent ...")
    s1 = run_single_agent(model, questions, answers)
    print(f"    -> {s1.accuracy:.0f}% ({s1.correct}/{s1.total}) in {s1.wall_time_s:.1f}s")

    print("  Running non-latent 4-agent ...")
    s2 = run_text_pipeline(model, questions, answers)
    print(f"    -> {s2.accuracy:.0f}% ({s2.correct}/{s2.total}) in {s2.wall_time_s:.1f}s")

    print("  Running latent 4-agent ...")
    s3 = run_latent_pipeline(model, questions, answers)
    print(f"    -> {s3.accuracy:.0f}% ({s3.correct}/{s3.total}) in {s3.wall_time_s:.1f}s")

    benchmarks.append(("ARC-Easy", [("Single Agent", s1), ("Non-Latent 4-Agent", s2), ("Latent 4-Agent", s3)]))

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print_table(benchmarks)

    # Token savings & speedup
    for name, results in benchmarks:
        text_tokens = results[1][1].generated_tokens
        latent_tokens = results[2][1].generated_tokens
        text_time = results[1][1].wall_time_s
        latent_time = results[2][1].wall_time_s
        savings = (1 - latent_tokens / text_tokens) * 100 if text_tokens > 0 else 0
        speedup = text_time / latent_time if latent_time > 0 else 0
        print(f"{name}: Token savings={savings:.0f}%, Speedup={speedup:.1f}x")


if __name__ == "__main__":
    main()
