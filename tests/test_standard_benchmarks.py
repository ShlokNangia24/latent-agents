"""Standard LLM benchmarks: Non-Latent vs Latent multi-agent pipelines.

Evaluates on three widely-used LLM benchmarks:
  1. GSM8K   -- Grade school math reasoning (extract final number)
  2. MMLU    -- Massive Multitask Language Understanding (multiple choice)
  3. ARC-Easy -- AI2 Reasoning Challenge (multiple choice, science)

4-agent pipeline:
  1. PLANNER  (Plan-and-Solve)    -- plan + acceptance checks
  2. SOLVER   (Self-consistency)  -- 3 candidate solutions, pick best
  3. CRITIC   (Self-Refine)       -- tighten / fix the chosen solution
  4. VERIFIER (Chain-of-Verification) -- verify claims, output final

Each benchmark is run under three pipeline configurations:
  - Single agent (baseline)
  - Non-latent 4-agent (text-based communication)
  - Latent 4-agent (KV-cache communication)

Run:
    pytest tests/test_standard_benchmarks.py -v -s --run-integration

Use -k to run a single benchmark:
    pytest tests/test_standard_benchmarks.py -v -s --run-integration -k gsm8k
    pytest tests/test_standard_benchmarks.py -v -s --run-integration -k mmlu
    pytest tests/test_standard_benchmarks.py -v -s --run-integration -k arc
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pytest
import torch

from datasets import load_dataset

from latent_agents import Agent, LatentModel, LatentPipeline, set_seed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SMALL_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Number of samples per benchmark (keep small for CPU; increase for GPU)
N_GSM8K = 30
N_MMLU = 50
N_ARC = 50


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(answer_str: str) -> str:
    """Extract the final numeric answer from GSM8K ground truth (after ####)."""
    match = re.search(r"####\s*(.+)", answer_str)
    if match:
        return match.group(1).strip().replace(",", "")
    return ""


def extract_number_from_text(text: str) -> str:
    """Extract the last number from model output for GSM8K comparison."""
    # Look for boxed answers first
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip().replace(",", "")
    # Look for "answer is X" pattern
    answer_is = re.search(r"(?:answer|result|total|equals?)\s*(?:is|=|:)\s*\$?\s*([-\d,]+(?:\.\d+)?)", text, re.I)
    if answer_is:
        return answer_is.group(1).strip().replace(",", "")
    # Fall back to last number in text
    numbers = re.findall(r"([-]?\d[\d,]*(?:\.\d+)?)", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def check_gsm8k(model_output: str, ground_truth: str) -> bool:
    """Check if model output contains the correct GSM8K answer."""
    expected = extract_gsm8k_answer(ground_truth)
    extracted = extract_number_from_text(model_output)
    if expected and extracted:
        try:
            return float(extracted) == float(expected)
        except ValueError:
            pass
    # Fallback: substring check
    return expected in model_output if expected else False


def check_mmlu(model_output: str, correct_idx: int, choices: List[str]) -> bool:
    """Check if model picked the correct MMLU answer."""
    labels = ["A", "B", "C", "D"]
    correct_letter = labels[correct_idx]
    correct_text = choices[correct_idx]
    text = model_output.strip().upper()

    # Check if the model output starts with or contains the correct letter
    if text and text[0] == correct_letter:
        return True
    # Check for "answer is A" patterns
    pattern = re.search(r"(?:answer|correct)\s*(?:is|:)\s*\(?([A-D])\)?", model_output, re.I)
    if pattern and pattern.group(1).upper() == correct_letter:
        return True
    # Check for the correct choice text
    if correct_text.lower() in model_output.lower():
        return True
    return False


def check_arc(model_output: str, correct_key: str, choices: Dict) -> bool:
    """Check if model picked the correct ARC answer."""
    text = model_output.strip().upper()
    labels = choices["label"]
    texts = choices["text"]
    correct_idx = labels.index(correct_key) if correct_key in labels else -1

    # Check letter match
    if text and text[0] == correct_key.upper():
        return True
    pattern = re.search(r"(?:answer|correct)\s*(?:is|:)\s*\(?([A-D])\)?", model_output, re.I)
    if pattern and pattern.group(1).upper() == correct_key.upper():
        return True
    # Check text match
    if correct_idx >= 0 and texts[correct_idx].lower() in model_output.lower():
        return True
    return False


# ---------------------------------------------------------------------------
# Prompt templates – 4-agent pipeline
#   1. PLANNER  (Plan-and-Solve)
#   2. SOLVER   (Self-consistency, N=3)
#   3. CRITIC   (Self-Refine)
#   4. VERIFIER (Chain-of-Verification)
# ---------------------------------------------------------------------------

# ---- Single-agent baselines ----

def single_agent_gsm8k(question: str, context: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": (
            "You are an expert math tutor. Solve the problem step by step, showing "
            "all arithmetic clearly. After your reasoning, write the final numeric "
            "answer on the last line in the format: The answer is <NUMBER>."
        )},
        {"role": "user", "content": question},
    ]

def single_agent_mcq(question: str, context: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": (
            "You are a knowledgeable assistant. Read the question and all options "
            "carefully. Briefly reason about each option, then clearly state the "
            "correct answer as a single letter (A, B, C, or D) on the last line."
        )},
        {"role": "user", "content": question},
    ]

# ---- Non-latent (text-based) 4-agent pipeline ----

def planner_text(question: str, context: str) -> List[Dict[str, str]]:
    """Agent 1 – PLANNER (Plan-and-Solve)."""
    return [
        {"role": "system", "content": (
            "You are PLANNER.\n"
            "Create a plan before solving.\n\n"
            "Output only:\n"
            "PLAN: 3-7 numbered steps, each with a concrete deliverable.\n"
            "CHECKS: bullet list of acceptance criteria and how to validate correctness."
        )},
        {"role": "user", "content": f"Task: {question}"},
    ]

def solver_text(question: str, prior_text: str) -> List[Dict[str, str]]:
    """Agent 2 – SOLVER (Self-consistency, N=3)."""
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
            f"--- PLANNER output ---\n{prior_text}\n--- end ---\n\n"
            f"Task: {question}"
        )},
    ]

def critic_text(question: str, prior_text: str) -> List[Dict[str, str]]:
    """Agent 3 – CRITIC (Self-Refine)."""
    return [
        {"role": "system", "content": (
            "You are CRITIC.\n"
            "Improve the CHOSEN ANSWER with minimal edits.\n\n"
            "Output only:\n"
            "TOP ISSUES (max 5): specific problems or gaps\n"
            "PATCH: the revised answer (clean, final-draft quality)"
        )},
        {"role": "user", "content": (
            f"--- Previous agents ---\n{prior_text}\n--- end ---\n\n"
            f"Task: {question}"
        )},
    ]

def verifier_text_gsm8k(question: str, prior_text: str) -> List[Dict[str, str]]:
    """Agent 4 – VERIFIER (CoVe) for math."""
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
            f"--- Previous agents ---\n{prior_text}\n--- end ---\n\n"
            f"Task: {question}"
        )},
    ]

def verifier_text_mcq(question: str, prior_text: str) -> List[Dict[str, str]]:
    """Agent 4 – VERIFIER (CoVe) for MCQ."""
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
            f"--- Previous agents ---\n{prior_text}\n--- end ---\n\n"
            f"Task: {question}"
        )},
    ]

# ---- Latent 4-agent pipeline (KV-cache communication) ----

def latent_planner(question: str, context: str) -> List[Dict[str, str]]:
    """Latent Agent 1 – PLANNER."""
    return [
        {"role": "system", "content": (
            "You are PLANNER.\n"
            "Create a plan before solving.\n\n"
            "Think about:\n"
            "PLAN: 3-7 numbered steps, each with a concrete deliverable.\n"
            "CHECKS: acceptance criteria and how to validate correctness."
        )},
        {"role": "user", "content": f"Task: {question}"},
    ]

def latent_solver(question: str, context: str) -> List[Dict[str, str]]:
    """Latent Agent 2 – SOLVER."""
    return [
        {"role": "system", "content": (
            "You are SOLVER. You have received internal reasoning context from PLANNER.\n"
            "Use the plan to consider 3 independent solution approaches.\n"
            "Think about which attempt is most consistent with the acceptance checks.\n"
            "Settle on the best candidate answer."
        )},
        {"role": "user", "content": f"Task: {question}"},
    ]

def latent_critic(question: str, context: str) -> List[Dict[str, str]]:
    """Latent Agent 3 – CRITIC."""
    return [
        {"role": "system", "content": (
            "You are CRITIC. You have received internal reasoning context from "
            "PLANNER and SOLVER.\n"
            "Identify up to 5 specific problems or gaps in the chosen answer.\n"
            "Think about how to patch and improve it to final-draft quality."
        )},
        {"role": "user", "content": f"Task: {question}"},
    ]

def latent_verifier_gsm8k(question: str, context: str) -> List[Dict[str, str]]:
    """Latent Agent 4 – VERIFIER (final) for math."""
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
        {"role": "user", "content": f"Task: {question}"},
    ]

def latent_verifier_mcq(question: str, context: str) -> List[Dict[str, str]]:
    """Latent Agent 4 – VERIFIER (final) for MCQ."""
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
        {"role": "user", "content": f"Task: {question}"},
    ]


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_text_pipeline(
    model: LatentModel,
    question: str,
    agent_specs: List[Dict[str, Any]],
    max_intermediate: int = 300,
    max_final: int = 400,
) -> Tuple[str, RunStats]:
    stats = RunStats()
    accumulated_text = ""

    for spec in agent_specs:
        is_final = spec["is_final"]
        max_tok = max_final if is_final else max_intermediate
        messages = spec["prompt_fn"](question, accumulated_text)
        _, input_ids, attention_mask = model.prepare_chat_batch(
            [messages], add_generation_prompt=True,
        )
        stats.prompt_tokens += int(attention_mask.sum().item())

        t0 = time.perf_counter()
        texts, _ = model.generate_text_batch(
            input_ids, attention_mask,
            max_new_tokens=max_tok, temperature=0.6, top_p=0.95,
        )
        stats.wall_time_s += time.perf_counter() - t0

        output = texts[0].strip()
        gen_tok = count_tokens(model.tokenizer, output)
        stats.generated_tokens += gen_tok
        if not is_final:
            stats.intermediate_tokens += gen_tok
            accumulated_text += f"{output}\n"

    return output, stats


def run_latent_pipeline(
    model: LatentModel,
    pipeline: LatentPipeline,
    question: str,
) -> Tuple[str, RunStats]:
    stats = RunStats()
    for agent in pipeline.agents:
        msgs = agent.prompt_fn(question, "")
        rendered = model.render_chat(msgs, add_generation_prompt=True)
        stats.prompt_tokens += count_tokens(model.tokenizer, rendered)

    t0 = time.perf_counter()
    result = pipeline.run(question)
    stats.wall_time_s = time.perf_counter() - t0

    stats.generated_tokens = count_tokens(model.tokenizer, result.text)
    stats.intermediate_tokens = 0
    return result.text, stats


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_mcq(question: str, choices: List[str]) -> str:
    """Format a multiple-choice question with labeled options."""
    labels = ["A", "B", "C", "D"]
    opts = "\n".join(f"  {labels[i]}. {c}" for i, c in enumerate(choices))
    return f"{question}\n\n{opts}"


def format_arc_mcq(question: str, choices: Dict) -> str:
    labels = choices["label"]
    texts = choices["text"]
    opts = "\n".join(f"  {l}. {t}" for l, t in zip(labels, texts))
    return f"{question}\n\n{opts}"


def print_table(title: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))
    total_w = sum(widths.values()) + 3 * (len(columns) - 1)
    header = " | ".join(col.ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    print(f"\n{'=' * total_w}")
    print(f"  {title}")
    print(f"{'=' * total_w}")
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))
    print()


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    set_seed(42)
    print(f"\n  Loading model: {SMALL_MODEL} ...")
    m = LatentModel(SMALL_MODEL, device="cpu", realign=True)
    print("  Model loaded.\n")
    return m


# ===========================================================================
# GSM8K Benchmark
# ===========================================================================

@pytest.mark.integration
class TestGSM8K:
    """Grade School Math 8K -- the standard math reasoning benchmark."""

    @pytest.fixture(scope="class")
    def gsm8k_data(self):
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return list(ds.select(range(min(N_GSM8K, len(ds)))))

    def test_gsm8k_comparison(self, model, gsm8k_data):
        set_seed(42)
        n = len(gsm8k_data)
        print(f"\n  GSM8K: evaluating {n} questions\n")

        configs = {
            "Single Agent": {
                "type": "text",
                "specs": [{"prompt_fn": single_agent_gsm8k, "is_final": True}],
            },
            "Non-Latent 4-Agent": {
                "type": "text",
                "specs": [
                    {"prompt_fn": planner_text, "is_final": False},
                    {"prompt_fn": solver_text, "is_final": False},
                    {"prompt_fn": critic_text, "is_final": False},
                    {"prompt_fn": verifier_text_gsm8k, "is_final": True},
                ],
            },
            "Latent 4-Agent (steps=10)": {
                "type": "latent",
                "agents": [
                    Agent(name="Planner", role="planner", prompt_fn=latent_planner),
                    Agent(name="Solver", role="solver", prompt_fn=latent_solver),
                    Agent(name="Critic", role="critic", prompt_fn=latent_critic),
                    Agent(name="Verifier", role="verifier", prompt_fn=latent_verifier_gsm8k, is_final=True),
                ],
                "latent_steps": 10,
            },
        }

        summary = []

        for cfg_name, cfg in configs.items():
            set_seed(42)
            correct = 0
            total = RunStats()

            if cfg["type"] == "latent":
                pipeline = LatentPipeline(
                    model, cfg["agents"], latent_steps=cfg["latent_steps"],
                    max_new_tokens=400, temperature=0.6,
                )

            for sample in gsm8k_data:
                q = sample["question"]
                gt = sample["answer"]

                if cfg["type"] == "text":
                    answer, stats = run_text_pipeline(model, q, cfg["specs"])
                else:
                    answer, stats = run_latent_pipeline(model, pipeline, q)

                if check_gsm8k(answer, gt):
                    correct += 1
                total.prompt_tokens += stats.prompt_tokens
                total.generated_tokens += stats.generated_tokens
                total.intermediate_tokens += stats.intermediate_tokens
                total.wall_time_s += stats.wall_time_s

            acc = correct / n * 100
            summary.append({
                "Pipeline": cfg_name,
                "Accuracy": f"{correct}/{n} ({acc:.1f}%)",
                "Prompt Tok": total.prompt_tokens,
                "Intermed Tok": total.intermediate_tokens,
                "Final Tok": total.generated_tokens - total.intermediate_tokens,
                "Total Gen": total.generated_tokens,
                "All Tok": total.prompt_tokens + total.generated_tokens,
                "Time": f"{total.wall_time_s:.1f}s",
            })

        print_table(
            f"GSM8K BENCHMARK ({n} questions)",
            summary,
            ["Pipeline", "Accuracy", "Prompt Tok", "Intermed Tok", "Final Tok",
             "Total Gen", "All Tok", "Time"],
        )

        # Token savings
        text_gen = summary[1]["Total Gen"]
        lat_gen = summary[2]["Total Gen"]
        if text_gen > 0:
            print(f"  Token savings (latent vs non-latent): {(1 - lat_gen/text_gen)*100:.0f}%")
        if summary[1]["Time"] != "0.0s":
            t_time = float(summary[1]["Time"].rstrip("s"))
            l_time = float(summary[2]["Time"].rstrip("s"))
            if l_time > 0:
                print(f"  Speedup: {t_time/l_time:.2f}x\n")

        assert len(summary) == 3


# ===========================================================================
# MMLU Benchmark
# ===========================================================================

@pytest.mark.integration
class TestMMLU:
    """Massive Multitask Language Understanding -- the gold-standard LLM benchmark."""

    @pytest.fixture(scope="class")
    def mmlu_data(self):
        ds = load_dataset("cais/mmlu", "all", split="test")
        # Sample across subjects for diversity
        ds_shuffled = ds.shuffle(seed=42)
        return list(ds_shuffled.select(range(min(N_MMLU, len(ds_shuffled)))))

    def test_mmlu_comparison(self, model, mmlu_data):
        set_seed(42)
        n = len(mmlu_data)
        print(f"\n  MMLU: evaluating {n} questions\n")

        configs = {
            "Single Agent": {
                "type": "text",
                "specs": [{"prompt_fn": single_agent_mcq, "is_final": True}],
            },
            "Non-Latent 4-Agent": {
                "type": "text",
                "specs": [
                    {"prompt_fn": planner_text, "is_final": False},
                    {"prompt_fn": solver_text, "is_final": False},
                    {"prompt_fn": critic_text, "is_final": False},
                    {"prompt_fn": verifier_text_mcq, "is_final": True},
                ],
            },
            "Latent 4-Agent (steps=10)": {
                "type": "latent",
                "agents": [
                    Agent(name="Planner", role="planner", prompt_fn=latent_planner),
                    Agent(name="Solver", role="solver", prompt_fn=latent_solver),
                    Agent(name="Critic", role="critic", prompt_fn=latent_critic),
                    Agent(name="Verifier", role="verifier", prompt_fn=latent_verifier_mcq, is_final=True),
                ],
                "latent_steps": 10,
            },
        }

        summary = []
        subject_stats: Dict[str, Dict[str, List[bool]]] = {}

        for cfg_name, cfg in configs.items():
            set_seed(42)
            correct = 0
            total = RunStats()

            if cfg["type"] == "latent":
                pipeline = LatentPipeline(
                    model, cfg["agents"], latent_steps=cfg["latent_steps"],
                    max_new_tokens=256, temperature=0.6,
                )

            for sample in mmlu_data:
                q_text = format_mcq(sample["question"], sample["choices"])
                correct_idx = sample["answer"]

                if cfg["type"] == "text":
                    answer, stats = run_text_pipeline(model, q_text, cfg["specs"], max_final=256)
                else:
                    answer, stats = run_latent_pipeline(model, pipeline, q_text)

                hit = check_mmlu(answer, correct_idx, sample["choices"])
                if hit:
                    correct += 1

                subj = sample.get("subject", "unknown")
                subject_stats.setdefault(cfg_name, {}).setdefault(subj, []).append(hit)

                total.prompt_tokens += stats.prompt_tokens
                total.generated_tokens += stats.generated_tokens
                total.intermediate_tokens += stats.intermediate_tokens
                total.wall_time_s += stats.wall_time_s

            acc = correct / n * 100
            summary.append({
                "Pipeline": cfg_name,
                "Accuracy": f"{correct}/{n} ({acc:.1f}%)",
                "Prompt Tok": total.prompt_tokens,
                "Intermed Tok": total.intermediate_tokens,
                "Final Tok": total.generated_tokens - total.intermediate_tokens,
                "Total Gen": total.generated_tokens,
                "All Tok": total.prompt_tokens + total.generated_tokens,
                "Time": f"{total.wall_time_s:.1f}s",
            })

        print_table(
            f"MMLU BENCHMARK ({n} questions, diverse subjects)",
            summary,
            ["Pipeline", "Accuracy", "Prompt Tok", "Intermed Tok", "Final Tok",
             "Total Gen", "All Tok", "Time"],
        )

        text_gen = summary[1]["Total Gen"]
        lat_gen = summary[2]["Total Gen"]
        if text_gen > 0:
            print(f"  Token savings (latent vs non-latent): {(1 - lat_gen/text_gen)*100:.0f}%")
        t_time = float(summary[1]["Time"].rstrip("s"))
        l_time = float(summary[2]["Time"].rstrip("s"))
        if l_time > 0:
            print(f"  Speedup: {t_time/l_time:.2f}x")

        # Random baseline for 4-choice is 25%
        print(f"  (Random baseline for 4-choice: 25.0%)\n")

        assert len(summary) == 3


# ===========================================================================
# ARC-Easy Benchmark
# ===========================================================================

@pytest.mark.integration
class TestARC:
    """AI2 Reasoning Challenge (Easy) -- standard science reasoning benchmark."""

    @pytest.fixture(scope="class")
    def arc_data(self):
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        ds_shuffled = ds.shuffle(seed=42)
        return list(ds_shuffled.select(range(min(N_ARC, len(ds_shuffled)))))

    def test_arc_comparison(self, model, arc_data):
        set_seed(42)
        n = len(arc_data)
        print(f"\n  ARC-Easy: evaluating {n} questions\n")

        configs = {
            "Single Agent": {
                "type": "text",
                "specs": [{"prompt_fn": single_agent_mcq, "is_final": True}],
            },
            "Non-Latent 4-Agent": {
                "type": "text",
                "specs": [
                    {"prompt_fn": planner_text, "is_final": False},
                    {"prompt_fn": solver_text, "is_final": False},
                    {"prompt_fn": critic_text, "is_final": False},
                    {"prompt_fn": verifier_text_mcq, "is_final": True},
                ],
            },
            "Latent 4-Agent (steps=10)": {
                "type": "latent",
                "agents": [
                    Agent(name="Planner", role="planner", prompt_fn=latent_planner),
                    Agent(name="Solver", role="solver", prompt_fn=latent_solver),
                    Agent(name="Critic", role="critic", prompt_fn=latent_critic),
                    Agent(name="Verifier", role="verifier", prompt_fn=latent_verifier_mcq, is_final=True),
                ],
                "latent_steps": 10,
            },
        }

        summary = []

        for cfg_name, cfg in configs.items():
            set_seed(42)
            correct = 0
            total = RunStats()

            if cfg["type"] == "latent":
                pipeline = LatentPipeline(
                    model, cfg["agents"], latent_steps=cfg["latent_steps"],
                    max_new_tokens=256, temperature=0.6,
                )

            for sample in arc_data:
                q_text = format_arc_mcq(sample["question"], sample["choices"])
                correct_key = sample["answerKey"]

                if cfg["type"] == "text":
                    answer, stats = run_text_pipeline(model, q_text, cfg["specs"], max_final=256)
                else:
                    answer, stats = run_latent_pipeline(model, pipeline, q_text)

                hit = check_arc(answer, correct_key, sample["choices"])
                if hit:
                    correct += 1
                total.prompt_tokens += stats.prompt_tokens
                total.generated_tokens += stats.generated_tokens
                total.intermediate_tokens += stats.intermediate_tokens
                total.wall_time_s += stats.wall_time_s

            acc = correct / n * 100
            summary.append({
                "Pipeline": cfg_name,
                "Accuracy": f"{correct}/{n} ({acc:.1f}%)",
                "Prompt Tok": total.prompt_tokens,
                "Intermed Tok": total.intermediate_tokens,
                "Final Tok": total.generated_tokens - total.intermediate_tokens,
                "Total Gen": total.generated_tokens,
                "All Tok": total.prompt_tokens + total.generated_tokens,
                "Time": f"{total.wall_time_s:.1f}s",
            })

        print_table(
            f"ARC-Easy BENCHMARK ({n} questions)",
            summary,
            ["Pipeline", "Accuracy", "Prompt Tok", "Intermed Tok", "Final Tok",
             "Total Gen", "All Tok", "Time"],
        )

        text_gen = summary[1]["Total Gen"]
        lat_gen = summary[2]["Total Gen"]
        if text_gen > 0:
            print(f"  Token savings (latent vs non-latent): {(1 - lat_gen/text_gen)*100:.0f}%")
        t_time = float(summary[1]["Time"].rstrip("s"))
        l_time = float(summary[2]["Time"].rstrip("s"))
        if l_time > 0:
            print(f"  Speedup: {t_time/l_time:.2f}x")
        print(f"  (Random baseline for 4-choice: 25.0%)\n")

        assert len(summary) == 3


# ===========================================================================
# Grand Summary
# ===========================================================================

@pytest.mark.integration
class TestGrandSummary:
    """Run all three benchmarks and print a combined summary."""

    @pytest.fixture(scope="class")
    def all_data(self):
        gsm = list(load_dataset("openai/gsm8k", "main", split="test").select(range(N_GSM8K)))
        mmlu = list(load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42).select(range(N_MMLU)))
        arc = list(load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").shuffle(seed=42).select(range(N_ARC)))
        return {"gsm8k": gsm, "mmlu": mmlu, "arc": arc}

    def test_grand_summary(self, model, all_data):
        """Combined results across all benchmarks."""
        set_seed(42)

        pipelines = {
            "Single Agent": lambda bench: {
                "type": "text",
                "specs": [{"prompt_fn": single_agent_gsm8k if bench == "gsm8k" else single_agent_mcq, "is_final": True}],
            },
            "Non-Latent 4-Agent": lambda bench: {
                "type": "text",
                "specs": [
                    {"prompt_fn": planner_text, "is_final": False},
                    {"prompt_fn": solver_text, "is_final": False},
                    {"prompt_fn": critic_text, "is_final": False},
                    {"prompt_fn": verifier_text_gsm8k if bench == "gsm8k" else verifier_text_mcq, "is_final": True},
                ],
            },
            "Latent 4-Agent": lambda bench: {
                "type": "latent",
                "agents": [
                    Agent(name="Planner", role="planner", prompt_fn=latent_planner),
                    Agent(name="Solver", role="solver", prompt_fn=latent_solver),
                    Agent(name="Critic", role="critic", prompt_fn=latent_critic),
                    Agent(name="Verifier", role="verifier", is_final=True,
                          prompt_fn=latent_verifier_gsm8k if bench == "gsm8k" else latent_verifier_mcq),
                ],
                "latent_steps": 10,
            },
        }

        results = []

        for pipe_name, pipe_factory in pipelines.items():
            row = {"Pipeline": pipe_name}
            total_correct = 0
            total_n = 0
            total_gen = 0
            total_time = 0.0

            for bench_name, data in all_data.items():
                set_seed(42)
                cfg = pipe_factory(bench_name)
                n = len(data)
                correct = 0
                gen = 0
                t = 0.0

                if cfg["type"] == "latent":
                    pipeline = LatentPipeline(
                        model, cfg["agents"], latent_steps=cfg["latent_steps"],
                        max_new_tokens=400 if bench_name == "gsm8k" else 256,
                        temperature=0.6,
                    )

                for sample in data:
                    if bench_name == "gsm8k":
                        q = sample["question"]
                    elif bench_name == "mmlu":
                        q = format_mcq(sample["question"], sample["choices"])
                    else:
                        q = format_arc_mcq(sample["question"], sample["choices"])

                    if cfg["type"] == "text":
                        answer, stats = run_text_pipeline(
                            model, q, cfg["specs"],
                            max_final=400 if bench_name == "gsm8k" else 256,
                        )
                    else:
                        answer, stats = run_latent_pipeline(model, pipeline, q)

                    if bench_name == "gsm8k":
                        hit = check_gsm8k(answer, sample["answer"])
                    elif bench_name == "mmlu":
                        hit = check_mmlu(answer, sample["answer"], sample["choices"])
                    else:
                        hit = check_arc(answer, sample["answerKey"], sample["choices"])

                    if hit:
                        correct += 1
                    gen += stats.generated_tokens
                    t += stats.wall_time_s

                acc = correct / n * 100
                row[f"{bench_name.upper()} ({n})"] = f"{correct}/{n} ({acc:.0f}%)"
                total_correct += correct
                total_n += n
                total_gen += gen
                total_time += t

            overall_acc = total_correct / total_n * 100
            row["Overall"] = f"{total_correct}/{total_n} ({overall_acc:.0f}%)"
            row["Total Gen Tok"] = total_gen
            row["Total Time"] = f"{total_time:.0f}s"
            results.append(row)

        cols = ["Pipeline",
                f"GSM8K ({len(all_data['gsm8k'])})",
                f"MMLU ({len(all_data['mmlu'])})",
                f"ARC ({len(all_data['arc'])})",
                "Overall", "Total Gen Tok", "Total Time"]

        print_table("GRAND SUMMARY -- Standard LLM Benchmarks", results, cols)

        # Token and speed comparison
        if results[1]["Total Gen Tok"] > 0:
            savings = (1 - results[2]["Total Gen Tok"] / results[1]["Total Gen Tok"]) * 100
            print(f"  Latent vs Non-Latent token savings: {savings:.0f}%")
        t1 = float(results[1]["Total Time"].rstrip("s"))
        t2 = float(results[2]["Total Time"].rstrip("s"))
        if t2 > 0:
            print(f"  Latent vs Non-Latent speedup: {t1/t2:.1f}x\n")

        assert len(results) == 3
