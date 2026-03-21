#!/usr/bin/env python3
"""Example: using latent-agents with any HuggingFace model.

Usage:
    python example.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda
    python example.py --model HuggingFaceTB/SmolLM2-135M-Instruct --device cpu
    python example.py --model Qwen/Qwen3-4B --latent_steps 30
"""

import argparse
from latent_agents import Agent, LatentModel, LatentPipeline, set_seed


def planner_prompt(question: str, context: str):
    return [
        {"role": "system", "content": "You are a helpful planning assistant."},
        {"role": "user", "content": (
            f"You are a Planner Agent. Design a clear step-by-step plan to "
            f"solve this problem. Do NOT produce the final answer.\n\n"
            f"Question: {question}"
        )},
    ]


def critic_prompt(question: str, context: str):
    return [
        {"role": "system", "content": "You are a helpful critical reviewer."},
        {"role": "user", "content": (
            f"You are a Critic Agent. You have received latent context from "
            f"a Planner. Review the plan and provide constructive feedback.\n\n"
            f"Question: {question}"
        )},
    ]


def refiner_prompt(question: str, context: str):
    return [
        {"role": "system", "content": "You are a helpful assistant that refines plans."},
        {"role": "user", "content": (
            f"You are a Refiner Agent. Based on latent context from the "
            f"Planner and Critic, produce a refined plan.\n\n"
            f"Question: {question}"
        )},
    ]


def solver_prompt(question: str, context: str):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": (
            f"You have received latent reasoning context from previous agents. "
            f"Reason step by step and output the final answer inside "
            f"\\boxed{{YOUR_ANSWER}}.\n\n"
            f"Question: {question}"
        )},
    ]


def main():
    parser = argparse.ArgumentParser(description="latent-agents example")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent_steps", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--no_realign", action="store_true")
    parser.add_argument("--n_samples", type=int, default=1,
                        help="Self-consistency voting: generate N answers and majority-vote")
    parser.add_argument("--convergence_threshold", type=float, default=None,
                        help="Stop latent steps early when hidden-state change < threshold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", type=str,
                        default="If a train travels 60 km in 40 minutes, what is its speed in km/h?")
    args = parser.parse_args()

    set_seed(args.seed)

    print(f"Loading model: {args.model} on {args.device} ...")
    model = LatentModel(args.model, device=args.device, realign=not args.no_realign)
    print("Model loaded.\n")

    # Per-agent latent steps: Planner thinks more, Critic/Refiner less
    agents = [
        Agent(name="Planner", role="planner", prompt_fn=planner_prompt,
              latent_steps=args.latent_steps),       # uses CLI value
        Agent(name="Critic",  role="critic",  prompt_fn=critic_prompt,
              latent_steps=max(1, args.latent_steps // 2)),  # half as many
        Agent(name="Refiner", role="refiner", prompt_fn=refiner_prompt),  # uses pipeline default
        Agent(name="Solver",  role="solver",  prompt_fn=solver_prompt, is_final=True),
    ]

    pipeline = LatentPipeline(
        model, agents,
        latent_steps=args.latent_steps,
        max_new_tokens=args.max_new_tokens,
        convergence_threshold=args.convergence_threshold,
        n_samples=args.n_samples,
    )

    print(f"Question: {args.question}")
    print(f"Running {len(agents)} agents (default latent_steps={args.latent_steps}, "
          f"n_samples={args.n_samples}) ...\n")

    result = pipeline.run(args.question)

    print("=" * 60)
    print("AGENT TRACES")
    print("=" * 60)
    for trace in result.agent_traces:
        actual = trace.get("latent_steps", "-")
        configured = trace.get("latent_steps_configured", "")
        steps_str = f"actual={actual}"
        if configured:
            steps_str += f" (configured={configured})"
        print(f"\n--- {trace['name']} ({trace['role']}) | {steps_str} ---")
        if trace.get("candidates"):
            print(f"[Voted from {len(trace['candidates'])} candidates]")
        if trace["output"]:
            print(trace["output"][:500])
        else:
            print("[Latent only -- no text output]")

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.text)


if __name__ == "__main__":
    main()
