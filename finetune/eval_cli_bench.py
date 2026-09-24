#!/usr/bin/env python3
"""Benchmark a model (base or fine-tuned) against the held-out CLI syntax test set.

Usage:
    python3 eval_cli_bench.py \\
        --base-url http://192.168.1.168:8080/v1 \\
        --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --test-file ./cli_data/test.jsonl \\
        --output ./results_base.json
"""
import argparse
import difflib
import json
import re
import statistics
import sys
import time
from pathlib import Path

from openai import OpenAI


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().casefold())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default="not-needed")
    p.add_argument("--test-file", default="./cli_data/test.jsonl")
    p.add_argument("--output", required=True)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    examples = [json.loads(l) for l in Path(args.test_file).read_text().splitlines() if l.strip()]
    if args.limit:
        examples = examples[:args.limit]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    results = []
    for i, ex in enumerate(examples):
        messages = ex["messages"]
        expected = messages[-1]["content"]
        prompt_messages = messages[:-1]

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=prompt_messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            actual = (resp.choices[0].message.content or "").strip()
            usage = resp.usage
            prompt_tokens = usage.prompt_tokens if usage else None
            completion_tokens = usage.completion_tokens if usage else None
            error = None
        except Exception as e:
            actual = ""
            prompt_tokens = None
            completion_tokens = None
            error = str(e)
        elapsed = time.time() - t0
        tokens_per_sec = (completion_tokens / elapsed) if completion_tokens and elapsed > 0 else None

        exact_match = normalize(actual) == normalize(expected)
        similarity = difflib.SequenceMatcher(None, normalize(actual), normalize(expected)).ratio()

        results.append({
            "index": i,
            "user_input": next((m["content"] for m in prompt_messages if m["role"] == "user"), ""),
            "expected": expected,
            "actual": actual,
            "exact_match": exact_match,
            "similarity": round(similarity, 4),
            "elapsed_s": round(elapsed, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_sec": round(tokens_per_sec, 1) if tokens_per_sec else None,
            "error": error,
        })

        status = "OK " if exact_match else ("~~~" if similarity > 0.7 else "XXX")
        print(f"[{i+1}/{len(examples)}] {status} ({elapsed:.1f}s, {completion_tokens or 0} tok) {expected!r} <-> {actual!r}", file=sys.stderr)

    n = len(results)
    ok = [r for r in results if not r["error"]]
    exact_matches = sum(r["exact_match"] for r in results)
    mean_similarity = sum(r["similarity"] for r in results) / n if n else 0.0
    errors = sum(1 for r in results if r["error"])

    elapsed_values = [r["elapsed_s"] for r in ok]
    completion_token_values = [r["completion_tokens"] for r in ok if r["completion_tokens"] is not None]
    tps_values = [r["tokens_per_sec"] for r in ok if r["tokens_per_sec"] is not None]

    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "test_file": args.test_file,
        "n_examples": n,
        "exact_match_count": exact_matches,
        "exact_match_rate": round(exact_matches / n, 4) if n else 0.0,
        "mean_similarity": round(mean_similarity, 4),
        "errors": errors,
        "total_wall_time_s": round(sum(elapsed_values), 2),
        "mean_elapsed_s": round(statistics.mean(elapsed_values), 2) if elapsed_values else None,
        "median_elapsed_s": round(statistics.median(elapsed_values), 2) if elapsed_values else None,
        "p95_elapsed_s": round(statistics.quantiles(elapsed_values, n=20)[18], 2) if len(elapsed_values) >= 20 else None,
        "mean_completion_tokens": round(statistics.mean(completion_token_values), 1) if completion_token_values else None,
        "total_completion_tokens": sum(completion_token_values) if completion_token_values else None,
        "mean_tokens_per_sec": round(statistics.mean(tps_values), 1) if tps_values else None,
    }

    Path(args.output).write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nfull results written to {args.output}")


if __name__ == "__main__":
    main()
