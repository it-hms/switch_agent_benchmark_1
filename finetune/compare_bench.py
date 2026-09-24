#!/usr/bin/env python3
"""Compare two eval_cli_bench.py result files (e.g. base model vs fine-tuned)."""
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True, help="results file for the base model")
    p.add_argument("--after", required=True, help="results file for the fine-tuned model")
    args = p.parse_args()

    before = json.loads(open(args.before).read())
    after = json.loads(open(args.after).read())

    if before["summary"]["test_file"] != after["summary"]["test_file"]:
        print("WARNING: the two runs used different test files - comparison may not be apples-to-apples")

    print("=== summary ===")
    print(f"{'metric':<24}{'before':<14}{'after':<14}{'delta':<10}")
    for key in (
        "exact_match_rate",
        "mean_similarity",
        "mean_elapsed_s",
        "median_elapsed_s",
        "p95_elapsed_s",
        "mean_completion_tokens",
        "mean_tokens_per_sec",
        "total_wall_time_s",
        "total_completion_tokens",
    ):
        b, a = before["summary"].get(key), after["summary"].get(key)
        if b is None or a is None:
            continue
        print(f"{key:<24}{b:<14}{a:<14}{a - b:+.4f}")

    before_by_input = {r["user_input"]: r for r in before["results"]}
    after_by_input = {r["user_input"]: r for r in after["results"]}
    common = set(before_by_input) & set(after_by_input)

    improved = [i for i in common if not before_by_input[i]["exact_match"] and after_by_input[i]["exact_match"]]
    regressed = [i for i in common if before_by_input[i]["exact_match"] and not after_by_input[i]["exact_match"]]

    print(f"\n{len(improved)} examples flipped wrong -> right")
    for i in improved[:10]:
        print(f"  input: {i!r}")
        print(f"    expected: {after_by_input[i]['expected']!r}")
        print(f"    before:   {before_by_input[i]['actual']!r}")
        print(f"    after:    {after_by_input[i]['actual']!r}")

    print(f"\n{len(regressed)} examples flipped right -> wrong (regressions)")
    for i in regressed[:10]:
        print(f"  input: {i!r}")
        print(f"    expected: {after_by_input[i]['expected']!r}")
        print(f"    before:   {before_by_input[i]['actual']!r}")
        print(f"    after:    {after_by_input[i]['actual']!r}")


if __name__ == "__main__":
    main()
