#!/usr/bin/env python3
"""Split the CLI syntax JSONL dataset into train/valid/test for mlx_lm.lora."""
import argparse
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(
        Path.home() / "hammerhead/hammerhead_istax_copy/docs/datasets/rl-cli-commands.jsonl"
    ))
    p.add_argument("--out-dir", default="./cli_data")
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    lines = [l.strip() for l in Path(args.source).read_text().splitlines() if l.strip()]
    examples = [json.loads(l) for l in lines]

    # Dedup on the raw (user, assistant) content pair.
    seen = set()
    deduped = []
    for ex in examples:
        key = tuple(m["content"] for m in ex["messages"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ex)
    dropped = len(examples) - len(deduped)

    random.Random(args.seed).shuffle(deduped)

    n = len(deduped)
    n_test = round(n * args.test_frac)
    n_valid = round(n * args.valid_frac)
    n_train = n - n_valid - n_test

    train = deduped[:n_train]
    valid = deduped[n_train:n_train + n_valid]
    test = deduped[n_train + n_valid:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")

    print(f"source examples: {len(examples)} (dropped {dropped} exact duplicates)")
    print(f"train: {len(train)}  valid: {len(valid)}  test: {len(test)}")
    print(f"written to {out_dir}/")


if __name__ == "__main__":
    main()
