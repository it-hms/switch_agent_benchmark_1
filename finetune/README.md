# CLI syntax fine-tuning benchmark

Fine-tunes the local MLX model on hh_switch CLI syntax examples and measures
whether it actually helped, using held-out test data.

Source dataset: `~/hammerhead/hammerhead_istax_copy/docs/datasets/rl-cli-commands.jsonl`
(600 `{"messages": [...]}` chat-format examples, one CLI command per example).

Mac (`192.168.1.168`) runs `mlx_lm.server`; this box (WSL) runs the scripts below
against it over the network.

## 1. Split the data

```bash
cd ~/harness/finetune
python3 prepare_data.py
```

Writes `cli_data/train.jsonl` (480), `cli_data/valid.jsonl` (60), `cli_data/test.jsonl` (60).
Re-run any time the source dataset changes — it dedupes and reshuffles with a fixed seed.

## 2. Baseline benchmark (before fine-tuning)

Confirm the server is up first:

```bash
curl -s -o /dev/null -w "HTTP_STATUS:%{http_code}\n" http://192.168.1.168:8080/v1/models
```

Then run the benchmark against the current (un-tuned) model:

```bash
python3 eval_cli_bench.py \
  --base-url http://192.168.1.168:8080/v1 \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --output ./results_base.json
```

Takes ~10s/example (60 examples ≈ 10 min). Prints per-example progress to stderr
and writes full results + summary (exact-match rate, similarity, timing, tokens/sec)
to the `--output` file.

## 3. Copy the split data to the Mac

```bash
scp -r cli_data <mac-user>@192.168.1.168:~/cli_data
```

## 4. Fine-tune (on the Mac, via SSH)

Since SSH sessions don't stay open, run it detached:

```bash
source ~/mlx-env/bin/activate
nohup mlx_lm.lora \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --train --data ~/cli_data \
  --iters 1000 --batch-size 4 --learning-rate 1e-5 \
  --adapter-path ~/adapters \
  > ~/lora_train.log 2>&1 &
disown
```

Watch progress with `tail -f ~/lora_train.log` — check that validation loss keeps
dropping alongside train loss; if val loss starts climbing, stop early (fewer `--iters`).

## 5. Restart the server with the adapter loaded (on the Mac)

```bash
pkill -f "mlx_lm.server"
nohup mlx_lm.server \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --adapter-path ~/adapters \
  --host 0.0.0.0 --port 8080 \
  > ~/mlx_server.log 2>&1 &
disown
```

## 6. Re-run the benchmark and compare

```bash
python3 eval_cli_bench.py \
  --base-url http://192.168.1.168:8080/v1 \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --output ./results_finetuned.json

python3 compare_bench.py --before results_base.json --after results_finetuned.json
```

Shows before/after/delta for exact-match rate, similarity, timing, and token counts,
plus the specific examples that flipped wrong→right or right→wrong.

## Notes

- `eval_cli_bench.py` only compares generated text against the expected command — it
  does **not** execute generated commands against the real switch. Don't automate that;
  some generated commands could be destructive (e.g. `reload`, `write erase`) and should
  only be run against real hardware manually, with review.
- `cli_data/` and `results_*.json` are gitignored (regenerable / run outputs, not source).
