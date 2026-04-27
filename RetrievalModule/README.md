# VidAnomalyRetrieval

LoRA fine-tuning of `Qwen/Qwen3-VL-Embedding-2B` for text→video retrieval on UCF-Crime.

Two-phase pipeline:
1. **Phase 1 (warmup):** symmetric InfoNCE, in-batch negatives, category-stratified batches.
2. **Phase 2 (hard-neg mining):** re-mine K hard negatives per query before each epoch; `L_t2v^hard + α · L_v2t^in-batch`.

Design & plan:
- `docs/superpowers/specs/2026-04-24-refactor-2phase-design.md`
- `docs/superpowers/plans/2026-04-24-refactor-2phase-implementation.md`

## Layout

```
src/var/            # importable package
  config.py         # TOML → typed RunConfig
  data.py           # QueryVideoDataset, CategoryStratifiedSampler, ContrastiveCollator
  model.py          # QwenEmbeddingEngine + LoRA attach/load
  losses.py         # symmetric_infonce, hard_neg_infonce_t2v, phase2_combined_loss
  metrics.py        # R@K, MedR, mAP
  mining.py         # encode_corpus + mine_hard_negatives (with fallback ladder)
  trainer.py        # ContrastiveTrainer
scripts/
  prepare_data.py   # one-shot dedup
  smoke_test.py     # LoRA attach + 1 forward pass
  train.py          # phase1 / phase2 dispatcher
  evaluate.py       # on-the-fly R@K/MedR/mAP (zero-shot or adapter)
configs/
  phase1.toml
  phase2.toml
Qwen3-VL-Embedding/   # vendored upstream
data/                 # dataset JSON + prep scripts
```

## Quickstart

```bash
# 0. Install editable
pip install -e .

# 1. Prepare data (one-shot, dedup queries: 1610 → 1574)
python scripts/prepare_data.py

# 2. Zero-shot baseline (required — reference point for any fine-tuning)
python scripts/evaluate.py --config configs/phase1.toml --zero-shot

# 3. Smoke checks
python scripts/smoke_test.py --config configs/phase1.toml --skip-forward
python scripts/smoke_test.py --config configs/phase1.toml

# 4. Phase 1 training (warmup, symmetric InfoNCE, bs=16, 4 epochs)
python scripts/train.py --config configs/phase1.toml

# 5. Evaluate phase 1
python scripts/evaluate.py \
  --config configs/phase1.toml \
  --adapter outputs/phase1-warmup/final_adapter \
  --output-json outputs/eval_phase1.json

# 6. Phase 2 training (hard-neg mining, bs=4, 3 epochs, remine per epoch)
python scripts/train.py --config configs/phase2.toml

# 7. Evaluate phase 2 (final)
python scripts/evaluate.py \
  --config configs/phase2.toml \
  --adapter outputs/phase2-hardneg/final_adapter \
  --output-json outputs/eval_phase2.json
```

## Logs

Every entry script auto-writes its output to a log file (console is also tee'd), so `nohup python scripts/train.py --config configs/phase1.toml &` leaves a readable log behind without manual `> run.log`.

- `scripts/train.py`    → `outputs/<phase-dir>/logs/{phase1,phase2}-YYYYMMDD-HHMMSS.log`
- `scripts/evaluate.py` → `outputs/logs/eval-YYYYMMDD-HHMMSS.log` (or `eval_baseline-...` with `--zero-shot`)
- `scripts/smoke_test.py` → `outputs/logs/smoke-YYYYMMDD-HHMMSS.log`

Lines use the format `[HH:MM:SS] [scope] message`. Tail with `tail -f outputs/phase1-warmup/logs/phase1-*.log`.

## W&B

- Set `[training].wandb_project` in the config to enable. Pass `--no-wandb` to disable on a single run.
- Logged per step: `train/loss`, `train/grad_norm`, `train/lr`, `train/epoch`.
- Logged per eval step: `eval/loss`, `eval/top1`.
- Full config (model, lora, data, training, phase2) is attached to the W&B run config.

## Hugging Face Hub push

Optional `[hub]` section in each config:

```toml
[hub]
push_to_hub = false     # flip to true to push after training finishes
model_id = "your-name/qwen3-vl-embedding-phase1"
private = true
```

After `trainer` finishes (adapter saved to `final_adapter/`), `scripts/train.py` pushes to the hub if `push_to_hub=true`. Override with `--no-push` to skip on a single run. You must `huggingface-cli login` (or set `HF_TOKEN`) before the first push.

## Notes

- Single GPU, bf16, gradient checkpointing. No multi-GPU / DeepSpeed.
- Features are always encoded on-the-fly (no cache) — LoRA weights change every run, so any cached features would be stale.
- `server_prefix` in config points to where UCF-Crime video files live on disk.
