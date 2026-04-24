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

## Notes

- Single GPU, bf16, gradient checkpointing. No multi-GPU / DeepSpeed.
- Features are always encoded on-the-fly (no cache) — LoRA weights change every run, so any cached features would be stale.
- `server_prefix` in config points to where UCF-Crime video files live on disk.
- W&B: set `[training].wandb_project = ""` to disable, or pass `--no-wandb` to `scripts/train.py`.
