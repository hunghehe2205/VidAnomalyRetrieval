from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from torch.utils.data import DataLoader

from collator import QueryVideoCollator
from dataset import QueryVideoDataset
from embedding import QwenEmbeddingEngine
from lora_utils import attach_lora, count_parameters, load_lora_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test LoRA attachment/loading on top of Qwen3-VL-Embedding."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.toml"),
        help="Path to config TOML.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=None,
        help="Optional path to a saved PEFT adapter. If omitted, initialize a fresh LoRA adapter.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Optional data file override for smoke test.",
    )
    parser.add_argument(
        "--server-prefix",
        type=str,
        default=None,
        help="Optional server prefix for relative video paths.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2,
        help="Number of rows to load for the forward smoke test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the forward smoke test.",
    )
    parser.add_argument(
        "--all-batches",
        action="store_true",
        help="Run all batches in the loaded subset instead of only the first batch.",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Only verify adapter attachment/loading without running a forward pass.",
    )
    parser.add_argument(
        "--trainable",
        action="store_true",
        help="Keep the loaded adapter trainable.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_data_path(repo_root: Path, data_cfg: Dict[str, Any], data_file: Path | None) -> Path:
    resolved = data_file or Path(str(data_cfg["train_file"]))
    if resolved.is_absolute():
        return resolved
    return repo_root / resolved


def build_loader(
    *,
    data_path: Path,
    data_cfg: Dict[str, Any],
    server_prefix: str,
    num_samples: int,
    batch_size: int,
    collator: QueryVideoCollator,
) -> Tuple[QueryVideoDataset, DataLoader]:
    dataset = QueryVideoDataset(
        data_path=str(data_path),
        query_column=str(data_cfg.get("query_column", "query")),
        video_column=str(data_cfg.get("video_column", "video")),
        server_prefix=server_prefix,
        max_samples=num_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty: {data_path}")

    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=False,
        collate_fn=collator,
    )
    return dataset, loader


def select_batches(loader: DataLoader, run_all_batches: bool) -> Iterable[Tuple[int, Dict[str, Any]]]:
    if run_all_batches:
        return enumerate(loader, start=1)

    first_batch = next(iter(loader), None)
    if first_batch is None:
        raise RuntimeError("No batch was produced by DataLoader.")
    return [(1, first_batch)]


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root / args.config)
    lora_cfg = config.get("lora", {})
    if not lora_cfg.get("enabled", False):
        raise ValueError("LoRA is disabled in the config.")

    engine = QwenEmbeddingEngine.from_config(config=config, repo_root=repo_root)
    base_model = engine.embedder.model

    if args.adapter_path is not None:
        adapter_path = args.adapter_path
        if not adapter_path.is_absolute():
            adapter_path = repo_root / adapter_path
        engine.embedder.model = load_lora_adapter(
            base_model,
            adapter_path=adapter_path,
            is_trainable=args.trainable,
        )
        mode = f"loaded adapter from {adapter_path}"
    else:
        engine.embedder.model = attach_lora(base_model, lora_cfg)
        if not args.trainable:
            engine.embedder.model.eval()
        mode = "attached fresh LoRA adapter from config"

    trainable_params, total_params = count_parameters(engine.embedder.model)
    print(f"Adapter test mode: {mode}")
    print(f"Wrapped model class: {engine.embedder.model.__class__.__name__}")
    print(f"Device: {engine.device}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")

    active_adapters = getattr(engine.embedder.model, "active_adapters", None)
    if active_adapters is not None:
        print(f"Active adapters: {active_adapters}")

    print_trainable_parameters = getattr(engine.embedder.model, "print_trainable_parameters", None)
    if callable(print_trainable_parameters):
        print_trainable_parameters()

    if args.skip_forward:
        print("Adapter attach/load smoke test succeeded.")
        return

    data_cfg = config["data"]
    server_prefix = (
        args.server_prefix if args.server_prefix is not None else str(data_cfg.get("server_prefix", ""))
    )
    data_path = resolve_data_path(repo_root=repo_root, data_cfg=data_cfg, data_file=args.data_file)
    collator = QueryVideoCollator(
        embedder=engine.embedder,
        fps=float(data_cfg.get("fps", 1)),
        max_frames=int(data_cfg.get("max_frames", 16)),
    )
    dataset, loader = build_loader(
        data_path=data_path,
        data_cfg=data_cfg,
        server_prefix=server_prefix,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        collator=collator,
    )

    print(f"Data file: {data_path}")
    print(f"Running adapter forward smoke test on {len(dataset)} samples.")

    for step, batch in select_batches(loader, args.all_batches):
        query_embeddings = engine.encode_processed_inputs(batch["query_inputs"])
        video_embeddings = engine.encode_processed_inputs(batch["video_inputs"])
        score_matrix = query_embeddings @ video_embeddings.T

        print(f"[Batch {step}]")
        print(f"  query/text shape : {tuple(query_embeddings.shape)}")
        print(f"  video shape      : {tuple(video_embeddings.shape)}")
        print(f"  score matrix     : {tuple(score_matrix.shape)}")
        print(f"  score diag       : {score_matrix.diag().detach().cpu().tolist()}")

    print("Adapter attach/load forward smoke test succeeded.")


if __name__ == "__main__":
    main()
