from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from collator import QueryVideoCollator
from dataset import QueryVideoDataset
from embedder_adapter import QwenEmbedderAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test one forward pass for text/video embeddings."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.toml"),
        help="Path to phase-1 config TOML.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help=(
            "Optional data file override for smoke test. "
            "If omitted, use config[data].smoke_file (fallback to config[data].train_file)."
        ),
    )
    parser.add_argument(
        "--server-prefix",
        type=str,
        default=None,
        help="Optional server prefix for relative video paths (overrides config[data].server_prefix).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of dataset rows to smoke-test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for smoke-test DataLoader.",
    )
    parser.add_argument(
        "--all-batches",
        action="store_true",
        help="Iterate all batches in the selected subset (default: only first batch).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def load_qwen_embedder_class(repo_root: Path):
    qwen_root = repo_root / "Qwen3-VL-Embedding"
    if not qwen_root.exists():
        raise FileNotFoundError(f"Missing folder: {qwen_root}")
    qwen_root_str = str(qwen_root.resolve())
    if qwen_root_str not in sys.path:
        sys.path.insert(0, qwen_root_str)
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

    return Qwen3VLEmbedder


def build_embedder(config: Dict[str, Any], repo_root: Path):
    model_cfg = config["model"]
    data_cfg = config["data"]
    Qwen3VLEmbedder = load_qwen_embedder_class(repo_root)

    init_kwargs: Dict[str, Any] = {
        "model_name_or_path": model_cfg["model_name_or_path"],
        "fps": data_cfg.get("fps", 1),
        "max_frames": data_cfg.get("max_frames", 16),
    }

    if torch.cuda.is_available():
        if "attn_implementation" in model_cfg:
            init_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
        init_kwargs["torch_dtype"] = torch.bfloat16

    return Qwen3VLEmbedder(**init_kwargs)


def move_tensor_dict_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def pool_last_hidden(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_positions = attention_mask.flip(dims=[1]).argmax(dim=1)
    col_idx = attention_mask.shape[1] - last_positions - 1
    row_idx = torch.arange(hidden_state.shape[0], device=hidden_state.device)
    pooled = hidden_state[row_idx, col_idx]
    return torch.nn.functional.normalize(pooled, p=2, dim=-1)


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root / args.config)

    raw_embedder = build_embedder(config, repo_root)
    embedder = QwenEmbedderAdapter(raw_embedder)

    data_cfg = config["data"]
    data_file_cfg = args.data_file or Path(
        str(data_cfg.get("smoke_file", data_cfg["train_file"]))
    )
    data_path = (
        data_file_cfg
        if data_file_cfg.is_absolute()
        else (repo_root / data_file_cfg)
    )
    server_prefix = args.server_prefix
    if server_prefix is None:
        server_prefix = data_cfg.get("server_prefix", "")

    dataset = QueryVideoDataset(
        data_path=str(data_path),
        query_column=str(data_cfg.get("query_column", "query")),
        video_column=str(data_cfg.get("video_column", "video")),
        server_prefix=str(server_prefix),
        max_samples=args.num_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty: {data_path}")

    num_samples = len(dataset)
    collator = QueryVideoCollator(
        embedder=embedder,
        fps=float(data_cfg.get("fps", 1)),
        max_frames=int(data_cfg.get("max_frames", 16)),
    )

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, num_samples),
        shuffle=False,
        collate_fn=collator,
    )

    device = embedder.model.device
    print(f"Device: {device}")
    print(f"Data file: {data_path}")
    print(f"Running smoke test on {num_samples} samples.")
    if args.all_batches:
        batch_iter = enumerate(loader, start=1)
    else:
        first_batch = next(iter(loader), None)
        if first_batch is None:
            raise RuntimeError("No batch was produced by DataLoader.")
        batch_iter = [(1, first_batch)]

    for step, batch in batch_iter:
        query_inputs = move_tensor_dict_to_device(batch["query_inputs"], device)
        video_inputs = move_tensor_dict_to_device(batch["video_inputs"], device)

        with torch.no_grad():
            query_outputs = embedder.forward(query_inputs)
            video_outputs = embedder.forward(video_inputs)

        query_embeddings = pool_last_hidden(
            query_outputs["last_hidden_state"], query_outputs["attention_mask"]
        )
        video_embeddings = pool_last_hidden(
            video_outputs["last_hidden_state"], video_outputs["attention_mask"]
        )
        score_matrix = query_embeddings @ video_embeddings.T

        print(f"[Batch {step}]")
        print(f"  query/text shape : {tuple(query_embeddings.shape)}")
        print(f"  video shape      : {tuple(video_embeddings.shape)}")
        print(f"  score matrix     : {tuple(score_matrix.shape)}")
        print(f"  score diag       : {score_matrix.diag().detach().cpu().tolist()}")

    print("Embedding smoke test succeeded.")


if __name__ == "__main__":
    main()
