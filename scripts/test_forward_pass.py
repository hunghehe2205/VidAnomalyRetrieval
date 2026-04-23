from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model
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
        "--server-prefix",
        type=str,
        default=None,
        help="Optional server prefix for relative video paths (overrides config[data].server_prefix).",
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
        "trust_remote_code": model_cfg.get("trust_remote_code", True),
        "fps": data_cfg.get("fps", 1),
        "max_frames": data_cfg.get("max_frames", 16),
    }

    if torch.cuda.is_available():
        if "attn_implementation" in model_cfg:
            init_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
        init_kwargs["torch_dtype"] = torch.bfloat16

    return Qwen3VLEmbedder(**init_kwargs)


def attach_lora(embedder: Any, config: Dict[str, Any]) -> None:
    lora_cfg = config["lora"]
    alpha = lora_cfg.get("alpha", lora_cfg.get("lora_alpha", 32))
    peft_cfg = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(alpha),
        target_modules=list(lora_cfg["target_modules"]),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
    )
    embedder.model = get_peft_model(embedder.model, peft_cfg)
    embedder.model.eval()


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
    repo_root = Path(__file__).resolve().parent.parent
    config = load_config(repo_root / args.config)

    raw_embedder = build_embedder(config, repo_root)
    attach_lora(raw_embedder, config)
    embedder = QwenEmbedderAdapter(raw_embedder)

    data_cfg = config["data"]
    train_path = repo_root / data_cfg["train_file"]
    server_prefix = args.server_prefix
    if server_prefix is None:
        server_prefix = data_cfg.get("server_prefix", "")

    dataset = QueryVideoDataset(
        data_path=str(train_path),
        query_column=str(data_cfg.get("query_column", "query")),
        video_column=str(data_cfg.get("video_column", "video")),
        server_prefix=str(server_prefix),
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty: {train_path}")

    training_cfg = config["training"]
    batch_size = int(training_cfg.get("per_device_train_batch_size", 4))

    collator = QueryVideoCollator(
        embedder=embedder,
        max_frames=int(data_cfg.get("max_frames", 16)),
        fallback_to_dummy_video=bool(data_cfg.get("fallback_to_dummy_video", True)),
        strict_video_check=bool(data_cfg.get("strict_video_check", False)),
        warn_on_dummy_fallback=bool(data_cfg.get("warn_on_dummy_fallback", True)),
        dummy_num_frames=int(data_cfg.get("dummy_num_frames", 0)) or None,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))

    device = embedder.model.device
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

    print("Forward-pass smoke test succeeded.")
    print(f"Batch size: {query_embeddings.shape[0]}")
    print(f"Embedding dim: {query_embeddings.shape[1]}")
    print(f"Text embeddings shape: {tuple(query_embeddings.shape)}")
    print(f"Video embeddings shape: {tuple(video_embeddings.shape)}")


if __name__ == "__main__":
    main()
