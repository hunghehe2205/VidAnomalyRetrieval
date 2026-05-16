"""Push LoRA adapters (embedding ck-900 + reranker v6 ck-50) to HuggingFace Hub.

Run on the training server where /workspace/... checkpoints live and HF token
is already saved (hf auth login --token <token>).

Fixes two common issues before upload:
  1. README.md YAML has `base_model: /workspace/...` (local path → HF rejects).
  2. adapter_config.json has `base_model_name_or_path: /workspace/...`
     (would break downstream PeftModel.from_pretrained for users).

Both are rewritten to canonical HF model ids:
  - Qwen/Qwen3-VL-Embedding-2B
  - Qwen/Qwen3-VL-Reranker-2B

Usage:
  cd /workspace/VidAnomalyRetrieval/RetrievalModule
  python scripts/push_to_hub.py                  # push both
  python scripts/push_to_hub.py --only embed     # push embedding only
  python scripts/push_to_hub.py --only rerank    # push reranker only
  python scripts/push_to_hub.py --dry-run        # print what would happen
  python scripts/push_to_hub.py --include-state  # also upload trainer_state.pt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_folder


# --------------------------------------------------------------------------- #
# Targets — adjust paths/repos here if needed
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGETS = {
    "embed": {
        "local_path": REPO_ROOT / "outputs/Embedding/phase2-hardneg/checkpoint-900",
        "repo_id": "hungnghehe/qwen3-vl-embedding-phase2",
        "canonical_base": "Qwen/Qwen3-VL-Embedding-2B",
        "commit_msg": "P2 ck-900: t2v R@1=0.5556 mAP=0.6779 (UCF-Crime)",
    },
    "rerank": {
        "local_path": REPO_ROOT / "outputs/Reranker/rerank-phase1-v6/checkpoint-50",
        "repo_id": "hungnghehe/qwen3-vl-reranker-2b-ucf-rerank-phase1-v6",
        "canonical_base": "Qwen/Qwen3-VL-Reranker-2B",
        "commit_msg": "v6 ck-50: cascade R@1=0.5799 on stage-1 ck-900 (UCF-Crime)",
    },
}


# --------------------------------------------------------------------------- #
# Pre-upload fixes
# --------------------------------------------------------------------------- #

def fix_adapter_config(path: Path, canonical_base: str) -> bool:
    """Rewrite base_model_name_or_path to canonical HF id. Returns True if changed."""
    cfg_file = path / "adapter_config.json"
    if not cfg_file.exists():
        print(f"  [skip] no adapter_config.json at {cfg_file}")
        return False
    cfg = json.loads(cfg_file.read_text())
    cur = cfg.get("base_model_name_or_path", "")
    if cur == canonical_base:
        print(f"  [ok ] adapter_config base_model already canonical: {cur}")
        return False
    print(f"  [fix] adapter_config base_model: {cur!r} -> {canonical_base!r}")
    cfg["base_model_name_or_path"] = canonical_base
    cfg_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    return True


def fix_readme_yaml(path: Path, canonical_base: str) -> bool:
    """Rewrite YAML frontmatter `base_model:` line to canonical HF id.
    Returns True if file was modified."""
    rd = path / "README.md"
    if not rd.exists():
        print(f"  [skip] no README.md at {rd}")
        return False
    text = rd.read_text()
    if not text.startswith("---"):
        print(f"  [skip] README has no YAML frontmatter; leaving as-is")
        return False
    # Replace any `base_model: <value>` whose value starts with '/' or doesn't
    # match canonical, with the canonical HF id.
    pattern = re.compile(r"^(\s*base_model\s*:\s*)(.+)$", re.MULTILINE)

    def _sub(m: re.Match) -> str:
        prefix, val = m.group(1), m.group(2).strip()
        if val == canonical_base:
            return m.group(0)
        return f"{prefix}{canonical_base}"

    new_text, n = pattern.subn(_sub, text, count=1)
    if n == 0 or new_text == text:
        print(f"  [ok ] README YAML base_model already canonical or absent")
        return False
    print(f"  [fix] README YAML base_model -> {canonical_base!r}")
    rd.write_text(new_text)
    return True


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #

def upload_one(
    name: str,
    spec: dict,
    *,
    private: bool,
    include_state: bool,
    dry_run: bool,
) -> None:
    local: Path = spec["local_path"]
    repo_id: str = spec["repo_id"]
    canonical: str = spec["canonical_base"]
    commit_msg: str = spec["commit_msg"]

    print(f"\n=== {name.upper()} ===")
    print(f"  local : {local}")
    print(f"  repo  : {repo_id}  (private={private})")
    print(f"  base  : {canonical}")

    if not local.exists():
        print(f"  [ERR] local path does not exist; skipping")
        return

    # Always run fix pass (idempotent).
    fix_adapter_config(local, canonical)
    fix_readme_yaml(local, canonical)

    # Build ignore list. trainer_state.pt can be hundreds of MB; skip by default.
    ignore = []
    if not include_state:
        ignore.append("trainer_state.pt")
        ignore.append("optimizer.pt")
        ignore.append("scheduler.pt")
        ignore.append("rng_state.pth")

    if dry_run:
        print(f"  [dry] would upload_folder(repo_id={repo_id!r}, "
              f"folder_path={str(local)!r}, ignore_patterns={ignore})")
        return

    api = HfApi()
    # Ensure repo exists (private).
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    url = upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_msg,
        ignore_patterns=ignore if ignore else None,
    )
    print(f"  [done] {url}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=["embed", "rerank"], default=None,
                    help="Push only one target (default: both)")
    ap.add_argument("--public", action="store_true",
                    help="Upload as public repo (default: private)")
    ap.add_argument("--include-state", action="store_true",
                    help="Also upload trainer_state.pt + optimizer files "
                         "(default: skip — adapter-only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions without uploading")
    args = ap.parse_args()

    targets = TARGETS if args.only is None else {args.only: TARGETS[args.only]}
    private = not args.public

    for name, spec in targets.items():
        upload_one(name, spec,
                   private=private,
                   include_state=args.include_state,
                   dry_run=args.dry_run)

    print("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
