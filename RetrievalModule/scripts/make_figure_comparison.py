"""Generate two comparison figures (Stage-1 Embedding vs Reranker).

Figure 1 — Reranker improves rank:
    Stage-1: positive at rank 3  →  Reranker: positive at rank 1

Figure 2 — Reranker maintains rank:
    Stage-1: positive at rank 1  →  Reranker: positive at rank 1

Each figure layout:
    ┌─────────────────────────────┬─────────────────────────────┐
    │    Stage-1 Embedding        │         Reranker            │
    ├─────────────────────────────┼─────────────────────────────┤
    │ Top1 [f1][f2][f3][f4][f5]  │ Top1 [f1][f2][f3][f4][f5]  │
    │ Top2 [f1][f2][f3][f4][f5]  │ Top2 [f1][f2][f3][f4][f5]  │
    │ Top3 [f1][f2][f3][f4][f5]  │ Top3 [f1][f2][f3][f4][f5]  │
    └─────────────────────────────┴─────────────────────────────┘

Run on the training server:
    python scripts/make_figure_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ──────────────────────────────────────────────
VIDEO_ROOT = "/workspace/VidAnomalyRetrieval/UCF_Video"
OUT_DIR    = Path("outputs/case_studies")
N_FRAMES   = 5
# ──────────────────────────────────────────────

# ── FIGURE 1: Reranker improves (Stage-1 rank 3 → Rerank rank 1) ──────────────
CASE1 = dict(
    query=(
        "At night, in the gas station, a tanker truck exploded, "
        "accompanied by fire, and many equipment were knocked down."
    ),
    positive="Explosion013_x264.mp4",
    stage1=[
        ("Top1", "Explosion/Explosion021_x264.mp4",  False),
        ("Top2", "Explosion/Explosion033_x264.mp4",  False),
        ("Top3", "Explosion/Explosion013_x264.mp4",  True),   # ← rank 3
    ],
    rerank=[
        ("Top1", "Explosion/Explosion013_x264.mp4",  True),   # ← promoted to 1
        ("Top2", "Explosion/Explosion021_x264.mp4",  False),
        ("Top3", "Explosion/Explosion033_x264.mp4",  False),
    ],
    out="fig1_rerank_improved_explosion013.png",
)

# ── FIGURE 2: Reranker maintains rank (Stage-1 rank 1 → Rerank rank 1) ────────
CASE2 = dict(
    query=(
        "A fat man with a mask smashed the glass door of the store "
        "with a throwing object."
    ),
    positive="Vandalism017_x264.mp4",
    stage1=[
        ("Top1", "Vandalism/Vandalism017_x264.mp4",  True),   # ← rank 1
        ("Top2", "Vandalism/Vandalism015_x264.mp4",  False),
        ("Top3", "Robbery/Robbery048_x264.mp4",      False),
    ],
    rerank=[
        ("Top1", "Vandalism/Vandalism017_x264.mp4",  True),   # ← stays rank 1
        ("Top2", "Robbery/Robbery048_x264.mp4",      False),
        ("Top3", "Shooting/Shooting024_x264.mp4",    False),
    ],
    out="fig2_rerank_maintained_vandalism017.png",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def sample_frames(path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Cannot open / no frames: {path}")
    idxs = np.linspace(0, total - 1, n, dtype=int).tolist()
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed reading frame {i} from {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def render_panel(axes_block, entries, video_root, n_frames):
    """Fill a K×N_FRAMES block of axes with frames + colored borders."""
    for r, (label, relpath, is_gt) in enumerate(entries):
        path = str(Path(video_root) / relpath)
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        frames = sample_frames(path, n_frames)
        for c, fr in enumerate(frames):
            ax = axes_block[r, c]
            ax.imshow(fr)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(2.8 if is_gt else 0.8)
                spine.set_color("#2ca02c" if is_gt else "#888888")

        check = " ✓" if is_gt else ""
        vname = Path(relpath).stem.replace("_x264", "")
        axes_block[r, 0].set_ylabel(
            f"{label}{check}\n{vname}",
            rotation=0, ha="right", va="center", labelpad=52,
            fontsize=10, fontweight="bold",
            color="#2ca02c" if is_gt else "#333333",
        )


def make_figure(case: dict, video_root: str, n_frames: int, out_dir: Path):
    K = len(case["stage1"])       # number of rows (3)
    N = n_frames                  # frames per video (5)

    # Grid: K rows × (N + gap + N) cols
    # We use two separate GridSpec columns joined by a divider
    fig_w = 1.8 + 2 * (N * 2.1)
    fig_h = 1.6 + K * 2.0

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Title (query)
    fig.suptitle(
        f"Query : {case['query']}",
        fontsize=11, fontweight="bold", color="#a8002a", y=0.98,
        wrap=True,
    )

    # Two GridSpec sub-areas side by side
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        K, 2 * N,
        figure=fig,
        left=0.13, right=0.98,
        top=0.88, bottom=0.04,
        wspace=0.06, hspace=0.12,
    )

    # Build axis arrays for left and right panels
    axes_left  = np.array([[fig.add_subplot(gs[r, c])     for c in range(N)]     for r in range(K)])
    axes_right = np.array([[fig.add_subplot(gs[r, N + c]) for c in range(N)]     for r in range(K)])

    # Panel headers
    mid_left  = (0.13 + (0.13 + N * (0.98 - 0.13) / (2 * N))) / 2
    mid_right = ((0.13 + N * (0.98 - 0.13) / (2 * N)) + 0.98) / 2
    for x, txt in [(0.30, "Stage-1 Embedding"), (0.73, "Reranker")]:
        fig.text(x, 0.91, txt, ha="center", va="bottom",
                 fontsize=12, fontweight="bold", color="#1f77b4")

    # Vertical divider line
    fig.add_artist(plt.Line2D(
        [0.555, 0.555], [0.03, 0.90],
        transform=fig.transFigure,
        color="#cccccc", linewidth=1.5, linestyle="--",
    ))

    render_panel(axes_left,  case["stage1"], video_root, n_frames)
    render_panel(axes_right, case["rerank"], video_root, n_frames)

    out_path = out_dir / case["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for case in [CASE1, CASE2]:
        print(f"\nGenerating: {case['out']}")
        make_figure(case, VIDEO_ROOT, N_FRAMES, OUT_DIR)


if __name__ == "__main__":
    main()
