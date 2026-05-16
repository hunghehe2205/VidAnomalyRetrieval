"""Render a single t2v case-study figure (hardcoded for the Shooting033 case).

Layout (matches the paper figure):
    Query: <query text>                        (red, bold, top)

    Top1 ✓     [frame1][frame2][frame3][frame4][frame5]   (green border = GT)
    Shooting033

    Top2       [frame1][frame2][frame3][frame4][frame5]
    Robbery102

    Top3       [frame1][frame2][frame3][frame4][frame5]
    Abuse028
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ----- EDIT THIS -----
VIDEO_ROOT = "/workspace/test/Testing_Anomaly_Videos"
OUT_PNG    = "outputs/case_studies/rerank_win_shooting033.png"
# ---------------------

QUERY = "At the gate, the robbers robbed the woman's bag and knocked the woman to the ground."

# (label, relative path under VIDEO_ROOT, is_gt)
TOP3 = [
    ("Top1", "Shooting/Shooting033_x264.mp4", True),
    ("Top2", "Robbery/Robbery102_x264.mp4",   False),
    ("Top3", "Abuse/Abuse028_x264.mp4",       False),
]

N_FRAMES = 5


def sample_frames(path: str, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"No frames in {path}")
    idxs = np.linspace(0, total - 1, n, dtype=int).tolist()
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed reading frame {i} of {path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    K, N = len(TOP3), N_FRAMES
    fig_w = 2.5 + N * 2.4
    fig_h = 1.4 + K * 2.0
    fig, axes = plt.subplots(
        K, N, figsize=(fig_w, fig_h),
        gridspec_kw={"wspace": 0.05, "hspace": 0.18},
    )

    fig.suptitle(
        f"Query : {QUERY}",
        fontsize=12, fontweight="bold", color="#a8002a", y=0.985,
    )

    for r, (label, relpath, is_gt) in enumerate(TOP3):
        path = str(Path(VIDEO_ROOT) / relpath)
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        frames = sample_frames(path, N)
        for c, fr in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(fr)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(2.5 if is_gt else 0.8)
                spine.set_color("#2ca02c" if is_gt else "#888")

        check = " ✓" if is_gt else ""
        vname = Path(relpath).stem.replace("_x264", "")
        axes[r, 0].set_ylabel(
            f"{label}{check}\n{vname}",
            rotation=0, ha="right", va="center", labelpad=48,
            fontsize=12, color="#a8002a", fontweight="bold",
        )

    out = Path(OUT_PNG)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0.05, 0.0, 1.0, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[ok] saved → {out}")


if __name__ == "__main__":
    main()
