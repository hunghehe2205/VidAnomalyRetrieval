"""Case B figure: Reranker top-1 wrong, positive at rank 2.

Query : At night, the two cars collided violently in the middle of the crossroad
        and crashed into the side of the road.

Rerank result:
  Top1  [✗]  Shooting037          (wrong — reranker confused)
  Top2  [✓]  RoadAccidents131     (correct positive)
  Top3  [✗]  Normal_Videos_902    (wrong)

Run on the training server where UCF_Video lives:
    python scripts/make_figure_roadaccidents131.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ----- EDIT IF NEEDED -----
VIDEO_ROOT = "/workspace/VidAnomalyRetrieval/UCF_Video"
OUT_PNG    = "outputs/case_studies/rerank_fail_roadaccidents131.png"
# --------------------------

QUERY = ("At night, the two cars collided violently in the middle of "
         "the crossroad and crashed into the side of the road.")

# (row_label, relative_path_under_VIDEO_ROOT, is_gt)
TOP3 = [
    ("Top1",   "Shooting/Shooting037_x264.mp4",                              False),
    ("Top2",   "RoadAccidents/RoadAccidents131_x264.mp4",                    True),
    ("Top3",   "Testing_Normal_Videos_Anomaly/Normal_Videos_902_x264.mp4",   False),
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
                spine.set_color("#2ca02c" if is_gt else "#888888")

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
