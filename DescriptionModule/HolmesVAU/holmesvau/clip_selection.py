import os
from typing import List, Tuple, Optional

import numpy as np


def select_clips(
    anomaly_score: np.ndarray,
    K: int = 3,
    clip_length: Optional[int] = None,
    clip_sec: float = 16.0,
    fps: Optional[float] = None,
    snippet_size: int = 16,
) -> List[Tuple[int, int]]:
    """Select K fixed-length, non-overlapping clips by greedy NMS over snippet scores.

    Algorithm:
      1. Sort snippet indices by score descending.
      2. Greedy pick with NMS: accept a snippet only if its distance to every
         already-picked snippet is >= clip_length (snippets). Stop at K picks.
      3. Center a fixed-width window on each picked snippet, clamp to [0, T],
         then sort by start time.

    Args:
        anomaly_score: shape (T,), per-snippet scores in [0, 1].
        K: number of clips to return.
        clip_length: clip width in snippets. If None, computed from clip_sec/fps.
        clip_sec: clip width in seconds (used only if clip_length is None).
        fps: video fps (required if clip_length is None).
        snippet_size: frames per snippet (default 16, matching dense_sample_freq).

    Returns:
        List of (start_snippet, end_snippet) tuples, end-exclusive, sorted by start.
        May contain fewer than K clips if T is too small for K non-overlapping windows.
    """
    score = np.asarray(anomaly_score).reshape(-1)
    T = score.shape[0]
    if T == 0:
        return []

    if clip_length is None:
        if fps is None:
            raise ValueError("Provide clip_length (in snippets) or fps (with clip_sec).")
        clip_length = max(1, int(round(clip_sec * fps / snippet_size)))

    if clip_length >= T:
        return [(0, T)]

    half = clip_length // 2
    sorted_idx = np.argsort(-score, kind="stable")

    picked: List[int] = []
    for idx in sorted_idx:
        if len(picked) >= K:
            break
        idx = int(idx)
        if all(abs(idx - p) >= clip_length for p in picked):
            picked.append(idx)

    clips: List[Tuple[int, int]] = []
    for idx in picked:
        start = idx - half
        end = start + clip_length
        if start < 0:
            start, end = 0, clip_length
        elif end > T:
            start, end = T - clip_length, T
        clips.append((start, end))

    clips.sort(key=lambda c: c[0])
    return clips


def pick_frames(
    score: np.ndarray,
    num_picks: int,
    tau: float = 0.1,
    offset: int = 0,
) -> List[int]:
    """Density-aware frame picking via cumsum inverse-CDF sampling.

    Same algorithm as Holmes-VAU's `Temporal_Sampler.density_aware_sample`:
    add `tau` to every score so flat regions are still reachable, take the
    cumulative sum, then invert it at `num_picks` evenly-spaced quantiles.
    High-score regions get picked densely, low-score regions sparsely. Falls
    back to uniform spacing when the slice is too short or near-zero.

    Args:
        score: 1D array (typically a slice of the per-frame anomaly map).
        num_picks: number of indices to return.
        tau: smoothing constant; higher -> more uniform. Default mirrors
            `Temporal_Sampler.tau = 0.1`.
        offset: added to every returned index — convenient when `score` is a
            slice and you want absolute frame indices.

    Returns:
        Sorted list of int indices (with offset), length == num_picks.
    """
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    T = score.shape[0]
    if T == 0 or num_picks <= 0:
        return []
    if T <= num_picks or score.sum() < 1.0:
        idxs = np.rint(np.linspace(0, T - 1, num_picks)).astype(int).tolist()
    else:
        smoothed = score + tau
        cumsum = np.concatenate(([0.0], np.cumsum(smoothed)))
        max_cum = float(cumsum[-1])
        sample_x = np.linspace(1.0, max_cum, num_picks)
        sampled = np.interp(sample_x, cumsum, np.arange(T + 1, dtype=np.float64))
        idxs = [min(T - 1, max(0, int(i))) for i in sampled]
    return [int(i) + offset for i in idxs]


def upsample_to_frames(
    snippet_score: np.ndarray,
    num_frames: int,
    snippet_size: int = 16,
) -> np.ndarray:
    """Linearly interpolate per-snippet scores to a per-frame array of length num_frames.

    Snippet i is anchored at its center frame (i * snippet_size + snippet_size/2).
    Boundary frames take the value of the nearest snippet center (np.interp default).
    """
    snippet_score = np.asarray(snippet_score, dtype=np.float64).reshape(-1)
    T = snippet_score.shape[0]
    if T == 0 or num_frames <= 0:
        return np.zeros(max(num_frames, 0), dtype=np.float64)
    centers = np.arange(T) * snippet_size + snippet_size / 2.0
    return np.interp(np.arange(num_frames), centers, snippet_score)


def snippets_to_frames(
    clips: List[Tuple[int, int]],
    snippet_size: int = 16,
    max_frame: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Convert (start_snippet, end_snippet) clips to (start_frame, end_frame) ranges."""
    frame_clips = []
    for s, e in clips:
        sf = s * snippet_size
        ef = e * snippet_size
        if max_frame is not None:
            ef = min(ef, max_frame)
            sf = min(sf, ef)
        frame_clips.append((int(sf), int(ef)))
    return frame_clips


def load_gt_segments(
    video_path: str,
    annotation_file: str,
) -> List[Tuple[int, int]]:
    """Read UCF temporal annotations and return GT frame ranges for the given video.

    Annotation format: `video_name  class  start1  end1  start2  end2`
    where -1 marks an absent segment. Returns [] for normal videos or unknown names.
    """
    name = os.path.splitext(os.path.basename(video_path))[0]
    with open(annotation_file, "r") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != name:
                continue
            nums = [int(x) for x in parts[2:6]]
            segments = []
            for s, e in [(nums[0], nums[1]), (nums[2], nums[3])]:
                if s >= 0 and e >= 0 and e > s:
                    segments.append((s, e))
            return segments
    return []
