"""Description runner for HolmesVAU on UCF-Crime, supporting URDMU or VadCLIP scorer.

Usage in notebook:
    from description_runner import load_pipeline, run_inference

    mllm, tok, gen_cfg, scorer = load_pipeline(
        scorer_type='vadclip',                       # or 'urdmu'
        mllm_path='./ckpts/HolmesVAU-2B',
        urdmu_ckpt='.../anomaly_scorer.pth',         # only for urdmu
        vadclip_ckpt='.../model_ucf.pth',            # only for vadclip
        vadclip_features_dir='.../UCFClipFeatures',  # only for vadclip
        device=torch.device('cuda'),
    )
    pred, frame_indices, anomaly_score = run_inference(
        video_path, prompt, mllm, tok, gen_cfg, scorer,
        select_frames=12, use_ATS=True,
    )
"""
import sys
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image
from scipy import interpolate
from transformers import AutoModel, AutoTokenizer

# --- Path setup so we can import HolmesVAU & VadCLIP code without modifying them ---
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "HolmesVAU"))
sys.path.insert(0, str(_HERE / "VadCLIP" / "src"))

from holmesvau.ATS.Temporal_Sampler import Temporal_Sampler
from holmesvau.internvl_utils import build_transform, dynamic_preprocess, get_index

from model import CLIPVAD                                       # VadCLIP/src/model.py
from utils.tools import get_batch_mask, get_prompt_text, process_split  # VadCLIP/src/utils/tools.py


_UCF_LABEL_MAP = {
    'Normal': 'Normal', 'Abuse': 'Abuse', 'Arrest': 'Arrest', 'Arson': 'Arson',
    'Assault': 'Assault', 'Burglary': 'Burglary', 'Explosion': 'Explosion',
    'Fighting': 'Fighting', 'RoadAccidents': 'RoadAccidents', 'Robbery': 'Robbery',
    'Shooting': 'Shooting', 'Shoplifting': 'Shoplifting', 'Stealing': 'Stealing',
    'Vandalism': 'Vandalism',
}
_UCF_CATEGORIES = list(_UCF_LABEL_MAP.keys())


def _get_pixel_values(vr, frame_indices, input_size=448, max_num=1):
    transform = build_transform(input_size=input_size)
    pv_list, num_patches_list = [], []
    for fi in frame_indices:
        img = Image.fromarray(vr[fi].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pv = torch.stack([transform(t) for t in tiles])
        num_patches_list.append(pv.shape[0])
        pv_list.append(pv)
    return torch.cat(pv_list), num_patches_list


def density_aware_sample(anomaly_score: np.ndarray, select_frames: int, tau: float = 0.1):
    """Pick `select_frames` indices into `anomaly_score`, weighted by score density.

    Same algorithm as HolmesVAU's Temporal_Sampler.density_aware_sample, just standalone.
    """
    num_frames = anomaly_score.shape[0]
    if num_frames <= select_frames or float(np.sum(anomaly_score)) < 1.0:
        return list(np.rint(np.linspace(0, num_frames - 1, select_frames)).astype(int))
    scores = [s + tau for s in anomaly_score]
    score_cumsum = np.concatenate((np.zeros((1,), dtype=float), np.cumsum(scores)), axis=0)
    max_score_cumsum = np.round(score_cumsum[-1]).astype(int)
    f_upsample = interpolate.interp1d(score_cumsum, np.arange(num_frames + 1),
                                      kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.linspace(1, max_score_cumsum, select_frames)
    sampled_idxs = f_upsample(scale_x)
    return [min(num_frames - 1, max(0, int(i))) for i in sampled_idxs]


# ============================================================
# Scorers
# ============================================================

class BaseScorer:
    name = 'base'

    def score(self, video_path, dense_frame_indices, pixel_values, mllm) -> np.ndarray:
        raise NotImplementedError


class URDMUScorer(BaseScorer):
    """HolmesVAU's default scorer — InternVL2 ViT CLS tokens → URDMU."""
    name = 'urdmu'

    def __init__(self, ckpt_path: str, device: torch.device):
        self._sampler = Temporal_Sampler(ckpt_path, device)

    def score(self, video_path, dense_frame_indices, pixel_values, mllm):
        return self._sampler.get_anomaly_scores(pixel_values, mllm)


class VadCLIPScorer(BaseScorer):
    """Precomputed CLIP-ViT-B/16 features + CLIPVAD checkpoint.

    Resolves feature path as <features_dir>/<Category>/<VideoName>__5.npy.
    Expects snippet stride = 16 frames (matches HolmesVAU dense_sample_freq=16).
    """
    name = 'vadclip'

    def __init__(self, ckpt_path: str, features_dir: str, device: torch.device,
                 visual_length: int = 256):
        self.device = device
        self.features_dir = Path(features_dir)
        self.visual_length = visual_length
        self.prompt_text = get_prompt_text(_UCF_LABEL_MAP)

        self.model = CLIPVAD(
            num_class=14, embed_dim=512, visual_length=visual_length,
            visual_width=512, visual_head=1, visual_layers=2,
            attn_window=8, prompt_prefix=10, prompt_postfix=10, device=device,
        ).to(device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.eval()

    @staticmethod
    def _infer_category(stem: str) -> str:
        for cat in sorted(_UCF_CATEGORIES, key=len, reverse=True):
            if stem.startswith(cat):
                return cat
        if stem.lower().startswith('normal'):
            return 'Normal'
        raise ValueError(f"Cannot infer UCF category from filename: {stem}")

    def _resolve_feature_path(self, video_path: str) -> Path:
        stem = Path(video_path).stem
        return self.features_dir / self._infer_category(stem) / f"{stem}__5.npy"

    @torch.no_grad()
    def _forward_clipvad(self, features: np.ndarray) -> np.ndarray:
        L = self.visual_length
        T = features.shape[0]
        clip_feature, _ = process_split(features, L)
        if clip_feature.ndim == 2:
            clip_feature = clip_feature[None]                       # [1, L, 512]
        visual = torch.tensor(clip_feature, dtype=torch.float32, device=self.device)

        n_chunks = visual.shape[0]
        lengths = torch.zeros(n_chunks, dtype=torch.long)
        remaining = T
        for j in range(n_chunks):
            lengths[j] = max(0, min(L, remaining))
            remaining -= int(lengths[j].item())
        padding_mask = get_batch_mask(lengths, L).to(self.device)

        _, logits1, _ = self.model(visual, padding_mask, self.prompt_text, lengths)
        prob1 = torch.sigmoid(logits1.reshape(-1)).cpu().numpy()    # [n_chunks * L]
        return prob1[:T]

    def score(self, video_path, dense_frame_indices, pixel_values, mllm):
        feat_path = self._resolve_feature_path(video_path)
        if not feat_path.exists():
            raise FileNotFoundError(f"VadCLIP feature not found: {feat_path}")
        features = np.load(feat_path)
        if features.ndim == 3:                                       # [T, n_crop, 512] -> avg over crops
            features = features.mean(axis=1)
        scores = self._forward_clipvad(features)                     # [T_feat]

        N = len(dense_frame_indices)
        if scores.shape[0] == N:
            return scores
        if scores.shape[0] > N:
            return scores[:N]
        pad_val = scores[-1] if scores.size > 0 else 0.0
        return np.concatenate([scores, np.full(N - scores.shape[0], pad_val)])


# ============================================================
# Pipeline
# ============================================================

def load_pipeline(scorer_type: str,
                  mllm_path: str,
                  device: torch.device,
                  urdmu_ckpt: str = None,
                  vadclip_ckpt: str = None,
                  vadclip_features_dir: str = None,
                  use_flash_attn: bool = False):
    """Load InternVL2 MLLM + tokenizer + chosen scorer.

    Returns: (mllm, tokenizer, generation_config, scorer)
    """
    mllm = AutoModel.from_pretrained(
        mllm_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn, trust_remote_code=True,
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(mllm_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    if scorer_type == 'urdmu':
        if urdmu_ckpt is None:
            raise ValueError("urdmu_ckpt is required for scorer_type='urdmu'")
        scorer = URDMUScorer(urdmu_ckpt, device)
    elif scorer_type == 'vadclip':
        if vadclip_ckpt is None or vadclip_features_dir is None:
            raise ValueError("vadclip_ckpt and vadclip_features_dir are required for scorer_type='vadclip'")
        scorer = VadCLIPScorer(vadclip_ckpt, vadclip_features_dir, device)
    else:
        raise ValueError(f"Unknown scorer_type: {scorer_type!r}. Use 'urdmu' or 'vadclip'.")

    return mllm, tokenizer, generation_config, scorer


def run_inference(video_path: str, prompt: str,
                  mllm, tokenizer, generation_config, scorer: BaseScorer,
                  dense_sample_freq: int = 16, select_frames: int = 12,
                  use_ATS: bool = True):
    """Generate description for a video. Mirrors HolmesVAU's generate() but
    delegates anomaly scoring to the scorer plug-in.
    """
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    print("Frame Number: ", len(vr))

    if use_ATS and len(vr) > dense_sample_freq * select_frames:
        print(f"Anomaly-focused Temporal Sampling [{scorer.name}]...")
        dense_frame_indices = list(range(len(vr)))[::dense_sample_freq]
        pixel_values, num_patches_list = _get_pixel_values(vr, dense_frame_indices)
        anomaly_score = scorer.score(video_path, dense_frame_indices, pixel_values, mllm)
        sampled_idxs = density_aware_sample(anomaly_score, select_frames)
        sparse_pixel_values = pixel_values[sampled_idxs]
        frame_indices = [dense_frame_indices[i] for i in sampled_idxs]
        num_patches_list = [num_patches_list[i] for i in sampled_idxs]
        print('Sampled frames: ', frame_indices)
    else:
        print("Uniform Sampling...")
        frame_indices = get_index(bound=None, fps=float(vr.get_avg_fps()),
                                  max_frame=len(vr) - 1, first_idx=0,
                                  num_segments=select_frames)
        frame_indices = list(map(int, frame_indices))
        sparse_pixel_values, num_patches_list = _get_pixel_values(vr, frame_indices)
        anomaly_score = None

    sparse_pixel_values = sparse_pixel_values.to(torch.bfloat16).to(mllm.device)
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + prompt
    response, _ = mllm.chat(
        tokenizer, sparse_pixel_values, question, generation_config,
        num_patches_list=num_patches_list, history=None, return_history=True,
    )
    return response, frame_indices, anomaly_score
