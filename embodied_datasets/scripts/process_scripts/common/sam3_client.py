"""Local-inference SAM3 client for Check2 video-state consistency, backed
by HuggingFace transformers' Sam3Model/Sam3Processor. See
docs/superpowers/specs/2026-07-28-sam3-check2-camera-calibration-design.md
section 9 for why this is box+text prompted (not point-prompted like
SAM/SAM2) and requires the Python 3.12 / lerobot 0.6.0 / torch 2.11.0 /
transformers 5.14.1 combination this repo now pins.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class LocalSam3Client:
    """VideoStateConsistencyClient backed by a local SAM3 model. Model
    weights are loaded lazily -- only on the first .segment() call, not at
    construction -- and cached on the instance for reuse across calls.
    Constructing a LocalSam3Client is therefore cheap and dependency-free;
    only code paths that actually call .segment() need torch/transformers
    importable and (network + gated-access) the facebook/sam3 weights
    downloadable.
    """

    def __init__(self, model_id: str, text_prompt: str, hf_token: str):
        self._model_id = model_id
        self._text_prompt = text_prompt
        self._hf_token = hf_token
        self._predictor = None

    def segment(self, frame: np.ndarray, box_prompt: Tuple[float, float, float, float]) -> np.ndarray:
        if self._predictor is None:
            self._predictor = self._load_predictor()
        return self._run_box_prompt(self._predictor, frame, box_prompt)

    def _load_predictor(self):
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(self._model_id, token=self._hf_token)
        model.eval()
        processor = AutoProcessor.from_pretrained(self._model_id, token=self._hf_token)
        return model, processor

    def _run_box_prompt(self, predictor, frame: np.ndarray, box_prompt: Tuple[float, float, float, float]) -> np.ndarray:
        import torch
        from PIL import Image

        model, processor = predictor
        image = Image.fromarray(frame)
        x0, y0, x1, y1 = box_prompt
        inputs = processor(
            images=image,
            text=self._text_prompt,
            input_boxes=[[[x0, y0, x1, y1]]],
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=0.3, target_sizes=[image.size[::-1]]
        )[0]

        if len(results["masks"]) == 0:
            return np.zeros(frame.shape[:2], dtype=bool)

        query_cx, query_cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        boxes = results["boxes"]
        box_centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
        box_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
        distances = (box_centers_x - query_cx) ** 2 + (box_centers_y - query_cy) ** 2
        best_idx = int(torch.argmin(distances).item())
        return results["masks"][best_idx].cpu().numpy().astype(bool)
