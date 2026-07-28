#!/usr/bin/env python3
"""
Water-mask segmentation for Dream Fusion.

Wraps a SegFormer model finetuned on ADE20K and produces a per-frame boolean
water mask at the original image resolution. Any ADE20K class whose label
matches one of WATER_KEYWORDS is treated as water.

The mask is computed at the model's native logit resolution (H/4 x W/4) and
upsampled to the image size with bilinear interpolation, so we never
materialise per-class logits at full resolution (the camera frames are 5320 x
3032, so a full-res logit tensor would be multiple GB).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


WATER_KEYWORDS = {
    "water",
    "sea",
    "river",
    "lake",
    "pool",
    "swimming pool",
    "waterfall",
    "falls",
}


def find_water_class_ids(id2label: dict) -> List[int]:
    """Return every class id whose label (or any comma-separated synonym) is
    in WATER_KEYWORDS."""
    ids: List[int] = []
    for cls_id, name in id2label.items():
        parts = [p.strip().lower() for p in str(name).split(",")]
        if any(p in WATER_KEYWORDS for p in parts):
            ids.append(int(cls_id))
    return sorted(ids)


def load_segformer(model_id: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
    model.to(device)
    model.eval()
    water_ids = find_water_class_ids(model.config.id2label)
    if not water_ids:
        raise RuntimeError(
            f"No water-like classes found in {model_id} "
            f"(id2label keys: {list(model.config.id2label.items())[:5]}...). "
            "This model does not appear to be trained on ADE20K."
        )
    water_names = [model.config.id2label[i] for i in water_ids]
    print(f"SegFormer water classes: {list(zip(water_ids, water_names))}")
    return processor, model, water_ids


@torch.no_grad()
def compute_water_mask(processor, model, image: Image.Image,
                       water_ids: List[int], device: str) -> np.ndarray:
    """Return a boolean (H, W) water mask at the input image's full resolution."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    outputs = model(**inputs)

    logits = outputs.logits  # (1, C, h_small, w_small)
    semantic_small = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)

    mask_small = np.isin(semantic_small, water_ids).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_small, mode="L")
    # Bilinear upsample keeps edges smooth so the vid2 water tint doesn't look
    # blocky; thresholded back to bool for downstream fraction math.
    mask_full_pil = mask_pil.resize(image.size, Image.BILINEAR)
    return np.asarray(mask_full_pil) > 127


def water_fraction(mask: np.ndarray, bbox_xyxy: Tuple[float, float, float, float]) -> float:
    """Fraction of a bbox that is water (True) in the mask."""
    h, w = mask.shape
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(w, int(round(x2)))
    y2 = min(h, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(region.mean())
