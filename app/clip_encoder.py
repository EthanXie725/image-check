from __future__ import annotations

import threading
from io import BytesIO
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import open_clip
import torch
from PIL import Image

from app.config import CLIP_MODEL_NAME, CLIP_PRETRAINED


class ClipEncoder:
    """Thread-safe CLIP image encoder with L2-normalized outputs."""

    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        pretrained: str = CLIP_PRETRAINED,
        device: str | None = None,
    ) -> None:
        self._lock = threading.Lock()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess

    def _load_image(self, source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        if isinstance(source, Image.Image):
            img = source
        elif isinstance(source, (str, Path)):
            img = Image.open(source)
        elif isinstance(source, bytes):
            img = Image.open(BytesIO(source))
        else:
            raise TypeError(f"Unsupported image source type: {type(source)}")
        return img.convert("RGB")

    @torch.inference_mode()
    def encode_image(self, source: Union[str, Path, bytes, Image.Image]) -> np.ndarray:
        """Encode a single image to a float32 L2-normalized vector (1, dim)."""
        img = self._load_image(source)
        with self._lock:
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            feat = self.model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_images(
        self, sources: Sequence[Union[str, Path, bytes, Image.Image]]
    ) -> np.ndarray:
        """Batch encode images to float32 L2-normalized vectors (N, dim)."""
        if not sources:
            return np.zeros((0, 0), dtype=np.float32)

        tensors = []
        for source in sources:
            img = self._load_image(source)
            tensors.append(self.preprocess(img))

        batch = torch.stack(tensors).to(self.device)
        with self._lock:
            feat = self.model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            return feat.detach().cpu().numpy().astype(np.float32)
