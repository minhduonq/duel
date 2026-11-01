"""DINOv3 image embeddings for embeddings.npz (the judge's Stage 2A best-view pick).

Same recipe as AI-3D-Worker-Example's DinoEmbedder: pooled output (CLS token when there is no pooler),
L2-normalised, float32. Runs on CPU when no GPU is present; one bundle is only 9 images.
"""

from __future__ import annotations

import io
import threading
from typing import Any

import numpy as np

MODEL_ID = "Tooony133/dinov3-vits16-pretrain-lvd1689m"
REVISION = "fc6921f7a0b44d5b33ab4482cfed5443db6ccd81"


class Dino:
    def __init__(self, model_id: str = MODEL_ID, revision: str = REVISION, device: str | None = None) -> None:
        self.model_id, self.revision, self.device = model_id, revision, device
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id, revision=self.revision)
        self._model = AutoModel.from_pretrained(self.model_id, revision=self.revision).eval().to(device)
        self.device = device

    def embed(self, pngs: list[bytes]) -> np.ndarray:
        """(N, D) float32 L2-normalised embeddings of the given PNGs."""
        import torch
        from PIL import Image

        with self._lock:
            if self._model is None:
                self._load()
            images = [Image.open(io.BytesIO(png)).convert("RGB") for png in pngs]
            inputs = self._processor(images=images, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self._model(**inputs)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = out.last_hidden_state[:, 0]
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            return pooled.detach().to("cpu").float().numpy()
