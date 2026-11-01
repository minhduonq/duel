"""View bundles, produced the way upstream does before a duel: renders from render-service-js plus DINOv3 embeddings.

Layout of one bundle (the judge reads it by convention):
    white/<view>.png   8 views on white: front_left, front_right, front_below, front_above, right, back, left, top_down
    gray/<view>.png    the 4 front views on mid-gray
    grid.png           the render service's default 2x2 orbit grid
    embeddings.npz     DINOv3 vectors: `prompt` + `view_<name>` for the 8 white views
    bundle.json        render settings, or the renderer's rejection (then the side has no views and loses by default)
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from .dino import Dino

WHITE_VIEWS = (
    ("front_left", 30, 0),
    ("front_right", 330, 0),
    ("front_below", 0, 15),
    ("front_above", 0, -30),
    ("right", 90, 0),
    ("back", 180, 0),
    ("left", 270, 0),
    ("top_down", 0, -90),
)
GRAY_VIEWS = WHITE_VIEWS[:4]
GRID_THETAS, GRID_PHIS = (24, 120, 216, 312), (-15, -15, -15, -15)


@dataclass(frozen=True)
class RenderSettings:
    url: str = "http://127.0.0.1:8000"
    img_size: int = 518
    lighting: str = "neutral"  # the scoring pipeline renders with neutral lighting (docs 05)
    cam_radius: float = 2.0
    cam_fov_deg: float = 49.1
    gap: int = 5


class Rejected(Exception):
    """The renderer refused the module (HTTP 422)."""


class Renderer:
    """`settings.url` may list several render services, comma-separated; each request goes to a free one."""

    def __init__(self, settings: RenderSettings, per_url: int = 4) -> None:
        self.settings = settings
        self._http = httpx.AsyncClient(timeout=240, trust_env=False)
        self._free: asyncio.Queue[str] = asyncio.Queue()
        for _ in range(per_url):
            for url in settings.url.split(","):
                if url.strip():
                    self._free.put_nowait(url.strip().rstrip("/"))

    async def render(self, source: str, thetas: tuple, phis: tuple, bg: str) -> bytes:
        s = self.settings
        params = {
            "thetas": ",".join(str(float(t)) for t in thetas),
            "phis": ",".join(str(float(p)) for p in phis),
            "img_size": str(s.img_size),
            "cam_radius": str(s.cam_radius),
            "cam_fov_deg": str(s.cam_fov_deg),
            "gap": str(s.gap),
            "bg_color": bg,
            "lighting": s.lighting,
        }
        url = await self._free.get()
        try:
            response = await self._http.post(url + "/render/grid", params=params, json={"source": source})
        finally:
            self._free.put_nowait(url)
        if response.status_code == 422:
            raise Rejected(response.text[:4000])
        response.raise_for_status()
        return response.content

    async def aclose(self) -> None:
        await self._http.aclose()


def bundle_key(source: str, prompt_png: bytes, settings: RenderSettings) -> str:
    h = hashlib.sha256()
    # The renderer address is not part of the key: the same module renders the same wherever it runs.
    config = {k: v for k, v in settings.__dict__.items() if k != "url"}
    for part in (source.encode(), prompt_png, json.dumps(config, sort_keys=True).encode()):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:24]


async def build_bundle(renderer: Renderer, dino: Dino, source: str, prompt_png: bytes, folder: Path) -> bool:
    """Render and embed one module into `folder` (skipped when already complete). True if the bundle has views."""
    meta_path = folder / "bundle.json"
    if meta_path.exists():
        return bool(json.loads(meta_path.read_text())["ok"])
    (folder / "white").mkdir(parents=True, exist_ok=True)
    (folder / "gray").mkdir(parents=True, exist_ok=True)
    meta = {"settings": renderer.settings.__dict__, "ok": False}
    try:
        white = await asyncio.gather(*(renderer.render(source, (t,), (p,), "ffffff") for _, t, p in WHITE_VIEWS))
        gray = await asyncio.gather(*(renderer.render(source, (t,), (p,), "808080") for _, t, p in GRAY_VIEWS))
        grid = await renderer.render(source, GRID_THETAS, GRID_PHIS, "808080")
    except Rejected as rejection:
        meta["rejected"] = str(rejection)
        meta_path.write_text(json.dumps(meta, indent=1))
        return False
    for (name, _, _), png in zip(WHITE_VIEWS, white):
        (folder / "white" / f"{name}.png").write_bytes(png)
    for (name, _, _), png in zip(GRAY_VIEWS, gray):
        (folder / "gray" / f"{name}.png").write_bytes(png)
    (folder / "grid.png").write_bytes(grid)
    vectors = await asyncio.to_thread(dino.embed, [prompt_png, *white])
    buf = io.BytesIO()
    np.savez(buf, prompt=vectors[0], **{f"view_{name}": vec for (name, _, _), vec in zip(WHITE_VIEWS, vectors[1:])})
    (folder / "embeddings.npz").write_bytes(buf.getvalue())
    meta["ok"] = True
    meta_path.write_text(json.dumps(meta, indent=1))
    return True
