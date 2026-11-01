"""Local files under URLs the official judge can use unchanged.

The judge builds asset URLs as `{views}/white/{name}.png`, `{views}/gray/{name}.png`, `{views}/grid.png` and
`{views}/embeddings.npz`, fetches some of them over HTTP and sends the rest to the VLM as image URLs. Every
asset here lives under `BASE`, which maps to a folder on disk:
- the judge's own HTTP reads go through `transport()`, an httpx transport that serves those files;
- image URLs in VLM requests become base64 data URLs (`data_url`), so the VLM never has to reach this machine.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

BASE = "http://duel.local/"


class AssetStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._data_urls: dict[str, str] = {}

    def url(self, path: Path) -> str:
        return BASE + path.resolve().relative_to(self.root).as_posix()

    def path(self, url: str) -> Path | None:
        if not url.startswith(BASE):
            return None
        path = (self.root / url[len(BASE) :]).resolve()
        return path if path.is_relative_to(self.root) else None

    def data_url(self, url: str) -> str:
        """The asset as a data URL; anything that is not a local asset is returned unchanged."""
        path = self.path(url)
        if path is None:
            return url
        if url not in self._data_urls:
            mime = "image/png" if path.suffix == ".png" else "application/octet-stream"
            self._data_urls[url] = f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        return self._data_urls[url]

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            path = self.path(str(request.url))
            if path is None or not path.is_file():
                return httpx.Response(404)
            return httpx.Response(200, content=path.read_bytes())

        return httpx.MockTransport(handle)
