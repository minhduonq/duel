"""The VLM client handed to the official judge: GLM behind an OpenAI-compatible endpoint (vLLM, as in the
competition and AI-3D-Worker-Example), plus a fake for dry runs.

The judge calls `client.chat.completions.create(model="glm-4.6v-flash", messages=..., ...)`. This wrapper
- inlines local asset URLs as data URLs (the endpoint cannot reach this machine),
- sends the endpoint's served model name,
- caps concurrent calls,
- counts failures: the judge silently scores a failed call as a neutral 5-5, so a broken endpoint would
  otherwise look like a run full of draws.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace
from typing import Any

from openai import AsyncOpenAI

from .assets import AssetStore


class GlmClient:
    def __init__(
        self, assets: AssetStore, base_url: str, api_key: str, model: str, concurrency: int = 32, timeout: float = 600
    ) -> None:
        self.assets, self.model = assets, model
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        self._gate = asyncio.Semaphore(concurrency)
        self.calls = 0
        self.errors = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _inline(self, messages: list[dict]) -> list[dict]:
        messages = copy.deepcopy(messages)
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image_url":
                        part["image_url"]["url"] = self.assets.data_url(part["image_url"]["url"])
        return messages

    async def _create(self, **kwargs: Any) -> Any:
        kwargs["model"] = self.model
        kwargs["messages"] = self._inline(kwargs["messages"])
        async with self._gate:
            self.calls += 1
            try:
                return await self._client.chat.completions.create(**kwargs)
            except Exception:
                self.errors += 1
                raise

    async def preflight(self) -> str:
        """One tiny call, so a wrong URL, key or model name fails loudly before any duel."""
        reply = await self._client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": "Reply with the word ok."}], max_tokens=64
        )
        return (reply.choices[0].message.content or "").strip()


class FakeGlm:
    """Schema-valid pseudo-random answers, deterministic per request: tests bundles and plumbing without a GPU."""

    def __init__(self, assets: AssetStore) -> None:
        self.assets = assets
        self.calls = 0
        self.errors = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls += 1
        for message in kwargs["messages"]:
            for part in message.get("content") if isinstance(message.get("content"), list) else []:
                if part.get("type") == "image_url" and not self.assets.data_url(part["image_url"]["url"]).startswith("data:"):
                    raise ValueError(f"image not resolvable: {part['image_url']['url']}")
        schema = kwargs["response_format"]["json_schema"]["schema"]
        key = hashlib.sha256(json.dumps(kwargs["messages"], sort_keys=True).encode()).hexdigest()
        content = json.dumps(_instance(schema, schema.get("$defs", {}), key))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    async def preflight(self) -> str:
        return "fake"


def _h(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)


def _instance(schema: dict, defs: dict, key: str) -> Any:
    if "$ref" in schema:
        return _instance(defs[schema["$ref"].split("/")[-1]], defs, key)
    if "enum" in schema:
        return schema["enum"][_h(key) % len(schema["enum"])]
    if "const" in schema:
        return schema["const"]
    kind = schema.get("type")
    if kind == "object":
        return {name: _instance(sub, defs, f"{key}.{name}") for name, sub in schema.get("properties", {}).items()}
    if kind == "array":
        count = max(schema.get("minItems", 0), min(schema.get("maxItems", 5), 5))
        return [_instance(schema.get("items", {}), defs, f"{key}[{i}]") for i in range(count)]
    if kind == "integer":
        return _h(key) % 11
    if kind == "number":
        return (_h(key) % 101) / 10
    if kind == "boolean":
        return _h(key) % 2 == 0
    return f"text {_h(key) % 1000}"
