"""Minimal stand-in for judge-service's generations module: only what multi_stage.py imports."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerationResult(BaseModel):
    """One side of a duel. `views` is the folder prefix holding white/, gray/, grid.png and embeddings.npz."""

    js: str | None = Field(default=None, description="Location of the Three.js module")
    views: str | None = Field(default=None, description="Folder prefix containing white/ and gray/ rendered views")
    failure_reason: str | None = Field(default=None, description="Why this prompt has no usable result")
    size: int = Field(default=0, description="Size of the JS module in bytes")
