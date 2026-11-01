"""Minimal stand-in for judge-service's match_report module: only what multi_stage.py imports."""

from enum import Enum


class DuelWinner(str, Enum):
    """Winner of a single prompt duel."""

    LEFT = "left"
    RIGHT = "right"
    DRAW = "draw"
    SKIPPED = "skipped"
