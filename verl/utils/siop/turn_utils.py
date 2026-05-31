"""Utilities for assistant-turn boundary detection in multi-turn agent rollouts."""

from typing import Sequence


def _normalize_mask(response_mask: Sequence[int]) -> list[int]:
    if hasattr(response_mask, "tolist"):
        response_mask = response_mask.tolist()
    return [int(x) for x in response_mask]


def get_assistant_turn_ranges(response_mask: Sequence[int]) -> list[tuple[int, int]]:
    """Return half-open token ranges for assistant-generated turns.

    The agent loop records assistant-generated tokens with ``1`` in ``response_mask``
    and tool/user observation tokens with ``0``. Each contiguous run of ``1`` values
    corresponds to one assistant turn.
    """

    mask = _normalize_mask(response_mask)
    ranges: list[tuple[int, int]] = []
    start: int | None = None

    for idx, value in enumerate(mask):
        if value == 1 and start is None:
            start = idx
        elif value == 0 and start is not None:
            ranges.append((start, idx))
            start = None

    if start is not None:
        ranges.append((start, len(mask)))

    return ranges


def get_assistant_turn_end_positions(response_mask: Sequence[int]) -> list[int]:
    """Return exclusive end token indices for assistant-generated turns."""

    return [end for _, end in get_assistant_turn_ranges(response_mask)]
