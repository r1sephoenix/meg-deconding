from __future__ import annotations

from typing import Iterable

import torch


def encode_words_ascii(words: Iterable[str], max_len: int = 32) -> torch.Tensor:
    rows = []
    for w in words:
        s = (w or "").lower()[:max_len]
        ids = [ord(c) if ord(c) < 128 else ord("?") for c in s]
        if len(ids) < max_len:
            ids.extend([0] * (max_len - len(ids)))
        rows.append(ids)
    return torch.tensor(rows, dtype=torch.long)
