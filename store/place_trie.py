"""Own Trie for place-name autocomplete.

- Keys are NFC-normalised, case-folded names (works for Tamil/Hindi code points).
- Every WORD START is inserted, so typing "road" finds "Avinashi Road".
- Each trie node stores its best ``TOP_N`` places, so ``suggest`` costs O(len(prefix)),
  independent of how many places share the prefix.
"""
from __future__ import annotations

import bisect
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

TOP_N = 20

KIND_IMPORTANCE = {
    "hospital": 5.0, "fire_station": 4.5, "police": 4.0, "clinic": 3.5,
    "station": 3.5, "junction": 3.0, "place": 3.0, "road": 2.0, "shop": 1.0, "other": 0.0,
}


@dataclass
class Place:
    name: str
    kind: str = "other"
    lat: float = 0.0
    lon: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def importance(self) -> float:
        return KIND_IMPORTANCE.get(self.kind, 0.0)


def normalise(s: str) -> str:
    return unicodedata.normalize("NFC", s).casefold().strip()


def _is_sep(c: str) -> bool:
    return c.isspace() or unicodedata.category(c).startswith("P")


def word_starts(s: str) -> list[int]:
    return [i for i, c in enumerate(s) if not _is_sep(c) and (i == 0 or _is_sep(s[i - 1]))]


class _Node:
    __slots__ = ("children", "top")

    def __init__(self) -> None:
        self.children: dict[str, _Node] = {}
        self.top: list[tuple[float, int]] = []  # (-score, place id), ascending => best first


class PlaceTrie:
    def __init__(self) -> None:
        self._root = _Node()
        self.places: list[Place] = []

    def __len__(self) -> int:
        return len(self.places)

    def add(self, place: Place, names: Iterable[str] | None = None) -> int:
        """Insert a place under its own name plus any extra names (name:en, name:ta, alt_name...)."""
        pid = len(self.places)
        self.places.append(place)
        seen: set[str] = set()
        for raw in [place.name, *(names or [])]:
            s = normalise(raw or "")
            if not s or s in seen:
                continue
            seen.add(s)
            for start in word_starts(s):
                score = place.importance + (1.0 if start == 0 else 0.0) - 0.001 * len(s)
                self._insert(s[start:], pid, score)
        return pid

    def _insert(self, key: str, pid: int, score: float) -> None:
        node = self._root
        for ch in key:
            node = node.children.setdefault(ch, _Node())
            self._update_top(node.top, pid, score)

    @staticmethod
    def _update_top(top: list[tuple[float, int]], pid: int, score: float) -> None:
        for i, (neg, p) in enumerate(top):
            if p == pid:
                if -score < neg:          # better score for the same place
                    del top[i]
                    break
                return
        bisect.insort(top, (-score, pid))
        if len(top) > TOP_N:
            top.pop()

    def suggest(self, prefix: str, limit: int = 8) -> list[Place]:
        s = normalise(prefix)
        if not s:
            return []
        node = self._root
        for ch in s:
            node = node.children.get(ch)
            if node is None:
                return []
        return [self.places[pid] for _, pid in node.top[:limit]]
