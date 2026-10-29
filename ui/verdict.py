"""The shared shape of a recommendation shown in the UI: a label, a tone, a headline, reasons and caveats.

Lives here (not in a feature-specific module) because more than one page constructs one directly -- the route
recommender builds its own Verdict from the route's own numbers, it does not borrow any scoring logic from elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Verdict:
    label: str
    tone: str                                   # good | mid | weak
    headline: str
    reasons: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    numbers: dict = field(default_factory=dict)
