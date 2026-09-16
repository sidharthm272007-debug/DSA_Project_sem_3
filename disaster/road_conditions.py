"""Road condition states and shared scenario state container.

Ensures that both directed routing and undirected structural connectivity
observe the identical single source of truth for edge conditions.
"""

from enum import Enum
from typing import Dict, Tuple, Any, Set, List
from config import CONDITION_TIME_MULTIPLIERS


class RoadCondition(str, Enum):
    """Operational status of a road segment under disaster impact."""
    NORMAL = "NORMAL"
    SLOWED = "SLOWED"
    SEVERELY_AFFECTED = "SEVERELY_AFFECTED"
    BLOCKED = "BLOCKED"

    @property
    def travel_time_multiplier(self) -> float:
        """Get simulation travel time penalty multiplier."""
        return CONDITION_TIME_MULTIPLIERS.get(self.value, 1.0)

    @property
    def is_traversable(self) -> bool:
        """Check if vehicle routing can traverse this edge."""
        return self != RoadCondition.BLOCKED


class ScenarioState:
    """Shared single source of truth for edge conditions across adapters.
    
    Both DirectedGraphAdapter and UndirectedGraphAdapter hold a reference to the
    same ScenarioState instance, ensuring zero drift between routing and connectivity.
    """

    def __init__(self, initial_conditions: Dict[Tuple[Any, Any, int], RoadCondition] = None):
        # Maps (u, v, key) -> RoadCondition
        self._conditions: Dict[Tuple[Any, Any, int], RoadCondition] = (
            dict(initial_conditions) if initial_conditions else {}
        )

    def get_condition(self, u: Any, v: Any, key: int = 0) -> RoadCondition:
        """Get the condition of edge (u, v, key), defaulting to NORMAL."""
        return self._conditions.get((u, v, key), RoadCondition.NORMAL)

    def set_condition(self, u: Any, v: Any, key: int, condition: RoadCondition) -> None:
        """Set the condition of edge (u, v, key)."""
        self._conditions[(u, v, key)] = condition

    def reset_all(self) -> None:
        """Reset all edge conditions back to NORMAL."""
        self._conditions.clear()

    def get_blocked_edges(self) -> Set[Tuple[Any, Any, int]]:
        """Return set of all edge tuples currently BLOCKED."""
        return {
            edge for edge, cond in self._conditions.items() if cond == RoadCondition.BLOCKED
        }

    def get_affected_edges(self) -> Set[Tuple[Any, Any, int]]:
        """Return set of edges that are either SLOWED or SEVERELY_AFFECTED."""
        return {
            edge
            for edge, cond in self._conditions.items()
            if cond in (RoadCondition.SLOWED, RoadCondition.SEVERELY_AFFECTED)
        }

    def count_by_condition(self) -> Dict[RoadCondition, int]:
        """Summary counts of non-normal edge conditions."""
        counts = {
            RoadCondition.NORMAL: 0,
            RoadCondition.SLOWED: 0,
            RoadCondition.SEVERELY_AFFECTED: 0,
            RoadCondition.BLOCKED: 0,
        }
        for cond in self._conditions.values():
            counts[cond] += 1
        return counts

    def get_all_modifications(self) -> Dict[Tuple[Any, Any, int], RoadCondition]:
        """Return shallow copy of all modified edges."""
        return dict(self._conditions)
