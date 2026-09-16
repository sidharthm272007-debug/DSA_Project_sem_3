"""Disaster Scenario Definition and Extensibility Interfaces.

Supports user-defined simulation scenarios in Phase 1 and establishes clean
abstractions for Phase 2 real-world feeds (weather APIs, government feeds, sensor data).
"""

from typing import Tuple, Set, Optional, Dict, Any, List
from dataclasses import dataclass, field
import uuid


@dataclass
class DisasterScenario:
    """Encapsulates all parameters defining a simulated disaster event.
    
    Attributes:
        disaster_type: 'Flood', 'Earthquake', 'Landslide', 'Fire', 'Storm/Cyclone', 'Custom Road Blockage'
        epicenter_lat: Latitude of disaster center
        epicenter_lon: Longitude of disaster center
        radius_m: Radius of direct impact in meters
        severity: Intensity rating from 1 (mild) to 5 (catastrophic)
        propagation_mode: 'geographic' (distance-based decay), 'bfs_topological' (network-hop spread), or 'hybrid'
        manual_blocked_edges: Set of specific (u, v, key) edges explicitly blocked by user
        scenario_id: Unique identifier
    """
    disaster_type: str = "Flood"
    epicenter_lat: float = 0.0
    epicenter_lon: float = 0.0
    radius_m: float = 500.0
    severity: int = 3
    propagation_mode: str = "geographic"  # 'geographic' | 'bfs_topological' | 'hybrid'
    manual_blocked_edges: Set[Tuple[Any, Any, int]] = field(default_factory=set)
    scenario_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""

    @property
    def epicenter(self) -> Tuple[float, float]:
        return self.epicenter_lat, self.epicenter_lon


class DisasterEventFeedInterface:
    """Abstract interface for Phase 2 live disaster and weather feeds.
    
    In Phase 2, external feeds (e.g. USGS Earthquakes, NOAA Weather, OpenWeather,
    Municipal Road Feeds) will implement this interface to produce DisasterScenario
    objects without altering any downstream graph algorithms or dashboard components.
    """

    def fetch_active_incidents(self, bbox: Tuple[float, float, float, float]) -> List[DisasterScenario]:
        raise NotImplementedError("Live incident ingestion is reserved for Phase 2.")
