"""
remedy/sku_profile_manager.py

SKU Profile Manager — loads per-SKU YAML configurations and applies
class_risk_overrides to the SeverityScorer at runtime.

Each YAML file under configs/sku_profiles/ defines:
  - class_risk_overrides: per-class risk weights (override global defaults)
  - rejection_area_thresholds: per-class area ratio thresholds
  - max_remediation_attempts: max retries before mandatory reject
  - preferred_stations: hint for TriageRouter station assignment

Usage:
    mgr = SKUProfileManager()
    profile = mgr.load("pouch_100g")
    scorer = SeverityScorer(class_risk_overrides=profile.class_risk_overrides)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from core.logging import get_logger
from core.schemas import DefectClass

log = get_logger(__name__)

# Path relative to this file's location: remedy/ → project_root/configs/sku_profiles/
_PROFILE_DIR = Path(__file__).parent.parent / "configs" / "sku_profiles"

_DEFAULT_RISK: dict[DefectClass, float] = {
    DefectClass.SURFACE_CONTAMINATION: 1.00,
    DefectClass.IMPROPER_FILLING: 0.75,
    DefectClass.PACKAGING_DAMAGE: 0.65,
    DefectClass.LABEL_MISALIGNMENT: 0.30,
}


@dataclass
class SKUProfile:
    """Parsed SKU configuration loaded from YAML."""

    sku_id: str
    sku_name: str
    product_category: str

    # Per-class risk weights, already merged with defaults
    class_risk_overrides: dict[DefectClass, float] = field(default_factory=dict)

    # Per-class area ratio at which a defect is considered critical
    rejection_area_thresholds: dict[DefectClass, float] = field(default_factory=dict)

    max_remediation_attempts: int = 2

    # Preferred remediation stations (hints for TriageRouter)
    preferred_stations: dict[DefectClass, str] = field(default_factory=dict)


def _parse_risk_map(raw: Optional[dict]) -> dict[DefectClass, float]:
    """Convert str-keyed risk dict to DefectClass-keyed, falling back to defaults."""
    if not raw:
        return dict(_DEFAULT_RISK)
    merged = dict(_DEFAULT_RISK)
    for k, v in raw.items():
        try:
            merged[DefectClass(k)] = float(v)
        except ValueError:
            log.warning("sku_profile_unknown_class", key=k)
    return merged


def _parse_area_map(raw: Optional[dict]) -> dict[DefectClass, float]:
    result: dict[DefectClass, float] = {}
    if not raw:
        return result
    for k, v in raw.items():
        try:
            result[DefectClass(k)] = float(v)
        except ValueError:
            log.warning("sku_profile_unknown_class_area", key=k)
    return result


def _parse_station_map(raw: Optional[dict]) -> dict[DefectClass, str]:
    result: dict[DefectClass, str] = {}
    if not raw:
        return result
    for k, v in raw.items():
        try:
            result[DefectClass(k)] = str(v)
        except ValueError:
            log.warning("sku_profile_unknown_class_station", key=k)
    return result


def _load_default_profile() -> SKUProfile:
    return SKUProfile(
        sku_id="default",
        sku_name="Default (No Profile)",
        product_category="generic",
        class_risk_overrides=dict(_DEFAULT_RISK),
        rejection_area_thresholds={},
        max_remediation_attempts=2,
        preferred_stations={},
    )


class SKUProfileManager:
    """
    Manages loading and caching of per-SKU YAML profiles.

    Profiles are loaded lazily and cached for the lifetime of the manager.
    If a requested SKU file is not found, the default profile is returned
    with a warning rather than raising.
    """

    def __init__(self, profile_dir: Optional[Path] = None) -> None:
        self._dir = profile_dir or _PROFILE_DIR
        self._cache: dict[str, SKUProfile] = {}
        self._available: list[str] | None = None

    def _list_available(self) -> list[str]:
        """Return list of available SKU IDs from YAML file stems (cached)."""
        if self._available is None:
            if not self._dir.exists():
                self._available = []
            else:
                self._available = [p.stem for p in self._dir.glob("*.yaml")]
        return self._available

    def load(self, sku_id: str) -> SKUProfile:
        """
        Load (and cache) the SKU profile for the given sku_id.

        Args:
            sku_id: Must match the YAML filename stem (e.g. 'pouch_100g')

        Returns:
            SKUProfile with merged risk overrides. Falls back to default
            profile if the file is not found.
        """
        if sku_id in self._cache:
            return self._cache[sku_id]

        if sku_id == "default":
            profile = _load_default_profile()
            self._cache[sku_id] = profile
            return profile

        yaml_path = self._dir / f"{sku_id}.yaml"
        if not yaml_path.exists():
            log.warning(
                "sku_profile_not_found",
                sku_id=sku_id,
                searched=str(yaml_path),
                available=self._list_available(),
            )
            profile = _load_default_profile()
            self._cache[sku_id] = profile
            return profile

        try:
            with yaml_path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            log.error("sku_profile_yaml_error", sku_id=sku_id, error=str(exc))
            profile = _load_default_profile()
            self._cache[sku_id] = profile
            return profile

        profile = SKUProfile(
            sku_id=raw.get("sku_id", sku_id),
            sku_name=raw.get("sku_name", sku_id),
            product_category=raw.get("product_category", "generic"),
            class_risk_overrides=_parse_risk_map(raw.get("class_risk_overrides")),
            rejection_area_thresholds=_parse_area_map(
                raw.get("rejection_area_thresholds")
            ),
            max_remediation_attempts=int(raw.get("max_remediation_attempts", 2)),
            preferred_stations=_parse_station_map(raw.get("preferred_stations")),
        )
        self._cache[sku_id] = profile
        log.info("sku_profile_loaded", sku_id=sku_id, category=profile.product_category)
        return profile

    def preload_all(self) -> None:
        """Load every YAML profile in the profile directory into the cache."""
        for sku_id in self._list_available():
            self.load(sku_id)
        log.info("sku_profiles_preloaded", count=len(self._cache))
