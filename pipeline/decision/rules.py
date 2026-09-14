"""
Inspection Rules and Threshold Configuration.
Defines asymmetric thresholds, class definitions, and decision constraints.
"""
from typing import List, Set
from pydantic import BaseModel, Field


class InspectionRulesConfig(BaseModel):
    """
    Industrial Quality Control Policy Configuration.
    Implements asymmetric thresholds to eliminate false defects while preventing
    faulty caps from passing.
    """
    pass_class: str = Field(default="good_cap", description="Only class that qualifies as a certified pass")
    
    defect_classes: List[str] = Field(
        default_factory=lambda: ["damaged_cap", "misplaced_cap", "no_cap", "open_cap", "wet_cap"],
        description="All classes that constitute a manufacturing defect"
    )

    # Asymmetric Thresholds
    pass_conf_threshold: float = Field(
        default=0.75,
        ge=0.50,
        le=1.0,
        description="Strict confidence required on good_cap to certify PASS (e.g. 0.75)"
    )

    defect_conf_threshold: float = Field(
        default=0.45,
        ge=0.20,
        le=1.0,
        description="Sensitivity threshold for defect classes to trigger REJECT (e.g. 0.45)"
    )

    uncertain_floor: float = Field(
        default=0.40,
        ge=0.10,
        le=0.75,
        description="Lower bound for uncertain/borderline cases (0.40 - 0.75) routed for review"
    )

    # Minimum mask area sanity check (in pixels)
    min_mask_area_px: float = Field(
        default=1000.0,
        description="Minimum expected pixel area for a valid cap mask (guards against tiny speckle noise)"
    )

    def is_defect(self, class_name: str) -> bool:
        return class_name in self.defect_classes

    def is_pass(self, class_name: str) -> bool:
        return class_name == self.pass_class
