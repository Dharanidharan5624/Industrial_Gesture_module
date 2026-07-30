"""Compliance detector package."""

from compliance.detectors.base import BaseDetector
from compliance.detectors.earbuds import EarbudsDetector
from compliance.detectors.phone import PhoneDetector
from compliance.detectors.shirt import ShirtButtonDetector
from compliance.detectors.spectacles import SpectaclesDetector
from compliance.detectors.writing import WritingDetector

__all__ = [
    "BaseDetector",
    "PhoneDetector",
    "ShirtButtonDetector",
    "EarbudsDetector",
    "SpectaclesDetector",
    "WritingDetector",
]
