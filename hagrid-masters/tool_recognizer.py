"""Feature-based hand-tool matcher using SIFT + FLANN.

Crops from the live camera feed are matched against reference tool images
stored in ``images/``. SIFT keypoints are computed for both crops and paired
with a FLANN KD-Tree matcher using Lowe's ratio test (0.7). The reference
with the most surviving inliers wins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from constants import FLANN_RATIO_THRESHOLD, TOOL_MATCH_MIN_INLIERS


@dataclass
class ToolMatch:
    name: str
    inliers: int
    homography: Optional[np.ndarray]
    corners: Optional[np.ndarray]


class ToolRecognizer:
    """Match live tool crops to reference images with SIFT + FLANN."""

    FLANN_INDEX_KDTREE = 1

    def __init__(
        self,
        reference_dir: str = "images",
        ratio_threshold: float = FLANN_RATIO_THRESHOLD,
        min_inliers: int = TOOL_MATCH_MIN_INLIERS,
    ):
        self.reference_dir = reference_dir
        self.ratio_threshold = ratio_threshold
        self.min_inliers = min_inliers
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.references: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._load_references()

    def _load_references(self) -> None:
        if not os.path.isdir(self.reference_dir):
            return
        for fname in sorted(os.listdir(self.reference_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(self.reference_dir, fname)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            kps, des = self.sift.detectAndCompute(image, None)
            if des is not None and len(des) > 0:
                name = os.path.splitext(fname)[0]
                self.references[name] = (des, np.asarray([kp.pt for kp in kps]))

    @property
    def available_references(self) -> List[str]:
        return list(self.references.keys())

    def match(self, crop: np.ndarray) -> Optional[ToolMatch]:
        """Return the best matching tool for ``crop`` or ``None``."""
        if crop is None or crop.size == 0 or not self.references:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, des_query = self.sift.detectAndCompute(gray, None)
        if des_query is None or len(des_query) < self.min_inliers:
            return None

        best: Optional[ToolMatch] = None
        for name, (ref_des, _) in self.references.items():
            matches = self.flann.knnMatch(des_query, ref_des, k=2)
            good = [m for m, n in matches if m.distance < self.ratio_threshold * n.distance]
            if len(good) < self.min_inliers:
                continue
            if best is None or len(good) > best.inliers:
                best = ToolMatch(name=name, inliers=len(good), homography=None, corners=None)
        return best

    def match_with_homography(self, crop: np.ndarray, ref_image: np.ndarray) -> Optional[ToolMatch]:
        """Match and compute a homography to localise the tool in frame."""
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray_ref = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY) if ref_image.ndim == 3 else ref_image
        kp_q, des_q = self.sift.detectAndCompute(gray_crop, None)
        kp_r, des_r = self.sift.detectAndCompute(gray_ref, None)
        if des_q is None or des_r is None:
            return None
        matches = self.flann.knnMatch(des_q, des_r, k=2)
        good = [m for m, n in matches if m.distance < self.ratio_threshold * n.distance]
        if len(good) < self.min_inliers:
            return None
        src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else len(good)
        h, w = gray_ref.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        return ToolMatch(name="match", inliers=inliers, homography=H, corners=corners)


# ---------------------------------------------------------------------------
# Backward-compatibility shim
# ---------------------------------------------------------------------------


class ToolMatcher:
    """Legacy wrapper around :class:`ToolRecognizer`.

    ``demo_ff.py`` and other original scripts use::

        tool_matcher = ToolMatcher(images_dir="images")
        name, count = tool_matcher.match(roi_img)

    This class maps that call signature to the new ``ToolRecognizer`` API so
    both the old and new code can coexist without modification.
    """

    def __init__(self, images_dir: str = "images", min_match_count: int = TOOL_MATCH_MIN_INLIERS):
        self._recognizer = ToolRecognizer(
            reference_dir=images_dir,
            ratio_threshold=FLANN_RATIO_THRESHOLD,
            min_inliers=min_match_count,
        )
        self.images_dir = images_dir

    def match(self, roi_img: np.ndarray, min_match_count: int = None):
        """Return ``(tool_name, inlier_count)`` or ``(None, 0)``."""
        result: Optional[ToolMatch] = self._recognizer.match(roi_img)
        if result is None:
            return None, 0
        return result.name, result.inliers
