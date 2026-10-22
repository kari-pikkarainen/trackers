# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Throwaway OC-SORT strong-birth/weak-confirmed-continuation experiment."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, cast

import numpy as np
import supervision as sv

from trackers import OCSORTTracker
from trackers.utils.detections import default_confidences
from trackers.utils.iou import IoU


@dataclass(frozen=True)
class ConfidencePolicy:
    """One predeclared scalar or split confidence policy."""

    name: str
    birth: float
    continuation: float
    split: bool = False


POLICIES = (
    ConfidencePolicy("native06", 0.6, 0.6),
    ConfidencePolicy("uniform02", 0.2, 0.2),
    ConfidencePolicy("uniform04", 0.4, 0.4),
    ConfidencePolicy("split02", 0.6, 0.2, True),
    ConfidencePolicy("split04", 0.6, 0.4, True),
)


class SplitContinuationTracker(OCSORTTracker):
    """Run native strong OCM/OCR first, then residual confirmed-only weak OCM/OCR.

    Extra weak stages intentionally change policy. CMC is unsupported. An observer receives independent snapshots with
    explicit stage names; the old two-stage observer must not be applied to this enabled adapter.
    """

    # Registry supports None; native OC-SORT narrows its annotation to str.
    tracker_id = None  # type: ignore[assignment]
    search_space: ClassVar[dict[str, dict]] = {}

    def __init__(
        self,
        continuation_threshold: float = 0.2,
        *,
        enabled: bool = True,
        association_observer: Callable[[dict], None] | None = None,
        **kwargs,
    ) -> None:
        """Keep native defaults except the declared continuation gate."""
        if continuation_threshold not in (0.2, 0.4):
            raise ValueError("continuation_threshold must be 0.2 or 0.4")
        if kwargs.get("enable_cmc", False):
            raise ValueError("experimental continuation supports CMC off only")
        if "high_conf_det_threshold" in kwargs and kwargs["high_conf_det_threshold"] != 0.6:
            raise ValueError("split birth threshold must stay 0.6")
        super().__init__(**kwargs)
        self.continuation_threshold = continuation_threshold
        self.enabled = enabled
        self.association_observer = association_observer

    def _associate(
        self,
        stage: str,
        pool_indices: list[int],
        proposal_indices: np.ndarray,
        matrix: np.ndarray,
        direction: np.ndarray | None,
        scores: np.ndarray,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        result = self._get_associated_indices(matrix, direction)
        if self.association_observer is not None:
            event = {
                "stage": stage,
                "pool": [
                    {
                        "track_index": i,
                        "id": self.tracks[i].tracker_id,
                        "time_since_update": self.tracks[i].time_since_update,
                        "seconds_since_update": self.tracks[i].time_since_update_seconds,
                        "state": self.tracks[i].state_estimator.get_state(),
                    }
                    for i in pool_indices
                ],
                "proposal_indices": proposal_indices.copy(),
                "raw_confidences": scores.copy(),
                "comparison_boxes": np.array(
                    [
                        self.tracks[i].get_state_bbox() if stage.endswith("ocm") else self.tracks[i].last_observation
                        for i in pool_indices
                    ]
                ).reshape(-1, 4),
                "raw_iou": matrix.copy(),
                "direction": None if direction is None else direction.copy(),
                "assignment_scores": (
                    matrix.copy() if direction is None else matrix + self.direction_consistency_weight * direction
                ),
                "threshold": self.minimum_iou_threshold,
                "result": copy.deepcopy(result),
            }
            self.association_observer(copy.deepcopy(event))
        return result

    def update(
        self,
        detections: sv.Detections,
        frame: np.ndarray | None = None,
        timestamp: float | None = None,
    ) -> sv.Detections:
        """Preserve native timing/birth/output paths and add two residual weak passes."""
        if not self.enabled:
            return super().update(detections, frame=frame, timestamp=timestamp)
        self._validate_detections(detections)
        self._warn_if_frame_unused(frame)
        timing = self._predict_timing(timestamp)
        if timing.skip_update:
            return self._detections_for_skipped_update(detections)
        if not self.tracks and not len(detections):
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
            return result

        confirmed_before = {t for t in self.tracks if t.tracker_id >= 0}
        boxes = detections.xyxy if len(detections) else np.empty((0, 4))
        confidences = default_confidences(detections)
        high_mask = np.ones(len(confidences), dtype=bool) if detections.confidence is None else confidences >= 0.6
        high_indices = np.where(high_mask)[0]
        low_indices = np.where(~high_mask)[0]
        weak_indices = np.where(~high_mask & (confidences >= self.continuation_threshold))[0]
        high_boxes = boxes[high_indices]
        high_scores = confidences[high_indices]
        out_indices: list[int] = []
        out_ids: list[int] = []

        predictions = self._predict_tracklets(self.tracks, timing, return_predictions=True)
        if self._lost_track_time_budget(timing, self.maximum_time_without_update) is not None:
            self.tracks = self._prune_expired_tracklets(timing)
        predicted = np.array(
            [predictions[t] for t in self.tracks]
            if not timing.skip_predict
            else [t.get_state_bbox() for t in self.tracks]
        )
        matrix = self.iou.compute(predicted, high_boxes)
        direction = (
            self._compute_direction_consistency_matrix(high_boxes, high_scores)
            if self.direction_consistency_weight != 0
            else None
        )
        matched, residual_tracks, residual_high = self._associate(
            "high_ocm", list(range(len(self.tracks))), high_indices, matrix, direction, high_scores
        )

        def apply_strong(track_index: int, detection_index: int) -> None:
            track = self.tracks[track_index]
            track.update(high_boxes[detection_index], timing)
            out_indices.append(int(high_indices[detection_index]))
            out_ids.append(
                track.resolve_tracker_id(self.minimum_consecutive_frames, self.frame_count, self._allocate_tracker_id)
            )

        for track_index, detection_index in matched:
            apply_strong(track_index, detection_index)
        if residual_tracks and residual_high:
            matrix = IoU().compute(
                np.array([self.tracks[i].last_observation for i in residual_tracks]), high_boxes[residual_high]
            )
            matched, remaining_tracks, remaining_high = self._associate(
                "high_ocr",
                residual_tracks,
                high_indices[residual_high],
                matrix,
                np.zeros_like(matrix),
                high_scores[residual_high],
            )
            for row, col in matched:
                apply_strong(residual_tracks[row], residual_high[col])
            residual_tracks = [residual_tracks[i] for i in remaining_tracks]
            residual_high = [residual_high[i] for i in remaining_high]

        # Strictly exclude expired tracks even in fixed-rate mode, whose native
        # high stage deliberately retains a final re-association opportunity.
        within_budget = set(self._prune_expired_tracklets(timing))
        weak_tracks = [
            i for i in residual_tracks if self.tracks[i] in confirmed_before and self.tracks[i] in within_budget
        ]
        weak_ids: dict[int, int] = {}

        def apply_weak(track_index: int, detection_index: int) -> None:
            track = self.tracks[track_index]
            if track not in confirmed_before or track.tracker_id < 0:
                raise RuntimeError("weak stage requires a previously confirmed track")
            track.update(boxes[detection_index], timing)
            # Native output maturity/suppression remains intact. Since the ID
            # already exists, resolve_tracker_id cannot allocate a new one.
            weak_ids[detection_index] = track.resolve_tracker_id(
                self.minimum_consecutive_frames, self.frame_count, self._allocate_tracker_id
            )

        if weak_tracks and len(weak_indices):
            matrix = self.iou.compute(predicted[weak_tracks], boxes[weak_indices])
            direction = (
                self._compute_direction_consistency_matrix(boxes[weak_indices], confidences[weak_indices])[weak_tracks]
                if self.direction_consistency_weight != 0
                else None
            )
            matched, remaining_tracks, remaining_weak = self._associate(
                "weak_ocm", weak_tracks, weak_indices, matrix, direction, confidences[weak_indices]
            )
            for row, col in matched:
                apply_weak(weak_tracks[row], int(weak_indices[col]))
            weak_tracks = [weak_tracks[i] for i in remaining_tracks]
            weak_indices = weak_indices[remaining_weak]
            if weak_tracks and len(weak_indices):
                matrix = IoU().compute(
                    np.array([self.tracks[i].last_observation for i in weak_tracks]), boxes[weak_indices]
                )
                matched, _, _ = self._associate(
                    "weak_ocr", weak_tracks, weak_indices, matrix, np.zeros_like(matrix), confidences[weak_indices]
                )
                for row, col in matched:
                    apply_weak(weak_tracks[row], int(weak_indices[col]))

        self._spawn_new_tracklets(high_boxes[residual_high])
        for detection_index in residual_high:
            out_indices.append(int(high_indices[detection_index]))
            out_ids.append(-1)
        for detection_index in low_indices:
            out_indices.append(int(detection_index))
            out_ids.append(weak_ids.get(int(detection_index), -1))
        self.tracks = self._prune_expired_tracklets(timing)
        if out_indices:
            result = cast(sv.Detections, detections[out_indices])
            result.tracker_id = np.array(out_ids, dtype=int)
        else:
            result = sv.Detections.empty()
            result.tracker_id = np.array([], dtype=int)
        self.frame_count += 1
        return result


def make_tracker(policy_name: str, **kwargs) -> OCSORTTracker:
    """Construct exactly one declared policy; uniform policies are literal native."""
    policy = next((p for p in POLICIES if p.name == policy_name), None)
    if policy is None:
        raise ValueError("unknown continuation-confidence policy")
    if kwargs.get("enable_cmc", False):
        raise ValueError("confidence comparison requires CMC off")
    if "high_conf_det_threshold" in kwargs:
        raise ValueError("policy controls the confidence threshold")
    if policy.split:
        return SplitContinuationTracker(policy.continuation, **kwargs)
    return OCSORTTracker(high_conf_det_threshold=policy.birth, **kwargs)
