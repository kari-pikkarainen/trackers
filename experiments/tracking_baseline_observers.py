# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Read-only native call-site observers; no association or update algorithm is replaced."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import textwrap
from pathlib import Path

import numpy as np

from experiments.continuation_confidence import SplitContinuationTracker
from experiments.tracking_validation_eval import jsonable, require
from trackers import BoTSORTTracker, CBIoUTracker, OCSORTTracker
from trackers.utils.state_representations import XCYCWHStateEstimator

SOURCE_HASHES = {
    "ocsort": "34b99c9b1336ff07f13e814e93a11418f93e57b9638f63fe3a8acc79652ad961",
    "botsort": "271cfb2bc62da24d0e4a5488c2f5df2540b4f740430358efe898f7011c2153c1",
    "cbiou": "af85deaa1d000915a89e392a81762f329a14dbe94c5e87d2b201aac920991e14",
}
NATIVE = {"ocsort": OCSORTTracker, "botsort": BoTSORTTracker, "cbiou": CBIoUTracker}


def snapshot_state(tracker):
    """Copy serializable native state and CMC arrays; name opaque estimator configuration exclusions.

    OpenCV feature/matcher objects are opaque, not an asserted equality of hidden C++ state. The admitted sparse-flow
    method has these fields set to None. Its mutable gray image, point arrays and initialization/failure counters
    remain.
    """
    if tracker is None:
        return None
    state = {k: v for k, v in vars(tracker).items() if not k.startswith("_baseline") and k != "association_observer"}
    cmc = state.get("cmc")
    if cmc is not None:
        fields = dict(vars(cmc))
        for key in ("detector", "extractor", "matcher"):
            value = fields.get(key)
            if value is not None:
                fields[key] = {"excluded_opaque_configuration": type(value).__name__}
        for key, value in fields.items():
            if isinstance(value, np.ndarray):
                copied = np.ascontiguousarray(value).copy()
                fields[key] = {
                    "array_shape": list(copied.shape),
                    "dtype": copied.dtype.str,
                    "bytes": copied.nbytes,
                    "sha256_C_order_bytes": hashlib.sha256(copied.tobytes()).hexdigest(),
                }
        state["cmc"] = fields
    return copy.deepcopy(jsonable(state))


def call_sites(family):
    """Pin native source and AST assignment names, rejecting changed or unrecognized stages."""
    native = NATIVE[family]
    path = Path(inspect.getfile(native))
    require(hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_HASHES[family], "native observer source changed")
    lines, start = inspect.getsourcelines(native.update)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    targets = (
        {"matched_indices": "ocm", "ocr_matched": "ocr"}
        if family == "ocsort"
        else {"matched": "first", "matched_uc": "unconfirmed"}
    )
    sites = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func
        if not isinstance(fn, ast.Attribute) or fn.attr != "_get_associated_indices":
            continue
        target = node.targets[0]
        require(isinstance(target, ast.Tuple) and isinstance(target.elts[0], ast.Name), "unknown association target")
        name = target.elts[0].id
        require(name in targets, "unknown native association stage")
        stage = targets[name]
        if (
            family != "ocsort"
            and stage == "first"
            and any(
                isinstance(arg, ast.Attribute) and arg.attr == "minimum_iou_threshold_second_assoc"
                for arg in node.value.args
            )
        ):
            stage = "second"
        for line in range(start + node.value.lineno - 1, start + node.value.end_lineno):
            sites[line] = stage
    require(
        set(sites.values()) == ({"ocm", "ocr"} if family == "ocsort" else {"first", "second", "unconfirmed"}),
        "missing native association stages",
    )
    return sites


class ObserverMixin:
    """Delegate unchanged methods while copying explicit source-bound call locals and lifecycle evidence."""

    def _setup_observer(self, family):
        if family == "split":
            from experiments.tracking_baselines import SPLIT_SHA256

            require(
                hashlib.sha256(Path(inspect.getfile(SplitContinuationTracker)).read_bytes()).hexdigest()
                == SPLIT_SHA256,
                "shared split source changed",
            )
        self._baseline_family = family
        self._baseline_sites = {} if family == "split" else call_sites(family)
        self._baseline_uids = {}
        self._baseline_events = []
        self._baseline_before = []
        self._baseline_predictions = []
        self._baseline_timing = None
        self._baseline_raw = None
        self._baseline_input = None
        if family == "split":
            self.association_observer = self._split_event

    def _uid(self, track):
        if track not in self._baseline_uids:
            self._baseline_uids[track] = len(self._baseline_uids)
        return self._baseline_uids[track]

    def _track_record(self, track):
        return {
            "uid": self._uid(track),
            "id": int(track.tracker_id),
            "time_since_update": track.time_since_update,
            "seconds_since_update": track.time_since_update_seconds,
            "consecutive_updates": track.number_of_successful_consecutive_updates,
            "successful_updates": getattr(track, "number_of_successful_updates", None),
            "comparison_box": track.get_state_bbox().copy(),
            "last_observation": copy.deepcopy(getattr(track, "last_observation", None)),
        }

    def begin_observation(self):
        """Clear observations outside native update and preserve shot-local stable object identity."""
        self._baseline_events = []
        self._baseline_predictions = []
        self._baseline_timing = None
        self._baseline_before = [self._track_record(t) for t in self.tracks]

    def end_observation(self):
        """Separate budget expiry from immature removal; do not label every disappearance an expiry."""
        after = [self._track_record(t) for t in self.tracks]
        alive = {r["uid"] for r in after}
        removed = []
        timing = self._baseline_timing
        predicted = {r["uid"]: r for r in self._baseline_predictions}
        for prior in self._baseline_before:
            if prior["uid"] in alive:
                continue
            latest = predicted.get(prior["uid"], prior)
            budget = self.maximum_time_without_update if timing is not None and timing.uses_elapsed_time else None
            expired = (
                latest["seconds_since_update"] > budget
                if budget is not None
                else latest["time_since_update"] > self.maximum_frames_without_update
            )
            removed.append(
                {
                    **latest,
                    "reason": "budget_expiry" if expired else "immature_or_native_removal",
                    "established_before": prior["id"] >= 0,
                }
            )
        return jsonable(
            {
                "events": self._baseline_events,
                "confidence_policy": {
                    "family": self._baseline_family,
                    "high_threshold": self.high_conf_det_threshold,
                    "low_threshold_exclusive": 0.1 if self._baseline_family in ("botsort", "cbiou") else None,
                    "birth_threshold": getattr(self, "track_activation_threshold", self.high_conf_det_threshold),
                    "weak_continuation_threshold": getattr(self, "continuation_threshold", None),
                },
                "lifecycle": {
                    "before": self._baseline_before,
                    "predicted_before_CMC_and_prune": self._baseline_predictions,
                    "after": after,
                    "removed": removed,
                    "births": [r for r in after if r["uid"] not in {b["uid"] for b in self._baseline_before}],
                    "newly_confirmed": [
                        r
                        for r in after
                        if r["id"] >= 0 and any(b["uid"] == r["uid"] and b["id"] < 0 for b in self._baseline_before)
                    ],
                    "timing": None if timing is None else vars(timing),
                },
            }
        )

    def update(self, detections, *args, **kwargs):
        """Bind original proposal metadata without modifying it and delegate the complete native update."""
        self._baseline_input = detections
        return super().update(detections, *args, **kwargs)

    def _predict_tracklets(self, tracklets, timing, *, return_predictions=False):
        self._baseline_timing = timing
        result = super()._predict_tracklets(tracklets, timing, return_predictions=return_predictions)
        self._baseline_predictions = [self._track_record(t) for t in tracklets]
        return result

    def _get_iou_matrix(self, *args, **kwargs):
        result = super()._get_iou_matrix(*args, **kwargs)
        self._baseline_raw = result.copy()
        return result

    def _biou_matrix(self, *args, **kwargs):
        result = super()._biou_matrix(*args, **kwargs)
        self._baseline_raw = result.copy()
        return result

    def _get_associated_indices(self, matrix, second):
        family = self._baseline_family
        if family == "split":
            return super()._get_associated_indices(matrix, second)
        frame = inspect.currentframe().f_back
        try:
            require(frame.f_code is NATIVE[family].update.__code__, "observer called outside pinned native update")
            require(frame.f_lineno in self._baseline_sites, "unexpected native association call site")
            stage, local = self._baseline_sites[frame.f_lineno], frame.f_locals
            if family == "ocsort":
                pool = self.tracks if stage == "ocm" else [self.tracks[i] for i in local["unmatched_tracks"]]
                indices = (
                    local["high_indices"] if stage == "ocm" else local["high_indices"][local["unmatched_detections"]]
                )
                comparison = local["predicted_boxes"] if stage == "ocm" else local["last_observation_of_tracks"]
                raw, direction = matrix.copy(), None if second is None else second.copy()
                assignment = matrix.copy() if second is None else matrix + self.direction_consistency_weight * second
                threshold, geometry = self.minimum_iou_threshold, "last_observation" if stage == "ocr" else "prediction"
            else:
                pool = local[
                    {"first": "strack_pool", "second": "remaining_tracked", "unconfirmed": "unconfirmed_tracks"}[stage]
                ]
                indices = local["low_indices"] if stage == "second" else local["high_indices"]
                if stage == "unconfirmed":
                    indices = indices[local["unmatched_high_list"]]
                comparison = np.array([local["predicted_state_boxes"][id(t)].copy() for t in pool]).reshape(-1, 4)
                raw, direction, assignment = self._baseline_raw.copy(), None, matrix.copy()
                threshold, geometry = second, "buffered_prediction" if family == "cbiou" else "prediction"
            require(matrix.shape == (len(pool), len(indices)), "association proposal/pool shape mismatch")
            require(raw.shape == matrix.shape, "raw/fused matrix mismatch")
            require(len(set(map(int, indices))) == len(indices), "duplicate proposal mapping")
            require(np.isfinite(matrix).all() and np.isfinite(raw).all(), "nonfinite observed matrix")
            data = self._baseline_input
            proposal = data.data["proposal_index"][indices].copy()
            event = {
                "stage": stage,
                "geometry": geometry,
                "pool": [self._track_record(t) for t in pool],
                "input_indices": indices.copy(),
                "proposal_indices": proposal,
                "comparison_boxes": comparison.copy(),
                "raw_iou": raw,
                "raw_confidences": np.ones(len(indices))
                if data.confidence is None
                else data.confidence[indices].copy(),
                "direction": direction,
                "assignment_scores": assignment.copy(),
                "gate_scores": matrix.copy(),
                "threshold": threshold,
                "score_fused": family != "ocsort" and stage != "second",
                "cmc_transform": copy.deepcopy(local.get("H")),
                "alive_at_stage": [self._uid(t) for t in self.tracks],
            }
            result = super()._get_associated_indices(matrix, second)
            event["result"] = copy.deepcopy(result)
            self._baseline_events.append(jsonable(event))
            return result
        finally:
            del frame

    def _split_event(self, source):
        event = copy.deepcopy(source)
        indices = event["proposal_indices"].copy()
        for record in event["pool"]:
            record["uid"] = self._uid(self.tracks[record["track_index"]])
        event["input_indices"] = indices
        event["proposal_indices"] = self._baseline_input.data["proposal_index"][indices].copy()
        event["geometry"] = "last_observation" if event["stage"].endswith("ocr") else "prediction"
        event["gate_scores"] = event["raw_iou"].copy()
        event["score_fused"] = False
        self._baseline_events.append(jsonable(event))


class ObservedOC(ObserverMixin, OCSORTTracker):
    """Unregistered native OC diagnostic delegate."""

    tracker_id = None  # type: ignore[assignment]


class ObservedBoT(ObserverMixin, BoTSORTTracker):
    """Unregistered native BoT diagnostic delegate."""

    tracker_id = None  # type: ignore[assignment]


class ObservedCBIoU(ObserverMixin, CBIoUTracker):
    """Unregistered native C-BIoU diagnostic delegate."""

    tracker_id = None  # type: ignore[assignment]


class ObservedSplit(ObserverMixin, SplitContinuationTracker):
    """Use the unchanged split callback for explicitly named residual stages."""

    tracker_id = None  # type: ignore[assignment]


def observed_baseline(name, *, frame_rate=5):
    """Construct diagnostic counterparts with exactly the predeclared native arguments."""
    if name in ("native06", "uniform04", "ocsort-common"):
        kwargs = {"high_conf_det_threshold": 0.4} if name == "uniform04" else {}
        if name == "ocsort-common":
            kwargs = {"minimum_consecutive_frames": 2, "state_estimator_class": XCYCWHStateEstimator}
        tracker, family = ObservedOC(frame_rate=frame_rate, **kwargs), "ocsort"
    elif name == "split02":
        tracker, family = ObservedSplit(0.2, frame_rate=frame_rate), "split"
    elif name in ("botsort-native", "botsort-cmc-off", "botsort-common"):
        kwargs = {} if name == "botsort-native" else {"enable_cmc": False}
        if name == "botsort-common":
            kwargs["track_activation_threshold"] = 0.6
        tracker, family = ObservedBoT(frame_rate=frame_rate, **kwargs), "botsort"
    else:
        require(name in ("cbiou-native", "cbiou-common"), "unfrozen observed baseline")
        kwargs = {"track_activation_threshold": 0.6} if name == "cbiou-common" else {}
        tracker, family = ObservedCBIoU(frame_rate=frame_rate, **kwargs), "cbiou"
    tracker._setup_observer(family)
    return tracker
