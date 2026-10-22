# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Lifecycle, score, metadata and native-equivalence synthetic policy checks."""

import copy
import warnings

import numpy as np
import pytest
import supervision as sv

from experiments.continuation_confidence import POLICIES, SplitContinuationTracker, make_tracker
from trackers import OCSORTTracker
from trackers.core.base import BaseTracker

BOX = np.array([10.0, 20.0, 30.0, 60.0])


def det(scores=(0.9,), offsets=None, missing=False):
    offsets = list(range(len(scores))) if offsets is None else offsets
    boxes = np.array([BOX + np.array([x, 0, x, 0]) for x in offsets], dtype=float).reshape(-1, 4)
    return sv.Detections(
        xyxy=boxes,
        confidence=None if missing else np.array(scores),
        class_id=np.arange(len(boxes)),
        data={"proposal_index": np.arange(len(boxes)), "tag": np.array(["rider"] * len(boxes))},
    )


def canonical(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if isinstance(value, type):
        return value.__module__ + "." + value.__name__
    if callable(value):
        return "callable"
    if hasattr(value, "__dict__"):
        return canonical(vars(value))
    return value


def state(tracker):
    # Compare every native-owned tracker/tracklet/estimator field, including IDs,
    # timestamps, allocator, frozen ORU state and all Kalman caches/matrices.
    return canonical(
        {
            k: v
            for k, v in vars(tracker).items()
            if k not in ["enabled", "continuation_threshold", "association_observer"]
        }
    )


def warm(tracker, start=0):
    for i in range(5):
        tracker.update(det(), timestamp=start + i * 0.2)
    assert tracker.tracks[0].tracker_id == 0
    return tracker


def equal_detections(a, b):
    for field in ["xyxy", "confidence", "class_id", "mask", "tracker_id"]:
        left, right = getattr(a, field), getattr(b, field)
        assert (left is None and right is None) or np.array_equal(left, right), field
    assert canonical(a.data) == canonical(b.data)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("clock", [False, True])
def test_exact_native_full_state_with_no_weak_matches(enabled, clock):
    native = OCSORTTracker(frame_rate=5)
    adapter = SplitContinuationTracker(enabled=enabled, frame_rate=5)
    stamps = [0, 0.2, 0.41, 0.41, 0.62, 0.9, 1.1, 1.3, 1.6, 1.8, 1.7, float("nan"), 2.0]
    for i, stamp in enumerate(stamps):
        d = det((0.9, 0.1), [i, 100]) if i % 4 else sv.Detections.empty()
        saved = copy.deepcopy(d)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = native.update(d, timestamp=stamp if clock else None)
            b = adapter.update(d, timestamp=stamp if clock else None)
        equal_detections(a, b)
        assert state(native) == state(adapter)
        equal_detections(d, saved)


@pytest.mark.parametrize("policy", [p.name for p in POLICIES])
def test_policy_factory_registry_and_no_weak_birth(policy):
    assert BaseTracker._registry["ocsort"].tracker_class is OCSORTTracker
    tracker = make_tracker(policy, frame_rate=5)
    if policy.startswith("split"):
        for i in range(8):
            assert tracker.update(det((0.4,)), timestamp=i * 0.2).tracker_id.tolist() == [-1]
        assert tracker.tracks == [] and tracker._next_track_id == 0
    else:
        assert type(tracker) is OCSORTTracker


@pytest.mark.parametrize("threshold", [0.2, 0.4])
@pytest.mark.parametrize("side", [-1, 0, 1])
def test_continuation_boundary_and_birth_boundary(threshold, side):
    tracker = warm(SplitContinuationTracker(threshold, frame_rate=5))
    score = np.nextafter(threshold, -np.inf) if side < 0 else np.nextafter(threshold, np.inf) if side > 0 else threshold
    out = tracker.update(det((score,)), timestamp=1.0)
    assert out.tracker_id.tolist() == ([0] if side >= 0 else [-1])
    assert tracker._next_track_id == 1
    for score in [np.nextafter(0.6, -np.inf), 0.6, np.nextafter(0.6, np.inf)]:
        fresh = SplitContinuationTracker(threshold, frame_rate=5)
        fresh.update(det((score,)), timestamp=0)
        assert len(fresh.tracks) == int(score >= 0.6)


def test_missing_confidence_and_unconfirmed_weak_maturity_exclusion():
    tracker = SplitContinuationTracker(frame_rate=5)
    tracker.update(det(missing=True), timestamp=0)
    assert len(tracker.tracks) == 1 and tracker.tracks[0].tracker_id == -1
    original = tracker.tracks[0]
    for i in range(1, 5):
        tracker.update(det((0.3,)), timestamp=i * 0.2)
        assert original.tracker_id == -1 and original.number_of_successful_consecutive_updates == 0
        assert tracker._next_track_id == 0


def test_strong_priority_and_metadata_one_row_per_input():
    tracker = warm(SplitContinuationTracker(frame_rate=5))
    data = det((0.3, 0.9, 0.1, 0.59), [0, 3, 200, 500])
    saved = copy.deepcopy(data)
    out = tracker.update(data, timestamp=1.0)
    ids = dict(zip(out.data["proposal_index"], out.tracker_id))
    assert ids == {0: -1, 1: 0, 2: -1, 3: -1}
    assert sorted(out.data["proposal_index"]) == list(range(4)) and len(out) == 4
    for i, index in enumerate(out.data["proposal_index"]):
        np.testing.assert_array_equal(out.xyxy[i], data.xyxy[index])
        assert out.confidence[i] == data.confidence[index]
        assert out.class_id[i] == data.class_id[index] and out.data["tag"][i] == "rider"
    equal_detections(data, saved)


def test_buffered_weak_oru_preserves_native_maturity_and_allocator():
    tracker = warm(SplitContinuationTracker(frame_rate=5))
    track = tracker.tracks[0]
    tracker.update(sv.Detections.empty(), timestamp=1.0)
    tracker.update(sv.Detections.empty(), timestamp=1.2)
    assert track._frozen_state is not None
    first = tracker.update(det((0.3,)), timestamp=1.4)
    assert first.tracker_id.tolist() == [-1]
    assert track._frozen_state is None and track.tracker_id == 0 and tracker._next_track_id == 1
    tracker.update(det((0.3,)), timestamp=1.6)
    assert tracker.update(det((0.3,)), timestamp=1.8).tracker_id.tolist() == [0]
    assert tracker._next_track_id == 1


@pytest.mark.parametrize("timestamped", [False, True])
def test_weak_cannot_revive_expired(timestamped):
    tracker = warm(SplitContinuationTracker(frame_rate=5)) if timestamped else SplitContinuationTracker(frame_rate=5)
    if not timestamped:
        for _ in range(5):
            tracker.update(det())
    if timestamped:
        tracker.update(det((0.3,)), timestamp=2.0)
    else:
        for _ in range(5):
            tracker.update(sv.Detections.empty())
        tracker.update(det((0.3,)))
    assert tracker.tracks == [] and tracker._next_track_id == 1


def test_explicit_weak_ocr_raw_confidence_and_observer_nonmutation():
    events = []
    observed = warm(SplitContinuationTracker(frame_rate=5, association_observer=events.append))
    plain = warm(SplitContinuationTracker(frame_rate=5))
    for tracker in [observed, plain]:
        tracker.tracks[0].state_estimator.kf.state[0, 0] += 1000
    events.clear()
    data = det((0.3,))
    a = observed.update(data, timestamp=1.0)
    b = plain.update(data, timestamp=1.0)
    equal_detections(a, b)
    assert state(observed) == state(plain)
    assert [e["stage"] for e in events] == ["high_ocm", "weak_ocm", "weak_ocr"]
    assert events[-1]["raw_confidences"].tolist() == [0.3] and events[-1]["pool"][0]["id"] == 0
    assert events[-1]["raw_iou"][0, 0] == 1 and events[-1]["result"][0] == [(0, 0)]
    events[-1]["pool"][0]["state"]["state"][:] = 123
    assert state(observed) == state(plain)


def test_reset_independence_empty_invalid_and_configuration():
    first = warm(SplitContinuationTracker(frame_rate=5))
    second = SplitContinuationTracker(frame_rate=5)
    snapshot = state(first)
    with pytest.raises(ValueError):
        first.update(sv.Detections(xyxy=np.array([[0, 0, np.nan, 4.0]])))
    assert state(first) == snapshot
    first.reset()
    assert state(first) == state(second)
    assert first.update(sv.Detections.empty()).tracker_id.tolist() == []
    assert second.tracks == [] and second._next_track_id == 0
    with pytest.raises(ValueError):
        SplitContinuationTracker(enable_cmc=True)
    with pytest.raises(ValueError):
        SplitContinuationTracker(0.3)
    with pytest.raises(ValueError):
        make_tracker("other")
    with pytest.raises(ValueError):
        make_tracker("native06", high_conf_det_threshold=0.4)


@pytest.mark.parametrize("policy", ["native06", "uniform02", "uniform04"])
@pytest.mark.parametrize("side", [-1, 0, 1])
def test_native_uniform_cutoff_equality(policy, side):
    gate = next(p.birth for p in POLICIES if p.name == policy)
    score = np.nextafter(gate, -np.inf) if side < 0 else np.nextafter(gate, np.inf) if side > 0 else gate
    tracker = make_tracker(policy, frame_rate=5)
    tracker.update(det((score,)), timestamp=0)
    assert len(tracker.tracks) == int(side >= 0)


@pytest.mark.parametrize("side", [-1, 0, 1])
def test_exact_seconds_expiry_boundary_for_weak_reacquisition(side):
    tracker = warm(SplitContinuationTracker(frame_rate=5))
    timestamp = np.nextafter(1.8, -np.inf) if side < 0 else np.nextafter(1.8, np.inf) if side > 0 else 1.8
    out = tracker.update(det((0.3,)), timestamp=timestamp)
    assert len(tracker.tracks) == int(side <= 0)
    assert out.tracker_id.tolist() == ([0] if side <= 0 else [-1])
    assert tracker._next_track_id == 1


def test_weak_direction_uses_raw_confidence_and_duplicate_clock():
    events = []
    tracker = SplitContinuationTracker(frame_rate=5, association_observer=events.append)
    reference = OCSORTTracker(frame_rate=5)
    for i in range(5):
        tracker.update(det(offsets=[2 * i]), timestamp=i * 0.2)
        reference.update(det(offsets=[2 * i]), timestamp=i * 0.2)
    data = det((0.3,), offsets=[10])
    timing = reference._predict_timing(1.0)
    reference._predict_tracklets(reference.tracks, timing, return_predictions=True)
    expected = reference._compute_direction_consistency_matrix(data.xyxy, data.confidence)
    assert np.any(expected != 0)
    events.clear()
    saved = copy.deepcopy(data)
    tracker.update(data, timestamp=1.0)
    weak = next(e for e in events if e["stage"] == "weak_ocm")
    np.testing.assert_array_equal(weak["direction"], expected)
    assert weak["proposal_indices"].tolist() == [0] and weak["raw_confidences"].tolist() == [0.3]
    equal_detections(data, saved)
    age = tracker.tracks[0].age
    with pytest.warns(UserWarning, match="duplicate timestamp"):
        tracker.update(data, timestamp=1.0)
    assert tracker.tracks[0].age == age
    before = state(tracker)
    with pytest.warns(UserWarning, match="earlier than"):
        out = tracker.update(data, timestamp=0.9)
    assert out.tracker_id.tolist() == [-1] and state(tracker) == before
