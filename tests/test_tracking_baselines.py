# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Synthetic fixed-constructor, observer transparency, timing and panel contracts."""

from __future__ import annotations

import copy
import importlib
import itertools

import cv2
import numpy as np
import pytest
import supervision as sv

from experiments.tracking_baseline_analysis import compare_panels, evaluate_panel, startup_spans
from experiments.tracking_baseline_observers import snapshot_state
from experiments.tracking_baselines import (
    VARIANTS,
    balanced_orders,
    full_deep_status,
    make_baseline,
    preload,
    replay_sequence,
    timing_passes,
)
from experiments.tracking_validation_eval import account_frame
from experiments.tracking_validation_events import person_events
from experiments.tracking_validation_timing import unchanged
from trackers import BoTSORTTracker, CBIoUTracker, OCSORTTracker
from trackers.core.base import BaseTracker


def detections(boxes=([20, 20, 40, 60],), scores=(0.9,), indices=None):
    """Raw synthetic proposals with source indices deliberately unlike column indices."""
    return sv.Detections(
        xyxy=np.asarray(boxes, float).reshape(-1, 4),
        confidence=np.asarray(scores, float),
        class_id=np.zeros(len(scores), int),
        data={"proposal_index": np.asarray(indices if indices is not None else np.arange(len(scores)) + 17)},
    )


def truth(sample, *, shot=0, timestamp=None, source_frame=None):
    """Known physical principal with fixed immutable localization/accounting semantics."""
    return {
        "sample_index": sample,
        "source_frame_index": sample * 5 if source_frame is None else source_frame,
        "timestamp": sample / 5 if timestamp is None else timestamp,
        "pts": sample * 1000,
        "time_base": "1/5000",
        "shot": shot,
        "phase": "unknown",
        "height": 120,
        "width": 160,
        "uncertainties": [],
        "people": [
            {
                "person_id": f"fixture/shot{shot}/rider",
                "box": [20, 20, 40, 60],
                "box_uncertainty_pixels": 1,
                "role": "principal_rider",
                "identity_eligible": True,
                "detection_eligible": True,
                "evaluation_domain": "physical_person",
            }
        ],
    }


def sequence(count=4):
    """Small bounded fixture, never a private corpus dependency."""
    return {
        "id": "fixture",
        "source": "synthetic",
        "panel": "validation",
        "categories": ["fixture"],
        "partition": "development",
        "semantic_event_group": "one-physical-event",
        "items": [
            {"truth": truth(i), "detections": detections(), "image": np.zeros((120, 160, 3), np.uint8)}
            for i in range(count)
        ],
    }


@pytest.mark.parametrize("name", VARIANTS)
def test_observer_every_update_future_reset_and_nonmutation(name):
    """Diagnostics preserve outputs and complete accessible state through misses, clocks and future continuation."""
    native, observed = make_baseline(name), make_baseline(name, diagnostic=True)
    rng = np.random.default_rng(32)
    texture = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    inputs = [
        (0.0, 0.9, 0),
        (0.2, 0.9, 1),
        (0.4, 0.9, 2),
        (0.6, 0.4, 3),
        (0.8, None, 4),
        (1.0, 0.9, 5),
        (1.0, 0.9, 6),
        (0.9, 0.9, 7),
        (float("nan"), 0.9, 8),
        (None, 0.9, 9),
        (2.0, 0.9, 10),
        (3.0, None, 11),
        (3.00001, 0.9, 12),
        (3.2, 0.9, 13),
    ]
    for timestamp, score, step in inputs * 2:
        if step == 0:
            native.reset()
            observed.reset()
        data = detections() if score is not None else detections((), ())
        if score is not None:
            data.confidence[:] = score
            data.xyxy += [step * 0.2, 0, step * 0.2, 0]
        before = copy.deepcopy(data)
        frame = cv2.warpAffine(texture, np.float32([[1, 0, step], [0, 1, step / 3]]), (160, 120))
        saved_frame = frame.copy()
        kwargs = {"frame": frame} if name == "botsort-native" else {}
        observed.begin_observation()
        a = native.update(data, timestamp=timestamp, **kwargs)
        b = observed.update(copy.deepcopy(data), timestamp=timestamp, **kwargs)
        observed.end_observation()
        assert np.array_equal(a.xyxy, b.xyxy)
        assert np.array_equal(a.tracker_id, b.tracker_id)
        assert np.array_equal(a.confidence, b.confidence)
        assert snapshot_state(native) == snapshot_state(observed)
        unchanged(before, data)
        assert np.array_equal(saved_frame, frame)


def test_native_factories_and_registry_are_unchanged():
    """Literal classes and registry entries survive both import and reload of diagnostic subclasses."""
    before = dict(BaseTracker._registry)
    import experiments.tracking_baseline_observers as observers

    importlib.reload(observers)
    assert BaseTracker._registry == before
    assert type(make_baseline("native06")) is OCSORTTracker
    assert type(make_baseline("botsort-native")) is BoTSORTTracker
    assert type(make_baseline("cbiou-native")) is CBIoUTracker
    assert make_baseline("botsort-native").cmc is not None
    assert make_baseline("botsort-cmc-off").cmc is None
    assert make_baseline("uniform04").high_conf_det_threshold == 0.4
    assert make_baseline("botsort-common").track_activation_threshold == 0.6
    with pytest.raises(ValueError, match="unfrozen"):
        make_baseline("botsort-uniform02")


def test_cmc_accessible_state_really_updates():
    """Moving texture exercises actual sparse optical flow, not an injected identity transform."""
    tracker = make_baseline("botsort-native", diagnostic=True)
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for step in range(3):
        tracker.begin_observation()
        frame = cv2.warpAffine(image, np.float32([[1, 0, step * 4], [0, 1, step * 2]]), (320, 240))
        tracker.update(detections(), timestamp=step * 0.2, frame=frame)
        diagnostic = tracker.end_observation()
    assert tracker.cmc._initialized
    assert tracker.cmc._prev_frame_gray is not None
    assert tracker.cmc._prev_points is not None
    transform = diagnostic["events"][0]["cmc_transform"]
    assert transform is not None and abs(transform[0][2]) > 1
    snapshot = snapshot_state(tracker)
    assert snapshot["cmc"]["_prev_frame_gray"]
    assert snapshot["cmc"]["detector"] is None


def test_observer_source_and_call_site_fail_closed():
    """Direct or changed-source association may not silently receive invented stage mappings."""
    tracker = make_baseline("native06", diagnostic=True)
    with pytest.raises(ValueError, match="outside pinned"):
        tracker._get_associated_indices(np.zeros((1, 1)), np.zeros((1, 1)))
    tracker._baseline_sites = {}
    tracker.begin_observation()
    with pytest.raises(ValueError, match="unexpected native"):
        tracker.update(detections(), timestamp=0)


@pytest.mark.parametrize("name", VARIANTS)
def test_replay_cut_resets_and_snapshot_independence(name):
    fixture = sequence()
    fixture["items"][3]["truth"] = truth(3, shot=1)
    replay = replay_sequence(fixture, name, diagnostic=True)
    assert [r["reset"] for r in replay["frames"]] == [True, False, False, True]
    first = copy.deepcopy(replay["frames"][1])
    replay["final_state"]["tracks"].clear()
    assert replay["frames"][1] == first
    assert replay["frames"][1]["events"][0]["proposal_indices"] == [17]


def test_balanced_timing_excludes_io_and_diagnostics():
    fixture = sequence(2)
    orders = balanced_orders()
    assert len(orders) == 18
    for position in range(9):
        assert sorted(o[position] for o in orders) == sorted(VARIANTS * 2)
    ticks = itertools.count(0, 100)
    result = timing_passes([fixture], clock=lambda: next(ticks))
    assert len(result["samples"]) == 18 * 9 * 2
    assert all(s["duration_ns"] == 100 for s in result["samples"])
    assert all(v["frames"] == 36 for v in result["summary"].values())
    assert all(r["ratio"] == 1 for v in result["summary"].values() for r in v["ratios"]["native06"])
    with pytest.raises(ValueError, match="excluded"):
        replay_sequence(fixture, "native06", diagnostic=True, clock=lambda: 0)
    clock = iter([2, 1])
    with pytest.raises(ValueError, match="nonmonotonic"):
        replay_sequence(sequence(1), "native06", clock=lambda: next(clock))


def test_preload_and_deep_fail_closed():
    fixture = sequence(1)
    fixture["items"][0].pop("image")
    fixture["items"][0]["truth"]["image_path"] = "synthetic-image"
    with pytest.raises(ValueError, match="missing actual"):
        preload(fixture, read_image=lambda p: None)
    with pytest.raises(ValueError, match="CMC requires"):
        replay_sequence(fixture, "botsort-native")
    assert full_deep_status()["status"] == "unavailable"
    with pytest.raises(ValueError, match="source permission"):
        full_deep_status(enabled=True)


def test_startup_censor_and_panel_grouping(tmp_path):
    fixture = sequence(4)
    frames = [i["truth"] for i in fixture["items"]]
    outputs = [[], [], [{"box": [20, 20, 40, 60], "id": 0, "confidence": 0.9, "proposal_index": 17}], []]
    accounts = [account_frame(f, o) for f, o in zip(frames, outputs, strict=True)]
    events = person_events(frames, accounts)
    start = startup_spans(frames, events)[0]
    assert start["observations_to_first"] == 3
    assert start["source_frames_to_first"] == 10
    assert start["seconds_to_first"] == 0.4
    never = startup_spans(frames, person_events(frames, [account_frame(f, []) for f in frames]))[0]
    assert never["never_established"] and never["seconds_to_first"] is None
    assert never["seconds_lower_bound"] == 0.6
    replay = replay_sequence(fixture, "native06")
    evaluated = evaluate_panel([fixture], [replay], tmp_path)
    assert evaluated["panel"] == "validation"
    assert any(g["group_kind"] == "source" and g["group"] == "synthetic" for g in evaluated["groups"])
    assert compare_panels(evaluated, evaluated)[0]["coverage"]["lost"] == []
    other = {**evaluated, "panel": "original"}
    with pytest.raises(ValueError, match="histories"):
        compare_panels(evaluated, other)
    with pytest.raises(ValueError, match="cannot pool"):
        evaluate_panel([fixture, {**fixture, "panel": "original"}], [replay, replay], tmp_path)


@pytest.mark.parametrize("name", ("native06", "uniform04", "split02"))
def test_bounded_replay_matches_accepted_oc_orchestration(name):
    from experiments.tracking_validation_timing import replay_policy

    fixture = sequence(6)
    fixture["items"][3]["detections"].confidence[:] = 0.4
    old = replay_policy([fixture], name)["predictions"][0]
    new = replay_sequence(fixture, name)
    assert [r["output"] for r in old["frames"]] == [r["output"] for r in new["frames"]]
    assert old["final_state"] == new["final_state"]


def test_accuracy_runner_uses_panel_contract_and_never_adjudicates(tmp_path):
    from experiments.tracking_baseline_runner import run_accuracy_panel

    fixture = sequence(5)
    fixture["items"][3]["detections"].confidence[:] = 0.4
    reports = run_accuracy_panel([fixture], tmp_path, names=("native06", "uniform04"))
    assert set(reports) == {"native06", "uniform04"}
    assert not reports["native06"]["audit_complete"]
    assert reports["native06"]["causal_candidates"]
    assert all(not r["confirmed_motion_cause"] for r in reports["native06"]["causal_candidates"])
    assert (tmp_path / "native06" / "fixture" / "predictions.json").is_file()
    assert (tmp_path / "uniform04" / "fixture" / "evaluation.json").is_file()


def test_complete_warmup_population_precedes_first_timed_update():
    """No clip measurement starts until every selected clip and variant has had a discarded warmup."""
    clips = [sequence(1), sequence(1)]
    events = []
    for index, clip in enumerate(clips):
        clip["id"] = f"clip{index}"
        clip["items"][0].pop("image")
        clip["items"][0]["truth"]["image_path"] = clip["id"]

    def reader(path):
        events.append(("read", path))
        return np.zeros((120, 160, 3), np.uint8)

    ticks = itertools.count()

    def clock():
        events.append(("timer", None))
        return next(ticks)

    timing_passes(clips, names=("native06", "uniform04", "split02"), clock=clock, read_image=reader)
    first_timer = next(i for i, event in enumerate(events) if event[0] == "timer")
    assert events[:first_timer] == [("read", "clip0"), ("read", "clip1"), ("read", "clip0")]


def test_original_and_former_test_timing_require_explicit_development_warmup():
    measured = sequence(1)
    measured.update(id="original", panel="original", partition="development_exposed")
    warmup = sequence(1)
    warmup["id"] = "development"
    ticks = itertools.count()
    with pytest.raises(ValueError, match="development-panel warmup"):
        timing_passes([measured], names=("native06", "uniform04", "split02"), clock=lambda: next(ticks))
    report = timing_passes(
        [measured], warmup_sequences=[warmup], names=("native06", "uniform04", "split02"), clock=lambda: next(ticks)
    )
    assert report["warmup_population"][0]["clip"] == "development"
    assert report["measurement_population"][0]["clip"] == "original"
    assert all(row["clip"] == "original" for row in report["samples"])
    with pytest.raises(ValueError, match="development-panel warmup"):
        timing_passes([measured], warmup_sequences=[{**warmup, "partition": "test"}])
