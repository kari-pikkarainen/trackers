# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Injected timing scopes and exact prediction/state invariants, never speed rankings."""

import copy
import itertools

import numpy as np
import pytest
import supervision as sv

from experiments.tracking_validation_timing import (
    POLICY_NAMES,
    POLICY_ORDERS,
    benchmark,
    predictions_only,
    replay_policy,
    timing_summary,
)


def sequence(partition="development"):
    return {
        "id": "synthetic",
        "source": "synthetic",
        "partition": partition,
        "categories": ["test"],
        "items": [
            {
                "truth": {"sample_index": n, "shot": int(n >= 4), "timestamp": n * 0.2},
                "detections": sv.Detections(
                    xyxy=np.array([[10.0, 10.0, 30.0, 50.0]]),
                    confidence=np.array([0.9 if n % 3 else 0.3]),
                    class_id=np.array([0]),
                    data={"proposal_index": np.array([7])},
                ),
            }
            for n in range(8)
        ],
    }


def test_six_orders_warmups_scope_and_milliseconds():
    sequences = [sequence()]
    original = copy.deepcopy(sequences)
    ordinary = {p: replay_policy(sequences, p, diagnostic=True) for p in POLICY_NAMES}
    calls = []
    counter = itertools.count()

    def timer():
        calls.append(1)
        return next(counter) * 1000

    result = benchmark(sequences, sequences, ordinary, clock=timer)
    assert len(result["passes"]) == 18
    assert len(calls) == 18 * 8 * 2  # Warmup/construction/copy/check/logging never calls this timer.
    assert [r["updates"] for r in result["warmup"]] == [8, 8, 8]
    assert all(r["state_discarded"] for r in result["warmup"])
    assert [tuple(result["passes"][n * 3]["order"]) for n in range(6)] == list(POLICY_ORDERS)
    for p in POLICY_NAMES:
        stats = result["summary"]["distributions"][p]
        assert stats["updates"] == 48 and stats["median_ms"] == 0.001
    assert all(row["ratio_vs_native"] == 1 for row in result["summary"]["paired_clip_pass_ratios"])
    for old, new in zip(original[0]["items"], sequences[0]["items"]):
        np.testing.assert_array_equal(old["detections"].xyxy, new["detections"].xyxy)
        np.testing.assert_array_equal(
            old["detections"].data["proposal_index"], new["detections"].data["proposal_index"]
        )


@pytest.mark.parametrize("policy", POLICY_NAMES)
def test_observed_and_plain_predictions_and_fresh_shot_ids(policy):
    data = [sequence()]
    observed = replay_policy(data, policy, diagnostic=True)
    plain = replay_policy(data, policy)
    assert predictions_only(observed) == predictions_only(plain)
    assert observed["predictions"][0]["final_state"] == plain["predictions"][0]["final_state"]
    assert plain["measurements"] == []
    ids = [row["output"][0]["id"] for row in plain["predictions"][0]["frames"]]
    assert ids[4] == -1  # New shot cannot inherit prior confirmed state.


def test_warmup_must_not_consume_test_as_development():
    data = [sequence("test")]
    ordinary = {p: replay_policy(data, p) for p in POLICY_NAMES}
    with pytest.raises(ValueError, match="warmup"):
        benchmark(data, data, ordinary)


def test_zero_native_timing_has_undefined_ratio():
    record = {"pass": 0, "policy": "native06", "measurements": [{"clip": "x", "duration_ns": 0}]}
    assert timing_summary([record])["paired_clip_pass_ratios"][0]["ratio_vs_native"] is None


@pytest.mark.parametrize("policy", POLICY_NAMES)
def test_initial_empty_inputs_preserve_empty_output_without_metadata(policy):
    data = sequence()
    empty = sv.Detections.empty()
    empty.data["proposal_index"] = np.array([], int)
    data["items"][0]["detections"] = empty
    result = replay_policy([data], policy, diagnostic=True)
    assert result["predictions"][0]["frames"][0]["output"] == []
