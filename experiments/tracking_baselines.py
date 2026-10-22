# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Fixed baseline factories and bounded, preloaded update-only experiment runners."""

from __future__ import annotations

import copy
import hashlib
import time
from pathlib import Path

import numpy as np

from experiments.continuation_confidence import make_tracker
from experiments.tracking_validation_eval import jsonable, require
from experiments.tracking_validation_timing import detection_output, unchanged
from trackers import BoTSORTTracker, CBIoUTracker, OCSORTTracker
from trackers.utils.state_representations import XCYCWHStateEstimator

VARIANTS = (
    "native06",
    "uniform04",
    "split02",
    "botsort-native",
    "cbiou-native",
    "botsort-cmc-off",
    "botsort-common",
    "cbiou-common",
    "ocsort-common",
)
PRIMARY = VARIANTS[:5]
SPLIT_SHA256 = "f1be86060a4621d76bb38a4652c27b8b4a7d5fad53a021b30ce979d164892f9a"


def make_baseline(name, *, diagnostic=False, frame_rate=5):
    """Use literal native constructors; only the accepted split policy changes OC updates."""
    require(name in VARIANTS, "unfrozen baseline")
    if name == "split02":
        path = Path(__file__).with_name("continuation_confidence.py")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == SPLIT_SHA256, "shared split policy changed")
    if diagnostic:
        from experiments.tracking_baseline_observers import observed_baseline

        return observed_baseline(name, frame_rate=frame_rate)
    if name == "native06":
        return OCSORTTracker(frame_rate=frame_rate)
    if name == "uniform04":
        return OCSORTTracker(frame_rate=frame_rate, high_conf_det_threshold=0.4)
    if name == "split02":
        return make_tracker("split02", frame_rate=frame_rate)
    if name == "botsort-native":
        return BoTSORTTracker(frame_rate=frame_rate)
    if name == "cbiou-native":
        return CBIoUTracker(frame_rate=frame_rate)
    if name == "botsort-cmc-off":
        return BoTSORTTracker(frame_rate=frame_rate, enable_cmc=False)
    if name == "botsort-common":
        return BoTSORTTracker(frame_rate=frame_rate, enable_cmc=False, track_activation_threshold=0.6)
    if name == "cbiou-common":
        return CBIoUTracker(frame_rate=frame_rate, track_activation_threshold=0.6)
    return OCSORTTracker(
        frame_rate=frame_rate, minimum_consecutive_frames=2, state_estimator_class=XCYCWHStateEstimator
    )


def full_deep_status(*, enabled=False):
    """Fail closed without importing any unapproved Deep-EIoU source or appearance model."""
    prerequisites = [
        "official whole-tracker source permission",
        "official appearance artifact and bound embeddings",
        "validated adapter and bounded execution authorization",
    ]
    require(not enabled, "full Deep-EIoU unavailable: " + "; ".join(prerequisites))
    return {"status": "unavailable", "prerequisites": prerequisites, "ablation": "not assigned"}


def balanced_orders(names=VARIANTS):
    """Every variant occupies every position twice, across forward/reverse rotations."""
    names = tuple(names)
    require(len(names) >= 3 and len(set(names)) == len(names), "need at least three unique variants")
    return tuple(order[i:] + order[:i] for order in (names, names[::-1]) for i in range(len(names)))


def preload(sequence, *, read_image=None):
    """Load one bounded clip outside timing; retain decoded actual frames for native CMC."""
    if read_image is None:
        import cv2

        read_image = cv2.imread
    result = {**sequence, "items": []}
    for item in sequence["items"]:
        image = item.get("image")
        if image is None:
            image = read_image(item["truth"]["image_path"])
        require(isinstance(image, np.ndarray) and image.ndim == 3, "missing actual decoded frame")
        truth = item["truth"]
        require(image.shape[:2] == (truth["height"], truth["width"]), "decoded frame dimensions differ")
        result["items"].append({**item, "image": image})
    return result


def replay_sequence(sequence, name, *, diagnostic=False, clock=None):
    """Fresh shot-local state; only native update lies inside the injectable timer."""
    require(not (diagnostic and clock is not None), "diagnostics excluded from timing")
    from experiments.tracking_baseline_observers import snapshot_state

    tracker, shot = None, None
    rows, measurements = [], []
    for item in sequence["items"]:
        truth = item["truth"]
        reset = tracker is None or truth["shot"] != shot
        if reset:
            rate = sequence.get("synthetic_reference_rate", 5)
            require(rate == 5 or sequence.get("population") == "synthetic_source_clock", "recorded cadence changed")
            tracker, shot = make_baseline(name, diagnostic=diagnostic, frame_rate=rate), truth["shot"]
        require(tracker is not None, "missing tracker")
        frame = item.get("image") if name == "botsort-native" else None
        require(name != "botsort-native" or isinstance(frame, np.ndarray), "CMC requires preloaded actual frame")
        saved_frame = None if frame is None else frame.copy()
        data, before = copy.deepcopy(item["detections"]), copy.deepcopy(item["detections"])
        prior_count = len(tracker.tracks)
        if diagnostic:
            tracker.begin_observation()
        begin = clock() if clock is not None else None
        output = tracker.update(data, frame=frame, timestamp=truth["timestamp"])
        duration = None if clock is None else clock() - begin
        require(duration is None or duration >= 0, "nonmonotonic clock")
        unchanged(before, data)
        require(saved_frame is None or np.array_equal(saved_frame, frame), "input image mutation")
        row = {"sample_index": truth["sample_index"], "shot": shot, "reset": reset, "output": detection_output(output)}
        if diagnostic:
            row.update(tracker.end_observation())
            row["state_after"] = snapshot_state(tracker)
        rows.append(row)
        if duration is not None:
            measurements.append(
                {
                    "clip": sequence["id"],
                    "source": sequence["source"],
                    "sample_index": truth["sample_index"],
                    "duration_ns": duration,
                    "detection_count": len(data),
                    "active_tracks_before": prior_count,
                    "active_tracks_after": len(tracker.tracks),
                }
            )
    return {
        "clip": sequence["id"],
        "frames": rows,
        "measurements": measurements,
        "final_state": snapshot_state(tracker) if tracker is not None else None,
    }


def timing_passes(sequences, *, warmup_sequences=None, names=VARIANTS, clock=time.perf_counter_ns, read_image=None):
    """Discard a complete population warmup before measurement; bound memory to one preloaded clip."""
    sequences = tuple(sequences)
    warmups = tuple(sequences if warmup_sequences is None else warmup_sequences)
    require(
        bool(warmups) and all(s.get("panel") == "validation" and s.get("partition") == "development" for s in warmups),
        "explicit development-panel warmup required",
    )
    orders, samples = balanced_orders(names), []
    for sequence in warmups:
        loaded = preload(sequence, read_image=read_image)
        for name in names:
            replay_sequence(loaded, name)
    for sequence in sequences:
        loaded = preload(sequence, read_image=read_image)
        for pass_index, order in enumerate(orders):
            for position, name in enumerate(order):
                replay = replay_sequence(loaded, name, clock=clock)
                samples.extend(
                    {**m, "variant": name, "pass": pass_index, "position": position} for m in replay["measurements"]
                )
    return {
        "orders": orders,
        "warmups_discarded": True,
        "warmup_population": [
            {
                "clip": s["id"],
                "source": s["source"],
                "panel": s["panel"],
                "partition": s["partition"],
                "frames": len(s["items"]),
            }
            for s in warmups
        ],
        "measurement_population": [
            {
                "clip": s["id"],
                "source": s["source"],
                "panel": s["panel"],
                "partition": s["partition"],
                "frames": len(s["items"]),
            }
            for s in sequences
        ],
        "preload_scope": "one clip",
        "timed_region": "native update only",
        "samples": samples,
        "summary": timing_summary(samples),
    }


def timing_summary(samples):
    """Retain populations and paired clip/pass ratios against both faithful baselines."""
    groups, totals = {}, {}
    for row in samples:
        key = row["variant"]
        groups.setdefault(key, []).append(row["duration_ns"])
        pair = (row["clip"], row["pass"])
        totals.setdefault(key, {}).setdefault(pair, 0)
        totals[key][pair] += row["duration_ns"]
    summary = {}
    for name, times in groups.items():
        summary[name] = {
            "frames": len(times),
            "total_ns": sum(times),
            "mean_ns": float(np.mean(times)),
            **{f"p{p}_ns": float(np.percentile(times, p)) for p in (50, 90, 95)},
            "ratios": {},
        }
        for reference in ("native06", "botsort-native"):
            pairs = sorted(set(totals[name]) & set(totals.get(reference, {})))
            summary[name]["ratios"][reference] = [
                {"clip": c, "pass": p, "ratio": totals[name][c, p] / totals[reference][c, p]}
                for c, p in pairs
                if totals[reference][c, p] > 0
            ]
    return jsonable(summary)
