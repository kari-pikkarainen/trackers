# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Preloaded update-only local timing with six fixed order permutations and prediction parity."""

from __future__ import annotations

import copy
import itertools
import time
from collections import defaultdict

import numpy as np

from experiments.continuation_confidence import make_tracker
from experiments.tracking_validation_eval import jsonable, require

POLICY_NAMES = ("native06", "uniform04", "split02")
POLICY_ORDERS = tuple(itertools.permutations(POLICY_NAMES))


def native_state(tracker):
    """Snapshot all tracker fields except the observational callback."""
    return jsonable({k: v for k, v in vars(tracker).items() if k != "association_observer"})


def detection_output(detections):
    """Retain every row and raw metadata; ID -1 is distinct from valid ID zero."""
    if not len(detections):
        return []
    return [
        {"box": box.tolist(), "id": int(ident), "confidence": float(score), "proposal_index": int(proposal)}
        for box, ident, score, proposal in zip(
            detections.xyxy,
            detections.tracker_id,
            detections.confidence,
            detections.data["proposal_index"],
            strict=True,
        )
    ]


def unchanged(before, after):
    """Check complete input metadata outside the timed region."""
    for field in ["xyxy", "confidence", "class_id", "tracker_id", "mask"]:
        a, b = getattr(before, field), getattr(after, field)
        require((a is None and b is None) or np.array_equal(a, b), f"input mutation: {field}")
    require(jsonable(before.data) == jsonable(after.data), "input data mutation")


def replay_policy(sequences, policy, *, clock=None, diagnostic=False):
    """Fresh shot state; timer encloses update only, not copy/check/log/construction."""
    require(policy in POLICY_NAMES, "unfrozen policy")
    timer = time.perf_counter_ns if clock is None else clock
    wall_start = time.perf_counter_ns()
    construction_ns = 0
    records, measurements = [], []
    for sequence in sequences:
        tracker = None
        shot = None
        rows = []
        events = []
        uids = {}
        for item in sequence["items"]:
            frame = item["truth"]
            if tracker is None or frame["shot"] != shot:
                start = time.perf_counter_ns()
                tracker = make_tracker(policy, frame_rate=5)
                construction_ns += time.perf_counter_ns() - start
                shot = frame["shot"]
                uids = {}
                if diagnostic and policy == "split02":
                    tracker.association_observer = lambda e: events.append(jsonable(e))
            data = copy.deepcopy(item["detections"])
            saved = copy.deepcopy(data)
            events.clear()
            prior_objects = {t: t.tracker_id for t in tracker.tracks}
            for track in tracker.tracks:
                if track not in uids:
                    uids[track] = len(uids)
            count_before = len(tracker.tracks)
            begin = timer() if clock is not None else None
            if begin is None:
                result = tracker.update(data, timestamp=frame["timestamp"])
                duration = None
            else:
                result = tracker.update(data, timestamp=frame["timestamp"])
                duration = timer() - begin
                require(duration >= 0, "nonmonotonic timer")
            unchanged(saved, data)
            output = detection_output(result)
            weak_matches = []
            for event in events:
                if event["stage"].startswith("weak"):
                    for r, c in event["result"][0]:
                        source_row = int(event["proposal_indices"][c])
                        proposal = int(data.data["proposal_index"][source_row])
                        assigned = next(o["id"] for o in output if o["proposal_index"] == proposal)
                        weak_matches.append(
                            {
                                "stage": event["stage"],
                                "proposal_index": proposal,
                                "prior_id": event["pool"][r]["id"],
                                "raw_confidence": event["raw_confidences"][c],
                                "output_id": assigned,
                                "internal_only": assigned < 0,
                            }
                        )
            for track in tracker.tracks:
                if track not in uids:
                    uids[track] = len(uids)
            lifecycle = {
                "births": [{"uid": uids[t], "id": t.tracker_id} for t in tracker.tracks if t not in prior_objects],
                "expired": [
                    {"uid": uids[t], "id": ident} for t, ident in prior_objects.items() if t not in tracker.tracks
                ],
                "newly_confirmed": [
                    {"uid": uids[t], "id": t.tracker_id}
                    for t in tracker.tracks
                    if t in prior_objects and prior_objects[t] < 0 <= t.tracker_id
                ],
            }
            rows.append(
                {
                    "sample_index": frame["sample_index"],
                    "shot": shot,
                    "output": output,
                    "events": copy.deepcopy(events),
                    **({"weak_matches": weak_matches, "lifecycle": lifecycle} if diagnostic else {}),
                    **({"state_after": native_state(tracker)} if diagnostic else {}),
                }
            )
            if duration is not None:
                measurements.append(
                    {
                        "clip": sequence["id"],
                        "sample_index": frame["sample_index"],
                        "shot": shot,
                        "duration_ns": duration,
                        "detection_count": len(data),
                        "active_tracks_before": count_before,
                        "active_tracks_after": len(tracker.tracks),
                        "source": sequence["source"],
                        "categories": sequence["categories"],
                        "source_timestamp": frame["timestamp"],
                    }
                )
        records.append({"clip": sequence["id"], "frames": rows, "final_state": native_state(tracker)})
    return {
        "predictions": records,
        "measurements": measurements,
        "construction_reset_ns": construction_ns,
        "complete_replay_wall_ns": time.perf_counter_ns() - wall_start,
    }


def predictions_only(replay):
    """Compare predictions independently of diagnostic events or timing values."""
    return [
        {
            "clip": c["clip"],
            "frames": [
                {k: v for k, v in f.items() if k not in ["events", "state_after", "weak_matches", "lifecycle"]}
                for f in c["frames"]
            ],
        }
        for c in replay["predictions"]
    ]


def benchmark(sequences, warmup_sequences, ordinary, *, clock=time.perf_counter_ns):
    """Discard complete development warmups, then each of six permutations once."""
    require(all(c["partition"] == "development" for c in warmup_sequences), "warmup must be development only")
    warmup = []
    for policy in POLICY_NAMES:
        discarded = replay_policy(warmup_sequences, policy)
        warmup.append(
            {
                "policy": policy,
                "updates": sum(len(c["frames"]) for c in discarded["predictions"]),
                "state_discarded": True,
            }
        )
    passes = []
    for pass_index, order in enumerate(POLICY_ORDERS):
        for policy in order:
            result = replay_policy(sequences, policy, clock=clock)
            require(
                predictions_only(result) == predictions_only(ordinary[policy]), "benchmark/ordinary prediction mismatch"
            )
            passes.append(
                {
                    "pass": pass_index,
                    "order": list(order),
                    "policy": policy,
                    "ordinary_prediction_parity": True,
                    **{k: v for k, v in result.items() if k != "predictions"},
                }
            )
    return {
        "scope": (
            "tracker.update only, monotonic nanoseconds; all conversion/copy/I/O/observers/metrics/encoding excluded"
        ),
        "warmup": warmup,
        "passes": passes,
        "summary": timing_summary(passes),
        "limits": (
            "Local observational wall-clock measurements, not isolated hardware or production "
            "throughput; no speed gate. One update is one sampled input, not one source video "
            "frame. No filters changed, so no filter microbenchmark applies."
        ),
    }


def timing_summary(passes):
    """Distribution and paired clip/pass ratios; no independent-frame confidence interval."""
    values = defaultdict(list)
    totals = defaultdict(float)
    by_clip = defaultdict(list)
    grouped = defaultdict(list)
    for record in passes:
        policy = record["policy"]
        for m in record["measurements"]:
            values[policy].append(m["duration_ns"])
            totals[record["pass"], policy, m["clip"]] += m["duration_ns"]
            by_clip[policy, m["clip"]].append(m["duration_ns"])
            if "source" in m:
                grouped[policy, "source", m["source"]].append(m["duration_ns"])
                grouped[policy, "detection_count", str(m["detection_count"])].append(m["duration_ns"])
                for category in m["categories"]:
                    grouped[policy, "category", category].append(m["duration_ns"])
    distributions = {}
    for policy, durations in values.items():
        x = np.asarray(durations, float) / 1e6
        distributions[policy] = {
            "updates": len(x),
            "median_ms": float(np.median(x)),
            "p90_ms": float(np.percentile(x, 90)),
            "p95_ms": float(np.percentile(x, 95)),
            "mean_ms": float(x.mean()),
            "total_ms": float(x.sum()),
        }
    pairs = []
    for (pass_index, policy, clip), duration in sorted(totals.items()):
        native = totals[pass_index, "native06", clip]
        pairs.append(
            {
                "pass": pass_index,
                "policy": policy,
                "clip": clip,
                "total_ns": duration,
                "native_total_ns": native,
                "ratio_vs_native": duration / native if native else None,
            }
        )
    return {
        "distributions": distributions,
        "paired_clip_pass_ratios": pairs,
        "clip_update_counts": [
            {"policy": p, "clip": c, "timed_updates_all_six_passes": len(v)} for (p, c), v in sorted(by_clip.items())
        ],
        "group_summaries": [
            {
                "policy": p,
                "group_kind": kind,
                "group": name,
                "updates": len(v),
                "mean_ms": float(np.mean(v)) / 1e6,
                "median_ms": float(np.median(v)) / 1e6,
                "p90_ms": float(np.percentile(v, 90)) / 1e6,
                "p95_ms": float(np.percentile(v, 95)) / 1e6,
                "total_ms": float(np.sum(v)) / 1e6,
            }
            for (p, kind, name), v in sorted(grouped.items())
        ],
        "paired_ratio_ranges": {
            p: {"minimum": min(v) if v else None, "maximum": max(v) if v else None}
            for p in POLICY_NAMES
            for v in [[r["ratio_vs_native"] for r in pairs if r["policy"] == p and r["ratio_vs_native"] is not None]]
        },
        "visual_clip_count": len({c for _, c in by_clip}),
        "uncertainty_unit": (
            "Paired clip/pass ratios; correlated frames/repeats are not independent trials; "
            "replay-event groups remain correlated."
        ),
    }
