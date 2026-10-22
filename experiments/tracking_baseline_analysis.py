# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Frozen evaluator reuse, panel-local contrasts and conservative causal evidence records."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.tracking_validation import evaluate_sequence, grouped_summary
from experiments.tracking_validation_eval import jsonable, require
from experiments.tracking_validation_events import paired_events, paired_frames
from trackers.eval import box_iou


def startup_spans(frames, events):
    """Add exact opportunities/source-clock spans; never-established is censored, never zero delay."""
    by_sample = {f["sample_index"]: f for f in frames}
    result = []
    for run in events["runs"]:
        start = run["start"]
        end = run["first_match"]
        observations = [
            f
            for f in frames
            if f["sample_index"] >= start["sample_index"]
            and f["sample_index"] <= (end or run["last_assessable"])["sample_index"]
            and f["shot"] == start["shot"]
            and any(p["person_id"] == run["person_id"] and p["identity_eligible"] for p in f["people"])
        ]
        principal = any(
            p["person_id"] == run["person_id"] and p["role"] == "principal_rider"
            for p in by_sample[start["sample_index"]]["people"]
        )
        result.append(
            {
                "person_id": run["person_id"],
                "principal": principal,
                "start": start,
                "first_match": end,
                "never_established": end is None,
                "censored": end is None,
                "censor_reason": run["censor_reason"] if end is None else None,
                "observations_to_first": len(observations) if end is not None else None,
                "startup_delay_observations": len(observations) - 1 if end is not None else None,
                "observations_lower_bound": len(observations) if end is None else None,
                "source_frames_to_first": None
                if end is None
                else end["source_frame_index"] - start["source_frame_index"],
                "seconds_to_first": None if end is None else end["timestamp"] - start["timestamp"],
                "seconds_lower_bound": run["startup_observed_lower_bound_seconds"] if end is None else None,
            }
        )
    return result


def evaluate_panel(sequences, replays, destination):
    """Keep tracker histories and denominators panel-local, reusing every primary localization penalty."""
    require(len(sequences) == len(replays), "replay population differs")
    results, metrics = [], []
    panels = {s["panel"] for s in sequences}
    require(len(panels) == 1, "cannot pool original and validation histories")
    for sequence, replay in zip(sequences, replays, strict=True):
        require(sequence["id"] == replay["clip"], "clip history differs")
        result, raw = evaluate_sequence(sequence, replay, Path(destination) / sequence["id"])
        result["panel"] = sequence["panel"]
        result["startup"] = startup_spans([i["truth"] for i in sequence["items"]], result["events"])
        results.append(result)
        metrics.append(raw)
    groups = grouped_summary(results, metrics)
    uncertain = [i for i, result in enumerate(results) if result["source"] == "mykonos"]
    if uncertain:
        cluster_results = [
            {**results[i], "semantic_event_group": "mykonos-unresolved-source-cluster"} for i in uncertain
        ]
        clusters = grouped_summary(cluster_results, [metrics[i] for i in uncertain])
        groups.extend(
            {**g, "group_kind": "uncertain_event_cluster"}
            for g in clusters
            if g["group_kind"] == "physical_event_group"
        )
    return {
        "panel": next(iter(panels)),
        "results": results,
        "groups": groups,
        "aggregation_unit": "correlated clip histories; not independent physical jump counts",
        "exposure": "all panels exposed; no fresh holdout",
        "audit_complete": False,
    }


def compare_panels(baseline, candidate):
    """Paired coverage/event flags remain unaudited; added establishment is not automatically harm."""
    require(baseline["panel"] == candidate["panel"], "different panel histories")
    require([r["clip"] for r in baseline["results"]] == [r["clip"] for r in candidate["results"]], "different clips")
    return [
        {
            "clip": a["clip"],
            "source": a["source"],
            "semantic_event_group": a["semantic_event_group"],
            "coverage": paired_frames(a["frames"], b["frames"]),
            "events": paired_events(a["events"], b["events"]),
            "audit_complete": False,
            "established_harms": None,
        }
        for a, b in zip(baseline["results"], candidate["results"], strict=True)
    ]


def causal_probe(truth, detections, diagnostic, *, person_id, prior_uid=None, raw_saved_proposals=()):
    """Emit inspectable gate/path evidence, not a motion-cause conclusion from a tracker ranking.

    Same-person proposal eligibility uses the frozen primary IoU threshold. This cannot establish missing logits or fix
    loose localization. Confidence paths and actual gated assignment scores remain separate from raw geometry.
    """
    people = [
        p for p in truth["people"] if p["person_id"] == person_id and p["detection_eligible"] and p["identity_eligible"]
    ]
    base = {
        "person_id": person_id,
        "sample_index": truth["sample_index"],
        "phase": truth["phase"],
        "prior_uid": prior_uid,
        "source_audit_required": True,
        "confirmed_motion_cause": False,
    }
    if not people:
        return {**base, "category": "unknown_visibility_or_identity"}
    overlap = box_iou(np.asarray([people[0]["box"]]), detections.xyxy).reshape(-1)
    eligible = [int(detections.data["proposal_index"][i]) for i in np.flatnonzero(overlap >= 0.5)]
    base["eligible_saved_proposals"] = eligible
    base["other_saved_categories"] = [
        {"index": i, "kind": p["kind"]} for i, p in enumerate(raw_saved_proposals) if p["kind"] != "rider"
    ]
    base["localization_overlaps"] = overlap.tolist()
    if not eligible:
        return {
            **base,
            "category": "localization_miss" if np.any(overlap > 0) else "detection_stream_absence",
            "limitation": "no eligible same-person proposal in admitted saved stream; no claim about missing logits",
            "localization_only_possible": bool(np.any(overlap > 0)),
        }
    policy = diagnostic.get("confidence_policy", {})
    base["confidence_policy"] = policy
    base["eligible_confidences"] = [
        float(detections.confidence[i]) if detections.confidence is not None else 1.0
        for i, p in enumerate(detections.data["proposal_index"])
        if int(p) in eligible
    ]
    if diagnostic.get("reset"):
        return {**base, "category": "reset_boundary"}
    lifecycle = diagnostic["lifecycle"]
    removed = [r for r in lifecycle["removed"] if r["uid"] == prior_uid]
    if removed:
        return {**base, "category": removed[0]["reason"], "removal": removed[0]}
    stages = []
    for event in diagnostic["events"]:
        rows = [i for i, p in enumerate(event["pool"]) if p["uid"] == prior_uid]
        cols = [i for i, p in enumerate(event["proposal_indices"]) if p in eligible]
        for row in rows:
            for col in cols:
                chosen = [list(pair) for pair in event["result"][0]]
                stages.append(
                    {
                        "stage": event["stage"],
                        "geometry": event["geometry"],
                        "proposal_index": event["proposal_indices"][col],
                        "row": row,
                        "col": col,
                        "raw_iou": event["raw_iou"][row][col],
                        "assignment_score": event["assignment_scores"][row][col],
                        "gate_score": event["gate_scores"][row][col],
                        "threshold": event["threshold"],
                        "passes_actual_gate": event["gate_scores"][row][col] >= event["threshold"],
                        "matched": [row, col] in chosen,
                        "competing_pairs": [pair for pair in chosen if pair[0] == row or pair[1] == col],
                        "score_fused": event["score_fused"],
                        "cmc_transform": event.get("cmc_transform"),
                    }
                )
    base["stages"] = stages
    alive = any(r["uid"] == prior_uid for r in lifecycle["before"])
    if not alive:
        category = "not_established_or_reset"
    elif any(s["matched"] for s in stages):
        category = "matched_internal_or_public_check_localization_and_identity"
    elif any(s["passes_actual_gate"] and s["competing_pairs"] for s in stages):
        category = "assignment_competition"
    elif not stages:
        minimum = policy.get("weak_continuation_threshold")
        if minimum is None:
            minimum = policy.get("low_threshold_exclusive")
        if minimum is None:
            minimum = policy.get("high_threshold", 0.0)
        scores = base["eligible_confidences"]
        rejected = all(
            score <= minimum if policy.get("low_threshold_exclusive") is not None else score < minimum
            for score in scores
        )
        category = "tracker_confidence_rejection" if rejected else "confidence_stage_eligibility_rejection"
    elif all(s["geometry"] == "last_observation" for s in stages):
        category = "observation_matching_gate"
    elif any(s["geometry"] == "last_observation" for s in stages):
        category = "mixed_prediction_observation_or_shape_matching_gates"
    else:
        category = "possible_prediction_shape_fusion_or_CMC_gate"
    return jsonable({**base, "category": category})


def phase_cause_table(records):
    """Group source/event/phase evidence, preserving unknowns and unaudited categories."""
    counts = defaultdict(int)
    for row in records:
        counts[(row["panel"], row["source"], row["event_group"], row["phase"], row["category"])] += 1
    return [
        {"panel": p, "source": s, "event_group": e, "phase": phase, "cause": c, "count": n}
        for (p, s, e, phase, c), n in sorted(counts.items())
    ]


def phase_boundary_views(frames, *, shift=2):
    """Return fixed ±2-opportunity original-panel phase sensitivities, never replace primary unknowns or cross cuts."""
    require(shift == 2, "unfrozen phase sensitivity")
    result = {}
    for delta in (-shift, 0, shift):
        phases = []
        for index, frame in enumerate(frames):
            source = frames[min(max(index + delta, 0), len(frames) - 1)]
            phases.append(
                frame["phase"] if frame["phase"] == "unknown" or source["shot"] != frame["shot"] else source["phase"]
            )
        result[str(delta)] = phases
    return result
