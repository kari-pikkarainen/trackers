# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Exact source-frame withholding schedules and distinct coast/expiry/reacquisition observations."""

from __future__ import annotations

import copy
from fractions import Fraction

import numpy as np

from experiments.tracking_baselines import replay_sequence
from experiments.tracking_validation_eval import account_frame, require
from trackers.eval import box_iou

GAPS = (15, 30, 60)


def exact_window(frames, anchor_sample, gap):
    """B=A+G, with G-1 source frames strictly inside; never use a nearest cached endpoint."""
    require(gap in GAPS, "unfrozen source-frame gap")
    anchors = [f for f in frames if f["sample_index"] == anchor_sample]
    require(len(anchors) == 1, "anchor sample missing or duplicated")
    anchor = anchors[0]
    endpoints = [f for f in frames if f["source_frame_index"] == anchor["source_frame_index"] + gap]
    if len(endpoints) != 1:
        return {"status": "missing_exact_endpoint", "source_frame_difference": gap}
    end = endpoints[0]
    interior = [f for f in frames if anchor["source_frame_index"] < f["source_frame_index"] < end["source_frame_index"]]
    if any(f["shot"] != anchor["shot"] for f in [*interior, end]):
        return {"status": "crosses_cut", "source_frame_difference": gap}
    require(all(f["time_base"] == anchor["time_base"] for f in [*interior, end]), "time base changed within shot")
    elapsed = Fraction(end["pts"] - anchor["pts"]) * Fraction(anchor["time_base"])
    require(elapsed > 0, "source timestamps not increasing")
    return {
        "status": "recorded_candidate" if interior else "endpoint_only_diagnostic",
        "last_sample": anchor_sample,
        "return_sample": end["sample_index"],
        "last_source_frame": anchor["source_frame_index"],
        "return_source_frame": end["source_frame_index"],
        "source_frame_difference": gap,
        "strict_interior_source_frames": gap - 1,
        "actual_elapsed_fraction": str(elapsed),
        "actual_elapsed_seconds": float(elapsed),
        "last_pts": anchor["pts"],
        "return_pts": end["pts"],
        "time_base": anchor["time_base"],
        "interior_saved_samples": [f["sample_index"] for f in interior],
    }


def suppression_rows(truth, detections, person_id):
    """Suppress only unambiguous eligible same-person saved proposals; protect every other physical person."""
    people = [p for p in truth["people"] if p["detection_eligible"]]
    target = [i for i, p in enumerate(people) if p["person_id"] == person_id and p["identity_eligible"]]
    if len(target) != 1:
        return {"status": "visibility_or_identity_unassessable", "rows": []}
    overlaps = box_iou(np.asarray([p["box"] for p in people]), detections.xyxy)
    eligible = overlaps[target[0]] >= 0.5
    others = [i for i in range(len(people)) if i != target[0]]
    ambiguous = eligible & np.any(overlaps[others] >= 0.5, axis=0) if others else np.zeros(len(detections), bool)
    if ambiguous.any():
        return {"status": "ambiguous_saved_proposal", "rows": []}
    return {
        "status": "eligible" if eligible.any() else "no_eligible_suppressible_proposal",
        "rows": np.flatnonzero(eligible).tolist(),
    }


def run_recorded_window(sequence, variant, window, *, person_id):
    """Bounded prefix through one frozen return; failure is a result, never outcome-based reselection."""
    frames = [i["truth"] for i in sequence["items"]]
    exact = exact_window(frames, window["last_sample"], window["source_frame_difference"])
    base = {
        "variant": variant,
        "person_id": person_id,
        "window": exact,
        "population": sequence.get("population", "recorded_withholding"),
    }
    if exact["status"] != "recorded_candidate":
        return {**base, "status": exact["status"]}
    for key in ("return_sample", "last_pts", "return_pts", "interior_saved_samples"):
        require(exact[key] == window[key], "frozen stress selection changed: " + key)
    start, end = exact["last_sample"], exact["return_sample"]
    prefix = {**sequence, "items": [copy.deepcopy(i) for i in sequence["items"] if i["truth"]["sample_index"] <= end]}
    suppressions = []
    for item in prefix["items"]:
        sample = item["truth"]["sample_index"]
        if sample not in exact["interior_saved_samples"]:
            continue
        selection = suppression_rows(item["truth"], item["detections"], person_id)
        if selection["status"] in ("visibility_or_identity_unassessable", "ambiguous_saved_proposal"):
            return {**base, "status": selection["status"], "sample_index": sample}
        rows = selection["rows"]
        suppressions.append(
            {"sample_index": sample, "proposal_indices": item["detections"].data["proposal_index"][rows].tolist()}
        )
        keep = np.ones(len(item["detections"]), bool)
        keep[rows] = False
        item["detections"] = item["detections"][keep]
    if not any(r["proposal_indices"] for r in suppressions):
        return {**base, "status": "no_eligible_suppressible_proposal", "suppressions": suppressions}
    replay = replay_sequence(prefix, variant, diagnostic=True)
    by_sample = {i["truth"]["sample_index"]: i["truth"] for i in prefix["items"]}
    rows = {r["sample_index"]: r for r in replay["frames"]}
    accounts = {s: account_frame(by_sample[s], rows[s]["output"]) for s in rows}
    anchor = [r for r in accounts[start]["matches"] if r["person_id"] == person_id]
    if not anchor:
        return {**base, "status": "unestablished_at_anchor", "suppressions": suppressions, "replay": replay}
    prior_id = anchor[0]["id"]
    anchor_tracks = [r for r in rows[start]["lifecycle"]["after"] if r["id"] == prior_id]
    require(len(anchor_tracks) == 1, "anchor public/internal identity ambiguity")
    uid = anchor_tracks[0]["uid"]
    coast = []
    for sample in [*exact["interior_saved_samples"], end]:
        row = rows[sample]
        internal = [t for t in row["lifecycle"]["after"] if t["uid"] == uid]
        public = [r for r in accounts[sample]["matches"] if r["person_id"] == person_id]
        coast.append(
            {
                "sample_index": sample,
                "pts": by_sample[sample]["pts"],
                "internal_alive": bool(internal),
                "internal_coasting": bool(internal and internal[0]["time_since_update"] > 0),
                "matched_internal": bool(internal and internal[0]["time_since_update"] == 0),
                "internal_id": internal[0]["id"] if internal else None,
                "correct_public_id": public[0]["id"] if public else None,
                "removed": [r for r in row["lifecycle"]["removed"] if r["uid"] == uid],
            }
        )
    returned = coast[-1]["correct_public_id"]
    status = "no_reacquisition" if returned is None else "same_id" if returned == prior_id else "new_id"
    return {
        **base,
        "status": status,
        "prior_id": prior_id,
        "prior_uid": uid,
        "suppressions": suppressions,
        "coast": coast,
        "replay": replay,
    }


def synthetic_clock(*, source_rate, gap, anchor=5):
    """Explicit simulated source PTS; source-rate averages never masquerade as recorded VFR endpoints."""
    require(gap in GAPS, "unfrozen simulated gap")
    rate = Fraction(source_rate)
    require(rate > 0, "invalid simulated source clock")
    return {
        "population": "synthetic_source_clock",
        "simulated": True,
        "source_rate": str(rate),
        "anchor_source_frame": anchor,
        "return_source_frame": anchor + gap,
        "source_frame_difference": gap,
        "strict_interior_source_frames": gap - 1,
        "timestamps": [float(Fraction(i, 1) / rate) for i in range(anchor + gap + 1)],
        "actual_elapsed_fraction": str(Fraction(gap, 1) / rate),
        "limitation": "simulated, not recorded Day1 VFR or accuracy evidence",
    }


def synthetic_sequence(*, source_rate, gap, trajectory):
    """Build declared source-clock fixtures, not detections from a learned model or new tracking motion code."""
    import supervision as sv

    require(trajectory in ("stationary", "constant_velocity", "smooth_curved"), "unfrozen synthetic trajectory")
    clock = synthetic_clock(source_rate=source_rate, gap=gap)
    rate = Fraction(source_rate)
    items = []
    rng = np.random.default_rng(42)
    image = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    for index, timestamp in enumerate(clock["timestamps"]):
        x = 20 if trajectory == "stationary" else 20 + index * 0.5
        y = 20 + 0.005 * index**2 if trajectory == "smooth_curved" else 20
        box = [x, y, x + 20, y + 30]
        truth = {
            "sample_index": index,
            "source_frame_index": index,
            "pts": index * rate.denominator,
            "time_base": f"1/{rate.numerator}",
            "timestamp": timestamp,
            "shot": 0,
            "phase": "unknown",
            "height": 120,
            "width": 160,
            "uncertainties": [],
            "people": [
                {
                    "person_id": "synthetic/shot0/principal",
                    "box": box,
                    "box_uncertainty_pixels": 0,
                    "role": "principal_rider",
                    "identity_eligible": True,
                    "detection_eligible": True,
                    "evaluation_domain": "physical_person",
                }
            ],
        }
        data = sv.Detections(
            xyxy=np.asarray([box], float),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
            data={"proposal_index": np.array([0])},
        )
        items.append({"truth": truth, "detections": data, "image": image})
    return {
        "id": f"synthetic-{source_rate}-{gap}-{trajectory}",
        "source": "simulated",
        "panel": "synthetic",
        "population": "synthetic_source_clock",
        "synthetic_reference_rate": float(rate),
        "clock": clock,
        "items": items,
    }
