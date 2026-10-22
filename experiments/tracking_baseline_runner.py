# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Reusable bounded accuracy and stress orchestration; execution is an explicitly separate delivery phase."""

from __future__ import annotations

import copy
from pathlib import Path

from experiments.tracking_baseline_analysis import causal_probe, evaluate_panel, phase_boundary_views, phase_cause_table
from experiments.tracking_baseline_stress import run_recorded_window
from experiments.tracking_baselines import VARIANTS, preload, replay_sequence
from experiments.tracking_validation import write_json
from experiments.tracking_validation_data import read_json, sha256
from experiments.tracking_validation_eval import account_frame, require


def _source_point(frame):
    """Copy an observed source opportunity without deriving a new phase or synthetic clock."""
    return {
        key: frame[key]
        for key in ("sample_index", "source_frame_index", "pts", "time_base", "timestamp", "shot", "phase")
    }


def _identity_transition(item, row, person, prior, current):
    """Preserve the previous established identity and actual evidence before advancing person history."""
    record = causal_probe(
        item["truth"],
        item["detections"],
        row,
        person_id=person,
        prior_uid=prior["uid"],
        raw_saved_proposals=item.get("raw_saved_proposals", ()),
    )
    lifecycle = row["lifecycle"]
    previous_stages = [e for e in row["events"] if any(t["uid"] == prior["uid"] for t in e["pool"])]
    current_stages = [e for e in row["events"] if any(t["uid"] == current["uid"] for t in e["pool"])]
    return copy.deepcopy(
        {
            **record,
            "record_kind": "established_identity_transition",
            "category": "established_identity_change",
            "prior_track_cause": record["category"],
            "transition_type": "reacquisition_changed_public_identity"
            if prior["missing_points"]
            else "continuous_public_identity_change",
            "previous_uid": prior["uid"],
            "current_uid": current["uid"],
            "previous_public_id": prior["id"],
            "current_public_id": current["id"],
            "previous_public_match": prior["point"],
            "current_public_match": _source_point(item["truth"]),
            "previous_public_state": prior["state"],
            "current_public_state": current,
            "intervening_unmatched_opportunities": prior["missing_points"],
            "intervening_track_evidence": prior["missing_evidence"],
            "previous_track_before": [t for t in lifecycle["before"] if t["uid"] == prior["uid"]],
            "previous_track_after": [t for t in lifecycle["after"] if t["uid"] == prior["uid"]],
            "previous_track_removals": [t for t in lifecycle["removed"] if t["uid"] == prior["uid"]],
            "previous_track_stage_evidence": previous_stages,
            "current_track_stage_evidence": current_stages,
            "current_track_births": [t for t in lifecycle.get("births", []) if t["uid"] == current["uid"]],
            "established_harm": None,
        }
    )


def causal_sequence(sequence, replay):
    """Emit misses and established identity transitions before advancing visible shot-local public history."""
    require(len(sequence["items"]) == len(replay["frames"]), "causal replay population differs")
    previous = {}
    records = []
    for item, row in zip(sequence["items"], replay["frames"], strict=True):
        frame = item["truth"]
        require(frame["sample_index"] == row["sample_index"], "causal sample mapping differs")
        people = {p["person_id"]: p for p in frame["people"] if p["identity_eligible"] and p["detection_eligible"]}
        identity_unknown = frame.get("identity_visibility_unknown", False)
        previous = {} if row["reset"] or identity_unknown else {k: v for k, v in previous.items() if k in people}
        account = account_frame(frame, row["output"])
        matched = {m["person_id"]: m for m in account["matches"]}
        for person in people:
            prior = previous.get(person)
            context = {
                "panel": sequence["panel"],
                "source": sequence["source"],
                "clip": sequence["id"],
                "event_group": sequence["semantic_event_group"],
                "public_established_before": prior is not None,
                "raw_visibility": item.get("raw_visibility", "accepted validation saved stream"),
            }
            if person not in matched:
                record = causal_probe(
                    frame,
                    item["detections"],
                    row,
                    person_id=person,
                    prior_uid=None if prior is None else prior["uid"],
                    raw_saved_proposals=item.get("raw_saved_proposals", ()),
                )
                records.append({**record, **context})
                if prior is not None:
                    prior["missing_points"].append(_source_point(frame))
                    prior["missing_evidence"].append(
                        copy.deepcopy(
                            {
                                "point": _source_point(frame),
                                "probe": record,
                                "before": [t for t in row["lifecycle"]["before"] if t["uid"] == prior["uid"]],
                                "after": [t for t in row["lifecycle"]["after"] if t["uid"] == prior["uid"]],
                                "removed": [t for t in row["lifecycle"]["removed"] if t["uid"] == prior["uid"]],
                            }
                        )
                    )
            else:
                internal = [t for t in row["lifecycle"]["after"] if t["id"] == matched[person]["id"]]
                require(len(internal) == 1, "public/internal identity mapping unavailable")
                current = internal[0]
                if prior is not None and prior["id"] != matched[person]["id"]:
                    records.append({**_identity_transition(item, row, person, prior, current), **context})
                # Unknown identity boundaries never seed history for a later apparent transition.
                if not identity_unknown:
                    previous[person] = {
                        "uid": current["uid"],
                        "id": matched[person]["id"],
                        "point": _source_point(frame),
                        "state": copy.deepcopy(current),
                        "missing_points": [],
                        "missing_evidence": [],
                    }
    return records


def run_accuracy_panel(sequences, destination, *, names=VARIANTS, read_image=None):
    """Preload each clip once per variant, retain diagnostics and all primary/sensitivity evaluator artifacts.

    This function does not infer detections or adjudicate causal candidates. Call separately per admitted panel and
    partition. It never joins repeated image aliases into a favorable history or marks audit_complete.
    """
    require(bool(sequences), "empty accuracy panel")
    require(all(n in VARIANTS for n in names) and len(set(names)) == len(names), "unfrozen variants")
    require(len({s["panel"] for s in sequences}) == 1, "separate panel histories required")
    destination = Path(destination)
    reports = {}
    for name in names:
        replays, causes = [], []
        for sequence in sequences:
            loaded = preload(sequence, read_image=read_image)
            replay = replay_sequence(loaded, name, diagnostic=True)
            causes.extend(causal_sequence(sequence, replay))
            write_json(destination / name / sequence["id"] / "predictions.json", replay)
            if sequence["panel"] == "original":
                write_json(
                    destination / name / sequence["id"] / "phase-sensitivities.json",
                    phase_boundary_views([i["truth"] for i in sequence["items"]]),
                )
            replays.append(replay)
        report = evaluate_panel(sequences, replays, destination / name)
        report["causal_candidates"] = causes
        report["phase_cause_table"] = phase_cause_table(causes)
        write_json(destination / name / "panel.json", report)
        reports[name] = report
    return reports


def run_selected_stress(sequences, selection_path, destination, *, selection_sha256, names=VARIANTS, read_image=None):
    """Consume the exact pre-output B1 subset; report unavailable variants without replacement windows."""
    require(sha256(selection_path) == selection_sha256, "frozen stress selection changed")
    require(all(n in VARIANTS for n in names) and len(set(names)) == len(names), "unfrozen stress variants")
    selection = read_json(selection_path)
    by_clip = {(s["panel"], s["id"]): s for s in sequences}
    require(len(by_clip) == len(sequences), "duplicate panel clip histories")
    selected_keys = {(w["panel"], w["clip"]) for w in selection["selected"]}
    require(selected_keys.issubset(by_clip), "selected stress clip unavailable; do not silently omit")
    outcomes = []
    # Grouping by sequence bounds image memory; selection membership/order remains retained by its frozen ID.
    for key, sequence in by_clip.items():
        windows = [w for w in selection["selected"] if (w["panel"], w["clip"]) == key]
        if not windows:
            continue
        loaded = preload(sequence, read_image=read_image)
        for window in windows:
            for name in names:
                for person in window["physical_ids"]:
                    result = run_recorded_window(loaded, name, window, person_id=person)
                    path = Path(destination) / window["id"] / name / (person.replace("/", "_") + ".json")
                    write_json(path, result)
                    outcomes.append(
                        {
                            "window_id": window["id"],
                            "panel": key[0],
                            "clip": key[1],
                            "variant": name,
                            "person_id": person,
                            "status": result["status"],
                            "path": str(path),
                        }
                    )
    write_json(
        Path(destination) / "index.json",
        {"selection_sha256": selection_sha256, "selected_count": selection["selected_count"], "outcomes": outcomes},
    )
    return outcomes
