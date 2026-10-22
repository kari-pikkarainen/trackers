# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Falsifying diagnostic, saved-input and exact source-clock fixtures."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from experiments.tracking_baseline_analysis import causal_probe, phase_cause_table
from experiments.tracking_baseline_data import original_receipt, proposals_to_detections
from experiments.tracking_baseline_stress import exact_window, run_recorded_window, suppression_rows, synthetic_clock
from experiments.tracking_baselines import VARIANTS, make_baseline, replay_sequence
from experiments.tracking_validation_eval import account_frame
from tests.test_tracking_baselines import detections, sequence, truth


def probe(diagnostic, data=None, *, uid=0):
    """Query the fixed primary person with explicit native track identity."""
    return causal_probe(
        truth(4), detections() if data is None else data, diagnostic, person_id="fixture/shot0/rider", prior_uid=uid
    )


def test_absence_localization_and_confidence_rejection_are_distinct():
    tracker = make_baseline("native06", diagnostic=True)
    for i in range(3):
        tracker.begin_observation()
        tracker.update(detections(), timestamp=i * 0.2)
        tracker.end_observation()
    tracker.begin_observation()
    tracker.update(detections(scores=(0.4,)), timestamp=0.6)
    diagnostic = tracker.end_observation()
    assert probe(diagnostic, detections(scores=(0.4,)))["category"] == "tracker_confidence_rejection"
    absent = probe(diagnostic, detections((), ()))
    assert absent["category"] == "detection_stream_absence"
    assert not absent["localization_only_possible"]
    loose = probe(diagnostic, detections(([10, 10, 80, 90],)))
    assert loose["localization_only_possible"]
    assert loose["localization_overlaps"][0] < 0.5


@pytest.mark.parametrize("name", VARIANTS)
def test_expiry_exact_seconds_boundary_and_not_prediction_failure(name):
    tracker = make_baseline(name, diagnostic=True)
    for i in range(4):
        tracker.begin_observation()
        tracker.update(detections(), timestamp=i * 0.2)
        tracker.end_observation()
    established = [t for t in tracker.tracks if t.tracker_id >= 0]
    assert established
    uid = tracker._uid(established[0])
    kwargs = {"frame": np.zeros((120, 160, 3), np.uint8)} if name == "botsort-native" else {}
    tracker.begin_observation()
    tracker.update(detections((), ()), timestamp=1.6, **kwargs)
    at_boundary = tracker.end_observation()
    assert any(t["uid"] == uid for t in at_boundary["lifecycle"]["after"])
    tracker.begin_observation()
    tracker.update(detections(), timestamp=1.60000001, **kwargs)
    beyond = tracker.end_observation()
    assert probe(beyond, uid=uid)["category"] == "budget_expiry"
    assert beyond["lifecycle"]["removed"][0]["established_before"]


@pytest.mark.parametrize("name", ("botsort-cmc-off", "cbiou-native"))
def test_immature_removal_is_startup_not_expiry(name):
    tracker = make_baseline(name, diagnostic=True)
    tracker.begin_observation()
    tracker.update(detections((), ()), timestamp=0)
    tracker.end_observation()
    tracker.begin_observation()
    tracker.update(detections(), timestamp=0.2)
    first = tracker.end_observation()
    assert first["lifecycle"]["after"][0]["id"] == -1
    tracker.begin_observation()
    tracker.update(detections((), ()), timestamp=0.4)
    removed = tracker.end_observation()["lifecycle"]["removed"][0]
    assert removed["reason"] == "immature_or_native_removal"
    assert not removed["established_before"]


def test_real_ocr_rescue_and_copied_prediction_state():
    tracker = make_baseline("native06", diagnostic=True)
    for i, x in enumerate((20, 25, 30, 35)):
        tracker.begin_observation()
        tracker.update(detections(([x, 20, x + 20, 60],)), timestamp=i * 0.2)
        tracker.end_observation()
    tracker.begin_observation()
    tracker.update(detections(([35, 20, 55, 60],)), timestamp=1.4)
    diagnostic = tracker.end_observation()
    assert [e["stage"] for e in diagnostic["events"]] == ["ocm", "ocr"]
    assert diagnostic["events"][0]["result"][0] == []
    assert diagnostic["events"][1]["result"][0] == [[0, 0]]
    assert diagnostic["events"][1]["geometry"] == "last_observation"
    saved = copy.deepcopy(diagnostic)
    tracker.update(detections(), timestamp=1.5)
    assert diagnostic == saved


def event(*, geometry="prediction", score=0.1, raw=0.1, pairs=(), stage="first", fused=False):
    """Explicit causal fixture: actual assignment/gate scores, raw geometry, pools and source columns."""
    return {
        "stage": stage,
        "geometry": geometry,
        "pool": [{"uid": 0}],
        "proposal_indices": [17],
        "raw_iou": [[raw]],
        "assignment_scores": [[score]],
        "gate_scores": [[score]],
        "threshold": 0.3,
        "result": [list(pairs), [], []],
        "score_fused": fused,
    }


def diagnostic(events):
    return {"events": events, "lifecycle": {"before": [{"uid": 0}], "after": [{"uid": 0}], "removed": []}}


def test_actual_gate_competition_observation_and_mixed_unknown():
    assert probe(diagnostic([event()]))["category"] == "possible_prediction_shape_fusion_or_CMC_gate"
    assert probe(diagnostic([event(geometry="last_observation")]))["category"] == "observation_matching_gate"
    mixed = probe(diagnostic([event(), event(geometry="last_observation")]))
    assert mixed["category"] == "mixed_prediction_observation_or_shape_matching_gates"
    assert not mixed["confirmed_motion_cause"]
    competing = event(score=0.8, pairs=[[0, 1]])
    assert probe(diagnostic([competing]))["category"] == "assignment_competition"
    fused = probe(diagnostic([event(raw=0.8, score=0.2, fused=True)]))
    assert fused["stages"][0]["raw_iou"] == 0.8
    assert not fused["stages"][0]["passes_actual_gate"]
    assert fused["stages"][0]["score_fused"]
    assert probe(diagnostic([event(score=0.8, pairs=[[0, 0]])]))["category"].startswith("matched_internal")
    assert probe(diagnostic([]), uid=9)["category"] == "not_established_or_reset"


def test_saved_proposals_preserve_raw_indices_and_two_contracts():
    raw = [
        {"kind": "kite", "box": [1, 1, 3, 3], "confidence": 0.9},
        {"kind": "rider", "box": [20, 20, 40, 60], "confidence": 0.4},
        {"kind": "unclassified", "box": [20, 20, 40, 60], "confidence": 0.99},
        {"kind": "rider", "box": [60, 20, 80, 60], "confidence": 0.7},
    ]
    data = proposals_to_detections(raw)
    assert data.data["proposal_index"].tolist() == [1, 3]
    assert data.confidence.tolist() == [0.4, 0.7]
    frame = {"source": "mykonos", "source_frame_index": 10, "pts": 20, "time_base": "1/50", "image_path": "saved"}
    settings = {"prompt": "a kite. a person.", "box_threshold": 0.1, "text_threshold": 0.1}
    receipt = {
        **settings,
        "source_frame_index": 10,
        "source_pts": 20,
        "source_time_base": "1/50",
        "image_path": "saved",
        "proposals": raw,
    }
    assert original_receipt(frame, receipt) == (raw, "recognized_only_historical_rows")
    day_frame = {**frame, "source": "day1", "image_sha256": "image", "source_sha256": "source"}
    day = {
        "frame": {**receipt, "image_sha256": "image", "source_sha256": "source"},
        "features": {"settings": settings, "raw_proposals": raw},
    }
    assert original_receipt(day_frame, day)[1] == "full_saved_postprocessed_rows"
    receipt["source_pts"] = 21
    with pytest.raises(ValueError, match="source-clock"):
        original_receipt(frame, receipt)
    with pytest.raises(ValueError, match="confidence"):
        proposals_to_detections([{**raw[1], "confidence": float("nan")}])


def test_primary_localization_unknown_duplicate_and_wrong_attachment_are_preserved():
    frame = truth(0)
    public = {"box": [20, 20, 40, 60], "id": 0, "confidence": 0.9, "proposal_index": 17}
    account = account_frame(frame, [public, {**public, "id": 1, "proposal_index": 18}])
    assert [r["fate"] for r in account["outputs"]] == ["matched", "duplicate"]
    loose = account_frame(frame, [{**public, "box": [10, 10, 80, 90]}])
    assert loose["matches"] == []
    unknown = {**frame, "people": [], "uncertainties": [{"region_id": "splash", "box": [20, 20, 40, 60]}]}
    assert account_frame(unknown, [public])["outputs"][0]["fate"] == "localized_unknown"
    records = [
        {
            "panel": "original",
            "source": "day1",
            "event_group": "replay-cluster",
            "phase": "unknown",
            "category": "unknown",
        }
    ] * 2
    assert phase_cause_table(records)[0]["count"] == 2


@pytest.mark.parametrize("gap", (15, 30, 60))
def test_exact_source_frame_windows_off_by_one_missing_cuts_and_endpoint_only(gap):
    frames = [truth(i, source_frame=i * 5) for i in range(gap // 5 + 1)]
    window = exact_window(frames, 0, gap)
    assert window["return_source_frame"] == gap
    assert window["strict_interior_source_frames"] == gap - 1
    assert window["actual_elapsed_seconds"] == gap / 25
    assert exact_window(frames[:-1], 0, gap)["status"] == "missing_exact_endpoint"
    assert exact_window([frames[0], frames[-1]], 0, gap)["status"] == "endpoint_only_diagnostic"
    frames[1]["shot"] = 2
    assert exact_window(frames, 0, gap)["status"] == "crosses_cut"
    simulated = synthetic_clock(source_rate="19001/317", gap=gap)
    assert simulated["simulated"] and simulated["population"] == "synthetic_source_clock"
    assert simulated["return_source_frame"] - simulated["anchor_source_frame"] == gap


def test_actual_vfr_boundary_not_rounded_to_nominal_budget():
    frames = [truth(i, source_frame=i * 12) for i in range(6)]
    for i, frame in enumerate(frames):
        frame["time_base"], frame["pts"] = "1/1000", [0, 200, 400, 601, 801, 1002][i]
    window = exact_window(frames, 0, 60)
    assert window["actual_elapsed_seconds"] == 1.002
    assert window["actual_elapsed_fraction"] == "501/500"


def test_stress_unestablished_no_suppressible_and_reacquisition_coasting():
    fixture = sequence(10)
    window = exact_window([i["truth"] for i in fixture["items"]], 3, 15)
    outcome = run_recorded_window(fixture, "botsort-cmc-off", window, person_id="fixture/shot0/rider")
    assert outcome["status"] == "same_id"
    assert all(r["internal_coasting"] and r["correct_public_id"] is None for r in outcome["coast"][:-1])
    assert outcome["coast"][-1]["correct_public_id"] == outcome["prior_id"]
    oc = run_recorded_window(fixture, "native06", window, person_id="fixture/shot0/rider")
    assert oc["status"] == "no_reacquisition"
    assert oc["coast"][-1]["matched_internal"] and oc["coast"][-1]["correct_public_id"] is None
    missing = copy.deepcopy(fixture)
    for item in missing["items"]:
        if 3 < item["truth"]["sample_index"] < 6:
            item["detections"] = detections((), ())
    assert (
        run_recorded_window(missing, "native06", window, person_id="fixture/shot0/rider")["status"]
        == "no_eligible_suppressible_proposal"
    )
    weak = copy.deepcopy(fixture)
    for item in weak["items"]:
        item["detections"].confidence[:] = 0.4
    assert (
        run_recorded_window(weak, "native06", window, person_id="fixture/shot0/rider")["status"]
        == "unestablished_at_anchor"
    )
    uncertain = truth(1)
    uncertain["people"] = []
    assert (
        suppression_rows(uncertain, detections(), "fixture/shot0/rider")["status"]
        == "visibility_or_identity_unassessable"
    )
    shared = truth(1)
    shared["people"].append({**shared["people"][0], "person_id": "other"})
    assert suppression_rows(shared, detections(), "fixture/shot0/rider")["status"] == "ambiguous_saved_proposal"


def test_synthetic_fixture_population_and_reference_clock_are_explicit():
    from experiments.tracking_baseline_stress import synthetic_sequence

    fixture = synthetic_sequence(source_rate="25", gap=30, trajectory="smooth_curved")
    assert fixture["population"] == "synthetic_source_clock"
    assert fixture["synthetic_reference_rate"] == 25
    assert len(fixture["items"]) == 36
    frames = [i["truth"] for i in fixture["items"]]
    window = exact_window(frames, 5, 30)
    outcome = run_recorded_window(fixture, "botsort-cmc-off", window, person_id="synthetic/shot0/principal")
    assert outcome["population"] == "synthetic_source_clock"
    assert outcome["window"]["actual_elapsed_seconds"] == 1.2
    assert any(r["removed"] for r in outcome["coast"])
    assert outcome["status"] in ("new_id", "no_reacquisition")


def test_native_stage_pools_raw_fusion_and_competition_are_captured():
    tracker = make_baseline("botsort-cmc-off", diagnostic=True)
    # First update is empty, so newly born tracks are immature on the next nonempty update.
    tracker.begin_observation()
    tracker.update(detections((), ()), timestamp=0)
    tracker.end_observation()
    for timestamp, data in [(0.2, detections()), (0.4, detections()), (0.6, detections(scores=(0.4,)))]:
        tracker.begin_observation()
        tracker.update(data, timestamp=timestamp)
        record = tracker.end_observation()
        if timestamp == 0.4:
            assert [e["stage"] for e in record["events"]] == ["first", "second", "unconfirmed"]
            unconfirmed = record["events"][-1]
            assert unconfirmed["proposal_indices"] == [17]
            assert unconfirmed["raw_iou"] == [[1.0]]
            assert unconfirmed["gate_scores"] == [[0.9]]
        if timestamp == 0.6:
            second = record["events"][1]
            assert second["proposal_indices"] == [17]
            assert second["raw_iou"] == second["gate_scores"]
            assert not second["score_fused"]
    oc = make_baseline("native06", diagnostic=True)
    for sample in range(3):
        oc.begin_observation()
        oc.update(detections(([20, 20, 40, 60], [25, 20, 45, 60]), (0.9, 0.9)), timestamp=sample * 0.2)
        oc.end_observation()
    oc.begin_observation()
    oc.update(detections(), timestamp=0.6)
    record = oc.end_observation()
    stage = record["events"][0]
    assert len(stage["pool"]) == 2 and len(stage["proposal_indices"]) == 1
    loser = next(row for row in range(2) if row not in [p[0] for p in stage["result"][0]])
    assert (
        causal_probe(
            truth(3), detections(), record, person_id="fixture/shot0/rider", prior_uid=stage["pool"][loser]["uid"]
        )["category"]
        == "assignment_competition"
    )


def test_original_adapter_requires_accepted_manifest_and_exact_receipt(tmp_path):
    from experiments.tracking_baseline_data import load_original_panel
    from experiments.tracking_validation_data import sha256
    from tests.test_tracking_validation_data import synthetic_corpus, write

    _, _, frame, _ = synthetic_corpus(tmp_path)
    receipt = {
        "source_frame_index": frame["source_frame_index"],
        "source_pts": frame["pts"],
        "source_time_base": frame["time_base"],
        "image_path": frame["image_path"],
        "prompt": "a kite. a person.",
        "box_threshold": 0.1,
        "text_threshold": 0.1,
        "proposals": [{"kind": "rider", "box": [10, 10, 30, 50], "confidence": 0.4}],
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_sha = write(receipt_path, receipt)
    frame.update(source="mykonos", receipt_path=str(receipt_path), receipt_sha256=receipt_sha)
    labels = tmp_path / "admission" / "labels"
    labels.mkdir(parents=True)
    paths = ["original-panel.json", "admission/labels/m1.json", "stress-selection.json"]
    write(tmp_path / paths[0], {"clips": [{"id": "m1", "source": "mykonos", "frames": [frame]}]})
    write(tmp_path / paths[1], {"clip": "m1", "frames": [frame]})
    write(tmp_path / paths[2], {"selected": [{"panel": "original", "clip": "m1", "event_group": "uncertain"}]})
    manifest_sha = write(
        tmp_path / "delivery-manifest.json", {"files": [{"path": p, "sha256": sha256(tmp_path / p)} for p in paths]}
    )
    kwargs = {
        "manifest_sha256": manifest_sha,
        "schema_path": tmp_path / "schema.json",
        "schema_sha256": sha256(tmp_path / "schema.json"),
    }
    sequences, consumed = load_original_panel(tmp_path, **kwargs)
    assert sequences[0]["panel"] == "original"
    assert sequences[0]["semantic_event_group"] == "uncertain"
    assert sequences[0]["items"][0]["raw_visibility"] == "recognized_only_historical_rows"
    assert len(consumed) == 7
    receipt_path.write_text("{}")
    with pytest.raises(ValueError, match="input changed"):
        load_original_panel(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="unaccepted"):
        load_original_panel(tmp_path, **{**kwargs, "manifest_sha256": "wrong"})


def test_fixed_phase_sensitivity_preserves_unknown_and_cut():
    from experiments.tracking_baseline_analysis import phase_boundary_views

    frames = [truth(i) for i in range(5)]
    for frame, phase in zip(frames, ["approach", "takeoff", "unknown", "descent", "landing"], strict=True):
        frame["phase"] = phase
    frames[4]["shot"] = 1
    views = phase_boundary_views(frames)
    assert views["0"] == [f["phase"] for f in frames]
    assert all(v[2] == "unknown" for v in views.values())
    assert views["2"][3] == "descent"
    assert views["-2"][4] == "landing"


def test_temporal_wrong_attachment_and_unknown_censor_are_not_hidden():
    from experiments.tracking_validation_events import person_events

    frames = [truth(i) for i in range(4)]
    for frame in frames:
        frame["people"].append(
            {**frame["people"][0], "person_id": "fixture/shot0/other", "role": "other_person", "box": [80, 20, 100, 60]}
        )
    public = {"box": [20, 20, 40, 60], "id": 0, "confidence": 0.9, "proposal_index": 17}
    outputs = [[public], [{**public, "box": [80, 20, 100, 60]}], [], []]
    frames[2]["people"] = []
    accounts = [account_frame(f, o) for f, o in zip(frames, outputs, strict=True)]
    events = person_events(frames, accounts)
    assert events["wrong_attachment_candidates"][0]["previous_person"] == "fixture/shot0/rider"
    assert events["wrong_attachment_candidates"][0]["new_person"] == "fixture/shot0/other"
    rider_loss = next(e for e in events["losses"] if e["person_id"] == "fixture/shot0/rider")
    assert rider_loss["right_censored"]
    assert rider_loss["censor_reason"] == "visibility_or_identity_unassessable"
    assert rider_loss["observed_end"]["sample_index"] == 1


def test_selected_stress_omission_and_tamper_fail_before_execution(tmp_path):
    from experiments.tracking_baseline_runner import run_selected_stress
    from experiments.tracking_validation_data import sha256
    from tests.test_tracking_validation_data import write

    selection = tmp_path / "selection.json"
    write(selection, {"selected_count": 1, "selected": [{"panel": "validation", "clip": "missing"}]})
    expected = sha256(selection)
    with pytest.raises(ValueError, match="do not silently omit"):
        run_selected_stress([], selection, tmp_path / "output", selection_sha256=expected)
    assert not (tmp_path / "output").exists()
    with pytest.raises(ValueError, match="selection changed"):
        run_selected_stress([], selection, tmp_path / "output", selection_sha256="wrong")


def identity_transition_fixture():
    """Two native established tracks; physical people exchange positions while localization stays correct."""
    fixture = sequence(8)
    for index, item in enumerate(fixture["items"]):
        swapped = index in (4, 5)
        principal_box = [80, 20, 100, 60] if swapped else [20, 20, 40, 60]
        other_box = [20, 20, 40, 60] if swapped else [80, 20, 100, 60]
        principal = item["truth"]["people"][0]
        principal["box"] = principal_box
        item["truth"]["people"].append(
            {**principal, "person_id": "fixture/shot0/other", "role": "other_person", "box": other_box}
        )
        item["truth"]["phase"] = "descent" if index < 6 else "landing"
        item["detections"] = detections(([20, 20, 40, 60], [80, 20, 100, 60]), (0.9, 0.9))
    return fixture, replay_sequence(fixture, "botsort-cmc-off", diagnostic=True)


def test_established_identity_transition_keeps_prior_current_native_evidence(tmp_path):
    from experiments.tracking_baseline_analysis import evaluate_panel
    from experiments.tracking_baseline_runner import causal_sequence

    fixture, replay = identity_transition_fixture()
    unchanged = copy.deepcopy(replay)
    accounts = [account_frame(i["truth"], r["output"]) for i, r in zip(fixture["items"], replay["frames"], strict=True)]
    assert all(len(a["matches"]) == 2 for a in accounts)
    evaluated = evaluate_panel([fixture], [replay], tmp_path)
    assert evaluated["results"][0]["summary"]["id_switches"] == 4
    transitions = [
        r
        for r in causal_sequence(fixture, replay)
        if r.get("record_kind") == "established_identity_transition" and r["person_id"] == "fixture/shot0/rider"
    ]
    assert [(r["sample_index"], r["previous_public_id"], r["current_public_id"]) for r in transitions] == [
        (4, 0, 1),
        (6, 1, 0),
    ]
    first, second = transitions
    assert first["previous_uid"] == 0 and first["current_uid"] == 1
    assert second["previous_uid"] == 1 and second["current_uid"] == 0
    assert first["previous_public_match"]["sample_index"] == 3
    assert second["previous_public_match"]["sample_index"] == 5
    assert first["transition_type"] == "continuous_public_identity_change"
    assert first["previous_public_state"]["uid"] == 0
    assert first["previous_track_stage_evidence"][0]["proposal_indices"] == [17, 18]
    assert first["current_track_stage_evidence"][0]["pool"][1]["uid"] == 1
    assert first["stages"] and first["prior_track_cause"]
    assert first["source_audit_required"] and not first["confirmed_motion_cause"]
    assert first["established_harm"] is None
    table = phase_cause_table(transitions)
    assert {(r["phase"], r["count"]) for r in table} == {("descent", 1), ("landing", 1)}
    assert replay == unchanged
    assert causal_sequence(fixture, replay) == causal_sequence(fixture, replay)
    # The fixed evaluator is unchanged by diagnostic candidate construction.
    assert (
        evaluate_panel([fixture], [replay], tmp_path / "again")["results"][0]["summary"]
        == evaluated["results"][0]["summary"]
    )
    from experiments.tracking_validation import write_json

    write_json(
        tmp_path / "identity-transition-evidence.json",
        {
            "truth": [i["truth"] for i in fixture["items"]],
            "native_replay": replay,
            "transition_candidates": causal_sequence(fixture, replay),
            "phase_table": table,
            "unchanged_primary_summary": evaluated["results"][0]["summary"],
        },
    )


@pytest.mark.parametrize("boundary", ("cut", "identity_ineligible", "detection_ineligible", "absent", "frame_unknown"))
def test_identity_transition_history_stops_at_observed_boundaries(boundary):
    from experiments.tracking_baseline_runner import causal_sequence

    fixture, _ = identity_transition_fixture()
    if boundary == "cut":
        for item in fixture["items"][4:]:
            item["truth"]["shot"] = 1
            for person in item["truth"]["people"]:
                person["person_id"] = person["person_id"].replace("shot0", "shot1")
    elif boundary == "frame_unknown":
        fixture["items"][3]["truth"]["identity_visibility_unknown"] = True
    elif boundary == "absent":
        fixture["items"][3]["truth"]["people"] = [fixture["items"][3]["truth"]["people"][1]]
    else:
        key = "identity_eligible" if boundary == "identity_ineligible" else "detection_eligible"
        fixture["items"][3]["truth"]["people"][0][key] = False
    replay = replay_sequence(fixture, "botsort-cmc-off", diagnostic=True)
    records = causal_sequence(fixture, replay)
    principal = [
        r
        for r in records
        if r.get("record_kind") == "established_identity_transition" and r["person_id"].endswith("/rider")
    ]
    assert [r["sample_index"] for r in principal] == [6]
    assert principal[0]["previous_public_match"]["sample_index"] == 5
    assert principal[0]["previous_public_id"] == 1 and principal[0]["current_public_id"] == 0
    assert all(r["sample_index"] != 0 for r in records if r.get("record_kind") == "established_identity_transition")


def test_public_id_suppression_never_becomes_an_identity_transition():
    from experiments.tracking_baseline_runner import causal_sequence

    fixture = sequence(10)
    for item in fixture["items"][4:6]:
        item["detections"] = detections((), ())
    replay = replay_sequence(fixture, "native06", diagnostic=True)
    assert replay["frames"][6]["output"][0]["id"] == -1
    assert replay["frames"][8]["output"][0]["id"] == 0
    records = causal_sequence(fixture, replay)
    assert not any(r.get("record_kind") == "established_identity_transition" for r in records)
    suppressed = next(r for r in records if r["sample_index"] == 6)
    assert suppressed["prior_uid"] == 0 and suppressed["public_established_before"]
    assert suppressed["category"] == "matched_internal_or_public_check_localization_and_identity"


def test_reacquisition_transition_retains_intervening_actual_expiry_evidence():
    from experiments.tracking_baseline_runner import causal_sequence

    fixture = sequence(12)
    for item in fixture["items"][4:10]:
        item["detections"] = detections((), ())
    replay = replay_sequence(fixture, "botsort-cmc-off", diagnostic=True)
    records = causal_sequence(fixture, replay)
    transitions = [r for r in records if r.get("record_kind") == "established_identity_transition"]
    assert len(transitions) == 1
    transition = transitions[0]
    assert transition["sample_index"] == 11
    assert transition["previous_public_id"] == 0 and transition["current_public_id"] == 1
    assert transition["previous_uid"] == 0 and transition["current_uid"] == 1
    assert transition["transition_type"] == "reacquisition_changed_public_identity"
    assert transition["previous_public_match"]["sample_index"] == 3
    assert [p["sample_index"] for p in transition["intervening_unmatched_opportunities"]] == list(range(4, 11))
    removals = [t for e in transition["intervening_track_evidence"] for t in e["removed"]]
    assert len(removals) == 1 and removals[0]["reason"] == "budget_expiry"
    assert transition["current_track_stage_evidence"][-1]["stage"] == "unconfirmed"
    saved = copy.deepcopy(transition)
    replay["frames"][11]["events"][0]["pool"].clear()
    assert transition == saved
