# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Hand-calculated custom-view, geometry, identity and censoring regressions."""

import copy

import numpy as np
import pytest

from experiments.tracking_validation_eval import (
    account_frame,
    aggregate_metrics,
    boxes,
    export_run,
    identity_frame,
    native_metrics,
    scoring_runs,
    threshold_match,
    union_ioa,
)
from experiments.tracking_validation_events import paired_frames, person_events, qualification, summary
from trackers.eval import evaluate_mot_sequence


def person(ident="a", box=(10, 10, 30, 50), **kwargs):
    return {
        "person_id": ident,
        "box": list(box),
        "role": "principal_rider",
        "identity_eligible": True,
        "detection_eligible": True,
        "evaluation_domain": "physical_person",
        "box_uncertainty_pixels": 1,
        **kwargs,
    }


def prediction(ident=0, box=(10, 10, 30, 50), index=0):
    return {"id": ident, "box": list(box), "proposal_index": index, "confidence": 0.9}


def frame(n=0, people=None, unknown=None, shot=0, stamp=None):
    return {
        "sample_index": n,
        "source_frame_index": n * 12,
        "pts": int((n * 0.2 if stamp is None else stamp) * 1000),
        "time_base": "1/1000",
        "timestamp": n * 0.2 if stamp is None else stamp,
        "shot": shot,
        "phase": "unknown",
        "people": [person()] if people is None else people,
        "uncertainties": []
        if unknown is None
        else [{"box": list(b), "region_id": f"u{i}"} for i, b in enumerate(unknown)],
    }


def metrics(frames, outputs, variant="primary"):
    runs, _ = scoring_runs(frames, outputs, variant)
    return aggregate_metrics([native_metrics(r) for r in runs])


@pytest.mark.parametrize(
    ("outputs", "tp", "fp", "fn"),
    [
        ([prediction()], 1, 0, 0),
        ([], 0, 0, 1),
        ([prediction(-1)], 0, 0, 1),
        ([prediction(), prediction(1, index=1)], 1, 1, 0),
    ],
)
def test_hand_calculated_detection_and_native_counts(outputs, tp, fp, fn):
    f = frame()
    r = account_frame(f, outputs)
    assert len(r["matches"]) == tp
    m = metrics([f], [outputs])
    assert (m["CLEAR"]["CLR_TP"], m["CLEAR"]["CLR_FP"], m["CLEAR"]["CLR_FN"]) == (tp, fp, fn)
    assert sum(p["fate"] == "duplicate" for p in r["outputs"]) == fp


@pytest.mark.parametrize("output", [[], [prediction()]])
def test_all_negative_contributions_and_undefined_recall(output):
    f = frame(people=[])
    r = account_frame(f, output)
    m = metrics([f], [output])
    assert m["CLEAR"]["CLR_FP"] == len(output)
    assert m["CLEAR"]["CLR_Frames"] == 1
    s = summary([f], [r], person_events([f], [r]))
    assert s["all_person_recall"] is None
    assert s["matched_output_precision"] == (0 if output else None)


def test_thresholded_cardinality_precedes_iou_and_tie_is_lexicographic():
    assert threshold_match(np.array([[1, 0.5], [0.5, 0.49]])) == [(0, 1), (1, 0)]
    assert threshold_match(np.ones((2, 2))) == [(0, 0), (1, 1)]
    assert threshold_match(np.array([[0.5 - 1e-13, 0.5]])) == [(0, 1)]


def test_canonical_order_is_independent_of_input_permutation():
    f = frame(people=[person("b"), person("a")])
    out = [prediction(1, index=1), prediction(0)]
    assert account_frame(f, out) == account_frame(frame(people=list(reversed(f["people"]))), list(reversed(out)))


@pytest.mark.parametrize("bad", [[-1, 0, 10, 10], [0, 0, 1921, 10], [1, 1, 1, 2], [0, 0, float("nan"), 10]])
def test_bad_source_geometry_rejected(bad):
    with pytest.raises(ValueError):
        boxes([bad])


@pytest.mark.parametrize(("right", "expected"), [(4.999999, False), (5, True), (5.000001, True)])
def test_union_ioa_equality_and_not_iou(right, expected):
    f = frame(people=[], unknown=[[0, 0, right, 10]])
    p = prediction(box=[0, 0, 10, 10])
    assert (account_frame(f, [p])["outputs"][0]["fate"] == "localized_unknown") is expected
    assert union_ioa(np.array([0, 0, 10, 10]), boxes([[0, 0, 5, 10], [0, 0, 5, 10]])) == 0.5
    assert union_ioa(np.array([0, 0, 10, 10]), boxes([[0, 0, 5, 10], [5, 0, 10, 10]])) == 1
    # A much larger region has IoA1 but small IoU; suppression uses IoA.
    assert union_ioa(np.array([10, 10, 20, 20]), boxes([[0, 0, 100, 100]])) == 1


def test_remote_unknown_preserves_rider_but_overlap_excludes_identity_only():
    f = frame(unknown=[[100, 100, 120, 130]])
    assert identity_frame(f, [prediction()])["eligible"]
    assert not identity_frame(f, [], "closed_world")["eligible"]
    near = frame(unknown=[[29, 20, 50, 40]])
    assert not identity_frame(near, [prediction()])["eligible"]
    assert len(account_frame(near, [prediction()])["matches"]) == 1


@pytest.mark.parametrize("left", [29, 29.99999])
def test_positive_intersection_even_below_hota_threshold_is_protected(left):
    f = frame(unknown=[[31, 0, 100, 100]])
    p = prediction(box=[left, 10, 100, 50])
    prepared = identity_frame(f, [p])
    assert prepared["eligible"] and prepared["predictions"] == [p] and not prepared["suppressed"]
    m = metrics([f], [[p]])
    assert m["HOTA"]["HOTA_FP_array"][-1] == 1


def test_multiple_remote_outputs_and_outside_background_accounted():
    f = frame(unknown=[[100, 100, 150, 150]])
    outputs = [
        prediction(0),
        prediction(1, [100, 100, 120, 120], 1),
        prediction(2, [110, 110, 130, 130], 2),
        prediction(3, [200, 200, 220, 220], 3),
    ]
    r = account_frame(f, outputs)
    assert [p["fate"] for p in r["outputs"]].count("localized_unknown") == 2
    assert [p["fate"] for p in r["outputs"]].count("unmatched_assessable_candidate") == 1
    prepared = identity_frame(f, outputs)
    assert len(prepared["suppressed"]) == 2 and len(prepared["predictions"]) == 2


def test_duplicate_near_ignore_is_not_hidden():
    f = frame(unknown=[[31, 10, 100, 50]])
    output = [prediction(), prediction(1, [12, 10, 40, 50], 1)]
    assert sorted(p["fate"] for p in account_frame(f, output)["outputs"]) == ["duplicate", "matched"]
    assert len(identity_frame(f, output)["predictions"]) == 2


def test_identity_unknown_physical_and_known_depiction_domains():
    unknown = person("x", [100, 100, 120, 140], identity_eligible=False)
    depicted = person(
        "photo",
        [200, 100, 220, 140],
        detection_eligible=False,
        identity_eligible=False,
        evaluation_domain="graphic_depiction",
        role="depicted_person",
    )
    f = frame(people=[person(), unknown, depicted])
    output = [prediction(), prediction(1, unknown["box"], 1), prediction(2, depicted["box"], 2)]
    assert len(account_frame(f, output)["matches"]) == 2
    prepared = identity_frame(f, output)
    assert prepared["known_gt"] == 1 and len(prepared["suppressed"]) == 2
    assert {reason for p in prepared["suppressed"] for reason in p["reasons"]} == {
        "identity_unknown_person",
        "graphic_depiction",
    }


def test_graphic_only_and_live_duplicate_overlap_protection():
    photo = person(
        "photo",
        [100, 100, 150, 150],
        detection_eligible=False,
        identity_eligible=False,
        evaluation_domain="graphic_depiction",
        role="depicted_person",
    )
    f = frame(people=[photo])
    p = prediction(box=photo["box"])
    assert not account_frame(f, [p])["matches"]
    assert identity_frame(f, [p])["suppressed"][0]["reasons"] == ["graphic_depiction"]
    live = frame(people=[person(), photo])
    outputs = [prediction(), prediction(1, [20, 10, 120, 50], 1)]
    assert len(identity_frame(live, outputs)["predictions"]) == 2


def test_temporal_unknown_suppression_can_improve_global_score_without_proving_attachment():
    f = [frame(0, unknown=[[100, 100, 150, 150]]), frame(1, unknown=[[100, 100, 150, 150]])]
    output = [[prediction()], [prediction(0, [110, 110, 130, 130])]]
    masked = metrics(f, output)
    raw = metrics([frame(0), frame(1)], output)
    assert masked["Identity"]["IDFP"] < raw["Identity"]["IDFP"]
    assert masked["Identity"]["IDF1"] > raw["Identity"]["IDF1"]
    assert len(account_frame(f[1], output[1])["matches"]) == 0


def test_source_fixed_exclusion_runs_and_dense_later_public_export(tmp_path):
    f = [frame(0), frame(1, unknown=[[20, 20, 50, 60]]), frame(2), frame(3)]
    outputs = [[prediction()] for _ in f]
    original = copy.deepcopy(f)
    runs, excluded = scoring_runs(f, outputs)
    assert [len(r) for r in runs] == [1, 2] and len(excluded) == 1
    assert [len(r) for r in scoring_runs(f, [[] for _ in f])[0]] == [1, 2]
    assert f == original
    later = runs[1]
    assert export_run(later, tmp_path)
    public = evaluate_mot_sequence(tmp_path / "gt.txt", tmp_path / "tracks.txt", metrics=["CLEAR", "HOTA", "Identity"])
    core = native_metrics(later)
    for family in core:
        obj = getattr(public, family)
        values = obj.to_dict(include_arrays=True) if family == "HOTA" else obj.to_dict()
        assert values.keys() == core[family].keys()
        for key in values:
            np.testing.assert_allclose(values[key], core[family][key], atol=1e-12, rtol=0)
    assert public.CLEAR.CLR_Frames == 2
    assert (tmp_path / "gt.txt").read_text().startswith("1,")


def test_cut_resets_metric_identity_and_native_aggregate_keeps_negatives():
    f = [frame(0), frame(1, shot=1)]
    runs, _ = scoring_runs(f, [[prediction()], [prediction(2)]])
    assert len(runs) == 2
    success = native_metrics(runs[0])
    missed = metrics([frame()], [[]])
    negative = metrics([frame(people=[])], [[prediction()]])
    total = aggregate_metrics([success, missed, negative])
    assert (
        total["CLEAR"]["CLR_TP"],
        total["CLEAR"]["CLR_FN"],
        total["CLEAR"]["CLR_FP"],
        total["CLEAR"]["CLR_Frames"],
    ) == (1, 1, 1, 3)
    assert metrics(f, [[prediction()], [prediction(2)]])["CLEAR"]["IDSW"] == 0


@pytest.mark.parametrize(
    ("outputs", "switches"), [([[prediction()], [prediction(1)]], 1), ([[prediction()], [prediction()]], 0)]
)
def test_direct_identity_switch(outputs, switches):
    f = [frame(0), frame(1)]
    accounts = [account_frame(a, b) for a, b in zip(f, outputs)]
    assert len(person_events(f, accounts)["id_switches"]) == switches
    assert metrics(f, outputs)["CLEAR"]["IDSW"] == switches


@pytest.mark.parametrize("kind", ["visibility", "cut", "segment_end", "recovery"])
def test_gap_observed_end_is_last_missing_except_actual_recovery(kind):
    f = [frame(0, stamp=0), frame(1, stamp=0.2), frame(2, stamp=0.55)]
    out = [[prediction()], [], []]
    if kind == "visibility":
        f[2] = frame(2, people=[], stamp=0.55)
    elif kind == "cut":
        f[2] = frame(2, people=[person("b")], shot=1, stamp=0.55)
    elif kind == "recovery":
        out[2] = [prediction()]
    accounts = [account_frame(a, b) for a, b in zip(f, out)]
    e = person_events(f, accounts)
    gap = e["losses"][0]
    expected = 0.35 if kind in ["recovery", "segment_end"] else 0
    assert gap["observed_duration_seconds"] == pytest.approx(expected)
    assert gap["right_censored"] == (kind != "recovery")
    assert (
        gap["censor_boundary"] is None
        if kind in ["recovery", "segment_end"]
        else gap["censor_boundary"]["timestamp"] == 0.55
    )


def test_never_confirmed_startup_fragmentation_and_paired_physical_identity():
    f = [frame(i) for i in range(5)]
    outputs = [[], [prediction(4)], [], [prediction(9)], []]
    accounts = [account_frame(a, b) for a, b in zip(f, outputs)]
    events = person_events(f, accounts)
    assert events["runs"][0]["startup_delay_seconds"] == 0.2
    assert events["losses"][0]["recovery_kind"] == "different_id"
    s = summary(f, accounts, events)
    assert s["fragmentations"] == 1 and s["right_censored_losses"] == 1
    empty = [account_frame(a, []) for a in f]
    assert person_events(f, empty)["runs"][0]["never_confirmed"]
    paired = paired_frames(empty, accounts)
    assert len(paired["gained"]) == 2 and not paired["lost"]
    assert qualification(s, s)["status"] == "inconclusive"


def test_public_export_cannot_represent_trailing_fully_empty_frames(tmp_path):
    frames = [frame(0), frame(1, people=[]), frame(2, people=[])]
    runs, _ = scoring_runs(frames, [[prediction()], [], []])
    assert not export_run(runs[0], tmp_path)
    core = native_metrics(runs[0])
    public = evaluate_mot_sequence(tmp_path / "gt.txt", tmp_path / "tracks.txt", metrics=["CLEAR"])
    assert core["CLEAR"]["CLR_Frames"] == 3 and public.CLEAR.CLR_Frames == 1
    assert core["CLEAR"]["CLR_TP"] == public.CLEAR.CLR_TP == 1


def test_two_people_exchange_ids_is_not_extra_person_false_positive():
    f = [frame(i, people=[person("a"), person("b", [100, 10, 120, 50], role="other_person")]) for i in range(2)]
    out = [[prediction(0), prediction(1, [100, 10, 120, 50], 1)], [prediction(1), prediction(0, [100, 10, 120, 50], 1)]]
    accounts = [account_frame(a, b) for a, b in zip(f, out)]
    events = person_events(f, accounts)
    assert len(events["id_switches"]) == 2 and len(events["wrong_attachment_candidates"]) == 2
    assert all(p["fate"] == "matched" for a in accounts for p in a["outputs"])
    assert metrics(f, out)["CLEAR"]["IDSW"] == 2


def test_full_containment_sensitivity_preserves_partial_unknown_output():
    f = frame(people=[], unknown=[[0, 0, 6, 10]])
    p = prediction(box=[0, 0, 10, 10])
    assert identity_frame(f, [p])["suppressed"]
    assert identity_frame(f, [p], "full_containment")["predictions"] == [p]


def test_source_declared_unknown_visibility_splits_identity_but_not_other_people_detection():
    f = frame(people=[person("other", role="other_person")])
    f["identity_visibility_unknown"] = True
    assert not identity_frame(f, [prediction()])["eligible"]
    assert len(account_frame(f, [prediction()])["matches"]) == 1


def test_known_graphic_negative_not_dropped_by_closed_world_reference():
    photo = person(
        "photo",
        detection_eligible=False,
        identity_eligible=False,
        evaluation_domain="graphic_depiction",
        role="depicted_person",
    )
    f = frame(people=[photo])
    prepared = identity_frame(f, [], "closed_world")
    assert prepared["eligible"] and prepared["known_gt"] == 0
    assert metrics([f], [[]], "closed_world")["CLEAR"]["CLR_Frames"] == 1


def test_incompatible_censored_duration_bounds_never_rank_as_improvement():
    from experiments.tracking_validation_events import compare_loss_durations

    a = {
        "person_id": "a",
        "first_missing": {"sample_index": 1},
        "right_censored": True,
        "observed_duration_seconds": 0.2,
    }
    b = a | {"right_censored": False, "observed_duration_seconds": 0.5}
    assert compare_loss_durations(a, b)["delta_seconds"] is None
    a["right_censored"] = False
    assert compare_loss_durations(a, b)["delta_seconds"] == pytest.approx(0.3)
    b["first_missing"] = {"sample_index": 2}
    assert not compare_loss_durations(a, b)["comparable"]


def audited_summary(**changes):
    return {
        "principal_person_frames": 10,
        "physical_person_frames": 10,
        "principal_covered_frames": 5,
        "all_person_recall": 0.5,
        "identity_idf1": 0.5,
        "established_harms": [],
        "audit_complete": True,
        "sensitivity_status": "consistent",
        "per_clip_regressions": [],
        "established_id_switches": 0,
        "established_fragmentations": 0,
        **changes,
    }


def test_qualification_is_research_only_and_requires_replicated_partition_evidence():
    baseline = audited_summary()
    improved = audited_summary(principal_covered_frames=6, all_person_recall=0.6, identity_idf1=0.6)
    assert qualification(baseline, improved)["status"] == "qualifies_research_only"
    assert qualification(baseline, improved)["activation_authorized"] is False
    assert qualification(baseline, improved | {"audit_complete": False})["status"] == "inconclusive"
    assert qualification(baseline, improved | {"established_harms": None})["status"] == "inconclusive"
    with pytest.raises(ValueError, match="common GT"):
        qualification(baseline, improved | {"principal_person_frames": 11})


@pytest.mark.parametrize(
    "changes",
    [
        {"principal_covered_frames": 5},
        {"all_person_recall": 0.4},
        {"identity_idf1": 0.4},
        {"established_id_switches": 1},
        {"established_fragmentations": 1},
        {"established_harms": ["background event"]},
        {"per_clip_regressions": ["lost native-covered frame"]},
    ],
)
def test_qualification_cannot_net_away_identity_harm_or_per_clip_regression(changes):
    baseline = audited_summary()
    improved = audited_summary(principal_covered_frames=6, all_person_recall=0.6, identity_idf1=0.6)
    assert qualification(baseline, improved | changes)["status"] == "does_not_qualify"


def test_custom_preprocessed_nonperfect_core_public_arrays_match(tmp_path):
    f = [frame(4, unknown=[[100, 100, 150, 150]]), frame(5, unknown=[[100, 100, 150, 150]])]
    output = [
        [prediction(), prediction(9, [100, 100, 120, 120], 1), prediction(2, [200, 200, 220, 220], 2)],
        [prediction(1)],
    ]
    runs, _ = scoring_runs(f, output)
    assert export_run(runs[0], tmp_path)
    public = evaluate_mot_sequence(tmp_path / "gt.txt", tmp_path / "tracks.txt", metrics=["CLEAR", "HOTA", "Identity"])
    core = native_metrics(runs[0])
    for family, expected in core.items():
        obj = getattr(public, family)
        values = obj.to_dict(include_arrays=True) if family == "HOTA" else obj.to_dict()
        for key in values:
            np.testing.assert_allclose(values[key], expected[key], rtol=0, atol=1e-12)


@pytest.mark.parametrize("second_id", [0, 9])
def test_secondary_source_link_distinguishes_loose_box_miss_from_actual_id_change(second_id):
    from experiments.tracking_validation_eval import source_audited_accounts

    f = [frame(0), frame(1)]
    outputs = [[prediction()], [prediction(second_id, [29, 10, 60, 50])]]
    primary = [account_frame(a, b) for a, b in zip(f, outputs)]
    assert len(primary[1]["matches"]) == 0
    decisions = [{0: {"classification": "physical_person", "person_id": "a"}} for _ in f]
    secondary_frames, secondary = source_audited_accounts(f, outputs, decisions)
    assert len(secondary[1]["matches"]) == 1
    assert len(person_events(secondary_frames, secondary)["id_switches"]) == int(second_id != 0)
    assert len(person_events(secondary_frames, secondary)["losses"]) == 0
    assert len(primary[1]["matches"]) == 0  # No overwrite of primary localization penalty.


def test_secondary_source_audit_preserves_unresolved_attachment():
    from experiments.tracking_validation_eval import source_audited_accounts

    f = [frame(0), frame(1)]
    outputs = [[prediction()], [prediction()]]
    decisions = [
        {0: {"classification": "physical_person", "person_id": "a"}},
        {0: {"classification": "unresolved", "person_id": None}},
    ]
    secondary_frames, secondary = source_audited_accounts(f, outputs, decisions)
    assert secondary[1]["secondary_identity_unresolved"] and not secondary[1]["matches"]
    assert secondary_frames[1]["people"][0]["identity_eligible"] is False
    assert (
        person_events(secondary_frames, secondary)["runs"][0]["censor_reason"] == "visibility_or_identity_unassessable"
    )


@pytest.mark.parametrize("status", ["directional_reversal", "undefined", None])
def test_qualification_cannot_ignore_reversed_or_undefined_sensitivity(status):
    baseline = audited_summary()
    improved = audited_summary(
        principal_covered_frames=6, all_person_recall=0.6, identity_idf1=0.6, sensitivity_status=status
    )
    assert qualification(baseline, improved)["status"] == "inconclusive"
