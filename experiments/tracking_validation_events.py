# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Source-visible physical continuity, censored duration and paired policy accounting."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise

from experiments.tracking_validation_eval import physical_people, require


def _point(frame):
    return {
        k: frame[k] for k in ["sample_index", "source_frame_index", "timestamp", "pts", "time_base", "phase", "shot"]
    }


def person_events(frames, accounts):
    """One break per established loss; unknown/cut boundaries never extend observed duration."""
    require(len(frames) == len(accounts), "frame/account length mismatch")
    result = {"runs": [], "losses": [], "id_switches": [], "wrong_attachment_candidates": []}
    active = {}
    id_people = {}

    def close(person, boundary, reason):
        run = active.pop(person)
        if run["pending"] is not None:
            finish_gap(run, None, boundary, reason)
        run["never_confirmed"] = run["first_match"] is None
        run["startup_observed_lower_bound_seconds"] = (
            run["last_assessable"]["timestamp"] - run["start"]["timestamp"] if run["never_confirmed"] else None
        )
        run["right_censored"] = True
        run["censor_reason"] = reason
        run["censor_boundary"] = boundary
        run.pop("pending")
        result["runs"].append(run)

    def finish_gap(run, recovery, boundary, reason):
        gap = run["pending"]
        observed_end = recovery if recovery is not None else gap["last_missing"]
        gap.update(
            observed_end=observed_end,
            observed_duration_seconds=observed_end["timestamp"] - gap["first_missing"]["timestamp"],
            observed_source_frame_span=observed_end["source_frame_index"] - gap["first_missing"]["source_frame_index"],
            right_censored=recovery is None,
            censor_reason=reason,
            censor_boundary=boundary,
        )
        result["losses"].append(gap)
        run["pending"] = None

    for frame, account in zip(frames, accounts, strict=True):
        matches = {m["person_id"]: m["id"] for m in account["matches"]}
        people = {p["person_id"]: p for p in physical_people(frame) if p["identity_eligible"]}
        for person in list(active):
            if person not in people or active[person]["shot"] != frame["shot"]:
                reason = "cut" if active[person]["shot"] != frame["shot"] else "visibility_or_identity_unassessable"
                close(person, _point(frame), reason)
        id_people = {k: v for k, v in id_people.items() if k[0] == frame["shot"]}
        for person, truth in people.items():
            point = _point(frame)
            if person not in active:
                active[person] = {
                    "person_id": person,
                    "role": truth["role"],
                    "shot": frame["shot"],
                    "start": point,
                    "last_assessable": point,
                    "first_match": None,
                    "last_id": None,
                    "pending": None,
                    "assessable_frames": 0,
                    "matched_frames": 0,
                }
            run = active[person]
            run["assessable_frames"] += 1
            run["last_assessable"] = point
            ident = matches.get(person)
            if ident is None:
                if run["last_id"] is not None:
                    if run["pending"] is None:
                        run["pending"] = {
                            "person_id": person,
                            "role": truth["role"],
                            "first_missing": point,
                            "last_missing": point,
                            "prior_id": run["last_id"],
                            "missing_observations": 0,
                        }
                    run["pending"]["last_missing"] = point
                    run["pending"]["missing_observations"] += 1
                continue
            run["matched_frames"] += 1
            if run["first_match"] is None:
                run["first_match"] = point
                run["startup_delay_seconds"] = point["timestamp"] - run["start"]["timestamp"]
            if run["pending"] is not None:
                run["pending"].update(
                    recovery_id=ident, recovery_kind="same_id" if ident == run["last_id"] else "different_id"
                )
                finish_gap(run, point, None, None)
            if run["last_id"] is not None and run["last_id"] != ident:
                result["id_switches"].append(
                    {"person_id": person, "before": run["last_id"], "after": ident, "point": point}
                )
            prior = id_people.get((frame["shot"], ident))
            if prior is not None and prior != person:
                result["wrong_attachment_candidates"].append(
                    {
                        "id": ident,
                        "previous_person": prior,
                        "new_person": person,
                        "point": point,
                        "status": "temporal match-based candidate; source audit required",
                    }
                )
            id_people[frame["shot"], ident] = person
            run["last_id"] = ident
    for person in list(active):
        close(person, None, "segment_end")
    return result


def summary(frames, accounts, events):
    """Explicit denominators and undefined cases, without declaring audit candidates false."""
    fates = Counter(p["fate"] for row in accounts for p in row["outputs"])
    physical = sum(row["physical_gt"] for row in accounts)
    matched = sum(len(row["matches"]) for row in accounts)
    principal = sum(len(row["principal_gt"]) for row in accounts)
    covered = sum(m["role"] == "principal_rider" for row in accounts for m in row["matches"])
    positive = sum(p["id"] >= 0 for row in accounts for p in row["outputs"])
    elapsed = sum(b["timestamp"] - a["timestamp"] for a, b in pairwise(frames) if a["shot"] == b["shot"])
    losses = events["losses"]
    return {
        "sampled_frames": len(frames),
        "within_shot_observed_seconds": elapsed,
        "physical_person_frames": physical,
        "matched_person_frames": matched,
        "all_person_recall": matched / physical if physical else None,
        "principal_person_frames": principal,
        "principal_covered_frames": covered,
        "principal_coverage": covered / principal if principal else None,
        "public_outputs": positive,
        "matched_output_precision": matched / positive if positive else None,
        "fates": dict(fates),
        "unmatched_candidate_frames": sum(
            any(p["fate"] == "unmatched_assessable_candidate" for p in row["outputs"]) for row in accounts
        ),
        "unmatched_candidates_per_sample": fates["unmatched_assessable_candidate"] / len(frames) if frames else None,
        "unmatched_candidates_per_observed_second": fates["unmatched_assessable_candidate"] / elapsed
        if elapsed
        else None,
        "duplicate_principal_outputs": sum(
            p["fate"] == "duplicate" and p["role"] == "principal_rider" for row in accounts for p in row["outputs"]
        ),
        "duplicate_other_person_outputs": sum(
            p["fate"] == "duplicate" and p["role"] != "principal_rider" for row in accounts for p in row["outputs"]
        ),
        "identity_runs": len(events["runs"]),
        "never_confirmed_runs": sum(r["never_confirmed"] for r in events["runs"]),
        "established_losses": len(losses),
        "right_censored_losses": sum(g["right_censored"] for g in losses),
        "observed_loss_seconds_sum": sum(g["observed_duration_seconds"] for g in losses),
        "same_id_recoveries": sum(g.get("recovery_kind") == "same_id" for g in losses),
        "different_id_recoveries": sum(g.get("recovery_kind") == "different_id" for g in losses),
        "fragmentations": sum(not g["right_censored"] for g in losses),
        "id_switches": len(events["id_switches"]),
        "losses_by_phase": dict(Counter(g["first_missing"]["phase"] for g in losses)),
        "audit_limit": (
            "Unmatched/background and wrong-attachment candidates require source audit; localized "
            "unknowns never count automatically correct or false. Censored duration sums are lower"
            " bounds, not comparable full durations."
        ),
    }


def paired_frames(native, candidate):
    """Common source denominators and every gained/lost principal-person frame."""
    require(len(native) == len(candidate), "paired length mismatch")
    gained, lost, changed = [], [], []
    for a, b in zip(native, candidate, strict=True):
        require(
            a["principal_gt"] == b["principal_gt"] and a["physical_gt"] == b["physical_gt"],
            "policy-dependent GT denominator",
        )
        hits_a = {m["person_id"] for m in a["matches"] if m["role"] == "principal_rider"}
        hits_b = {m["person_id"] for m in b["matches"] if m["role"] == "principal_rider"}
        gained.extend({"sample_index": a["sample_index"], "person_id": p} for p in sorted(hits_b - hits_a))
        lost.extend({"sample_index": a["sample_index"], "person_id": p} for p in sorted(hits_a - hits_b))

        def shape(row):
            return [(p["proposal_index"], p["id"] >= 0, p["fate"], p["person_id"]) for p in row["outputs"]]

        if shape(a) != shape(b):
            changed.append(a["sample_index"])
    return {
        "gained": gained,
        "lost": lost,
        "changed_output_samples": changed,
        "identity_note": (
            "Raw numerical IDs are not cross-policy physical identity; events are compared by source person/shot."
        ),
    }


def qualification(native, candidate):
    """Predeclared research gate; incomplete audits/identity never qualify a feature."""
    missing = [
        k
        for k in [
            "identity_idf1",
            "established_harms",
            "audit_complete",
            "per_clip_regressions",
            "established_id_switches",
            "established_fragmentations",
            "sensitivity_status",
        ]
        if k not in candidate
    ]
    if (
        missing
        or not candidate.get("audit_complete")
        or not native.get("audit_complete")
        or any(
            candidate.get(k) is None
            for k in [
                "established_harms",
                "per_clip_regressions",
                "established_id_switches",
                "established_fragmentations",
            ]
        )
        or any(native.get(k) is None for k in ["established_id_switches", "established_fragmentations"])
        or candidate.get("sensitivity_status") != "consistent"
        or candidate.get("identity_idf1") is None
        or native.get("identity_idf1") is None
    ):
        return {
            "status": "inconclusive",
            "reasons": ["missing identity/audit evidence or unresolved sensitivity", *missing],
        }
    require(
        native["principal_person_frames"] == candidate["principal_person_frames"]
        and native["physical_person_frames"] == candidate["physical_person_frames"],
        "qualification requires common GT denominators",
    )
    conditions = {
        "coverage_increase": candidate["principal_person_frames"] > 0
        and candidate["principal_covered_frames"] > native["principal_covered_frames"],
        "all_person_guard": candidate["all_person_recall"] is not None
        and candidate["all_person_recall"] >= native["all_person_recall"],
        "identity_guard": candidate["identity_idf1"] >= native["identity_idf1"]
        and candidate["established_id_switches"] <= native["established_id_switches"]
        and candidate["established_fragmentations"] <= native["established_fragmentations"],
        "no_new_established_harm": not candidate["established_harms"],
        "no_per_clip_regression": not candidate["per_clip_regressions"],
    }
    return {
        "status": "qualifies_research_only" if all(conditions.values()) else "does_not_qualify",
        "conditions": conditions,
        "activation_authorized": False,
    }


def compare_loss_durations(baseline, candidate):
    """Rank only fully observed common-onset interruptions, never incompatible lower bounds."""
    same_origin = (
        baseline["person_id"] == candidate["person_id"]
        and baseline["first_missing"]["sample_index"] == candidate["first_missing"]["sample_index"]
    )
    if not same_origin or baseline["right_censored"] or candidate["right_censored"]:
        return {
            "comparable": False,
            "delta_seconds": None,
            "reason": "different origin or censored boundaries; no duration ranking",
        }
    return {
        "comparable": True,
        "delta_seconds": candidate["observed_duration_seconds"] - baseline["observed_duration_seconds"],
        "reason": "same physical person/source onset, both actual observed recoveries",
    }


def paired_events(baseline, candidate):
    """Source-person event contrasts and explicit per-clip regression flags."""
    regressions = []
    base_runs = {(r["person_id"], r["start"]["sample_index"]): r for r in baseline["runs"]}
    for run in candidate["runs"]:
        key = run["person_id"], run["start"]["sample_index"]
        previous = base_runs[key]
        if previous["first_match"] is not None and (
            run["first_match"] is None or run["first_match"]["timestamp"] > previous["first_match"]["timestamp"]
        ):
            regressions.append(
                {
                    "kind": "startup_regression",
                    "person_id": run["person_id"],
                    "baseline_first_match": previous["first_match"],
                    "candidate_first_match": run["first_match"],
                }
            )
    known_switches = {(e["person_id"], e["point"]["sample_index"]) for e in baseline["id_switches"]}
    for event in candidate["id_switches"]:
        if (event["person_id"], event["point"]["sample_index"]) not in known_switches:
            regressions.append({"kind": "new_match_based_switch", "event": event, "status": "source audit required"})
    known_attachments = {
        (e["previous_person"], e["new_person"], e["point"]["sample_index"])
        for e in baseline["wrong_attachment_candidates"]
    }
    for event in candidate["wrong_attachment_candidates"]:
        if (event["previous_person"], event["new_person"], event["point"]["sample_index"]) not in known_attachments:
            regressions.append(
                {"kind": "new_wrong_attachment_candidate", "event": event, "status": "source audit required"}
            )
    known_fragments = {
        (e["person_id"], e["first_missing"]["sample_index"], e["observed_end"]["sample_index"])
        for e in baseline["losses"]
        if not e["right_censored"]
    }
    for event in candidate["losses"]:
        key = event["person_id"], event["first_missing"]["sample_index"], event["observed_end"]["sample_index"]
        if not event["right_censored"] and key not in known_fragments:
            regressions.append(
                {
                    "kind": "new_fragmentation_opportunity",
                    "event": event,
                    "status": (
                        "May follow additional establishment; compare coverage/visible opportunity and source "
                        "audit, never equate raw loss count with harm."
                    ),
                }
            )
    comparisons = []
    for a in baseline["losses"]:
        for b in candidate["losses"]:
            if (
                a["person_id"] == b["person_id"]
                and a["first_missing"]["sample_index"] == b["first_missing"]["sample_index"]
            ):
                record = {
                    "person_id": a["person_id"],
                    "first_missing_sample": a["first_missing"]["sample_index"],
                    **compare_loss_durations(a, b),
                }
                comparisons.append(record)
                if record["comparable"] and record["delta_seconds"] > 0:
                    regressions.append({"kind": "increased_fully_observed_interruption", **record})
    return {"regression_flags": regressions, "duration_comparisons": comparisons}
