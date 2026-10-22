# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Experiment-only physical-person accounting and custom known-identity native metrics."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from trackers.eval import (
    aggregate_clear_metrics,
    aggregate_hota_metrics,
    aggregate_identity_metrics,
    box_iou,
    compute_clear_metrics,
    compute_hota_metrics,
    compute_identity_metrics,
)

TOLERANCE = 1e-12
DETECTION_IOU = 0.5
IGNORE_IOA = 0.5
# Finite, pre-outcome diagnostics, not tunable evaluator configuration.
SENSITIVITIES = ("primary", "closed_world", "full_containment", "expanded_boxes", "contracted_boxes")


def require(condition, message):
    """Reject invalid experiment inputs without changing native library behavior."""
    if not condition:
        raise ValueError(message)


def jsonable(value):
    """Convert native metric arrays and tracker snapshots to strict JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, type):
        return value.__module__ + "." + value.__name__
    if callable(value):
        return "callable"
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    raise TypeError(type(value).__name__)


def boxes(values):
    """Return finite source-coordinate boxes, preserving explicit empty shapes."""
    result = np.asarray(values, dtype=float).reshape(-1, 4)
    require(np.isfinite(result).all(), "nonfinite box")
    require(np.all(result[:, 2:] > result[:, :2]), "degenerate box")
    require(np.all(result[:, :2] >= 0) and np.all(result[:, 2:] <= [1920, 1080]), "out-of-image box")
    return result


def intersection(a, b):
    """Positive intersection area, without IoU thresholding."""
    left = np.maximum(a[:2], b[:2])
    right = np.minimum(a[2:], b[2:])
    return float(np.maximum(right - left, 0).prod())


def union_ioa(prediction, regions):
    """Exact clipped rectangle-union area divided by prediction area."""
    clipped = []
    for region in regions:
        left = np.maximum(prediction[:2], region[:2])
        right = np.minimum(prediction[2:], region[2:])
        if np.all(right > left):
            clipped.append([*left, *right])
    xs = sorted({x for b in clipped for x in [b[0], b[2]]})
    area = 0.0
    for lo, hi in pairwise(xs):
        intervals = sorted((b[1], b[3]) for b in clipped if b[0] < hi and b[2] > lo)
        end = -float("inf")
        for bottom, top in intervals:
            area += (hi - lo) * max(0, top - max(bottom, end))
            end = max(end, top)
    return float(area / np.prod(prediction[2:] - prediction[:2]))


def _optimal(matrix):
    rows, columns = matrix.shape
    if not rows or not columns:
        return [], (0, 0.0)
    valid = matrix >= DETECTION_IOU
    weight = np.where(valid, min(rows, columns) + 1 + matrix, -1e6)
    weight = np.concatenate([weight, np.zeros((rows, rows))], axis=1)
    ri, ci = linear_sum_assignment(-weight)
    pairs = [(int(r), int(c)) for r, c in zip(ri, ci) if c < columns and valid[r, c]]
    return pairs, (len(pairs), sum(float(matrix[r, c]) for r, c in pairs))


def threshold_match(matrix):
    """Max cardinality then total IoU; lexicographic canonical row/column tie rule.

    Input ordering must already be canonical. Tolerance is 1e-12 for equality of total IoU only; the inclusive 0.5
    eligibility threshold has no epsilon.
    """
    matrix = np.asarray(matrix, dtype=float)
    remaining = list(range(matrix.shape[1]))
    pairs = []
    for row in range(matrix.shape[0]):
        _, optimum = _optimal(matrix[row:, remaining])
        candidates = [c for c in remaining if matrix[row, c] >= DETECTION_IOU] + [None]
        for column in candidates:
            rest = [c for c in remaining if c != column]
            _, value = _optimal(matrix[row + 1 :, rest])
            count = value[0] + int(column is not None)
            score = value[1] + (float(matrix[row, column]) if column is not None else 0)
            if count == optimum[0] and abs(score - optimum[1]) <= TOLERANCE:
                if column is not None:
                    pairs.append((row, column))
                    remaining.remove(column)
                break
        else:
            raise ValueError("assignment tie resolution failed")
    return pairs


def physical_people(frame):
    """All eligible physical people, including generic other_person roles."""
    return sorted([p for p in frame["people"] if p["detection_eligible"]], key=lambda p: p["person_id"])


def public_outputs(output):
    """Retain ID zero; use stable source proposal/geometry ordering for ties."""
    records = [dict(p) for p in output if p["id"] >= 0]
    return sorted(records, key=lambda p: (tuple(p["box"]), p["proposal_index"], p["id"]))


def regions(frame, identity=False):
    """Source-fixed localized unknown and graphic domains with reason provenance."""
    result = [{"box": r["box"], "reason": "uncertainty", "entity": r["region_id"]} for r in frame["uncertainties"]]
    for p in frame["people"]:
        if p["evaluation_domain"] == "graphic_depiction":
            result.append({"box": p["box"], "reason": "graphic_depiction", "entity": p["person_id"]})
        elif identity and not p["identity_eligible"]:
            result.append({"box": p["box"], "reason": "identity_unknown_person", "entity": p["person_id"]})
    return result


def account_frame(frame, output, ignore_threshold=IGNORE_IOA):
    """Account every raw output once; unmatched assessable outputs are audit candidates."""
    gt = physical_people(frame)
    predicted = public_outputs(output)
    gb = boxes([p["box"] for p in gt])
    pb = boxes([p["box"] for p in predicted])
    similarity = box_iou(gb, pb)
    matches = threshold_match(similarity)
    matched_outputs = {c: r for r, c in matches}
    matched_gt = {r for r, _ in matches}
    local = regions(frame)
    records = []
    for column, p in enumerate(predicted):
        record = p | {"person_id": None, "role": None, "fate": "unmatched_assessable_candidate"}
        if column in matched_outputs:
            person = gt[matched_outputs[column]]
            record.update(fate="matched", person_id=person["person_id"], role=person["role"])
        else:
            duplicates = [r for r in matched_gt if similarity[r, column] >= DETECTION_IOU]
            protected = any(intersection(pb[column], b) > 0 for b in gb)
            if duplicates:
                person = gt[min(duplicates)]
                record.update(fate="duplicate", person_id=person["person_id"], role=person["role"])
            elif not protected and union_ioa(pb[column], boxes([r["box"] for r in local])) >= ignore_threshold:
                reasons = sorted({r["reason"] for r in local if intersection(pb[column], np.array(r["box"])) > 0})
                record.update(fate="localized_unknown", ignore_reasons=reasons)
        records.append(record)
    records.extend(p | {"fate": "unconfirmed", "person_id": None, "role": None} for p in output if p["id"] < 0)
    return {
        "sample_index": frame["sample_index"],
        "shot": frame["shot"],
        "timestamp": frame["timestamp"],
        "physical_gt": len(gt),
        "identity_unknown_gt": sum(not p["identity_eligible"] for p in gt),
        "principal_gt": [p["person_id"] for p in gt if p["role"] == "principal_rider"],
        "raw_overlap_context": [
            {
                "person_id": p["person_id"],
                "max_overlapping_raw_confidence": max(
                    (
                        o["confidence"]
                        for o in output
                        if box_iou(boxes([p["box"]]), boxes([o["box"]]))[0, 0] >= DETECTION_IOU
                    ),
                    default=None,
                ),
            }
            for p in gt
        ],
        "matches": [
            {
                "person_id": gt[r]["person_id"],
                "id": predicted[c]["id"],
                "proposal_index": predicted[c]["proposal_index"],
                "role": gt[r]["role"],
            }
            for r, c in matches
        ],
        "outputs": records,
        "unknown_region_count": len(local),
        "source_uncertainties": frame["uncertainties"],
    }


def expanded(box, amount):
    """Clip a declared estimated-boundary sensitivity to the source image."""
    result = np.asarray(box, float) + np.array([-amount, -amount, amount, amount])
    result[:2] = np.maximum(result[:2], 0)
    result[2:] = np.minimum(result[2:], [1920, 1080])
    # A contracted tiny box keeps a positive extent; no GT is removed.
    if np.any(result[2:] <= result[:2]):
        center = (np.asarray(box)[:2] + np.asarray(box)[2:]) / 2
        result = np.r_[center - 0.25, center + 0.25]
    return result.tolist()


def identity_frame(frame, output, variant="primary"):
    """Source-fixed eligibility plus output-dependent localized suppression."""
    require(variant in SENSITIVITIES, "unknown finite sensitivity")
    known = [dict(p) for p in physical_people(frame) if p["identity_eligible"]]
    local = regions(frame, identity=True)
    if frame.get("identity_visibility_unknown"):
        return {"eligible": False, "reason": "source_declared_visibility_unknown", "known_gt": len(known)}
    if variant == "closed_world" and any(r["reason"] != "graphic_depiction" for r in local):
        return {"eligible": False, "reason": "closed_world_unknown", "known_gt": len(known)}
    ambiguity = any(
        intersection(np.array(expanded(p["box"], p["box_uncertainty_pixels"])), np.array(r["box"])) > 0
        for p in known
        for r in local
    )
    if ambiguity:
        return {"eligible": False, "reason": "source_boundary_overlap", "known_gt": len(known)}
    if variant in ("expanded_boxes", "contracted_boxes"):
        for p in known:
            p["box"] = expanded(p["box"], p["box_uncertainty_pixels"] * (1 if variant == "expanded_boxes" else -1))
    predicted = public_outputs(output)
    kept = []
    suppressed = []
    for p in predicted:
        protected = any(intersection(np.array(p["box"]), np.array(g["box"])) > 0 for g in known)
        ratio = union_ioa(np.array(p["box"]), boxes([r["box"] for r in local]))
        threshold = 1.0 if variant == "full_containment" else IGNORE_IOA
        if not protected and ratio >= threshold:
            reasons = sorted({r["reason"] for r in local if intersection(np.array(p["box"]), np.array(r["box"])) > 0})
            suppressed.append(p | {"ioa_union": ratio, "reasons": reasons})
        else:
            kept.append(p)
    return {
        "eligible": True,
        "reason": None,
        "gt": known,
        "predictions": kept,
        "suppressed": suppressed,
        "known_gt": len(known),
    }


def scoring_runs(frames, outputs, variant="primary"):
    """Split source-fixed eligible runs; preserve empty/negative frames."""
    result = []
    current = []
    excluded = []
    for frame, output in zip(frames, outputs, strict=True):
        prepared = identity_frame(frame, output, variant)
        if current and (not prepared["eligible"] or frame["shot"] != current[-1]["frame"]["shot"]):
            result.append(current)
            current = []
        if prepared["eligible"]:
            current.append({"frame": frame, "prepared": prepared})
        else:
            excluded.append(
                {"sample_index": frame["sample_index"], "reason": prepared["reason"], "known_gt": prepared["known_gt"]}
            )
    if current:
        result.append(current)
    return result, excluded


def run_arrays(run):
    """Remap known GT and output IDs independently to contiguous local integers."""
    gt_map = {v: i for i, v in enumerate(sorted({g["person_id"] for row in run for g in row["prepared"]["gt"]}))}
    pred_map = {v: i for i, v in enumerate(sorted({p["id"] for row in run for p in row["prepared"]["predictions"]}))}
    gt_ids, pred_ids, similarities = [], [], []
    for row in run:
        gt = row["prepared"]["gt"]
        predicted = row["prepared"]["predictions"]
        gt_ids.append(np.array([gt_map[g["person_id"]] for g in gt], int))
        pred_ids.append(np.array([pred_map[p["id"]] for p in predicted], int))
        similarities.append(box_iou(boxes([g["box"] for g in gt]), boxes([p["box"] for p in predicted])))
    return gt_ids, pred_ids, similarities


def native_metrics(run):
    """Native cores retain full low-IoU candidates for all HOTA thresholds."""
    args = run_arrays(run)
    return {
        "CLEAR": compute_clear_metrics(*args),
        "HOTA": compute_hota_metrics(*args),
        "Identity": compute_identity_metrics(*args),
    }


def aggregate_metrics(metrics):
    """Pool native counts/arrays, never average clip percentages."""
    return {
        "CLEAR": aggregate_clear_metrics([m["CLEAR"] for m in metrics]),
        "HOTA": aggregate_hota_metrics([m["HOTA"] for m in metrics]),
        "Identity": aggregate_identity_metrics([m["Identity"] for m in metrics]),
    }


def export_run(run, destination):
    """Dense run_position+1 MOT export with original source/PTS sidecar."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    gt_ids, pred_ids, _ = run_arrays(run)
    gt_rows, pred_rows, sidecar = [], [], []
    for position, row in enumerate(run):
        frame = row["frame"]
        sidecar.append(
            {
                "run_position": position + 1,
                "sample_index": frame["sample_index"],
                "source_frame_index": frame["source_frame_index"],
                "pts": frame["pts"],
                "time_base": frame["time_base"],
            }
        )
        for entities, ids, target, is_gt in [
            (row["prepared"]["gt"], gt_ids[position], gt_rows, True),
            (row["prepared"]["predictions"], pred_ids[position], pred_rows, False),
        ]:
            for entity, ident in zip(entities, ids, strict=True):
                x, y, right, bottom = entity["box"]
                target.append(
                    ",".join(
                        format(float(v), ".17g")
                        for v in [
                            position + 1,
                            ident + 1,
                            x + 1,
                            y + 1,
                            right - x,
                            bottom - y,
                            1 if is_gt else entity.get("confidence", 1),
                            1,
                            1,
                        ]
                    )
                )
    (destination / "gt.txt").write_text("\n".join(gt_rows) + ("\n" if gt_rows else ""))
    (destination / "tracks.txt").write_text("\n".join(pred_rows) + ("\n" if pred_rows else ""))
    (destination / "source.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    last = run[-1]["prepared"] if run else {"gt": [], "predictions": []}
    return bool(gt_rows and pred_rows and (last["gt"] or last["predictions"]))


def sensitivity_frames(frames, variant):
    """Finite GT-boundary diagnostics preserve every source person/domain/identity."""
    if variant not in ("expanded_boxes", "contracted_boxes"):
        return frames
    result = []
    for frame in frames:
        people = []
        for p in frame["people"]:
            value = dict(p)
            if p["detection_eligible"]:
                value["box"] = expanded(
                    p["box"], p["box_uncertainty_pixels"] * (1 if variant == "expanded_boxes" else -1)
                )
            people.append(value)
        result.append(frame | {"people": people})
    return result


def source_audited_accounts(frames, outputs, assignments, excluded_samples=()):
    """Secondary source-linked continuity; never replace primary IoU/native metrics.

    Every confirmed proposal needs an explicit shared visual decision. A source person is linked only when
    class/identity is unambiguous. Unknown or multiple IDs on one person censor secondary identity evidence rather than
    choosing an identity. Callers can share the union of excluded samples across policies.
    """
    accounts, prepared_frames = [], []
    for frame, output, decisions in zip(frames, outputs, assignments, strict=True):
        account = account_frame(frame, output)
        gt = {p["person_id"]: p for p in physical_people(frame)}
        linked = {}
        unresolved = frame["sample_index"] in excluded_samples
        for p in public_outputs(output):
            require(p["proposal_index"] in decisions, "missing confirmed-proposal source audit")
            decision = decisions[p["proposal_index"]]
            require(
                decision["classification"] in ["physical_person", "graphic_depiction", "nonperson", "unresolved"],
                "invalid source audit classification",
            )
            if decision["classification"] == "physical_person":
                person_id = decision.get("person_id")
                if person_id is None:
                    unresolved = True
                else:
                    require(
                        person_id in gt and gt[person_id]["identity_eligible"],
                        "source audit invented or unresolved identity",
                    )
                    linked.setdefault(person_id, []).append(p)
            elif decision["classification"] == "unresolved":
                unresolved = True
        unresolved = unresolved or any(len(v) > 1 for v in linked.values())
        account["matches"] = [
            {
                "person_id": person_id,
                "id": values[0]["id"],
                "proposal_index": values[0]["proposal_index"],
                "role": gt[person_id]["role"],
            }
            for person_id, values in sorted(linked.items())
            if len(values) == 1
        ]
        account["match_basis"] = (
            "secondary manually reviewed source identity, independent of IoU; "
            "primary localization penalties remain unchanged"
        )
        account["secondary_identity_unresolved"] = unresolved
        accounts.append(account)
        prepared_frames.append(
            frame | {"people": [p | {"identity_eligible": False} if unresolved else p for p in frame["people"]]}
        )
    return prepared_frames, accounts
