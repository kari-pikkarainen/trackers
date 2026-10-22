# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Admission-bound original panel adapter, preserving its two actual saved-proposal contracts."""

from __future__ import annotations

import copy
from fractions import Fraction
from pathlib import Path

import numpy as np
import supervision as sv

from experiments.tracking_validation_data import read_json, sha256, validate_labels
from experiments.tracking_validation_eval import boxes, require


def proposals_to_detections(raw):
    """Use recognized rider rows only, retaining raw row order, geometry, score and index."""
    riders = [(i, p) for i, p in enumerate(raw) if p["kind"] == "rider"]
    scores = np.asarray([p["confidence"] for _, p in riders], float)
    require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), "invalid saved confidence")
    return sv.Detections(
        xyxy=boxes([p["box"] for _, p in riders]),
        confidence=scores,
        class_id=np.zeros(len(riders), int),
        data={"proposal_index": np.array([i for i, _ in riders], int)},
    )


def original_receipt(frame, receipt):
    """Validate native receipt metadata; never synthesize generic settings or recover dropped proposals."""
    day1 = frame["source"] == "day1"
    if day1:
        saved = receipt["frame"]
        settings = receipt["features"]["settings"]
        raw = receipt["features"]["raw_proposals"]
        require(saved["image_sha256"] == frame["image_sha256"], "Day1 image binding differs")
        require(saved["source_sha256"] == frame["source_sha256"], "Day1 source binding differs")
    else:
        require(frame["source"] == "mykonos", "unadmitted original source")
        saved, settings, raw = receipt, receipt, receipt["proposals"]
    require(
        settings["prompt"] == "a kite. a person."
        and settings["box_threshold"] == 0.1
        and settings["text_threshold"] == 0.1,
        "actual saved detector settings differ",
    )
    require(
        saved["source_frame_index"] == frame["source_frame_index"]
        and saved["source_pts"] == frame["pts"]
        and saved["source_time_base"] == frame["time_base"],
        "receipt source-clock binding differs",
    )
    require(saved["image_path"] == frame["image_path"], "receipt image path differs")
    return raw, "full_saved_postprocessed_rows" if day1 else "recognized_only_historical_rows"


def load_original_panel(phase1, *, manifest_sha256, schema_path, schema_sha256):
    """Read only manifest-bound labels/panel/selection and hash all consumed external inputs.

    B1 source admission remains a prerequisite, not a replacement for runtime image/receipt bindings. Different original
    and validation clip histories remain separate, including exact overlapping frames.
    """
    root = Path(phase1)
    manifest_path = root / "delivery-manifest.json"
    require(sha256(manifest_path) == manifest_sha256, "unaccepted admission manifest")
    manifest = read_json(manifest_path)
    index = {row["path"]: row["sha256"] for row in manifest["files"]}
    consumed = {str(manifest_path): manifest_sha256}

    def admitted(relative):
        path = root / relative
        require(relative in index and sha256(path) == index[relative], "admission file changed: " + relative)
        consumed[str(path)] = index[relative]
        return read_json(path)

    require(sha256(schema_path) == schema_sha256, "label schema changed")
    consumed[str(schema_path)] = schema_sha256
    schema = read_json(schema_path)
    panel = admitted("original-panel.json")
    selection = admitted("stress-selection.json")
    groups = {r["clip"]: r["event_group"] for r in selection["selected"] if r["panel"] == "original"}
    sequences = []
    for clip in panel["clips"]:
        labels = admitted(f"admission/labels/{clip['id']}.json")
        validate_labels(labels, schema)
        require(len(clip["frames"]) == len(labels["frames"]), "original label/panel length differs")
        items = []
        for metadata, truth in zip(clip["frames"], labels["frames"], strict=True):
            for key in (
                "sample_index",
                "source_frame_index",
                "pts",
                "time_base",
                "image_path",
                "image_sha256",
                "receipt_path",
                "receipt_sha256",
                "width",
                "height",
            ):
                require(metadata[key] == truth[key], "original label/panel binding differs: " + key)
            for key, digest in (("image_path", "image_sha256"), ("receipt_path", "receipt_sha256")):
                require(sha256(metadata[key]) == metadata[digest], "original input changed: " + key)
                consumed[metadata[key]] = metadata[digest]
            receipt = read_json(metadata["receipt_path"])
            raw, visibility = original_receipt(metadata, receipt)
            require(truth["timestamp"] == float(truth["pts"] * Fraction(truth["time_base"])), "truth clock differs")
            items.append(
                {
                    "truth": copy.deepcopy(truth),
                    "detections": proposals_to_detections(raw),
                    "raw_saved_proposals": raw,
                    "raw_visibility": visibility,
                }
            )
        sequences.append(
            {
                "id": clip["id"],
                "source": clip["source"],
                "panel": "original",
                "partition": "development_exposed",
                "categories": ["original_jump_panel"],
                "semantic_event_group": groups[clip["id"]],
                "items": items,
            }
        )
    return sequences, [{"path": p, "sha256": s} for p, s in sorted(consumed.items())]
