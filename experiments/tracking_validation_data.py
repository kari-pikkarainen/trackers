# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Frozen offline corpus/schema/saved-proposal admission; no inference fallback."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import supervision as sv

from experiments.tracking_validation_eval import boxes, require


def read_json(path):
    """Read a local explicit input; this module has no model or network imports."""
    return json.loads(Path(path).read_text())


def sha256(path):
    """Stream large source files without materializing their bytes in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_clips(corpus, partition="development"):
    """Generic explicit partition API; P2 executions use development only."""
    require(partition in ["development", "test", "all"], "unknown partition")
    return [c for c in corpus["clips"] if partition == "all" or c["partition"] == partition]


def validate_labels(labels, schema):
    """Enforce the admitted schema, domains, source geometry and exact PTS."""
    previous = -float("inf")
    for n, frame in enumerate(labels["frames"]):
        require(set(schema["frame_required"]).issubset(frame), "missing required frame field")
        require(
            {"timestamp", "pts", "time_base", "source_frame_index", "width", "height"}.issubset(frame),
            "missing source timing/dimension field",
        )
        require(frame["phase"] in schema["phase"], "invalid phase")
        require(frame["sample_index"] == n, "noncontiguous clip sample index")
        stamp = float(Fraction(frame["time_base"]) * frame["pts"])
        require(
            np.isfinite(stamp) and abs(stamp - frame["timestamp"]) <= 1e-12 and stamp > previous,
            "missing/invalid/nonincreasing source timestamp",
        )
        require((frame["width"], frame["height"]) == (1920, 1080), "unsupported source dimensions")
        previous = stamp
        identities = []
        for person in frame["people"]:
            require(set(schema["person_required"]).issubset(person), "missing required person field")
            for name in ["role", "visibility", "source_medium", "evaluation_domain"]:
                require(person[name] in schema[name], f"invalid {name}")
            boxes([person["box"]])
            physical = person["evaluation_domain"] == "physical_person"
            require(person["detection_eligible"] == physical, "physical-domain eligibility mismatch")
            require(
                person["identity_eligible"] == (physical and person["identity_status"] == "known"),
                "identity eligibility mismatch",
            )
            require(
                person["person_id"].startswith(f"{labels['clip']}/shot{frame['shot']}/"), "cross-shot person identity"
            )
            identities.append(person["person_id"])
            if physical:
                require(
                    person["source_medium"] == "live_video" and person["role"] != "depicted_person",
                    "physical medium mismatch",
                )
            else:
                require(
                    person["role"] == "depicted_person"
                    and person["source_medium"] == "graphic_photo"
                    and bool(person["out_of_domain_ignore_reason"]),
                    "depiction domain mismatch",
                )
        require(len(identities) == len(set(identities)), "duplicate GT identity")
        for region in frame["uncertainties"]:
            require(set(schema["uncertainty_required"]).issubset(region), "missing required uncertainty field")
            boxes([region["box"]])
            require(region["identity_status"] == "unknown" and bool(region["reason"]), "invalid uncertainty")


def _metadata(row):
    return row.get("frame", row)


def _key(frame):
    return (
        frame["image_sha256"],
        frame["source_frame_index"],
        frame.get("pts", frame.get("source_pts")),
        frame.get("time_base", frame.get("source_time_base")),
    )


def load_sequences(corpus_path, detection_path, partition="development"):
    """Preload only selected labels/proposals; missing saved detections fail clearly.

    Saved detections accept the admitted legacy registry/receipt index or a generic version1 manifest with source/image
    hashes, frame/PTS/timebase and raw_proposals. Later explicit test/all use requires no source edit; P2 must not
    execute it.
    """
    corpus_path, detection_path = Path(corpus_path), Path(detection_path)
    corpus = read_json(corpus_path)
    schema_path = corpus_path.parent / "schema.json"
    schema = read_json(schema_path)
    require(schema["version"] == 4 and "other_person" in schema["role"], "unsupported admitted schema")
    saved = read_json(detection_path)
    index = {}
    for row in saved["frames"]:
        key = _key(_metadata(row))
        require(key not in index, "duplicate saved source-frame key")
        index[key] = row
    consumed = {str(corpus_path), str(schema_path), str(detection_path)}
    sequences = []
    checked_sources = set()
    for clip in select_clips(corpus, partition):
        source_path = corpus["sources"][clip["source"]]["path"]
        if source_path not in checked_sources:
            require(sha256(source_path) == clip["source_sha256"], "source video hash mismatch")
            consumed.add(source_path)
            checked_sources.add(source_path)
        label_path = Path(clip["label_path"])
        require(sha256(label_path) == clip["label_sha256"], "label hash mismatch")
        label = read_json(label_path)
        validate_labels(label, schema)
        require(label["partition"] == clip["partition"], "label partition mismatch")
        require(len(label["frames"]) == clip["frames"], "clip frame count mismatch")
        consumed.add(str(label_path))
        items = []
        for original_frame in label["frames"]:
            frame = dict(original_frame)
            frame["identity_visibility_unknown"] = any(
                lo <= frame["sample_index"] <= hi for lo, hi, reason in label.get("principal_visibility_ranges", [])
            )
            key = _key(frame)
            require(
                key in index,
                f"missing saved detections for {clip['id']}/{frame['sample_index']}; inference is disabled",
            )
            row = index[key]
            metadata = _metadata(row)
            require(metadata["source_sha256"] == clip["source_sha256"], "detection source hash mismatch")
            require(sha256(frame["image_path"]) == frame["image_sha256"], "source image hash mismatch")
            consumed.add(frame["image_path"])
            if "receipt_path" in row:
                path = Path(row["receipt_path"])
                require(sha256(path) == row["receipt_sha256"], "raw receipt hash mismatch")
                receipt = read_json(path)
                require(receipt["frame"] == metadata, "receipt source binding mismatch")
                settings = receipt["features"]["settings"]
                require(
                    settings["prompt"] == "a kite. a person."
                    and settings["box_threshold"] == 0.1
                    and settings["text_threshold"] == 0.1,
                    "legacy detector contract mismatch",
                )
                raw = receipt["features"]["raw_proposals"]
                consumed.add(str(path))
            else:
                settings = row.get("settings", saved.get("settings", {}))
                require(
                    settings.get("prompt") == "a kite. a person."
                    and settings.get("box_threshold") == 0.1
                    and settings.get("text_threshold") == 0.1,
                    "generic detector contract mismatch",
                )
                raw = row["raw_proposals"]
            riders = [(i, p) for i, p in enumerate(raw) if p["kind"] == "rider"]
            coordinates = boxes([p["box"] for _, p in riders])
            scores = np.asarray([p["confidence"] for _, p in riders], float)
            require(np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all(), "invalid raw confidence")
            detections = sv.Detections(
                xyxy=coordinates,
                confidence=scores,
                class_id=np.zeros(len(riders), int),
                data={"proposal_index": np.array([i for i, _ in riders], int)},
            )
            items.append({"truth": frame, "detections": detections})
        sequences.append(
            {
                "id": clip["id"],
                "source": clip["source"],
                "partition": clip["partition"],
                "categories": clip["categories"],
                "semantic_event_group": clip["semantic_event_group"],
                "items": items,
            }
        )
    require(bool(sequences), "empty selected partition")
    return sequences, [{"path": path, "sha256": sha256(path)} for path in sorted(consumed)]
