# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Synthetic partition/schema/saved-detection admission; no real held-out predictions."""

import copy
import hashlib
import json

import pytest

from experiments.tracking_validation_data import load_sequences, select_clips, validate_labels


def write(path, data):
    path.write_text(json.dumps(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_corpus(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic source provenance, not a real held-out video")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    image = tmp_path / "frame.bin"
    image.write_bytes(b"synthetic decoded pixels; loader hashes, rendering is tested separately")
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    schema = {
        "version": 4,
        "frame_required": ["sample_index", "phase", "people", "uncertainties"],
        "person_required": ["person_id", "box", "role"],
        "uncertainty_required": ["box", "identity_status", "reason"],
        "phase": ["unknown"],
        "role": ["principal_rider", "other_person"],
        "visibility": ["visible"],
        "source_medium": ["live_video"],
        "evaluation_domain": ["physical_person"],
    }
    write(tmp_path / "schema.json", schema)
    f = {
        "sample_index": 0,
        "source_frame_index": 42,
        "pts": 200,
        "time_base": "1/1000",
        "timestamp": 0.2,
        "width": 1920,
        "height": 1080,
        "shot": 0,
        "phase": "unknown",
        "people": [],
        "uncertainties": [],
        "image_path": str(image),
        "image_sha256": image_sha,
    }
    labels = tmp_path / "d.json"
    label_sha = write(labels, {"clip": "d", "partition": "development", "frames": [f]})
    corpus = {
        "sources": {"synthetic": {"path": str(source)}},
        "clips": [
            {
                "id": "d",
                "source": "synthetic",
                "source_sha256": source_sha,
                "partition": "development",
                "frames": 1,
                "label_path": str(labels),
                "label_sha256": label_sha,
                "categories": ["negative"],
                "semantic_event_group": "synthetic",
            },
            {"id": "t", "partition": "test", "label_path": str(tmp_path / "MUST_NOT_OPEN_TEST_LABELS.json")},
        ],
    }
    write(tmp_path / "corpus.json", corpus)
    saved = {
        "settings": {"prompt": "a kite. a person.", "box_threshold": 0.1, "text_threshold": 0.1},
        "frames": [
            {
                "image_sha256": image_sha,
                "source_frame_index": 42,
                "pts": 200,
                "time_base": "1/1000",
                "source_sha256": source_sha,
                "raw_proposals": [
                    {"kind": "rider", "box": [10, 10, 30, 50], "confidence": 0.4},
                    {"kind": "kite", "box": [100, 100, 120, 130], "confidence": 0.8},
                ],
            }
        ],
    }
    write(tmp_path / "detections.json", saved)
    return corpus, schema, f, saved


def test_default_dev_does_not_open_test_labels_and_preserves_raw_scores(tmp_path):
    synthetic_corpus(tmp_path)
    sequences, consumed = load_sequences(tmp_path / "corpus.json", tmp_path / "detections.json")
    assert len(sequences) == 1 and sequences[0]["partition"] == "development"
    detections = sequences[0]["items"][0]["detections"]
    assert detections.confidence.tolist() == [0.4] and detections.data["proposal_index"].tolist() == [0]
    assert not any("TEST_LABELS" in p["path"] for p in consumed)


def test_generic_partition_selection_requires_no_source_edit(tmp_path):
    corpus, _, _, _ = synthetic_corpus(tmp_path)
    assert [c["id"] for c in select_clips(corpus, "test")] == ["t"]
    assert [c["id"] for c in select_clips(corpus, "all")] == ["d", "t"]
    with pytest.raises(ValueError):
        select_clips(corpus, "invalid")


@pytest.mark.parametrize(
    "mutation", ["missing_detection", "wrong_source", "wrong_settings", "duplicate_source", "invalid_score"]
)
def test_saved_detections_fail_without_inference_fallback(tmp_path, mutation):
    _, _, _, saved = synthetic_corpus(tmp_path)
    if mutation == "missing_detection":
        saved["frames"] = []
    elif mutation == "wrong_source":
        saved["frames"][0]["source_sha256"] = "wrong"
    elif mutation == "wrong_settings":
        saved["settings"]["box_threshold"] = 0.2
    elif mutation == "duplicate_source":
        saved["frames"].append(copy.deepcopy(saved["frames"][0]))
    else:
        saved["frames"][0]["raw_proposals"][0]["confidence"] = float("nan")
    write(tmp_path / "detections.json", saved)
    with pytest.raises(ValueError):
        load_sequences(tmp_path / "corpus.json", tmp_path / "detections.json")


@pytest.mark.parametrize(
    "mutation", ["missing_phase", "invalid_phase", "missing_timestamp", "inconsistent_pts", "nonincreasing_time"]
)
def test_timestamp_and_schema_admission(tmp_path, mutation):
    _, schema, f, _ = synthetic_corpus(tmp_path)
    frames = [f]
    if mutation == "missing_phase":
        del f["phase"]
    elif mutation == "invalid_phase":
        f["phase"] = "made_up"
    elif mutation == "missing_timestamp":
        del f["timestamp"]
    elif mutation == "inconsistent_pts":
        f["timestamp"] += 0.01
    else:
        frames.append(copy.deepcopy(f))
        frames[1]["sample_index"] = 1
    with pytest.raises(ValueError):
        validate_labels({"clip": "d", "frames": frames}, schema)


def test_one_command_synthetic_run_preserves_locked_labels_and_synchronized_video(tmp_path, monkeypatch):
    from PIL import Image

    from experiments.tracking_validation import main

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _, _, _, saved = synthetic_corpus(inputs)
    image = inputs / "frame.bin"
    Image.new("RGB", (1920, 1080), "black").save(image, format="PNG")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    labels = json.loads((inputs / "d.json").read_text())
    labels["frames"][0]["image_sha256"] = digest
    label_sha = write(inputs / "d.json", labels)
    corpus = json.loads((inputs / "corpus.json").read_text())
    corpus["clips"][0]["label_sha256"] = label_sha
    write(inputs / "corpus.json", corpus)
    saved["frames"][0]["image_sha256"] = digest
    write(inputs / "detections.json", saved)
    frozen_paths = [inputs / p for p in ["corpus.json", "schema.json", "d.json"]]
    write(
        inputs / "truth-freeze.json",
        {"files": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in frozen_paths]},
    )
    output = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tracking_validation",
            "--corpus",
            str(inputs / "corpus.json"),
            "--detections",
            str(inputs / "detections.json"),
            "--output",
            str(output),
        ],
    )
    main()
    result = json.loads((output / "run.json").read_text())
    assert result["partition"] == "development" and result["frames"] == 1
    video = json.loads((output / "videos/d.json").read_text())
    assert video["frame_count"] == 1 and video["panel_order"] == ["native06", "uniform04", "split02"]
    assert not (inputs / "MUST_NOT_OPEN_TEST_LABELS.json").exists()
