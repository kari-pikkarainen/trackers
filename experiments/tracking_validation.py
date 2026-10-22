# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Offline experiment CLI: frozen corpus, three confidence policies, metrics and synchronized evidence.

Run with python -m experiments.tracking_validation --help. This is not a public library feature or production
integration. No inference/network fallback exists.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from experiments.tracking_validation_data import load_sequences, read_json, sha256
from experiments.tracking_validation_eval import (
    SENSITIVITIES,
    account_frame,
    aggregate_metrics,
    export_run,
    jsonable,
    native_metrics,
    require,
    scoring_runs,
    sensitivity_frames,
)
from experiments.tracking_validation_events import paired_events, paired_frames, person_events, qualification, summary
from experiments.tracking_validation_timing import POLICY_NAMES, benchmark, replay_policy

ANALYSIS = {
    "policies": {
        "native06": {"birth": 0.6, "continuation": 0.6},
        "uniform04": {"birth": 0.4, "continuation": 0.4},
        "split02": {"birth": 0.6, "continuation": 0.2},
    },
    "source_visibility": (
        "Declared unresolvable principal-visibility ranges exclude/split identity scoring while "
        "retaining full all-person detection/event accounting; no identity bridge is inferred through those intervals."
    ),
    "reference_frame_rate": 5,
    "CMC": False,
    "matching": (
        "All physical people, inclusive IoU>=0.5; maximum cardinality then total IoU; "
        "canonical person_id and (box,proposal_index,id) ordering; lexicographic ties "
        "within1e-12 total-IoU equality."
    ),
    "unknowns": (
        "Union of clipped localized regions; IoA>=0.5; positive-intersection candidate "
        "protection before suppression, including duplicates. Reasons retained separately."
    ),
    "identity": (
        "Primary known-identity view B with custom localized preprocessing; reference "
        "closed-world view A. Source boundary-estimate envelopes intersecting uncertainty "
        "exclude/split identity runs. Original unexpanded GT is used for primary native IoU. "
        "Empty/negative frames retained."
    ),
    "sensitivities": list(SENSITIVITIES),
    "sensitivity_definitions": {
        "primary": "IoA0.5 localized known-identity view",
        "closed_world": (
            "Any unresolved uncertainty/identity-unknown region excludes frame; known graphic-only negatives remain"
        ),
        "full_containment": "IoA1.0 for unprotected outputs, same source-fixed eligibility",
        "expanded_boxes": "Expand known GT by each declared pixel boundary estimate; same primary source membership",
        "contracted_boxes": "Contract by declared estimate, floor0.5pixel extent; same primary source membership",
    },
    "sensitivity_interpretation": (
        "Exploratory estimated-boundary perturbations, not confidence intervals. No tuning. "
        "Qualification reversal/undefined alternative means inconclusive; never choose the "
        "favorable view."
    ),
    "qualification": (
        "Separately development and locked test: strictly higher common-denominator principal "
        "coverage; all-person recall not lower; defined primary IDF1 not lower; no increased "
        "established switches/fragments; no new established "
        "wrong-attachment/background/duplicate events or frames; unresolved potentially "
        "reversing outputs =>inconclusive. Flag every lost native-covered frame, slower "
        "assessable startup, new switch/fragment or increased compatible fully-observed "
        "interruption. Censored lower bounds do not rank incompatible durations. Per-clip "
        "regression prevents unattended promotion. Test must replicate; never pool partitions "
        "for qualification."
    ),
    "choice": (
        "Prefer existing uniform0.4 configuration if it qualifies on locked test and split has"
        " no distinct replicated benefit. Split must show greater coverage without worse "
        "identity/harm, or better continuity without less coverage/harm, versus uniform0.4 "
        "separately on locked test. Tradeoffs/undefined evidence/per-clip regressions imply no"
        " automatic winner. No activation/default change authorized."
    ),
    "undefined": (
        "Recall/coverage undefined for zero GT; precision undefined for zero outputs; "
        "IDF1/association validity requires nonzero known GT and matched identity "
        "opportunities. Preserve native numeric zero conventions separately; absence of "
        "evidence never proves identity preservation."
    ),
    "duration": (
        "Recovered interruption ends at actual observed recovery; censored interruption lower "
        "bound ends at last assessable missing observation in same visible run. First "
        "unknown/cut is separate censor boundary. One loss per established episode; "
        "never-confirmed/startup separate."
    ),
    "MOT_export": (
        "Dense run_position+1, source sample/frame/PTS sidecar. Native cores handle empties; "
        "public filepath parity only representable nonempty runs. Not whole-corpus "
        "MOTChallenge equivalence."
    ),
    "timing": (
        "Complete discarded development warmup per policy; six lexicographic policy-order "
        "permutations, fresh clip/shot trackers; monotonic ns around update only, no "
        "I/O/copy/observer/metrics; observational paired clip/pass ratios, no release speed "
        "gate."
    ),
}


def write_json(path, value):
    """Write strict finite reproducible JSON; output is always outside source."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def evaluate_sequence(sequence, replay, destination):
    """Full detection/events plus each finite predeclared native identity view."""
    frames = [item["truth"] for item in sequence["items"]]
    outputs = [row["output"] for row in replay["frames"]]
    accounts = [account_frame(f, o) for f, o in zip(frames, outputs, strict=True)]
    events = person_events(frames, accounts)
    views = {}
    raw_metrics = {}
    for variant in SENSITIVITIES:
        runs, excluded = scoring_runs(frames, outputs, variant)
        computed = [native_metrics(run) for run in runs]
        raw_metrics[variant] = computed
        run_records = []
        for index, (run, result) in enumerate(zip(runs, computed, strict=True)):
            supported = export_run(run, destination / "MOT" / variant / str(index))
            run_records.append(
                {
                    "frames": len(run),
                    "sample_indices": [row["frame"]["sample_index"] for row in run],
                    "known_gt_boxes": sum(row["prepared"]["known_gt"] for row in run),
                    "suppression": [
                        {"sample_index": row["frame"]["sample_index"], "outputs": row["prepared"]["suppressed"]}
                        for row in run
                    ],
                    "metrics": result,
                    "nonempty_file_parity_supported": supported,
                }
            )
        diagnostic_frames = sensitivity_frames(frames, variant)
        diagnostic_accounts = [
            account_frame(f, o, 1.0 if variant == "full_containment" else 0.5)
            for f, o in zip(diagnostic_frames, outputs, strict=True)
        ]
        diagnostic_events = person_events(diagnostic_frames, diagnostic_accounts)
        views[variant] = {
            "full_detection_event_sensitivity": summary(diagnostic_frames, diagnostic_accounts, diagnostic_events),
            "runs": run_records,
            "excluded_frames": excluded,
            "native_aggregate": aggregate_metrics(computed),
            "known_gt_denominator": sum(r["known_gt_boxes"] for r in run_records),
            "included_frames": sum(len(r) for r in runs),
        }
    result = {
        "clip": sequence["id"],
        "source": sequence["source"],
        "categories": sequence["categories"],
        "semantic_event_group": sequence["semantic_event_group"],
        "frames": accounts,
        "events": events,
        "summary": summary(frames, accounts, events),
        "identity_views": views,
    }
    primary = views["primary"]["native_aggregate"]
    result["summary"]["identity_idf1"] = primary["Identity"]["IDF1"] if primary["Identity"]["IDTP"] > 0 else None
    result["summary"].update(audit_complete=False, established_harms=None, per_clip_regressions=None)
    write_json(destination / "evaluation.json", result)
    return result, raw_metrics


def grouped_summary(results, metrics):
    """Native aggregate by clip/source/category/physical-event group with pooled counts."""
    groups = defaultdict(list)
    for index, result in enumerate(results):
        groups["overall", "selected_partition"].append(index)
        groups["source", result["source"]].append(index)
        groups["physical_event_group", result["semantic_event_group"]].append(index)
        for category in result["categories"]:
            groups["category", category].append(index)
    out = []
    for (kind, name), indices in sorted(groups.items()):
        pooled = {
            key: sum(results[i]["summary"][key] for i in indices)
            for key in [
                "sampled_frames",
                "physical_person_frames",
                "matched_person_frames",
                "principal_person_frames",
                "principal_covered_frames",
                "public_outputs",
                "established_losses",
                "right_censored_losses",
                "observed_loss_seconds_sum",
                "fragmentations",
                "id_switches",
                "same_id_recoveries",
                "different_id_recoveries",
                "never_confirmed_runs",
            ]
        }
        pooled["principal_coverage"] = (
            pooled["principal_covered_frames"] / pooled["principal_person_frames"]
            if pooled["principal_person_frames"]
            else None
        )
        pooled["all_person_recall"] = (
            pooled["matched_person_frames"] / pooled["physical_person_frames"]
            if pooled["physical_person_frames"]
            else None
        )
        native = {
            variant: aggregate_metrics([m for i in indices for m in metrics[i][variant]]) for variant in SENSITIVITIES
        }
        pooled["identity_idf1"] = (
            native["primary"]["Identity"]["IDF1"] if native["primary"]["Identity"]["IDTP"] > 0 else None
        )
        out.append(
            {
                "group_kind": kind,
                "group": name,
                "clips": [results[i]["clip"] for i in indices],
                "counts": pooled,
                "identity_views": native,
            }
        )
    return out


def render_video(sequence, replays, destination):
    """Synchronized three-panel sampled playback; actual source PTS stays visible/sidecar."""
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{sequence['id']}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (1920, 400))
    require(writer.isOpened(), "video encoder unavailable")
    sidecar = []
    try:
        for index, item in enumerate(sequence["items"]):
            frame = item["truth"]
            raw = cv2.imread(frame["image_path"])
            require(raw is not None and raw.shape[:2] == (1080, 1920), "source decode failed")
            panels = []
            for policy in POLICY_NAMES:
                image = raw.copy()
                for p in frame["people"]:
                    x1, y1, x2, y2 = map(round, p["box"])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for p in replays[policy]["frames"][index]["output"]:
                    if p["id"] >= 0:
                        x1, y1, x2, y2 = map(round, p["box"])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(
                            image, str(p["id"]), (x1, max(y1 - 4, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1
                        )
                panel = np.zeros((400, 640, 3), np.uint8)
                panel[40:] = cv2.resize(image, (640, 360))
                cv2.putText(
                    panel,
                    f"{policy}  sample{frame['sample_index']}  PTS{frame['pts']}",
                    (5, 23),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
                panels.append(panel)
            writer.write(np.concatenate(panels, axis=1))
            sidecar.append(
                {k: frame[k] for k in ["sample_index", "source_frame_index", "pts", "time_base", "timestamp", "shot"]}
            )
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    require(decoded == len(sequence["items"]), "encoded video frame mismatch")
    write_json(
        destination / f"{sequence['id']}.json",
        {
            "frame_count": decoded,
            "panel_order": list(POLICY_NAMES),
            "playback": (
                "5Hz sampled playback, not variable-PTS real-time playback; synchronized panel indices/actual PTS match"
            ),
            "frames": sidecar,
        },
    )


def main():
    """Run the generic frozen offline CLI; P2 command selects development only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=["development", "test", "all"], default="development")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup-detections", type=Path)
    parser.add_argument("--no-videos", action="store_true")
    args = parser.parse_args()
    require(args.output.is_absolute(), "absolute output path required")
    source = Path(__file__).resolve().parents[1]
    require(
        not args.output.resolve().is_relative_to(args.corpus.resolve().parent), "output must not enter frozen admission"
    )
    freeze_path = args.corpus.parent / "truth-freeze.json"
    freeze = read_json(freeze_path)
    for row in freeze["files"]:
        require(sha256(row["path"]) == row["sha256"], "frozen input changed")
    require(not args.output.resolve().is_relative_to(source), "outputs must stay outside source")
    require(not args.output.exists() or not any(args.output.iterdir()), "output must be a fresh directory")
    args.output.mkdir(parents=True, exist_ok=True)
    # Publish finite definitions before any policy result, without modifying truth.
    write_json(args.output / "analysis-definitions.json", ANALYSIS)
    preload_start = time.perf_counter_ns()
    sequences, inputs = load_sequences(args.corpus, args.detections, args.partition)
    inputs.append({"path": str(freeze_path), "sha256": sha256(freeze_path)})
    preload_ns = time.perf_counter_ns() - preload_start
    ordinary = {policy: replay_policy(sequences, policy, diagnostic=True) for policy in POLICY_NAMES}
    all_results, all_metrics = {}, {}
    for policy in POLICY_NAMES:
        results, computed = [], []
        for sequence, replay in zip(sequences, ordinary[policy]["predictions"], strict=True):
            directory = args.output / policy / sequence["id"]
            write_json(directory / "trajectories.json", replay)
            result, metrics = evaluate_sequence(sequence, replay, directory)
            results.append(result)
            computed.append(metrics)
        all_results[policy], all_metrics[policy] = results, computed
        write_json(args.output / policy / "groups.json", grouped_summary(results, computed))
    for index in range(len(sequences)):
        for variant in SENSITIVITIES:
            expected = all_results["native06"][index]["identity_views"][variant]
            for policy in POLICY_NAMES:
                value = all_results[policy][index]["identity_views"][variant]
                require(
                    value["known_gt_denominator"] == expected["known_gt_denominator"]
                    and value["included_frames"] == expected["included_frames"]
                    and value["excluded_frames"] == expected["excluded_frames"]
                    and [r["sample_indices"] for r in value["runs"]] == [r["sample_indices"] for r in expected["runs"]],
                    "policy-dependent identity denominator/run membership",
                )
    differences = []
    for candidate, baseline in [("uniform04", "native06"), ("split02", "native06"), ("split02", "uniform04")]:
        for a, b in zip(all_results[baseline], all_results[candidate], strict=True):
            differences.append(
                {
                    "baseline": baseline,
                    "candidate": candidate,
                    "clip": a["clip"],
                    **paired_frames(a["frames"], b["frames"]),
                    **paired_events(a["events"], b["events"]),
                    "qualification": qualification(a["summary"], b["summary"]),
                }
            )
    write_json(args.output / "differences.json", differences)
    if args.benchmark:
        if args.partition == "development":
            warmup = sequences
        else:
            require(
                args.warmup_detections is not None,
                "explicit development warmup detections required outside development",
            )
            warmup, warm_inputs = load_sequences(args.corpus, args.warmup_detections, "development")
            inputs.extend(warm_inputs)
        write_json(args.output / "timing.json", benchmark(sequences, warmup, ordinary))
    if not args.no_videos:
        for index, sequence in enumerate(sequences):
            render_video(sequence, {p: ordinary[p]["predictions"][index] for p in POLICY_NAMES}, args.output / "videos")
    modules = [
        *sorted(source.glob("experiments/tracking_validation*.py")),
        source / "experiments/continuation_confidence.py",
    ]
    write_json(
        args.output / "run.json",
        {
            "partition": args.partition,
            "policies": list(POLICY_NAMES),
            "clips": len(sequences),
            "frames": sum(len(s["items"]) for s in sequences),
            "inputs": inputs,
            "source_modules": [{"path": str(p), "sha256": sha256(p)} for p in modules],
            "python": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "opencv_threads": cv2.getNumThreads(),
            "trackers_import": __import__("trackers").__file__,
            "versions": {
                p: __import__("importlib.metadata", fromlist=["version"]).version(p)
                for p in ["numpy", "scipy", "supervision", "opencv-python", "trackers"]
            },
            "thread_environment": {
                k: os.environ.get(k)
                for k in [
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]
            },
            "preload_ns": preload_ns,
            "measurement_reproducibility": (
                "Prediction/evaluation JSON exact; measured preload/timing/wall durations "
                "intentionally observational, not byte-reproduction targets."
            ),
            "source_fixed_identity_denominators": True,
            "no_inference": True,
        },
    )
    print(
        json.dumps(
            {"output": str(args.output), "clips": len(sequences), "frames": sum(len(s["items"]) for s in sequences)}
        )
    )


if __name__ == "__main__":
    main()
