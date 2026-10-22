# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Invariants for the throwaway vertical-noise intervention."""

import numpy as np
import pytest
import supervision as sv

from experiments.jump_motion_noise import VerticalNoiseStateEstimator, noise_estimator
from trackers import OCSORTTracker
from trackers.core.ocsort.tracklet import OCSORTTracklet
from trackers.utils.predict_timing import PredictTiming
from trackers.utils.state_representations import XCYCSRStateEstimator


@pytest.mark.parametrize("step", [0.25, 0.5, 1.0, 1.02, 2.5, 5.0, 7.5])
def test_vertical_only_symmetric_psd_with_refresh(step: float) -> None:
    bbox = np.array([10.0, 20.0, 30.0, 60.0])
    baseline = OCSORTTracklet(bbox).state_estimator
    changed = OCSORTTracklet(bbox, state_estimator_class=VerticalNoiseStateEstimator).state_estimator
    reference = baseline.motion.process_noise.baseline_Q.copy()
    for _ in range(3):
        changed.set_kf_covariances(process_noise=reference)
        baseline.predict(step, frame_rate=5)
        changed.predict(step, frame_rate=5)
        expected = baseline.kf.process_noise.copy()
        expected[np.ix_([1, 5], [1, 5])] *= 4
        np.testing.assert_array_equal(changed.kf.process_noise, expected)
        np.testing.assert_array_equal(changed.kf.transition_mtx, baseline.kf.transition_mtx)
        np.testing.assert_array_equal(changed.kf.state, baseline.kf.state)
        np.testing.assert_array_equal(changed.kf.measurement_noise, baseline.kf.measurement_noise)
        np.testing.assert_allclose(changed.kf.process_noise, changed.kf.process_noise.T, atol=1e-12)
        assert float(np.linalg.eigvalsh(changed.kf.process_noise)[0]) >= -1e-10
        assert float(np.linalg.eigvalsh(changed.kf.state_covariance)[0]) >= -1e-8
        assert np.isfinite(changed.kf.state_covariance).all()
        np.testing.assert_array_equal(reference, baseline.motion.process_noise.baseline_Q)


def test_restore_and_oru_do_not_lose_or_compound_vertical_noise() -> None:
    bbox = np.array([10.0, 20.0, 30.0, 60.0])
    tracklet = OCSORTTracklet(bbox, state_estimator_class=VerticalNoiseStateEstimator)
    estimator = tracklet.state_estimator
    estimator.predict(1.0, 5)
    expected = estimator.kf.process_noise.copy()
    saved = estimator.get_state()
    estimator.predict(3.0, 5)
    estimator.set_state(saved)
    estimator.predict(1.0, 5)
    np.testing.assert_array_equal(estimator.kf.process_noise, expected)
    timing = PredictTiming(frame_step=1.0, elapsed_seconds=0.2, frame_rate=5)
    tracklet.update(bbox + np.array([0, 2, 0, 2]), timing)
    tracklet.predict(timing)
    tracklet.predict(timing)
    tracklet.update(bbox + np.array([0, 6, 0, 6]), timing)
    assert tracklet._frozen_state is None
    estimator.predict(1.0, 5)
    np.testing.assert_array_equal(estimator.kf.process_noise, expected)
    assert np.isfinite(estimator.kf.state_covariance).all()
    assert float(np.linalg.eigvalsh(estimator.kf.state_covariance)[0]) >= -1e-8


def test_disabled_estimator_is_native_and_outputs_are_exact() -> None:
    assert noise_estimator() is XCYCSRStateEstimator
    first = OCSORTTracker(frame_rate=5)
    second = OCSORTTracker(frame_rate=5, state_estimator_class=noise_estimator(False))
    for i, timestamp in enumerate([0.0, 0.2, 0.41, 0.61, 0.9, 1.1, 1.3, 1.5]):
        detections = (
            sv.Detections.empty()
            if i in (3, 4)
            else sv.Detections(
                xyxy=np.array([[10 + i, 20 + i * i, 30 + i, 60 + i * i]], dtype=float),
                confidence=np.array([0.9]),
            )
        )
        native = first.update(detections, timestamp=timestamp)
        disabled = second.update(detections, timestamp=timestamp)
        np.testing.assert_array_equal(native.xyxy, disabled.xyxy)
        np.testing.assert_array_equal(native.tracker_id, disabled.tracker_id)
        assert len(first.tracks) == len(second.tracks)
        for a, b in zip(first.tracks, second.tracks):
            for key, value in a.state_estimator.get_state().items():
                np.testing.assert_array_equal(value, b.state_estimator.get_state()[key])
