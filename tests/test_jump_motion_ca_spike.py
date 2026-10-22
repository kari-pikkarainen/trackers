# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Synthetic invariants for the throwaway OC-SORT acceleration experiment."""

import numpy as np
import pytest
import supervision as sv

from experiments.jump_motion_ca import NATIVE_INDICES, VerticalAccelerationStateEstimator, ca_estimator
from trackers import OCSORTTracker
from trackers.core.ocsort.tracklet import OCSORTTracklet
from trackers.utils.predict_timing import PredictTiming
from trackers.utils.state_representations import XCYCSRStateEstimator

BBOX = np.array([10.0, 20.0, 30.0, 60.0])


def estimator() -> VerticalAccelerationStateEstimator:
    result = OCSORTTracklet(BBOX, state_estimator_class=VerticalAccelerationStateEstimator).state_estimator
    assert isinstance(result, VerticalAccelerationStateEstimator)
    return result


@pytest.mark.parametrize("fps", [5, 30, 60])
def test_analytic_irregular_acceleration(fps: int) -> None:
    changed = estimator()
    changed.kf.state[4, 0] = 12
    changed.kf.state[6, 0] = 18 / fps
    total = 0.0
    for dt in [0.0, 0.013, 0.033, 0.2, 0.017, 0.4]:
        changed.predict(dt * fps, fps)
        total += dt
        assert changed.kf.state[1, 0] == pytest.approx(40 + 18 * total + 6 * total**2)
        assert changed.kf.state[6, 0] == pytest.approx((18 + 12 * total) / fps)
        assert changed.kf.state[4, 0] == 12
        np.testing.assert_allclose(changed.kf.state_covariance, changed.kf.state_covariance.T, atol=1e-9)
        assert np.linalg.eigvalsh(changed.kf.state_covariance)[0] >= -1e-8


@pytest.mark.parametrize("step", [0, 0.25, 1, 1.02, 2.5, 5, 7.5])
def test_exact_native_mapped_blocks_and_zero_acceleration_mean(step: float) -> None:
    native = OCSORTTracklet(BBOX).state_estimator
    changed = estimator()
    mapping = np.ix_(NATIVE_INDICES, NATIVE_INDICES)
    np.testing.assert_array_equal(changed.kf.state_covariance[mapping], native.kf.state_covariance)
    np.testing.assert_array_equal(changed.kf.process_noise[mapping], native.kf.process_noise)
    np.testing.assert_array_equal(changed.kf.measurement_noise, native.kf.measurement_noise)
    native.kf.state[4:, 0] = [2, -3, 4]
    changed.kf.state[NATIVE_INDICES, 0] = native.kf.state[:, 0]
    native.predict(step, 5)
    changed.predict(step, 5)
    np.testing.assert_array_equal(changed.kf.state[NATIVE_INDICES], native.kf.state)
    np.testing.assert_array_equal(changed.kf.transition_mtx[mapping], native.kf.transition_mtx)
    np.testing.assert_array_equal(changed.kf.process_noise[mapping], native.kf.process_noise)
    # Acceleration uncertainty changes only vertical marginals/cross-covariances.
    stable = [0, 2, 3, 4, 6]
    mapped = NATIVE_INDICES[stable]
    np.testing.assert_array_equal(
        changed.kf.state_covariance[np.ix_(mapped, mapped)], native.kf.state_covariance[np.ix_(stable, stable)]
    )
    assert np.linalg.eigvalsh(changed.kf.process_noise)[0] >= -1e-10


@pytest.mark.parametrize("fps", [30, 60])
@pytest.mark.parametrize("sign", [-1, 1])
def test_learns_signed_acceleration_from_observations(fps: int, sign: int) -> None:
    changed = estimator()
    for i in range(1, fps * 10 + 1):
        time = i / fps
        changed.predict(1, fps)
        changed.update(BBOX + np.array([0, sign * 6 * time**2, 0, sign * 6 * time**2]))
    # Retained native per-frame velocity Q slows identification at high FPS.
    # Require meaningful signed learning, not an unjustified convergence rate.
    assert 3 < sign * changed.kf.state[4, 0] < 15
    assert np.isfinite(changed.kf.state_covariance).all()


def test_scale_clamp_restore_cache_and_oru() -> None:
    track = OCSORTTracklet(BBOX, state_estimator_class=VerticalAccelerationStateEstimator)
    changed = track.state_estimator
    assert isinstance(changed, VerticalAccelerationStateEstimator)
    changed.kf.state[7, 0] = -1000
    changed.kf.state[6, 0] = -123
    changed.clamp_velocity(2)
    assert changed.kf.state[7, 0] == 0
    assert changed.kf.state[6, 0] == -123
    changed.kf.state[4, 0] = 7
    changed.predict(1, 60)
    saved = changed.get_state()
    changed.predict(3, 60)
    changed.set_state(saved)
    for key, value in saved.items():
        np.testing.assert_array_equal(changed.get_state()[key], value)
    assert changed.motion.cached_step is None
    changed.predict(1)
    assert changed.motion.reference_fps == 60
    timing = PredictTiming(frame_step=1, elapsed_seconds=1 / 60, frame_rate=60)
    track.update(BBOX, timing)
    track.predict(timing)
    track.predict(timing)
    assert track._frozen_state is not None
    assert track._frozen_state["state"].shape == (8, 1)
    track.update(BBOX + np.array([0, 6, 0, 6]), timing)
    assert track._frozen_state is None
    assert changed.motion.reference_fps == 60
    assert np.isfinite(changed.kf.state_covariance).all()
    assert np.linalg.eigvalsh(changed.kf.state_covariance)[0] >= -1e-8


def test_zero_acceleration_uncertainty_nests_cv_through_refresh_and_updates() -> None:
    native = OCSORTTracklet(BBOX).state_estimator
    changed = estimator()
    covariance = changed.kf.state_covariance.copy()
    covariance[4, 4] = 0
    noise = changed.kf.process_noise.copy()
    noise[4, 4] = 0
    changed.set_kf_covariances(process_noise=noise, state_covariance=covariance)
    mapping = np.ix_(NATIVE_INDICES, NATIVE_INDICES)
    for i, step in enumerate([1.0, 0.3, 2.0, 1.0, 4.0]):
        # Recalibration must not move the native scale-velocity tuning to ay.
        changed.set_kf_covariances(process_noise=noise.copy())
        native.predict(step, 5)
        changed.predict(step, 5)
        box = BBOX + np.array([i, i * i, i, i * i])
        native.update(box)
        changed.update(box)
        np.testing.assert_allclose(changed.kf.state[NATIVE_INDICES], native.kf.state, atol=1e-10)
        np.testing.assert_allclose(changed.kf.state_covariance[mapping], native.kf.state_covariance, atol=1e-10)
        assert changed.kf.state[4, 0] == 0


def test_disabled_parity_and_ca_empty_duplicate_timestamps() -> None:
    assert ca_estimator() is XCYCSRStateEstimator
    trackers = [
        OCSORTTracker(frame_rate=5),
        OCSORTTracker(frame_rate=5, state_estimator_class=ca_estimator(False)),
        OCSORTTracker(frame_rate=5, state_estimator_class=ca_estimator(True)),
    ]
    for i, timestamp in enumerate([0, 0.2, 0.4, 0.4, 0.61, 0.9, 1.1, 1.3, 1.5]):
        detections = (
            sv.Detections.empty()
            if i in [0, 4, 5]
            else sv.Detections(xyxy=np.array([BBOX + np.array([0, i * i, 0, i * i])]), confidence=np.array([0.9]))
        )
        results = [tracker.update(detections, timestamp=timestamp) for tracker in trackers]
        np.testing.assert_array_equal(results[0].xyxy, results[1].xyxy)
        np.testing.assert_array_equal(results[0].tracker_id, results[1].tracker_id)
        for left, right in zip(trackers[0].tracks, trackers[1].tracks):
            for key, value in left.state_estimator.get_state().items():
                np.testing.assert_array_equal(value, right.state_estimator.get_state()[key])
        for track in trackers[2].tracks:
            assert np.isfinite(track.state_estimator.kf.state_covariance).all()


@pytest.mark.parametrize("step,fps", [(-1, 5), (float("nan"), 5), (1, 0), (1, float("inf"))])
def test_invalid_timing_rejected(step: float, fps: float) -> None:
    with pytest.raises(ValueError):
        estimator().predict(step, fps)
