# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Throwaway estimated vertical acceleration for OC-SORT with CMC disabled."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackers.utils.converters import xyxy_to_xcycsr
from trackers.utils.kalman_filter import KalmanFilter
from trackers.utils.motion_models import KalmanMotionModel
from trackers.utils.state_representations import XCYCSRStateEstimator

ACCELERATION_PRIOR_HEIGHT_FRACTION = 0.5
ACCELERATION_WALK_HEIGHT_FRACTION = 0.05
NATIVE_INDICES = np.array([0, 1, 2, 3, 5, 6, 7])


@dataclass
class VerticalAccelerationMotion(KalmanMotionModel):
    """Keep native velocity in pixels/frame, and acceleration in pixels/second²."""

    reference_fps: float = 5.0

    def apply(self, kf: KalmanFilter, frame_step: float, frame_rate: float | None = None) -> None:
        """Use physical dt for acceleration and remember FPS for native ORU replay."""
        if not np.isfinite(frame_step) or frame_step < 0:
            raise ValueError("frame_step must be finite and nonnegative")
        if frame_rate is not None:
            if not np.isfinite(frame_rate) or frame_rate <= 0:
                raise ValueError("frame_rate must be finite and positive")
            if frame_rate != self.reference_fps:
                self.reset_cache()
            self.reference_fps = float(frame_rate)
        # Native Q still receives None during ORU, preserving its fallback band.
        super().apply(kf, frame_step, frame_rate)
        dt = frame_step / self.reference_fps
        transition = kf.transition_mtx.copy()
        transition[1, 4] = 0.5 * dt * dt
        transition[6, 4] = dt / self.reference_fps
        noise = kf.process_noise.copy()
        # Independent acceleration random walk; original seven-state Q is exact.
        noise[4, 4] = self.process_noise.baseline_Q[4, 4] * dt
        kf.transition_mtx = transition
        kf.process_noise = noise


class VerticalAccelerationStateEstimator(XCYCSRStateEstimator):
    """State [xc,yc,s,r,ay,vx,vy,vs]; no fixed gravity or production API change.

    Only default OC-SORT without CMC is supported. Reference FPS is fixed per tracker; changing FPS does not rescale an
    already learned native velocity.
    """

    motion: VerticalAccelerationMotion

    _VEL_IDX = np.array([5, 6, 7], dtype=np.int64)

    def __init__(self, bbox: np.ndarray) -> None:
        """Install acceleration-aware transitions before tracklet calibration."""
        super().__init__(bbox)
        self.motion = VerticalAccelerationMotion(
            dim_x=8,
            pos_idx=self._POS_IDX,
            vel_idx=self._VEL_IDX,
            process_noise=self.motion.process_noise,
        )

    def _create_filter(self, bbox: np.ndarray) -> KalmanFilter:
        kf = self._init_cv_filter(8, xyxy_to_xcycsr(bbox))
        height = max(float(bbox[3] - bbox[1]), 1.0)
        # Compensate native OC-SORT's subsequent P[4:,4:]*10000 and Q[4:,4:]*.01.
        # VS remains last, retaining its separate extra factor .01 exactly.
        kf.state_covariance[4, 4] = (ACCELERATION_PRIOR_HEIGHT_FRACTION * height) ** 2 / 10000
        kf.process_noise[4, 4] = (ACCELERATION_WALK_HEIGHT_FRACTION * height) ** 2 / 0.01
        return kf

    def clamp_velocity(self, frame_step: float = 1.0) -> None:
        """Preserve native scale guard with VS moved from index6 to index7."""
        if self.kf.state[2, 0] + frame_step * self.kf.state[7, 0] <= 0:
            self.kf.state[7, 0] = 0.0

    def get_state(self) -> dict:
        """Retain full acceleration covariance and reference clock in ORU snapshots."""
        return super().get_state() | {"ca_reference_fps": self.motion.reference_fps}

    def set_state(self, state: dict) -> None:
        """Restore acceleration, clock and matrices, invalidating prediction caches."""
        super().set_state(state)
        self.motion.reference_fps = float(state["ca_reference_fps"])


def ca_estimator(enabled: bool = False) -> type[XCYCSRStateEstimator]:
    """Return literal native class when disabled for exact parity."""
    return VerticalAccelerationStateEstimator if enabled else XCYCSRStateEstimator
