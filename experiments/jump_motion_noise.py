# ------------------------------------------------------------------------
# Trackers
# Copyright (c) 2026 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Throwaway jump-motion spike; no stable API or default tracker change."""

from __future__ import annotations

import numpy as np

from trackers.utils.motion_models import ScalableProcessNoise
from trackers.utils.state_representations import XCYCSRStateEstimator

VERTICAL_VARIANCE_FACTOR = 4.0
_VERTICAL = np.ix_([1, 5], [1, 5])


class VerticalProcessNoise(ScalableProcessNoise):
    """Increase only the vertical principal covariance block at every build."""

    def build_Q(self, frame_step: float, frame_rate: float | None = None) -> np.ndarray:
        """Build native time-scaled Q, then add a positive vertical covariance block."""
        process_noise = super().build_Q(frame_step, frame_rate)
        # A principal block of PSD Q is PSD. Adding its positive multiple,
        # embedded in otherwise-zero entries, preserves PSD even with cross terms.
        process_noise[_VERTICAL] *= VERTICAL_VARIANCE_FACTOR
        return process_noise


class VerticalNoiseStateEstimator(XCYCSRStateEstimator):
    """Keep native CV state/means and replace only the process-noise builder."""

    def __init__(self, bbox: np.ndarray) -> None:
        """Install the wrapper before tracklet noise calibration/refresh occurs."""
        super().__init__(bbox)
        native = self.motion.process_noise
        wrapped = VerticalProcessNoise(
            dim_x=native.dim_x,
            pos_idx=native.pos_idx.copy(),
            vel_idx=native.vel_idx.copy(),
            baseline_Q=native.baseline_Q.copy(),
            sigma_a2=native.sigma_a2.copy(),
            extra_q_diagonal=native.extra_q_diagonal.copy(),
        )
        wrapped.calibrate(native.baseline_Q)
        self.motion.process_noise = wrapped


def noise_estimator(enabled: bool = False) -> type[XCYCSRStateEstimator]:
    """Return the literal native estimator when disabled, for exact parity."""
    return VerticalNoiseStateEstimator if enabled else XCYCSRStateEstimator
