"""Lightweight Turn360 command contract and CPU playback state (no torch/warp)."""

import math
from dataclasses import dataclass

import numpy as np


TURN360_EPISODE_LENGTH_S = 8.0


@dataclass(kw_only=True)
class Turn360Goal:
    target_angle: float = 2 * math.pi
    angle_tolerance: float = math.radians(10)
    progress_tilt: float = math.radians(30)
    success_tilt: float = math.radians(15)
    failure_tilt: float = math.radians(45)
    success_radius: float = 0.06
    failure_radius: float = 0.20
    stop_angular_speed: float = 0.3
    stop_linear_speed: float = 0.05
    hold_time: float = 0.3
    max_air_time: float = 0.06


def yaw_from_quat(quaternion):
    w, x, y, z = quaternion
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Turn360Playback:
    """Single-robot counterpart of the training command/termination state.

    Caller supplies simulation time, contact classification and robot state.
    A terminal reason requests a handoff to the walking/recovery policy.
    """

    def __init__(self, position, quaternion, goal=None):
        self.goal = goal or Turn360Goal()
        self.origin = np.array(position[:2], dtype=float)
        self.last_yaw = yaw_from_quat(quaternion)
        self.angle = self.frontier = self.elapsed = self.hold = self.air_time = 0.0
        self.hold = np.float32(0.0)
        self.command = np.array([1., 0., 0.], dtype=np.float32)

    def update(self, dt, position, quaternion, linear_velocity, angular_velocity,
               supported, body_contact):
        if not np.isfinite(np.concatenate((position, quaternion, linear_velocity, angular_velocity))).all():
            return "invalid state"
        goal = self.goal
        yaw = yaw_from_quat(quaternion)
        delta = yaw - self.last_yaw
        self.angle += (delta + math.pi) % (2 * math.pi) - math.pi
        self.last_yaw = yaw
        self.elapsed += dt
        self.air_time = 0.0 if supported else self.air_time + dt
        _, x, y, _ = quaternion
        tilt = math.acos(np.clip(1 - 2 * (x * x + y * y), -1, 1))
        offset = self.origin - np.asarray(position[:2])
        drift = float(np.linalg.norm(offset))
        c, s = math.cos(yaw), math.sin(yaw)
        self.command[:] = np.clip([
            (goal.target_angle - self.angle) / (2 * math.pi),
            (c * offset[0] + s * offset[1]) / goal.success_radius,
            (-s * offset[0] + c * offset[1]) / goal.success_radius,
        ], -2, 2)
        if body_contact or tilt > goal.failure_tilt or drift > goal.failure_radius:
            return "failed posture/contact"
        if self.elapsed > 0.25 and self.air_time > goal.max_air_time:
            return "airborne"
        valid = supported and tilt < goal.progress_tilt
        if valid:
            self.frontier = max(self.frontier, np.clip(self.angle, 0, goal.target_angle))
        stable = (
            valid and abs(self.angle - goal.target_angle) < goal.angle_tolerance
            and self.frontier >= goal.target_angle - goal.angle_tolerance
            and tilt < goal.success_tilt and drift < goal.success_radius
            and np.linalg.norm(angular_velocity) < goal.stop_angular_speed
            and np.linalg.norm(linear_velocity) < goal.stop_linear_speed
        )
        # Match training's float32 accumulation at the hold-time boundary.
        self.hold = np.float32(self.hold + np.float32(dt)) if stable else np.float32(0.0)
        if stable and self.hold >= np.float32(goal.hold_time):
            return "success"
        if self.elapsed >= TURN360_EPISODE_LENGTH_S:
            return "timeout"
        return None
