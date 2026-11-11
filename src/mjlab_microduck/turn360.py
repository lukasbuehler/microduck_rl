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
    finish_joint_tolerance: float = 0.25  # rad, every servo relative to HOME
    finish_joint_speed: float = 1.0  # rad/s, every servo
    # Training-only soft limits. None disables these speed costs.
    max_yaw_rate: float | None = None
    max_joint_speed: float | None = None

    def __post_init__(self):
        for name in ("max_yaw_rate", "max_joint_speed", "finish_joint_tolerance", "finish_joint_speed"):
            value = getattr(self, name)
            if value is None and name in ("max_yaw_rate", "max_joint_speed"):
                continue
            if value is None or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


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
        self.peak_yaw_rate = self.peak_joint_speed = 0.0
        self.command = np.array([1., 0., 0.], dtype=np.float32)

    def update(self, dt, position, quaternion, linear_velocity, angular_velocity,
               supported, body_contact, joint_velocity=None, joint_error=None, both_feet=False):
        if not np.isfinite(np.concatenate((position, quaternion, linear_velocity, angular_velocity))).all():
            return "invalid state"
        goal = self.goal
        yaw = yaw_from_quat(quaternion)
        delta = yaw - self.last_yaw
        increment = (delta + math.pi) % (2 * math.pi) - math.pi
        self.angle += increment
        if dt > 0:
            self.peak_yaw_rate = max(self.peak_yaw_rate, abs(increment) / dt)
        if joint_velocity is not None:
            self.peak_joint_speed = max(self.peak_joint_speed, float(np.max(np.abs(joint_velocity))))
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
        if body_contact:
            return "failed non-foot ground contact"
        if tilt > goal.failure_tilt:
            return f"failed tilt ({math.degrees(tilt):.1f} deg)"
        if drift > goal.failure_radius:
            return f"failed drift ({drift:.3f} m)"
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
            and both_feet and joint_error is not None and joint_velocity is not None
            and np.max(np.abs(joint_error)) < goal.finish_joint_tolerance
            and np.max(np.abs(joint_velocity)) < goal.finish_joint_speed
        )
        # Match training's float32 accumulation at the hold-time boundary.
        self.hold = np.float32(self.hold + np.float32(dt)) if stable else np.float32(0.0)
        if stable and self.hold >= np.float32(goal.hold_time):
            return "success"
        if self.elapsed >= TURN360_EPISODE_LENGTH_S:
            return "timeout"
        return None
