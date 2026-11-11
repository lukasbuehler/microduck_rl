"""Diagnostics must preserve simultaneous finish checks and continuous holds."""

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.turn360 import Turn360Goal
from mjlab_microduck.turn360_evaluation import EpisodeDiagnosis, finish_snapshot


def command():
    return SimpleNamespace(
        cfg=Turn360Goal(), angle=torch.tensor([2 * math.pi]), elapsed=torch.tensor([1.]),
        hold=torch.tensor([0.]), tilt=torch.tensor([0.]), drift=torch.tensor([0.]),
        angular_speed=torch.tensor([0.]), linear_speed=torch.tensor([0.]),
        pose_error=torch.zeros(1, 14), joint_velocity=torch.zeros(1, 14),
        metrics={"finish_support_ok": torch.tensor([1.])}, rotation_completed=torch.tensor([True]),
    )


NAMES = [f"servo_{index}" for index in range(14)]


def test_snapshot_reports_worst_joints_and_exact_failed_thresholds():
    cmd = command()
    cmd.pose_error[0, 6] = -0.5
    cmd.joint_velocity[0, 2] = 3.
    cmd.linear_speed[0] = 0.1
    sample = finish_snapshot(cmd, NAMES)
    assert set(sample["failed_checks"]) == {"joint_pose", "joint_speed", "base_linear_speed"}
    assert sample["worst_pose_joint"] == "servo_6"
    assert sample["fastest_joint"] == "servo_2"
    assert sample["values"]["joint_pose"] == 0.5
    assert sample["limits"]["joint_pose"] == cmd.cfg.finish_joint_tolerance


def test_best_sample_does_not_combine_individually_passed_checks_into_success():
    cmd = command()
    diagnosis = EpisodeDiagnosis()
    cmd.pose_error[0, 0] = 0.5
    diagnosis.observe(finish_snapshot(cmd, NAMES))  # angle right, pose wrong
    cmd.pose_error.zero_()
    cmd.angle[0] = 0.
    diagnosis.observe(finish_snapshot(cmd, NAMES))  # pose right, angle wrong
    result = diagnosis.result("timeout")
    assert result["best_sample"]["failed_checks"]
    assert result["max_continuous_hold_s"] == 0.
    assert result["near_goal_steps"] == 1
    assert result["pass_fraction_while_near_goal"]["joint_pose"] == 0.


def test_continuous_hold_keeps_peak_instead_of_summing_disjoint_holds():
    cmd = command()
    diagnosis = EpisodeDiagnosis()
    for hold in (0.1, 0.2, 0., 0.1):
        cmd.hold[0] = hold
        diagnosis.observe(finish_snapshot(cmd, NAMES))
    assert diagnosis.result("timeout")["max_continuous_hold_s"] == pytest.approx(0.2)


def test_exact_float32_threshold_remains_a_failed_check():
    cmd = command()
    cmd.drift[0] = cmd.cfg.success_radius
    sample = finish_snapshot(cmd, NAMES)
    assert not sample["checks"]["drift"]
