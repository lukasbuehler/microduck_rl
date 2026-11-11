"""CPU checks for heading wrap, reward accounting and asynchronous resets."""

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp


class Scene(dict):
    pass


def make_env(n=2):
    data = SimpleNamespace(
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
        root_link_pos_w=torch.zeros(n, 3),
        root_link_ang_vel_b=torch.zeros(n, 3),
        root_link_lin_vel_b=torch.zeros(n, 3),
        actuator_force=torch.zeros(n, 14),
        joint_vel=torch.zeros(n, 14),
        joint_pos=torch.full((n, 14), 0.1),
        default_joint_pos=torch.full((n, 14), 0.1),
    )
    scene = Scene(robot=SimpleNamespace(data=data, find_joints=lambda _: (list(range(14)), [])))
    scene.sensors = {
        name: SimpleNamespace(data=SimpleNamespace(found=torch.full((n, 2, 1), value)))
        for name, value in (("feet_ground_contact", 1.), ("turn_body_ground", 0.))
    }
    env = SimpleNamespace(num_envs=n, device="cpu", scene=scene, step_dt=0.02,
                          common_step_counter=0)
    command = mdp.Turn360CommandCfg(resampling_time_range=(1e9, 1e9)).build(env)
    env.command_manager = SimpleNamespace(get_term=lambda _: command)
    command.reset(torch.arange(n))
    command.compute(0.0)
    return env, command


def advance(env, command, degrees, tilt=0.):
    yaw = torch.deg2rad(torch.as_tensor(degrees, dtype=torch.float32).expand(env.num_envs))
    # yaw rotation composed with a body-x tilt, wxyz convention.
    half = yaw / 2
    t = math.radians(tilt) / 2
    env.scene["robot"].data.root_link_quat_w[:] = torch.stack(
        (half.cos() * math.cos(t), half.cos() * math.sin(t),
         half.sin() * math.sin(t), half.sin() * math.cos(t)), dim=-1,
    )
    env.common_step_counter += 1
    command.update()


def test_heading_unwraps_and_progress_cannot_be_farmed():
    env, cmd = make_env()
    paid = 0.
    for angle in (90, 170, 190, 150, 190, 270, 360, 400):
        advance(env, cmd, angle)
        paid += float(mdp.turn360_reward(env, "progress")[0]) * env.step_dt
    assert cmd.angle[0] == pytest.approx(math.radians(400))
    assert paid == pytest.approx(1.0)
    # Rotation beyond one revolution earns nothing; remaining angle is signed.
    assert cmd.command[0, 0] < 0
    assert not cmd.succeeded.any()


def test_readers_do_not_integrate_twice_and_wrong_direction_does_not_pay():
    env, cmd = make_env()
    advance(env, cmd, -30)
    assert not cmd.progress.any()
    advance(env, cmd, 30)
    angle, hold, elapsed = cmd.angle.clone(), cmd.hold.clone(), cmd.elapsed.clone()
    for _ in range(3):
        mdp.turn360_done(env, True)
        mdp.turn360_reward(env, "progress")
        cmd.compute(env.step_dt)
    assert torch.equal(cmd.angle, angle)
    assert torch.equal(cmd.hold, hold)
    assert torch.equal(cmd.elapsed, elapsed)
    assert cmd.frontier[0] == pytest.approx(math.pi / 6)


def test_success_requires_a_full_turn_and_a_continuous_stable_stop():
    env, cmd = make_env()
    for _ in range(20):
        advance(env, cmd, 0)
    assert not cmd.succeeded.any()  # same heading at spawn != a revolution
    for angle in (90, 180, 270, 360):
        advance(env, cmd, angle)
    env.scene["robot"].data.root_link_ang_vel_b[:, 2] = 1.
    for _ in range(20):
        advance(env, cmd, 360)
    assert not cmd.succeeded.any()
    env.scene["robot"].data.root_link_ang_vel_b.zero_()
    for _ in range(16):
        advance(env, cmd, 360)
    assert cmd.succeeded.all()
    assert torch.allclose(mdp.turn360_reward(env, "success") * env.step_dt,
                          torch.ones(env.num_envs))


def test_partial_reset_captures_new_origin_without_cross_episode_rotation():
    env, cmd = make_env()
    advance(env, cmd, [120, 60])
    # _reset_idx calls commands.reset before sim.forward. Mimic stale data,
    # then replace it with the new pose before command_manager.compute.
    cmd.reset(torch.tensor([0]))
    env.scene["robot"].data.root_link_pos_w[0, :2] = torch.tensor([5., 3.])
    env.scene["robot"].data.root_link_quat_w[0] = torch.tensor([0., 0., 0., 1.])
    cmd.compute(env.step_dt)
    assert cmd.angle[0] == 0
    assert cmd.frontier[0] == 0
    assert cmd.angle[1] == pytest.approx(math.pi / 3)
    assert torch.equal(cmd.origin[0], torch.tensor([5., 3.]))
    advance(env, cmd, [190, 70])
    assert torch.allclose(cmd.angle, torch.deg2rad(torch.tensor([10., 70.])), atol=1e-6)


@pytest.mark.parametrize("violation", ["tilt", "body", "drift", "flight"])
def test_failure_blocks_success_and_progress(violation):
    env, cmd = make_env()
    if violation == "body":
        env.scene.sensors["turn_body_ground"].data.found[:] = 1
    elif violation == "drift":
        env.scene["robot"].data.root_link_pos_w[:, 0] = 0.3
    elif violation == "flight":
        env.scene.sensors["feet_ground_contact"].data.found[:] = 0
        cmd.elapsed[:] = 1.
        cmd.air_time[:] = 0.1
    advance(env, cmd, 90, tilt=60 if violation == "tilt" else 0)
    assert cmd.failed.all()
    assert not cmd.succeeded.any()
    assert not cmd.progress.any()
    assert (mdp.turn360_reward(env, "failure") > 0).all()  # negative weight in cfg


def test_command_origin_error_is_body_relative():
    env, cmd = make_env()
    env.scene["robot"].data.root_link_pos_w[:, 0] = cmd.cfg.success_radius
    advance(env, cmd, 90)
    assert torch.allclose(cmd.command[:, 1:], torch.tensor([[0., 1.]]).expand(2, 2), atol=1e-6)


def test_cfg_preserves_layout_and_has_real_body_contacts():
    import mujoco
    from mjlab.scene import Scene
    from mjlab_microduck.tasks.microduck_turn360_env_cfg import (
        make_microduck_turn360_env_cfg, MicroduckTurn360RlCfg,
    )
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

    base = make_microduck_velocity_env_cfg()
    for play in (False, True):
        cfg = make_microduck_turn360_env_cfg(play=play)
        assert list(cfg.observations["actor"].terms) == list(base.observations["actor"].terms)
        assert "push_robot" not in cfg.events
        assert not cfg.curriculum
        assert cfg.rewards["turn_failure"].weight < 0
        assert cfg.rewards["turn_progress"].weight > 0
        assert cfg.rewards["turn_success"].weight > 0
        assert "track_angular_velocity" not in cfg.rewards
        assert "angular_momentum" not in cfg.rewards
        assert cfg.commands["twist"].target_angle == 2 * math.pi
        spec = cfg.scene.entities["robot"].spec_fn()
        model = spec.compile()
        assert model.nu == 14
        sensor = next(s for s in cfg.scene.sensors if s.name == "turn_body_ground")
        colliders = {
            geom.name for geom in spec.geoms
            if (geom.contype or geom.conaffinity)
            and geom.name not in ("left_foot_collision", "right_foot_collision")
        }
        assert "" not in colliders
        assert set(sensor.primary.pattern) == colliders
        assert len(sensor.primary.pattern) == len(set(sensor.primary.pattern))
        assert not {"left_foot_collision", "right_foot_collision"} & colliders
        # Compile the complete scene, including expanded sensor definitions.
        # Robot-only compilation missed repeated names from anonymous geoms.
        cfg.scene.num_envs = 1
        scene_model = Scene(cfg.scene, device="cpu").compile()
        sensors = [mujoco.mj_id2name(scene_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
                   for i in range(scene_model.nsensor)]
        body_sensors = [name for name in sensors if name.startswith("turn_body_ground_")]
        assert len(body_sensors) == len(colliders)
        assert len(body_sensors) == len(set(body_sensors))
    assert MicroduckTurn360RlCfg.algorithm.symmetry_cfg is None


def test_naming_anonymous_contacts_does_not_change_physics():
    import numpy as np
    from mjlab_microduck.robot.microduck_constants import get_allcollisions_spec
    from mjlab_microduck.tasks.microduck_turn360_env_cfg import get_turn360_spec

    original, named = get_allcollisions_spec().compile(), get_turn360_spec().compile()
    for field in ("geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_condim",
                  "geom_contype", "geom_conaffinity", "body_mass", "body_inertia"):
        np.testing.assert_array_equal(getattr(original, field), getattr(named, field))


def test_cpu_playback_commands_and_success_match_training():
    import numpy as np
    from mjlab_microduck.turn360 import Turn360Playback

    env, cmd = make_env(n=1)
    data = env.scene["robot"].data
    playback = Turn360Playback(data.root_link_pos_w[0].numpy(), data.root_link_quat_w[0].numpy())
    for angle in [70, 130, 190, 250, 310, 355, 360] + [360] * 16:
        data.root_link_pos_w[:, 0] = 0.01
        advance(env, cmd, angle)
        reason = playback.update(
            env.step_dt, data.root_link_pos_w[0].numpy(), data.root_link_quat_w[0].numpy(),
            data.root_link_lin_vel_b[0].numpy(), data.root_link_ang_vel_b[0].numpy(), True, False,
            joint_velocity=data.joint_vel[0].numpy(),
            joint_error=(data.joint_pos - data.default_joint_pos)[0].numpy(), both_feet=True,
        )
        np.testing.assert_allclose(playback.command, cmd.command[0].numpy(), atol=1e-6)
        assert bool(cmd.succeeded[0]) == (reason == "success")
        assert playback.angle == pytest.approx(float(cmd.angle[0]), abs=1e-6)
    assert reason == "success"


def test_speed_costs_are_off_by_default_and_charge_only_excess():
    env, cmd = make_env(n=1)
    data = env.scene["robot"].data
    data.joint_vel[:] = 10.
    advance(env, cmd, math.degrees(0.12))  # 6 rad/s over a 20 ms step
    assert mdp.turn360_reward(env, "yaw_speed").item() == 0
    assert mdp.turn360_reward(env, "joint_speed").item() == 0
    cmd.cfg.max_yaw_rate = 3.
    cmd.cfg.max_joint_speed = 5.
    assert mdp.turn360_reward(env, "yaw_speed").item() == pytest.approx(1., abs=1e-5)
    assert mdp.turn360_reward(env, "joint_speed").item() == pytest.approx(1.)
    data.joint_vel[:] = 4.
    advance(env, cmd, math.degrees(0.14))  # 1 rad/s
    assert mdp.turn360_reward(env, "yaw_speed").item() == 0
    assert mdp.turn360_reward(env, "joint_speed").item() == 0
    assert cmd.metrics["peak_yaw_rate_rad_s"].item() == pytest.approx(6., abs=1e-5)
    assert cmd.metrics["peak_joint_speed_rad_s"].item() == 10.
    cmd.reset(torch.tensor([0]))
    assert cmd.metrics["peak_joint_speed_rad_s"].item() == 0


@pytest.mark.parametrize("limit", [0., -1., math.inf, math.nan])
def test_invalid_speed_limits_are_rejected(limit):
    with pytest.raises(ValueError):
        mdp.Turn360CommandCfg(resampling_time_range=(1e9, 1e9), max_yaw_rate=limit)


@pytest.mark.parametrize("name", ["finish_joint_tolerance", "finish_joint_speed"])
@pytest.mark.parametrize("value", [None, 0., -1., math.inf, math.nan])
def test_invalid_finish_thresholds_are_rejected(name, value):
    from mjlab_microduck.turn360 import Turn360Goal

    with pytest.raises(ValueError):
        Turn360Goal(**{name: value})


@pytest.mark.parametrize("violation", ["head", "leg", "moving", "one_foot"])
def test_finish_requires_home_pose_slow_joints_and_both_feet_in_training_and_playback(violation):
    from mjlab_microduck.turn360 import Turn360Playback

    env, cmd = make_env(n=1)
    data = env.scene["robot"].data
    playback = Turn360Playback(data.root_link_pos_w[0].numpy(), data.root_link_quat_w[0].numpy())
    if violation in ("head", "leg"):
        data.joint_pos[0, 6 if violation == "head" else 0] += 0.4
    elif violation == "moving":
        data.joint_vel[0, 0] = 2.
    else:
        env.scene.sensors["feet_ground_contact"].data.found[:, 1] = 0.

    def step(angle):
        advance(env, cmd, angle)
        reason = playback.update(
            env.step_dt, data.root_link_pos_w[0].numpy(), data.root_link_quat_w[0].numpy(),
            data.root_link_lin_vel_b[0].numpy(), data.root_link_ang_vel_b[0].numpy(), True, False,
            joint_velocity=data.joint_vel[0].numpy(),
            joint_error=(data.joint_pos - data.default_joint_pos)[0].numpy(),
            both_feet=bool(env.scene.sensors["feet_ground_contact"].data.found.all()),
        )
        assert bool(cmd.succeeded[0]) == (reason == "success")
        return reason

    for angle in [90, 180, 270, 360] + [360] * 20:
        assert step(angle) is None
    data.joint_pos.copy_(data.default_joint_pos)
    data.joint_vel.zero_()
    env.scene.sensors["feet_ground_contact"].data.found[:] = 1.
    for _ in range(15):
        assert step(360) is None
    assert step(360) == "success"


def test_finish_costs_only_apply_near_end_and_measure_configured_pose():
    env, cmd = make_env(n=1)
    data = env.scene["robot"].data
    data.joint_pos += cmd.cfg.finish_joint_tolerance
    data.joint_vel[:] = cmd.cfg.finish_joint_speed
    for angle in [90, 180, 270]:
        advance(env, cmd, angle)
        assert mdp.turn360_reward(env, "finish_pose").item() == 0.
        assert mdp.turn360_reward(env, "finish_motion").item() == 0.
    advance(env, cmd, 360)
    assert mdp.turn360_reward(env, "finish_pose").item() == pytest.approx(1.)
    assert mdp.turn360_reward(env, "finish_motion").item() == pytest.approx(1.)
