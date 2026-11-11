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


@pytest.mark.parametrize("violation", ["tilt", "drift"])
def test_failure_blocks_success_and_progress(violation):
    env, cmd = make_env()
    if violation == "drift":
        env.scene["robot"].data.root_link_pos_w[:, 0] = 0.3
    advance(env, cmd, 90, tilt=60 if violation == "tilt" else 0)
    assert cmd.failed.all()
    assert not cmd.succeeded.any()
    assert not cmd.progress.any()
    assert (mdp.turn360_reward(env, "failure") > 0).all()  # negative weight in cfg


@pytest.mark.parametrize("airborne,body_contact", [(True, False), (False, True), (True, True)])
def test_hops_and_nonfoot_contacts_do_not_interrupt_turn_but_finish_requires_landing(airborne, body_contact):
    from mjlab_microduck.turn360 import Turn360Playback

    env, cmd = make_env(n=1)
    data = env.scene["robot"].data
    playback = Turn360Playback(data.root_link_pos_w[0].numpy(), data.root_link_quat_w[0].numpy())
    env.scene.sensors["feet_ground_contact"].data.found[:] = float(not airborne)
    env.scene.sensors["turn_body_ground"].data.found[:] = float(body_contact)

    def step(angle, landed=False):
        advance(env, cmd, angle)
        return playback.update(
            env.step_dt, data.root_link_pos_w[0].numpy(), data.root_link_quat_w[0].numpy(),
            data.root_link_lin_vel_b[0].numpy(), data.root_link_ang_vel_b[0].numpy(),
            not airborne or landed, body_contact, joint_velocity=data.joint_vel[0].numpy(),
            joint_error=(data.joint_pos - data.default_joint_pos)[0].numpy(),
            both_feet=not airborne or landed,
        )

    for angle in (90, 180, 270, 360):
        assert step(angle) is None
        assert cmd.progress.item() > 0.
        assert not cmd.failed.any()
    if airborne:
        for _ in range(40):  # exceed the old airborne grace and time limit
            assert step(360) is None
            assert not cmd.failed.any()
            assert not cmd.succeeded.any()
    env.scene.sensors["feet_ground_contact"].data.found[:] = 1.
    for _ in range(16):
        reason = step(360, landed=True)
    assert reason == "success"
    assert cmd.succeeded.all()


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
        assert cfg.commands["twist"].max_joint_speed == 12.
        assert cfg.commands["twist"].max_yaw_rate is None
        assert cfg.commands["twist"].finish.discount == MicroduckTurn360RlCfg.algorithm.gamma
        assert cfg.rewards["turn_finish_potential"].weight > 0
        assert 0 < cfg.rewards["turn_finish_credit"].weight < -cfg.rewards["turn_failure"].weight
        assert cfg.rewards["turn_success"].weight > cfg.rewards["turn_progress"].weight
        assert "turn_braking" not in cfg.rewards
        assert "turn_finish_motion" not in cfg.rewards
        assert "turn_time" not in cfg.rewards
        assert "time_out" not in cfg.terminations
        assert not cfg.terminations["turn_timeout"].time_out
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


def test_speed_costs_can_be_disabled_and_servo_cost_prices_worst_joint_near_budget():
    env, cmd = make_env(n=1)
    cmd.cfg.max_joint_speed = None
    data = env.scene["robot"].data
    data.joint_vel[:] = 10.
    advance(env, cmd, math.degrees(0.12))  # 6 rad/s over a 20 ms step
    assert mdp.turn360_reward(env, "yaw_speed").item() == 0
    assert mdp.turn360_reward(env, "joint_speed").item() == 0
    cmd.cfg.max_yaw_rate = 3.
    cmd.cfg.max_joint_speed = 5.
    assert mdp.turn360_reward(env, "yaw_speed").item() == pytest.approx(1., abs=1e-5)
    assert mdp.turn360_reward(env, "joint_speed").item() == pytest.approx(36.)
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


@pytest.mark.parametrize("name", ["finish_joint_tolerance", "finish_joint_speed", "pose_scale"])
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


def test_home_cost_is_gentle_and_does_not_ramp_with_turn_angle():
    env, cmd = make_env(n=1)
    data = env.scene["robot"].data
    data.joint_pos += cmd.cfg.finish_joint_tolerance
    data.joint_vel[:] = cmd.cfg.finish_joint_speed
    for angle in [0, 90, 180, 270, 360]:
        advance(env, cmd, angle)
        assert mdp.turn360_reward(env, "finish_pose").item() == pytest.approx(0.25)


def test_completion_pays_once_and_partial_turn_has_no_completion_bonus():
    env, cmd = make_env(n=1)
    for angle in [90, 180, 234, 270, 340]:
        advance(env, cmd, angle)
        assert mdp.turn360_reward(env, "completion").item() == 0.
    advance(env, cmd, 355)
    assert mdp.turn360_reward(env, "completion").item() * env.step_dt == pytest.approx(1.)
    for angle in [360, 355, 340, 355, 360]:
        advance(env, cmd, angle)
        assert mdp.turn360_reward(env, "completion").item() == 0.
    cmd.reset(torch.tensor([0]))
    assert not cmd.rotation_completed.any()


@pytest.mark.parametrize("tilt,drift,feet,earns_bonus", [
    (0., 0., 1, True), (20., 0., 1, False), (0., 0.1, 1, False), (0., 0., 2, False),
])
def test_pirouette_bonus_only_pays_for_new_stable_one_foot_progress(tilt, drift, feet, earns_bonus):
    env, cmd = make_env(n=1)
    env.scene.sensors["feet_ground_contact"].data.found[:, 1] = float(feet == 2)
    env.scene["robot"].data.root_link_pos_w[:, 0] = drift
    advance(env, cmd, 90, tilt=tilt)
    assert not cmd.failed.any()  # one foot is valid support during the maneuver
    bonus = mdp.turn360_reward(env, "pirouette").item() * env.step_dt
    assert bonus == pytest.approx(0.25 if earns_bonus else 0.)
    assert mdp.turn360_reward(env, "progress").item() > 0.
    for angle in [90, 60, 90]:
        advance(env, cmd, angle, tilt=tilt)
        assert mdp.turn360_reward(env, "pirouette").item() == 0.


@pytest.mark.parametrize("speed,fails", [(9., False), (12., False), (12.01, True)])
def test_one_servo_exceeding_budget_fails_training_and_playback(speed, fails):
    from mjlab_microduck.turn360 import Turn360Playback

    env, cmd = make_env(n=1)
    data = env.scene["robot"].data
    playback = Turn360Playback(data.root_link_pos_w[0].numpy(), data.root_link_quat_w[0].numpy())
    data.joint_vel[0, 6] = -speed
    advance(env, cmd, 30)
    reason = playback.update(
        env.step_dt, data.root_link_pos_w[0].numpy(), data.root_link_quat_w[0].numpy(),
        data.root_link_lin_vel_b[0].numpy(), data.root_link_ang_vel_b[0].numpy(), True, False,
        joint_velocity=data.joint_vel[0].numpy(),
    )
    assert bool(cmd.failed[0]) == fails
    assert (reason is not None and "joint-speed" in reason) == fails
    if fails:
        assert cmd.progress.item() == 0.
        assert cmd.metrics["joint_speed_violation"].item() == 1.
    if speed == 12.:
        assert mdp.turn360_reward(env, "joint_speed").item() == pytest.approx(1.)


def test_partial_turn_at_deadline_is_penalized_terminal_failure_not_bootstrapped_timeout():
    env, cmd = make_env(n=2)
    env.episode_length_buf = torch.tensor([400, 399])
    env.max_episode_length = 400
    advance(env, cmd, 234)
    assert torch.equal(mdp.turn360_timeout(env), torch.tensor([True, False]))
    assert mdp.turn360_reward(env, "failure")[0].item() * env.step_dt == pytest.approx(1.)
    assert not cmd.succeeded.any()
    assert not cmd.rotation_completed.any()
    assert not cmd.failed.any()  # separate physical failure / incomplete deadline diagnostics


def test_passive_joint_speed_does_not_affect_budget_or_pose_cost():
    env, cmd = make_env(n=1)
    robot = env.scene["robot"]
    robot.find_joints = lambda _: (list(range(0, 28, 2)), [])
    env._servo_joint_ids_cache.clear()
    for name in ("joint_vel", "joint_pos", "default_joint_pos"):
        setattr(robot.data, name, torch.zeros(1, 28))
    robot.data.joint_vel[:, 1::2] = 1000.
    robot.data.joint_pos[:, 1::2] = 10.
    advance(env, cmd, 90)
    assert not cmd.failed.any()
    assert mdp.turn360_reward(env, "joint_speed").item() == 0.
    assert mdp.turn360_reward(env, "finish_pose").item() == 0.


def test_finish_quality_guides_angle_pose_and_braking_before_binary_success():
    env, cmd = make_env(n=1)
    data = env.scene["robot"].data
    qualities = []
    for angle in (90, 180, 270, 330, 360):
        advance(env, cmd, angle)
        qualities.append(cmd.finish_quality.item())
    assert qualities == sorted(qualities)
    assert qualities[-1] == pytest.approx(1.)
    assert not cmd.succeeded.any()  # dense feedback is available before a held finish
    for joint in (0, 6):  # one bad leg OR head joint must lower the score
        data.joint_pos[0, joint] += 1.
        advance(env, cmd, 360)
        assert cmd.finish_quality.item() < qualities[-1]
        data.joint_pos.copy_(data.default_joint_pos)
    for speed in (8., 4., 0.):
        data.joint_vel[0, 6] = speed
        advance(env, cmd, 360)
        qualities.append(cmd.finish_quality.item())
    assert qualities[-3] < qualities[-2] < qualities[-1]
    assert cmd.finish_quality.item() == pytest.approx(1.)


def test_finish_quality_uses_current_angle_and_never_pays_for_just_holding():
    env, cmd = make_env(n=1)
    for angle in (90, 180, 270, 360):
        advance(env, cmd, angle)
    advance(env, cmd, 360)
    holding = mdp.turn360_reward(env, "finish_potential").item() * env.step_dt
    assert holding == pytest.approx(cmd.cfg.finish.discount - 1.)
    advance(env, cmd, 450)
    assert cmd.frontier.item() == pytest.approx(2 * math.pi)
    assert cmd.finish_quality.item() < 1.
    assert mdp.turn360_reward(env, "finish_potential").item() < 0.


def test_discounted_finish_shaping_cannot_profit_from_jitter_or_external_termination():
    env, cmd = make_env(n=1)
    data = env.scene["robot"].data
    total = 0.
    for index, angle in enumerate([90, 180, 270, 360] + [360] * 10):
        data.joint_vel[0, 0] = 8. if index % 2 else 0.
        if index == 13:
            env.reset_buf = torch.ones(1, dtype=torch.bool)
        advance(env, cmd, angle)
        term = mdp.turn360_reward(env, "finish_potential").item() * env.step_dt
        total += cmd.cfg.finish.discount ** index * term
    # The signed, discounted changes telescope to zero from the reset potential.
    # An external terminal must cancel remaining potential, not pay it out.
    assert total == pytest.approx(0., abs=2e-7)


@pytest.mark.parametrize("ending", ["success", "timeout", "failure"])
def test_finish_potential_is_zero_at_every_task_terminal(ending):
    env, cmd = make_env(n=1)
    for angle in (90, 180, 270, 360):
        advance(env, cmd, angle)
    previous = cmd.finish_quality.item()
    if ending == "success":
        cmd.hold[:] = 0.29
    elif ending == "timeout":
        cmd.elapsed[:] = 7.99
    else:
        env.scene["robot"].data.root_link_pos_w[:, 0] = 0.3
    advance(env, cmd, 360)
    assert (cmd.succeeded | cmd.timed_out | cmd.failed).all()
    assert mdp.turn360_reward(env, "finish_potential").item() * env.step_dt == pytest.approx(-previous)


def test_partial_reset_clears_finish_history_only_for_reset_robot():
    env, cmd = make_env(n=2)
    for angle in (90, 180, 270, 360):
        advance(env, cmd, angle)
    score = cmd.finish_quality[1].clone()
    cmd.reset(torch.tensor([0]))
    cmd.compute(env.step_dt)
    assert cmd.finish_quality[0] == 0.
    assert cmd.previous_finish_quality[0] == 0.
    assert cmd.finish_quality[1] == score


@pytest.mark.parametrize("name,value", [("discount", 1.1), ("discount", math.nan),
                                        ("pose_scale", 0.), ("joint_speed_scale", -1.)])
def test_invalid_finish_shaping_configuration_is_rejected(name, value):
    with pytest.raises(ValueError):
        mdp.Turn360FinishCfg(**{name: value})


def test_deadline_finish_credit_grades_near_misses_without_requiring_success():
    env, cmd = make_env(n=2)
    for angle in (90, 180, 270, 355):
        advance(env, cmd, angle)
    assert mdp.turn360_reward(env, "finish_credit").eq(0).all()  # no standing payouts
    # Both have completed the turn, but only one is close to HOME and quiet.
    data = env.scene["robot"].data
    data.joint_pos[0, 6] += 1.
    data.joint_vel[0, 0] = 8.
    cmd.elapsed[:] = 7.99
    advance(env, cmd, 355)
    assert cmd.timed_out.all()
    assert not cmd.succeeded.any()
    credit = mdp.turn360_reward(env, "finish_credit") * env.step_dt
    assert 0 < credit[0] < credit[1] <= 1.


def test_physical_failure_cannot_cash_out_finish_credit():
    env, cmd = make_env(n=1)
    for angle in (90, 180, 270, 355):
        advance(env, cmd, angle)
    env.scene["robot"].data.joint_vel[0, 0] = 13.
    advance(env, cmd, 355)
    assert cmd.failed.all()
    assert mdp.turn360_reward(env, "finish_credit").item() == 0.
