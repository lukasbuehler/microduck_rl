"""One counterclockwise revolution on feet, then a stable stop.

The 61D layout is retained, but twist means [remaining turns, origin x/y in
body frame / radius], NOT velocity. Use `play` or infer_policy.py's dedicated
--turn360 slot, which generates these state-dependent commands.
"""

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_ROBOT_CFG,
    get_allcollisions_spec,
)
from mjlab_microduck.tasks import mdp
from mjlab_microduck.turn360 import TURN360_EPISODE_LENGTH_S
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)


EPISODE_LENGTH_S = TURN360_EPISODE_LENGTH_S
ENABLE_VELOCITY_PUSHES = False


def get_turn360_spec():
    """Name anonymous colliders without changing geometry or contact settings.

    mjlab creates one contact sensor per geom name. Anonymous primaries would
    all produce the same sensor name, so excluding only the feet is insufficient.
    No `_collision` suffix: that would change which CollisionCfg rules apply.
    """
    spec = get_allcollisions_spec()
    for index, geom in enumerate(spec.geoms):
        if not geom.name and (geom.contype or geom.conaffinity):
            geom.name = f"turn_contact_{index}"
    return spec


def make_microduck_turn360_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = deepcopy(make_microduck_velocity_env_cfg(play=play))
    # Feet-only walking collisions would let the body pass through the floor.
    cfg.scene.entities["robot"] = deepcopy(MICRODUCK_ALLCOLLISIONS_ROBOT_CFG)
    cfg.scene.entities["robot"].spec_fn = get_turn360_spec
    robot_spec = get_turn360_spec()  # keep the owning MjSpec alive while reading its geoms
    body_geoms = tuple(
        geom.name for geom in robot_spec.geoms
        if (geom.contype or geom.conaffinity)
        and geom.name not in ("left_foot_collision", "right_foot_collision")
    )
    cfg.sim.nconmax = 200
    cfg.scene.sensors += (
        ContactSensorCfg(
            name="turn_body_ground",
            primary=ContactMatch(
                mode="geom",
                pattern=body_geoms,
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        ),
    )
    cfg.episode_length_s = EPISODE_LENGTH_S
    # Keep actuator/sensor DR at the inherited initial ranges. No gait-command,
    # head-pose or smoothing curricula: this is one fixed, controlled task.
    cfg.curriculum.clear()
    if not ENABLE_VELOCITY_PUSHES:
        cfg.events.pop("push_robot", None)
    cfg.events["reset_base"].params["pose_range"] = {
        "z": (0.12, 0.12), "yaw": (-math.pi, math.pi),
    }
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.commands["twist"] = mdp.Turn360CommandCfg(
        resampling_time_range=(1e9, 1e9),
        finish=mdp.Turn360FinishCfg(discount=MicroduckRlCfg.algorithm.gamma),
    )
    for name, dim in (("head_pose", 4), ("body_pose", 6)):
        cfg.commands[name] = mdp.UniformPoseCommandCfg(
            resampling_time_range=(1e9, 1e9), ranges=((0.0, 0.0),) * dim,
        )

    # Remove velocity tracking, gait style and angular-momentum costs: none
    # defines our objective, and 3D angular-momentum costs oppose yaw rotation.
    cfg.rewards = {
        name: deepcopy(cfg.rewards[name])
        for name in ("body_ang_vel", "action_rate_l2", "self_collisions")
    }
    cfg.rewards["body_ang_vel"].weight = -0.05  # roll/pitch rates only
    cfg.rewards["action_rate_l2"].weight = -0.002
    # Costs are nonnegative; the signed finish potential has a positive weight.
    for name, weight in {
        "progress": 4.0,
        "completion": 4.0,
        "pirouette": 1.0,
        "success": 20.0,
        "failure": -8.0,
        "tilt": -0.5,
        "drift": -0.2,
        "effort": -0.001,
        "yaw_speed": -1.0,
        "joint_speed": -1.0,
        "finish_pose": -0.1,
        "finish_potential": 10.0,
        "finish_credit": 6.0,
    }.items():
        cfg.rewards[f"turn_{name}"] = RewardTermCfg(
            func=mdp.turn360_reward, weight=weight, params={"component": name},
        )
    cfg.terminations = {
        name: cfg.terminations[name] for name in ("nan_state",)
    }
    for name in ("success", "failure"):
        cfg.terminations[f"turn_{name}"] = TerminationTermCfg(
            func=mdp.turn360_done, params={"success": name == "success"},
        )
    # This is a finite task deadline, not a truncated continuing episode:
    # timeout is penalized and must not bootstrap a hypothetical future finish.
    cfg.terminations["turn_timeout"] = TerminationTermCfg(func=mdp.turn360_timeout)
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


MicroduckTurn360RlCfg = deepcopy(MicroduckRlCfg)
MicroduckTurn360RlCfg.experiment_name = "microduck_turn360"
MicroduckTurn360RlCfg.run_name = "turn360"
MicroduckTurn360RlCfg.max_iterations = 3000
MicroduckTurn360RlCfg.save_interval = 100
MicroduckTurn360RlCfg.logger = "tensorboard"
# A mirror changes the rotation direction, so this task is asymmetric.
MicroduckTurn360RlCfg.algorithm.symmetry_cfg = None
