"""Deterministic, headless Turn360 diagnosis using the task's actual checks."""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch


TASK = "Mjlab-Turn360-Flat-MicroDuck"


def finish_snapshot(cmd, joint_names):
    """Read the same cached state the command used for success/termination."""
    goal = cmd.cfg
    pose = cmd.pose_error[0].abs()
    speed = cmd.joint_velocity[0].abs()
    pose_index, speed_index = int(pose.argmax()), int(speed.argmax())
    measured = {
        "angle": (cmd.angle[0] - goal.target_angle).abs(),
        "tilt": cmd.tilt[0],
        "drift": cmd.drift[0],
        "base_angular_speed": cmd.angular_speed[0],
        "base_linear_speed": cmd.linear_speed[0],
        "joint_pose": pose[pose_index],
        "joint_speed": speed[speed_index],
    }
    limits = dict(zip(measured, (
        goal.angle_tolerance, goal.success_tilt, goal.success_radius,
        goal.stop_angular_speed, goal.stop_linear_speed,
        goal.finish_joint_tolerance, goal.finish_joint_speed,
    )))
    values = {name: float(value) for name, value in measured.items()}
    ratios = {name: value / limits[name] if math.isfinite(value) else math.inf
              for name, value in values.items()}
    # Compare as tensors: converting to Python floats first changes float32
    # boundary behavior (e.g. exactly 0.06 m) relative to the real success rule.
    checks = {name: bool(value < limits[name]) for name, value in measured.items()}
    checks["both_feet"] = bool(cmd.metrics["finish_support_ok"][0])
    checks["full_turn_reached"] = bool(cmd.rotation_completed[0])
    return {
        "time_s": float(cmd.elapsed[0]),
        "rotation_deg": math.degrees(float(cmd.angle[0])),
        "hold_s": float(cmd.hold[0]),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "values": values,
        "limits": limits,
        "worst_ratio": max(ratios.values()),
        "worst_pose_joint": joint_names[pose_index],
        "fastest_joint": joint_names[speed_index],
    }


class EpisodeDiagnosis:
    """Retain simultaneous checks, rather than combining peaks from different times."""

    def __init__(self):
        self.best = self.final = None
        self.max_hold = 0.0
        self.near_goal_steps = 0
        self.near_goal_passes = {}

    def observe(self, sample):
        self.final = sample
        self.max_hold = max(self.max_hold, sample["hold_s"])
        rank = lambda state: (len(state["failed_checks"]), state["worst_ratio"])
        if self.best is None or rank(sample) < rank(self.best):
            self.best = sample
        if sample["checks"]["angle"] and sample["checks"]["full_turn_reached"]:
            self.near_goal_steps += 1
            for name, passed in sample["checks"].items():
                self.near_goal_passes[name] = self.near_goal_passes.get(name, 0) + int(passed)

    def result(self, outcome):
        return {
            "outcome": outcome, "best_sample": self.best, "final_sample": self.final,
            "max_continuous_hold_s": self.max_hold,
            "near_goal_steps": self.near_goal_steps,
            "pass_fraction_while_near_goal": {
                name: count / self.near_goal_steps for name, count in self.near_goal_passes.items()
            },
        }


def evaluate(checkpoint, episodes=3, seed=42, device="cpu"):
    # Use the same actor loading, normalizer and action wrapper as `play`.
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    cfg.auto_reset = False  # inspect terminal state before resetting
    agent_cfg = load_rl_cfg(TASK)
    raw_env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)
        # Restart the reset/DR sequence after loading, identically for every checkpoint.
        raw_env.reset(seed=seed)
        obs = env.get_observations()
        _, names = raw_env.scene["robot"].find_joints(r"^(?!passive_).*")
        cmd = raw_env.command_manager.get_term("twist")
        results = []
        with torch.inference_mode():
            for index in range(episodes):
                diagnosis = EpisodeDiagnosis()
                for _ in range(raw_env.max_episode_length + 1):
                    obs, _, done, _ = env.step(policy(obs))
                    diagnosis.observe(finish_snapshot(cmd, names))
                    if bool(done[0]):
                        break
                else:
                    raise RuntimeError("Turn360 did not terminate by its configured deadline")
                outcome = ("success" if cmd.succeeded[0] else "timeout" if cmd.timed_out[0]
                           else "joint_speed_violation" if cmd.metrics["joint_speed_violation"][0]
                           else "failure")
                result = diagnosis.result(outcome)
                results.append(result)
                best = result["best_sample"]
                print(f"Episode {index + 1}: {outcome}; max hold={diagnosis.max_hold:.3f}s; "
                      f"near-goal steps={diagnosis.near_goal_steps}")
                print(f"  Closest sample: {best['rotation_deg']:.1f} deg at {best['time_s']:.2f}s; "
                      f"blocked by: {', '.join(best['failed_checks']) or 'hold duration only'}")
                print(f"  Worst pose: {best['worst_pose_joint']}={best['values']['joint_pose']:.3f} rad; "
                      f"fastest: {best['fastest_joint']}={best['values']['joint_speed']:.3f} rad/s")
                if index + 1 < episodes:
                    obs, _ = env.reset()
        return {
            "checkpoint": str(checkpoint.resolve()), "task": TASK, "seed": seed,
            "device": device, "num_envs": 1, "deterministic_actions": True,
            "observation_noise": False, "command_cfg": asdict(cmd.cfg), "episodes": results,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "evaluation_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True,
            ).strip(),
            "evaluation_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[2], text=True,
            ).strip()),
        }
    finally:
        raw_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.checkpoint_file.is_file():
        parser.error("checkpoint file does not exist")
    if args.episodes < 1:
        parser.error("episodes must be positive")
    result = evaluate(args.checkpoint_file, args.episodes, args.seed, args.device)
    output = args.output or args.checkpoint_file.parent / "evaluation" / f"{args.checkpoint_file.stem}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_finite_json(result), indent=2, allow_nan=False) + "\n")
    print(f"Detailed report: {output}")


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return value


if __name__ == "__main__":
    main()
