# One fast, stable 360° turn

Task: `Mjlab-Turn360-Flat-MicroDuck`. Start standing, turn counterclockwise once
on the feet, stop near the starting point and hold for 0.3 s. No wheels, jumping,
reference motion or predefined gait. This is the first custom experiment, not a
claim of a trained result.

## Setup and first run (Linux/CUDA)

From the repository root:

```bash
git pull --ff-only origin lukas
uv sync --locked --extra tracking

# Required first check: 64 environments, five PPO updates.
uv run --extra tracking train-tracked --run-name turn360-smoke \
  Mjlab-Turn360-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5 --agent.seed 42
```

The first Warp launch compiles kernels. Check for NaNs/configuration errors and
confirm the checkpoint and ONNX export complete before a longer run. This smoke
test must run on the desktop; the Mac CPU tests do not establish GPU compatibility.

Start the live UI in a second terminal, also from the repository root:

```bash
uv run --extra tracking mlflow ui \
  --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 on the desktop. Metrics are copied from TensorBoard
about every five seconds. No hosted account is required. To view from the laptop,
forward the port with `ssh -L 5000:127.0.0.1:5000 user@desktop`, then open the same
URL on the laptop. Keep the UI bound to loopback.

After the smoke test passes, benchmark 1024/4096 environments before choosing a
full run. A first learning run could be:

```bash
uv run --extra tracking train-tracked --run-name turn360-v1-seed42 \
  Mjlab-Turn360-Flat-MicroDuck \
  --env.scene.num-envs 1024 --agent.max-iterations 1000 --agent.seed 42
```

Changing environment count also changes PPO batch size. Keep it fixed when
comparing reward versions. Repeat promising comparisons across several seeds.

## Reward

Let theta be signed, unwrapped yaw relative to reset, and
`p_t = max(p_(t-1), clip(theta_t, 0, 2*pi))` on valid supported steps.

```text
reward = 8 * (p_t - p_(t-1)) / (2*pi)
       + 4 * success - 8 * failure
       - dt * (0.05
               + 0.5 * min(abs(2*pi - theta)/(2*pi), 2)
               + 0.5 * sin(tilt)^2
               + 0.2 * (distance_from_start / 0.06)^2
               + 0.05 * braking_gate * angular_speed^2
               + 0.001 * sum(actuator_torque^2))
       + inherited roll/pitch-rate, action-rate and self-collision penalties
```

`braking_gate` rises linearly from 0 at 270° to 1 at 360°. Progress only pays
while at least one foot touches the floor, tilt is below 30°, and there is no
failure. Backtracking cannot repay an already visited angle; spinning past 360°
earns no further progress. Staying still earns nothing and incurs time/error
costs. Unlike `completed_angle / elapsed_time` paid every step, old progress
cannot be collected repeatedly. The frontier is a bounded progress signal, not
a claim of policy-invariant potential shaping under PPO's discounting.

mjlab multiplies rewards by `dt`, so progress and terminal functions divide by
`dt` internally. Their weights are total payouts; other weights are per-second
cost coefficients. Every component is logged separately. These weights are an
initial hypothesis to evaluate, not established optimal values.

Success requires all of the following for 0.3 continuous seconds:

- Signed rotation within 10° of +360° (not heading modulo 360°).
- Tilt below 15° and horizontal drift below 6 cm.
- Angular speed below 0.3 rad/s and linear speed below 0.05 m/s.
- Foot support and no non-foot ground contact.

Failure ends the episode on non-foot ground contact, tilt above 45°, drift
above 20 cm, invalid state, or over 60 ms airborne after a 250 ms reset grace
period. Episodes time out at eight seconds. A failure penalty discourages ending
an episode early merely to escape running costs. The airborne tolerance allows
brief contact-sensor gaps; it is not a license for a ballistic maneuver. Inspect
videos as well as success metrics.

## Architecture and reproducibility

- MDP state/rewards: the Turn360 section in `tasks/mdp.py`, following repo convention.
- Configuration: `tasks/microduck_turn360_env_cfg.py`; registry: `tasks/__init__.py`.
- Uses the full-collision model so head/body contacts cannot pass through the
  floor. BAM actuators, actuator delays, encoder bias and IMU noise come from
  the walking configuration. Physics DR stays at its initial ranges, without
  inherited walking curricula or external pushes.
- The actor remains 61D. Its three `twist` slots now contain **remaining signed
  turns and body-frame x/y displacement back to the origin**, normalized by
  the success radius. Head/body command slots are zero. Remaining angle and
  displacement currently use simulator state; hardware would need an estimator.
- Same observation size does **not** mean the standard runtime understands
  these commands. Do not deploy this ONNX through generic velocity playback.

`train-tracked` launches the unmodified trainer with TensorBoard logging and
mirrors scalar events to MLflow, with CPU/RAM and (on NVIDIA Linux) GPU system
metrics sampled in the background. Each run records resolved environment/agent
parameters (including reward weights), task, seed, Git commit/dirty status,
invocation, lockfile and working-tree diff. Complete YAML files are artifacts;
small fields are also searchable MLflow parameters. Checkpoints, ONNX files and
recorded videos are copied at exit. Local SQLite and artifact storage are
ignored by Git. Preserve both `mlflow.db` and its artifacts when backing up.

For an existing server, supply `--tracking-uri http://host:5000` **before** the
task name, or set `MLFLOW_TRACKING_URI`. That explicitly sends experiment data
and artifacts to that server. An interrupted run is marked KILLED; trainer
errors mark it FAILED. Abrupt power loss/kill -9 cannot run cleanup, but local
TensorBoard logs and checkpoints remain.

Watch component rewards, `Metrics/twist/success`, turns, yaw error, duration,
peak drift/tilt, episode terminations, throughput and PPO losses. Training
episode averages are diagnostics, not a fixed-distribution benchmark. Record
the code revision, seed, number of episodes and checkpoint when evaluating.

## Playback and videos

Use the task's command generator, not `scripts/infer_policy.py`:

```bash
uv run play Mjlab-Turn360-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_turn360/YOUR_RUN/model_99.pt \
  --num-envs 1
```

Use the actual checkpoint filename; these are examples, not guaranteed outputs.
For periodic training videos, append `--video --video-interval 5000
--video-length 200` to the training command. Video rendering adds overhead;
leave it off for throughput comparisons. Videos are archived to MLflow at exit.

## CPU checks

```bash
uv run --extra tracking --with pytest pytest tests/test_turn360.py tests/test_tracking.py -q
```

Checks cover yaw wraparound, bounded/non-repeatable progress, stable completion,
failure conditions, partial resets, observation layout, model compilation and
live TensorBoard-to-MLflow logging with parameter/artifact capture.

Initial local physics check (CPU MuJoCo, nominal BAM M6, full-collision robot,
no DR): the published `velstand` v9 policy held standing for three seconds,
ending at trunk height 0.1159 m and tilt 0.31°, with only foot-floor contacts.
Holding HOME joint targets without a learned controller tipped over in three
trials. The new policy therefore has to learn balance as well as turning;
neither this check nor the unit tests demonstrates that it has learned the task.
