# One controlled, stable 360° turn

Task: `Mjlab-Turn360-Flat-MicroDuck`. Start standing, turn counterclockwise once,
then stand near the starting point and hold for 0.3 s. Hopping is allowed. No
wheels, reference motion or predefined gait. This is a custom experiment, not a
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

## Reward: v5, graded feedback for the standing finish

The v4 change removes contact-based interruption and progress gating in both
training and CPU playback. Airborne rotation and non-foot ground contact no
longer fail an attempt. The final standing hold still requires both feet down.
The v5 change adds finish-potential shaping and bounded terminal finish credit. Other reward
weights, hopping permissions, success thresholds and the 12 rad/s budget stay
unchanged, isolating the new finish feedback.

There is no target yaw rate or prescribed duration. The eight-second deadline
bounds an attempt; the servo-speed budget limits allowed motion. PPO's discount
still favors earlier rewards, so this is not mathematically timing-neutral.
The explicit time and remaining-angle running costs are removed.

Let theta be signed, unwrapped yaw relative to reset, and
`p_t = max(p_(t-1), clip(theta_t, 0, 2*pi))` on valid upright steps.

```text
reward = 4 * delta_progress / (2*pi)
       + 4 * first_reach_350_degrees
       + 1 * stable_one_foot_delta_progress / (2*pi)
       + 20 * success - 8 * (failure OR incomplete_deadline)
       + 10 * (0.99 * next_finish_potential - previous_finish_potential)
       + 6 * terminal_finish_quality_if_success_or_deadline
       - dt * (0.5 * sin(tilt)^2
               + 0.2 * (distance_from_start / 0.06)^2
               + 0.1 * mean((joint_error / 0.5)^2)
               + max((max_abs_servo_speed / 12 - 0.8) / 0.2, 0)^2
               + 0.001 * sum(actuator_torque^2))
       + roll/pitch-rate, action-rate and self-collision penalties
```

Progress pays while tilt is below 30 degrees and there is no failure, including
while airborne. Backtracking cannot repay visited angles;
spinning beyond 360 degrees earns no more progress. The once-per-episode rotation
bonus uses the same 10-degree tolerance as success. It does not require settling,
so completion can be learned before the final stance. A 65% turn earns partial
progress but no rotation bonus and no success bonus.

The optional style preference adds at most one reward unit over a full turn.
It pays only for new progress on exactly one foot, with tilt below 15 degrees
and drift below 6 cm. Parking or rocking on one foot cannot farm it. Waddling
still earns normal progress: the policy can discover either support strategy.
Two-foot support is required only for the final standing hold.

HOME regularization is gentle and constant throughout the episode, using a
0.5 rad scale. The v2 angle-dependent pose, joint-motion and braking costs are
removed: their growing costs near the finish likely contributed to stopping
short, although the desktop observations do not isolate a single cause.
The action-rate weight is reduced from -0.01 to -0.002 because its cost nearly
matched the progress reward in v2. The budget handles excessive speed directly.
No positive survival or standing reward is paid each step.

mjlab multiplies rewards by dt. Progress and impulse functions divide by dt;
the listed payouts are episode totals and the costs are per-second rates.
The frontier is bounded shaping, not policy-invariant potential shaping.
These weights are experimental hypotheses, not guarantees of convergence.

### Dense finish feedback

The old objective paid for rotation progress even when the policy never found
the strict standing finish. Its gentle HOME cost did not explicitly explain
how to brake after rotating. The user observed a jerky maneuver ending in
folded poses, with no successes. Sampled frames from the supplied recording
also show head-down/folded end states. They do not establish a collision or
interpenetration exploit; that would require contact/state telemetry.

The new score Phi describes the current finish, not the furthest angle reached:

```text
angle_cost   = ((theta - 2*pi) / (pi/2))^2
pose_cost    = (max_abs_joint_error / 0.75)^2
motion_cost  = (max_abs_joint_speed / 4)^2
             + (base_angular_speed / 3)^2 + (base_linear_speed / 0.2)^2
balance_cost = (tilt / radians(30))^2 + (drift / 0.12)^2
Phi = exp(-angle_cost - (pose_cost + motion_cost + balance_cost)/3)
      * (0.5 + 0.25 * number_of_grounded_feet)
```

Phi ranges from zero to one. The broad scales provide a gradient before the
much stricter success thresholds are reached. Worst-joint errors keep one bad
head or leg joint from disappearing into an average. Both undershoot and
overshoot reduce the score, as do moving limbs or base, poor posture, tilt and
drift. Foot support improves the finish score without forbidding hops. Reset
potential is defined as zero.

The shaping term uses `gamma * Phi(next) - Phi(previous)`, with gamma matching
PPO (0.99). Each absorbing terminal state has zero potential, including
success, failure, deadline and external terminations. The actual state score
is still logged at termination. A terminal shaping contribution can therefore
be negative even on success, alongside the unchanged +20 success bonus.

With matching discounts, these signed changes telescope over a discounted
episode. Repeated jitter, moving away and returning, crashing, or timing out
cannot retain an extra shaping return. This potential term improves local
reward/credit feedback without changing the underlying discounted objective.

Separately, success or an unfinished deadline pays `6 * Phi(actual end state)`
once. Physical failures receive no such credit. This is a deliberate change
to the task objective: a near-complete, quiet HOME stance is worth more than
a folded or moving partial turn even when neither meets strict success.
The maximum credit is below the unchanged -8 failure/deadline penalty, so the
terminal subtotal on a timeout stays negative. The +20 held-success bonus remains
the main finish payoff. This credit grades only the final state, so it cannot
be collected repeatedly by jittering during an attempt. Neither term guarantees
exploration will discover success; per-step standing payouts are avoided.

Scales and discount are recorded under `env.commands.twist.finish.*`; the
weights are `env.rewards.turn_finish_potential.weight` and
`env.rewards.turn_finish_credit.weight`. If overriding PPO gamma,
also set `--env.commands.twist.finish.discount` to the same value. MLflow logs
the signed `Episode_Reward/turn_finish_potential`, terminal
`Episode_Reward/turn_finish_credit`, and state diagnostics
`Metrics/twist/finish_quality` and `finish_hold_s`. The score is a diagnostic,
not a replacement for success. Evaluate actual video and the strict finish.

Success requires all of the following for 0.3 continuous seconds:

- Signed rotation within 10 degrees of +360 degrees, not heading modulo 360.
- Tilt below 15 degrees and horizontal drift below 6 cm.
- Angular speed below 0.3 rad/s and linear speed below 0.05 m/s.
- Both feet supported. Extra non-foot contact does not disqualify the finish.
- Every servo within 0.25 rad (about 14 degrees) of configured HOME.
- Every servo moving below 1.0 rad/s.

`joint_error` uses configured `default_joint_pos`, including all head and leg
joints. Exact world-space foot positions are not prescribed. These tolerances
allow balance corrections; their feasibility still needs learned-rollout tests.

Failure ends the attempt on tilt above 45 degrees, drift above 20 cm, invalid
state or any measured servo speed above its budget. The budget has no reset
grace period. Contact alone does not classify a fall; neither a body scrape nor
loss of foot support interrupts the maneuver. Physics and collisions remain
active. A fall is detected by the remaining posture/state checks.

An unfinished eight-second attempt receives the same -8 failure payout.
The deadline is a terminal task outcome, not a continuing-task truncation:
PPO does not bootstrap beyond it. Success on the deadline takes precedence;
a simultaneous physical failure receives only one failure payout.

## Joint-speed budget

The v3 default is **12 rad/s per servo**, configurable in training:

```text
--env.commands.twist.max-joint-speed 12.0
```

This is a first experimental choice informed by earlier v1 CPU peaks around
12 rad/s. Checkpoint 100's speed was not measured, and this is not a validated
hardware limit. A larger budget can be chosen if exploration is completely
blocked, but record it as a distinct experiment.

The worst servo, not the average across joints, determines both the soft cost
and violation. The soft cost starts at 80% of the budget (9.6 rad/s by default),
is one at the budget, and any value strictly above the budget fails the attempt
and blocks progress/completion rewards on that step. Passive joints are excluded.
Setting the optional budget to None disables both the speed cost and its failure
check. No yaw-speed budget is enabled; the optional `max-yaw-rate` remains a
training-only soft excess cost if explicitly configured.

The check uses measured velocity at 50 Hz. Substep peaks can be higher, and a
failed attempt has already exceeded the budget. **This is an episode constraint,
not a hard physical speed clamp or hardware safety guarantee.** CPU playback
hands back to walking on a violation and does not clamp actions, torque or speed.
A real motor-speed guarantee requires a matching lower-level controller.

The CPU runner defaults to the same 12 rad/s budget. For a different trained
budget, pass `--turn360-max-joint-speed VALUE` to `scripts/infer_policy.py`.
The runner applies current success/failure checks even to older ONNX files;
use the original code revision for an unchanged historical evaluation.

MLflow logs mean per-episode peak yaw/joint speeds, not the maximum across a run.
It also logs `rotation_completed`, `joint_speed_violation`, `timeout`, and
`finish_angle_ok`, `finish_pose_ok`, `finish_motion_ok`, `finish_support_ok` under
`Metrics/twist/`. The finish flags describe episode-end conditions, not a
continuous 0.3-second hold. They help distinguish partial turns, stance failures
and speed violations rather than interpreting zero success alone.

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
  these commands. CPU playback has a dedicated `--turn360` slot; the onboard
  runtime still needs a matching command generator before hardware deployment.

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

To inspect the task alone, use its training command generator:

```bash
uv run play Mjlab-Turn360-Flat-MicroDuck \
  --checkpoint-file logs/rsl_rl/microduck_turn360/YOUR_RUN/model_99.pt \
  --num-envs 1
```

Use the actual checkpoint filename; these are examples, not guaranteed outputs.

To trigger the learned spin alongside walking, use the ONNX automatically
exported into the training run's directory (named after that directory):

```bash
uv run scripts/infer_policy.py \
  --walking policies/velstand.onnx \
  --turn360 "logs/rsl_rl/microduck_turn360/YOUR_RUN/YOUR_RUN.onnx" \
  --new-cmd-obs
```

Press SPACE to stop, let the duck settle, then **I in the terminal** to start
the spin. Walking/recovery takes over after the trained stable-finish condition,
failure or eight-second simulation timeout. The next button press captures a
fresh heading/origin. You can also add `--roulade policies/roulade.onnx` and use
R for the existing roll. Turn360 defaults to the full-collision scene and uses
projected gravity; it cannot run on the roller model.

The CPU tracker shares goal thresholds with training and tests its command and
completion calculations against the training implementation. It uses simulator
pose/contact feedback, not a hardware estimator. Entry is restricted to settled,
upright foot support; this is not yet a learned walking-to-spin transition.

The standing-finish objective must be learned by retraining. A v1 policy may
still finish in an arbitrary head/leg configuration and fail the new criteria.
The walking handoff still needs rollout testing with the newly trained policy.
Entry while walking is deliberately blocked because training starts at rest.

### v5 finish-shaping experiment

Retraining is required to learn from the new finish feedback. Existing ONNX
actions and playback stopping rules are unchanged by the shaping term.

Pull branch `lukas`, then run the required five-iteration CUDA smoke test:

```bash
git pull --ff-only origin lukas
uv run --extra tracking train-tracked --run-name turn360-finish-v5-smoke \
  Mjlab-Turn360-Flat-MicroDuck \
  --env.scene.num-envs 64 --agent.max-iterations 5 --agent.seed 42
```

After it passes, a fresh comparison run uses 2048 environments (matching the
actual v2/v3 runs), seed 42 and 1000 iterations:

```bash
uv run --extra tracking train-tracked --run-name turn360-finish-v5-seed42 \
  Mjlab-Turn360-Flat-MicroDuck \
  --env.scene.num-envs 2048 --agent.max-iterations 1000 --agent.seed 42
```

Preserve v2 checkpoint 100 as the preferred qualitative motion baseline. Its
observed rotation was still only about 65%, not a full success. The final v2
checkpoint degraded to an awkward partial turn, with zero logged successes.
Inspect checkpoints during learning rather than assuming the last is best.
Multiple reward/constraint changes make v3 a revised objective, not an isolated
single-factor ablation. CPU tests do not establish CUDA execution or behavior.

The first v1 combined-runner measurements failed after 0.16 s at 62-73 degrees,
with peak yaw rates 11.35-12.84 rad/s and joint speeds 11.78-12.28 rad/s. These
were early posture/contact failures under the old rules. Non-foot contact no
longer stops the runner; none of these unit tests proves the learned
walking handoff works.

For periodic training videos, append `--video --video-interval 5000
--video-length 200` to the training command. Video rendering adds overhead;
leave it off for throughput comparisons. Videos are archived to MLflow at exit.

## CPU checks

```bash
uv run --extra tracking --with pytest pytest tests/test_turn360.py tests/test_tracking.py -q
```

Checks cover yaw wraparound, bounded/non-repeatable progress, stable completion,
failure conditions, partial resets, observation layout, complete scene/sensor compilation and
live TensorBoard-to-MLflow logging with parameter/artifact capture.

Initial local physics check (CPU MuJoCo, nominal BAM M6, full-collision robot,
no DR): the published `velstand` v9 policy held standing for three seconds,
ending at trunk height 0.1159 m and tilt 0.31°, with only foot-floor contacts.
Holding HOME joint targets without a learned controller tipped over in three
trials. The new policy therefore has to learn balance as well as turning;
neither this check nor the unit tests demonstrates that it has learned the task.
