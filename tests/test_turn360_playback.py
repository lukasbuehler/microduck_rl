"""CPU integration: actual MuJoCo contacts, ONNX sessions and policy handoff."""

import math
from pathlib import Path
import runpy

import mujoco
import numpy as np
import onnx
from onnx import helper, numpy_helper
import pytest


REPO = Path(__file__).resolve().parents[1]


def policy_file(path, name, value):
    tensor = numpy_helper.from_array(np.full((1, 14), value, dtype=np.float32))
    graph = helper.make_graph(
        [helper.make_node("Constant", [], [name + "_action"], value=tensor)], name,
        [helper.make_tensor_value_info(name + "_obs", onnx.TensorProto.FLOAT, [1, 61])],
        [helper.make_tensor_value_info(name + "_action", onnx.TensorProto.FLOAT, [1, 14])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, path)


@pytest.fixture
def playback(tmp_path):
    ip = runpy.run_path(str(REPO / "scripts/infer_policy.py"))
    model = mujoco.MjModel.from_xml_path(str(REPO / "src/mjlab_microduck/robot/microduck/scene_allcollisions.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("STAND").id)
    mujoco.mj_forward(model, data)
    # Feet are meshes, not axis-aligned boxes: lower until MuJoCo reports
    # actual contacts rather than inferring sole height from geom_size.
    for _ in range(200):
        mujoco.mj_forward(model, data)
        if data.ncon:
            break
        data.qpos[2] -= .0001
    assert data.ncon
    for name, value in (("walk", .01), ("spin", .02)):
        policy_file(tmp_path / f"{name}.onnx", name, value)
    policy = ip["PolicyInference"](
        model, data, walking_onnx_path=str(tmp_path / "walk.onnx"),
        turn360_onnx_path=str(tmp_path / "spin.onnx"), new_cmd_obs=True,
        use_projected_gravity=True,
    )
    return model, data, policy


def test_turn_button_generates_commands_and_returns_to_walking(playback, capsys):
    model, data, policy = playback
    policy.trigger_behavior("turn360")
    assert policy.behavior_mode == "turn360"
    np.testing.assert_array_equal(policy.command[:3], [1, 0, 0])
    np.testing.assert_array_equal(policy.command[3:], np.zeros(10))
    # Graphs deliberately have different IO names: session swaps must use the
    # active graph's names rather than the walking graph's cached names.
    np.testing.assert_allclose(policy.infer(), .02)
    for angle in list(range(30, 361, 30)) + [360] * 20:
        half = math.radians(angle) / 2
        data.qpos[3:7] = [math.cos(half), 0, 0, math.sin(half)]
        mujoco.mj_forward(model, data)
        # A huge wall-clock delta must not expire the simulation-time maneuver.
        policy.update_behavior(20., simulation_dt=.02)
        if policy.behavior_mode is None:
            break
    assert policy.behavior_mode is None
    assert policy.current_policy == "walking"
    assert policy.turn360 is None
    output = capsys.readouterr().out
    assert "Turn360: success" in output
    assert "Turn360 peaks: yaw=" in output
    np.testing.assert_array_equal(policy.command, np.zeros(13))
    np.testing.assert_allclose(policy.infer(), .01)
    # A second button press starts a fresh relative 360° goal.
    policy.trigger_behavior("turn360")
    assert policy.turn360.angle == 0
    assert policy.command[0] == 1


def test_requires_standing_entry_and_falls_hand_back_to_recovery(playback):
    model, data, policy = playback
    policy.set_vel_cmd(.1, 0, 0)
    policy.trigger_behavior("turn360")
    assert policy.behavior_mode is None
    policy.set_vel_cmd(0, 0, 0)
    policy.trigger_behavior("turn360")
    data.qpos[2] += .1
    mujoco.mj_forward(model, data)
    for _ in range(20):
        policy.update_behavior(.02, simulation_dt=.02)
    assert policy.behavior_mode == "turn360"  # a hop does not force handoff
    data.qpos[2] = .02
    mujoco.mj_forward(model, data)
    policy.update_behavior(.02, simulation_dt=.02)
    assert policy.behavior_mode == "turn360"  # contact alone is not a fall
    data.qpos[3:7] = [math.sqrt(.5), math.sqrt(.5), 0, 0]
    mujoco.mj_forward(model, data)
    policy.update_behavior(.02, simulation_dt=.02)
    assert policy.behavior_mode is None
    assert policy.current_policy == "walking"
