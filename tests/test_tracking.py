"""Real TensorBoard events and an isolated local MLflow store, no GPU/server."""

from pathlib import Path

import pytest

from mjlab_microduck.tracking import flatten_params, read_config, take_option


def test_config_reader_handles_python_yaml_without_instantiating_objects(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("actor:\n  dims: !!python/tuple [512, 256]\nfunc: !!python/name:os.system ''\n")
    config = read_config(path)
    assert config["actor"]["dims"] == [512, 256]
    assert config["func"] == "name:os.system"
    assert flatten_params(config)["actor.dims"] == "[512, 256]"


def test_trainer_option_spellings():
    assert take_option(["--agent.run_name=x", "--agent.seed", "42"], "--agent.run-name", "y") == (
        ["--agent.seed", "42"], "x",
    )
    with pytest.raises(ValueError):
        take_option(["--agent.run-name"], "--agent.run-name", "y")


def test_regex_selectors_are_preserved_as_json_values():
    params = flatten_params({"joint_pos": {".*": 0.0, ".*hip_yaw.*": 0.1},
                             "reward": {"weight": 8.0}}, "env")
    assert params["env.joint_pos"] == '{".*": 0.0, ".*hip_yaw.*": 0.1}'
    assert params["env.reward.weight"] == "8.0"
    assert not any("*" in key for key in params)


def test_bridge_waits_for_config_dump_to_finish(tmp_path):
    pytest.importorskip("mlflow")
    from mjlab_microduck.tracking import TensorboardBridge
    (tmp_path / "params").mkdir()
    (tmp_path / "params/env.yaml").write_text('unfinished: "')
    bridge = TensorboardBridge(None, "not-used", tmp_path)
    bridge.sync()  # no event writer yet: do not read a partially written YAML
    assert not bridge.params_logged


def test_live_bridge_logs_new_events_once_and_resolved_params(tmp_path):
    pytest.importorskip("mlflow")
    from mlflow import MlflowClient
    from torch.utils.tensorboard import SummaryWriter
    from mjlab_microduck.tracking import TensorboardBridge

    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'tracking.db'}")
    experiment = client.create_experiment("bridge-test", artifact_location=(tmp_path / "artifacts").as_uri())
    run_id = client.create_run(experiment).info.run_id
    directory = tmp_path / "training"
    (directory / "params").mkdir(parents=True)
    (directory / "params/agent.yaml").write_text("seed: 42\nalgorithm:\n  learning_rate: 0.001\n")
    # Exercise the actual environment config, including terrain/joint regex
    # dictionaries that simple synthetic configs did not cover.
    from dataclasses import asdict
    from mjlab.utils.os import dump_yaml
    from mjlab_microduck.tasks.microduck_turn360_env_cfg import make_microduck_turn360_env_cfg
    env_cfg = make_microduck_turn360_env_cfg()
    dump_yaml(directory / "params/env.yaml", asdict(env_cfg))
    writer = SummaryWriter(str(directory))
    writer.add_scalar("Episode_Reward/turn_progress", 0.25, 1)
    writer.flush()
    bridge = TensorboardBridge(client, run_id, directory)
    bridge.sync()
    bridge.sync()
    writer.add_scalar("Episode_Reward/turn_progress", 0.5, 2)
    writer.close()
    bridge.sync()
    history = client.get_metric_history(run_id, "Episode_Reward/turn_progress")
    assert [(m.step, m.value) for m in history] == [(1, 0.25), (2, 0.5)]
    assert client.get_run(run_id).data.params["agent.seed"] == "42"
    assert client.get_run(run_id).data.params["agent.algorithm.learning_rate"] == "0.001"
    params = client.get_run(run_id).data.params
    assert params["env.rewards.turn_progress.weight"] == str(env_cfg.rewards["turn_progress"].weight)
    assert params["env.commands.twist.max_joint_speed"] == "12.0"
    assert params["env.rewards.turn_success.weight"] == "20.0"
    assert params["env.commands.twist.finish.discount"] == "0.99"
    assert params["env.rewards.turn_finish_potential.weight"] == "10.0"
    assert params["env.rewards.turn_finish_credit.weight"] == "6.0"
    joint_key = "env.scene.terrain.init_state.joint_pos"
    assert ".*" in client.get_run(run_id).data.params[joint_key]
    assert client.list_artifacts(run_id, "params")[0].path == "params/agent.yaml"
    client.set_terminated(run_id)


@pytest.mark.parametrize("exit_code, status", [(0, "FINISHED"), (1, "FAILED")])
def test_launcher_tracks_trainer_exit_and_artifacts(tmp_path, monkeypatch, exit_code, status):
    mlflow = pytest.importorskip("mlflow")
    from torch.utils.tensorboard import SummaryWriter
    from mjlab_microduck import tracking

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tracking, "git_output", lambda *args: "test-commit" if args[0] == "rev-parse" else "")

    class Trainer:
        def __init__(self, command):
            name = command[command.index("--agent.run-name") + 1]
            directory = Path("logs/rsl_rl/turn360") / f"timestamp_{name}"
            directory.mkdir(parents=True)
            (directory / "model_4.pt").write_bytes(b"checkpoint")
            writer = SummaryWriter(str(directory))
            writer.add_scalar("Train/mean_reward", -1., 4)
            writer.close()

        def poll(self):
            return exit_code

    monkeypatch.setattr(tracking.subprocess, "Popen", Trainer)
    uri = f"sqlite:///{tmp_path / 'launcher.db'}"
    result = tracking.main(["--tracking-uri", uri, "--experiment", "launch-test",
                            "--run-name", "smoke", "Task", "--agent.seed", "42"])
    assert result == exit_code
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name("launch-test")
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.info.status == status
    assert run.data.metrics["Train/mean_reward"] == -1.
    assert client.list_artifacts(run.info.run_id, "training")[0].path == "training/model_4.pt"
