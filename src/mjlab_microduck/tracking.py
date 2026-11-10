"""Run the official trainer with a live TensorBoard -> MLflow bridge.

No trainer monkey-patching. Events remain the local source of truth; MLflow
receives every scalar step, resolved YAML parameters, code provenance and final
checkpoints/videos. Install with `uv sync --extra tracking`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import uuid


def flatten_params(value, prefix=""):
    """Indexable config; regex-keyed maps stay JSON values, not MLflow keys.

    Joint selectors such as `.*` are valid configuration keys but invalid
    MLflow parameter names. Keep these maps intact instead of lossy renaming.
    Full/large structures also remain available in the YAML artifacts.
    """
    result = {}
    if isinstance(value, dict) and all(re.fullmatch(r"[\w-]+", str(key)) for key in value):
        for key, child in value.items():
            result.update(flatten_params(child, f"{prefix}.{key}" if prefix else str(key)))
    else:
        text = json.dumps(value, default=str)
        if len(prefix) <= 250 and len(text) <= 500:
            result[prefix] = text
    return result


def read_config(path):
    """Read mjlab's yaml.dump output without constructing Python objects."""
    import yaml

    class ConfigLoader(yaml.SafeLoader):
        pass

    def python_tag(loader, suffix, node):
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        return loader.construct_scalar(node) or suffix

    ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", python_tag)
    with Path(path).open() as stream:
        return yaml.load(stream, Loader=ConfigLoader)


def git_output(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def take_option(args, name, default):
    """Consume one trainer option (both tyro spellings and --key=value)."""
    remaining, value = [], default
    index = 0
    while index < len(args):
        key, equals, inline = args[index].partition("=")
        if key.replace("_", "-") == name:
            if equals:
                value = inline
            else:
                index += 1
                if index == len(args):
                    raise ValueError(f"Missing value for {key}")
                value = args[index]
        else:
            remaining.append(args[index])
        index += 1
    return remaining, value


class TensorboardBridge:
    def __init__(self, client, run_id, run_dir):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        self.client, self.run_id, self.run_dir = client, run_id, Path(run_dir)
        self.reader = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
        self.seen = set()
        self.params_logged = set()

    def sync(self, final=False):
        from mlflow.entities import Metric, Param

        has_events = any(self.run_dir.glob("events.out.tfevents.*"))
        if not has_events and not final:
            return  # config dumps may still be in progress before writer creation
        # The trainer writes configs before emitting scalar events. Log resolved
        # defaults too, rather than recording only the user's CLI overrides.
        for path in sorted((self.run_dir / "params").glob("*.yaml")):
            if path in self.params_logged:
                continue
            config = read_config(path)
            params = [Param(k, v) for k, v in flatten_params(config, path.stem).items()]
            for start in range(0, len(params), 100):
                self.client.log_batch(self.run_id, params=params[start:start + 100])
            self.client.log_artifact(self.run_id, str(path), "params")
            self.params_logged.add(path)
        if not has_events:
            return
        self.reader.Reload()
        metrics, identifiers = [], []
        for tag in self.reader.Tags().get("scalars", []):
            for event in self.reader.Scalars(tag):
                key = (tag, event.step, event.wall_time)
                if key not in self.seen and math.isfinite(event.value):
                    # Keep reward components/losses/throughput separately queryable.
                    metrics.append(Metric(tag, event.value, int(event.wall_time * 1000), event.step))
                    identifiers.append(key)
        for start in range(0, len(metrics), 1000):
            self.client.log_batch(self.run_id, metrics=metrics[start:start + 1000])
            self.seen.update(identifiers[start:start + 1000])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    parser.add_argument("--experiment", default="microduck-turn360")
    parser.add_argument("--run-name", default="turn360")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("task")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if "--hf-jobs" in args.train_args:
        parser.error("Run train-tracked on the GPU host; HF Jobs needs its own tracking setup")
    try:
        import mlflow
        from mlflow import MlflowClient
    except ImportError:
        parser.error("Install MLflow first: uv sync --extra tracking")

    train_args, logger = take_option(args.train_args, "--agent.logger", "tensorboard")
    if logger != "tensorboard":
        parser.error("train-tracked requires --agent.logger tensorboard")
    train_args, train_name = take_option(train_args, "--agent.run-name", args.run_name)
    # Unique suffix also discovers the run directory without parsing ANSI logs.
    suffix = uuid.uuid4().hex[:12]
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in train_name)
    command = [sys.executable, "-m", "mjlab.scripts.train", args.task, *train_args,
               "--agent.logger", "tensorboard", "--agent.run-name", f"{safe_name}_{suffix}"]
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    client = MlflowClient()
    bridge = None
    process = None
    status = "FAILED"
    with mlflow.start_run(run_name=train_name, log_system_metrics=True) as run:
        run_id = run.info.run_id
        mlflow.set_tags({"task": args.task, "git.commit": git_output("rev-parse", "HEAD"),
                         "git.branch": git_output("branch", "--show-current"),
                         "git.dirty": str(bool(git_output("status", "--porcelain"))),
                         "host": os.uname().nodename})
        mlflow.log_params({"task_id": args.task, "python": sys.version.split()[0]})
        with tempfile.TemporaryDirectory() as temporary:
            provenance = Path(temporary)
            (provenance / "command.json").write_text(json.dumps(command, indent=2))
            (provenance / "working-tree.patch").write_text(git_output("diff", "HEAD"))
            (provenance / "git-status.txt").write_text(git_output("status", "--porcelain"))
            mlflow.log_artifacts(str(provenance), "provenance")
        for path in ("pyproject.toml", "uv.lock"):
            if Path(path).exists():
                mlflow.log_artifact(path, "provenance")
        print(f"MLflow run: {run_id}\nTracking URI: {args.tracking_uri}", flush=True)
        try:
            process = subprocess.Popen(command)
            while True:
                if bridge is None:
                    matches = list(Path("logs/rsl_rl").glob(f"*/*_{suffix}"))
                    if matches:
                        bridge = TensorboardBridge(client, run_id, matches[0])
                        mlflow.set_tag("training.log_dir", str(matches[0]))
                if bridge is not None:
                    bridge.sync()
                code = process.poll()
                if code is not None:
                    status = "FINISHED" if code == 0 else "FAILED"
                    break
                time.sleep(args.poll_seconds)
        except KeyboardInterrupt:
            status = "KILLED"
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            code = 130
        finally:
            # A tracking failure must not leave an unobserved GPU job running.
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            try:
                if bridge is not None:
                    bridge.sync(final=True)  # writer has flushed at process exit
                    for pattern in ("*.pt", "*.onnx", "videos/**/*.mp4", "git/*"):
                        for path in bridge.run_dir.glob(pattern):
                            if path.is_file():
                                client.log_artifact(run_id, str(path), str(Path("training") / path.relative_to(bridge.run_dir).parent))
            finally:
                mlflow.end_run(status=status)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
