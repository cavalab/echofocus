#!/usr/bin/env python3
"""Generate and optionally submit Slurm jobs for EchoFocus experiments."""

from __future__ import annotations

import itertools
import json
import re
import shlex
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import fire


TRANSFORMER_ALIASES = {
    "customtransformer": "standard",
    "standard": "standard",
    "query": "query",
    "multiquery": "multiquery",
}


def _coerce_list(value: Any, name: str, cast: type | None = None) -> list[Any]:
    """Accept list-like Fire inputs and coerce them to a list."""
    if value is None:
        out: list[Any] = []
    elif isinstance(value, (list, tuple)):
        out = list(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            out = []
        elif text.startswith("["):
            out = json.loads(text)
        elif "," in text:
            out = [piece.strip() for piece in text.split(",") if piece.strip()]
        else:
            out = [piece.strip() for piece in text.split() if piece.strip()]
    else:
        out = [value]

    if cast is None:
        return out
    try:
        return [cast(item) for item in out]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be castable to {cast.__name__}") from exc


def _coerce_dict(value: Any, name: str) -> dict[str, Any]:
    """Accept dict-like Fire inputs and coerce them to a dictionary."""
    if value in (None, "", {}):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"{name} must be a dictionary or JSON dictionary string")


def _sanitize_name(text: str) -> str:
    """Convert arbitrary text into a Slurm/job-name-safe token."""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:120] if len(text) > 120 else text


def _load_config_file(config_path: str) -> dict[str, Any]:
    """Load a JSON or YAML experiment config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install with: pip install pyyaml"
            ) from exc
        config = yaml.safe_load(path.read_text())
    else:
        config = json.loads(path.read_text())

    if not isinstance(config, dict):
        raise ValueError("Config file must parse to a dictionary/object.")
    return config


def _normalize_transformer_type(value: str) -> str:
    """Map user-facing transformer names onto EchoFocus CLI values."""
    normalized = TRANSFORMER_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        allowed = ", ".join(sorted(TRANSFORMER_ALIASES))
        raise ValueError(
            f"Unsupported transformer_type/head value {value!r}. "
            f"Expected one of: {allowed}"
        )
    return normalized


def _normalize_optional_int(value: Any) -> int | None:
    """Normalize Fire/YAML representations of optional integer values."""
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"none", "null", ""}:
            return None
        return int(lowered)
    return int(value)


class SubmitJobs:
    """Generate Slurm scripts and optionally submit them via sbatch."""

    def run(
        self,
        experiment_name: str = "end2end",
        operation: str = "train",
        dataset: str = "bch_internal",
        tasks: list[str] | tuple[str, ...] = ("chd", "fyler", "measure"),
        transformer_types: list[str] | tuple[str, ...] = (
            "customtransformer",
            "multiquery",
        ),
        learning_rates: list[float] | tuple[float, ...] = (1e-4, 3e-4, 1e-3),
        num_clips: list[int] | tuple[int, ...] = (4, 8, 16),
        clip_lengths: list[int] | tuple[int, ...] = (16, 32),
        max_videos_per_study: list[int | None] | tuple[int | None, ...] = (
            None,
            25,
            50,
        ),
        seeds_file: str = "seeds.txt",
        ntrials: int = 1,
        base_model_name: str = "echofocus_e2e",
        train_args: dict[str, Any] | None = None,
        account: str = "chip-lacava",
        partition: str = "gpu-chip-lacava,bch-gpu,bch-gpu-pe",
        gpus: int | str = 1,
        time_limit: str = "7-00:00:00",
        ntasks: int = 1,
        cpus_per_task: int = 8,
        mem: str = "600G",
        qos: str | None = "normal",
        command_prefix: str = "uv run",
        cli_script: str = "echofocus.py",
        jobs_dir: str = "slurm_jobs",
        logs_dir: str = "slurm_logs",
        venv_activate: str | None = None,
        submit: bool = False,
        print_only: bool = False,
    ):
        """Generate one Slurm job per hyperparameter combination and seed."""
        tasks_list = _coerce_list(tasks, "tasks", str)
        transformer_list = _coerce_list(transformer_types, "transformer_types", str)
        lr_list = _coerce_list(learning_rates, "learning_rates", float)
        clips_list = _coerce_list(num_clips, "num_clips", int)
        clip_len_list = _coerce_list(clip_lengths, "clip_lengths", int)
        videos_list = [
            _normalize_optional_int(value)
            for value in _coerce_list(max_videos_per_study, "max_videos_per_study")
        ]
        extra_train_args = _coerce_dict(train_args, "train_args")

        if not tasks_list:
            raise ValueError("At least one task is required.")
        if not transformer_list:
            raise ValueError("At least one transformer type is required.")
        if not lr_list:
            raise ValueError("At least one learning rate is required.")
        if not clips_list:
            raise ValueError("At least one num_clips value is required.")
        if not clip_len_list:
            raise ValueError("At least one clip length is required.")
        if not videos_list:
            raise ValueError("At least one max_videos_per_study value is required.")

        seeds = self._load_seeds(seeds_file, ntrials)

        jobs_path = Path(jobs_dir)
        logs_path = Path(logs_dir)
        jobs_path.mkdir(parents=True, exist_ok=True)
        logs_path.mkdir(parents=True, exist_ok=True)

        created_scripts: list[Path] = []
        submitted_job_ids: list[str] = []

        combos = itertools.product(
            seeds,
            tasks_list,
            transformer_list,
            lr_list,
            clips_list,
            clip_len_list,
            videos_list,
        )

        for seed, task, transformer, learning_rate, nclips, clip_len, max_videos in combos:
            normalized_transformer = _normalize_transformer_type(transformer)
            run_args = deepcopy(extra_train_args)
            run_args.update(
                {
                    "dataset": dataset,
                    "task": task,
                    "seed": seed,
                    "learning_rate": learning_rate,
                    "transformer_type": normalized_transformer,
                    "num_clips": nclips,
                    "clip_len": clip_len,
                    "max_videos_per_study": max_videos,
                }
            )
            model_name = self._build_model_name(
                base_model_name=base_model_name,
                experiment_name=experiment_name,
                task=task,
                transformer=transformer,
                learning_rate=learning_rate,
                num_clips=nclips,
                clip_len=clip_len,
                max_videos_per_study=max_videos,
                seed=seed,
            )
            run_args["model_name"] = model_name

            command = [*shlex.split(str(command_prefix)), cli_script, operation]
            command.extend(self._args_to_cli(run_args))
            command_str = " ".join(shlex.quote(piece) for piece in command)

            job_name = self._build_job_name(
                experiment_name=experiment_name,
                task=task,
                transformer=transformer,
                learning_rate=learning_rate,
                num_clips=nclips,
                clip_len=clip_len,
                max_videos_per_study=max_videos,
                seed=seed,
            )
            script_path = jobs_path / f"{job_name}.sbatch"
            script_text = self._build_script(
                job_name=job_name,
                log_dir=logs_path,
                account=account,
                partition=partition,
                gpus=str(gpus),
                time_limit=time_limit,
                ntasks=ntasks,
                cpus_per_task=cpus_per_task,
                mem=mem,
                qos=qos,
                venv_activate=venv_activate,
                command_str=command_str,
            )
            script_path.write_text(script_text)
            created_scripts.append(script_path)

            if print_only:
                print(f"[{job_name}] {command_str}")
                continue

            if submit:
                sbatch_cmd = ["sbatch", str(script_path)]
                try:
                    proc = subprocess.run(
                        sbatch_cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except subprocess.CalledProcessError as exc:
                    stdout = (exc.stdout or "").strip()
                    stderr = (exc.stderr or "").strip()
                    raise RuntimeError(
                        "sbatch failed for "
                        f"{script_path}\n"
                        f"command: {' '.join(sbatch_cmd)}\n"
                        f"stdout: {stdout or '<empty>'}\n"
                        f"stderr: {stderr or '<empty>'}"
                    ) from exc
                output = proc.stdout.strip()
                job_id = output.split()[-1] if output else ""
                submitted_job_ids.append(job_id)
                print(f"submitted {script_path} -> {output}")
            else:
                print(f"generated {script_path}")

        print(f"total_jobs={len(created_scripts)}")
        if submit and submitted_job_ids:
            print(f"submitted_jobs={len(submitted_job_ids)}")
            print(f"job_ids={','.join(submitted_job_ids)}")
        return {
            "total_jobs": len(created_scripts),
            "jobs_dir": str(jobs_path),
            "logs_dir": str(logs_path),
            "submitted_job_ids": submitted_job_ids,
        }

    def run_config(
        self,
        config: str,
        submit: bool | None = None,
        print_only: bool | None = None,
    ):
        """Load a JSON/YAML config and generate one or more experiment sweeps."""
        params = _load_config_file(config)
        if submit is not None:
            params["submit"] = submit
        if print_only is not None:
            params["print_only"] = print_only

        shared = {k: v for k, v in params.items() if k != "experiments"}
        experiments = params.get("experiments")

        if experiments is None:
            return self.run(**shared)

        if not isinstance(experiments, list) or not experiments:
            raise ValueError("'experiments' must be a non-empty list when provided.")

        all_job_ids: list[str] = []
        total_jobs = 0
        results = []
        for experiment in experiments:
            if not isinstance(experiment, dict):
                raise ValueError("Each entry in 'experiments' must be a dictionary.")
            merged = deepcopy(shared)
            shared_train_args = deepcopy(merged.get("train_args", {}))
            experiment_train_args = deepcopy(experiment.get("train_args", {}))
            merged.update(experiment)
            if shared_train_args or experiment_train_args:
                merged["train_args"] = {
                    **shared_train_args,
                    **experiment_train_args,
                }
            result = self.run(**merged)
            total_jobs += int(result["total_jobs"])
            all_job_ids.extend(result["submitted_job_ids"])
            results.append(result)

        return {
            "total_jobs": total_jobs,
            "submitted_job_ids": all_job_ids,
            "experiments": results,
        }

    @staticmethod
    def _load_seeds(seeds_file: str, ntrials: int) -> list[int]:
        seeds_path = Path(seeds_file)
        if not seeds_path.exists():
            raise FileNotFoundError(f"seeds file not found: {seeds_file}")

        seeds: list[int] = []
        for line in seeds_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            seeds.append(int(stripped))
            if len(seeds) >= ntrials:
                break

        if len(seeds) < ntrials:
            raise ValueError(
                f"Requested ntrials={ntrials}, but only found "
                f"{len(seeds)} seeds in {seeds_file}"
            )
        return seeds

    @staticmethod
    def _format_cli_value(value: Any) -> str:
        """Render a Python value into a Fire-friendly CLI token."""
        if value is None:
            return "None"
        if isinstance(value, bool):
            return "True" if value else "False"
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value)
        return str(value)

    @classmethod
    def _args_to_cli(cls, args: dict[str, Any]) -> list[str]:
        cli_args: list[str] = []
        for key, value in args.items():
            cli_args.extend([f"--{key}", cls._format_cli_value(value)])
        return cli_args

    @staticmethod
    def _build_job_name(
        *,
        experiment_name: str,
        task: str,
        transformer: str,
        learning_rate: float,
        num_clips: int,
        clip_len: int,
        max_videos_per_study: int | None,
        seed: int,
    ) -> str:
        lr_token = str(learning_rate).replace(".", "p")
        max_videos_token = "all" if max_videos_per_study is None else str(max_videos_per_study)
        return _sanitize_name(
            f"{experiment_name}_{task}_{transformer}"
            f"_lr{lr_token}_clips{num_clips}_len{clip_len}"
            f"_mv{max_videos_token}_s{seed}"
        )

    @staticmethod
    def _build_model_name(
        *,
        base_model_name: str,
        experiment_name: str,
        task: str,
        transformer: str,
        learning_rate: float,
        num_clips: int,
        clip_len: int,
        max_videos_per_study: int | None,
        seed: int,
    ) -> str:
        lr_token = str(learning_rate).replace(".", "p")
        max_videos_token = "all" if max_videos_per_study is None else str(max_videos_per_study)
        return _sanitize_name(
            f"{base_model_name}_{experiment_name}_{task}_{transformer}"
            f"_lr{lr_token}_clips{num_clips}_len{clip_len}"
            f"_mv{max_videos_token}_s{seed}"
        )

    @staticmethod
    def _build_script(
        *,
        job_name: str,
        log_dir: Path,
        account: str,
        partition: str,
        gpus: str,
        time_limit: str,
        ntasks: int,
        cpus_per_task: int,
        mem: str,
        qos: str | None,
        venv_activate: str | None,
        command_str: str,
    ) -> str:
        now = datetime.now().isoformat(timespec="seconds")
        lines = [
            "#!/bin/bash",
            f"# generated {now}",
            f"#SBATCH --account={account}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --gres=gpu:{gpus}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --output={log_dir}/%j_{job_name}.txt",
            f"#SBATCH --ntasks={ntasks}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --mem={mem}",
        ]
        if qos:
            lines.append(f"#SBATCH --qos={qos}")
        lines.append("set -euo pipefail")
        if venv_activate:
            lines.append(f"source {shlex.quote(venv_activate)}")
        lines.extend(
            [
                "hostname",
                command_str,
                "",
            ]
        )
        return "\n".join(lines)


if __name__ == "__main__":
    fire.Fire(SubmitJobs)
