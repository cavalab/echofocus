# -*- coding: utf-8 -*-
"""Main entry point for training and evaluating EchoFocus models.

Authors: Platon Lukyanenko, William La Cava
"""

import pandas as pd
import os
import json
import ast
from dataclasses import asdict

import torch
import time
from datetime import datetime
import sys
import torch.profiler
import subprocess
import torch.multiprocessing as mp
from collections import OrderedDict

# import cv2
import numpy as np
# from torchvision import tv_tensors
# from torchvision.transforms import resize, center_crop

# from torchvision.transforms import v2


import uuid

from . import utils
from . import data as data_ops
from . import evaluation as evaluation_ops
from . import monitoring as monitoring_ops
from . import training as training_ops
from .checkpoints import (
    load_model_and_random_state,
    save_nn,
)
from .state import RuntimeConfig
from .state import EvaluateConfig, EmbedConfig, ExplainConfig, TrainPingPongConfig


class EchoFocus:
    """Train, evaluate, and explain EchoFocus models."""

    @utils.initializer  # this decorator automatically sets arguments to class attributes.
    def __init__(
        self,
        model_name=None,
        dataset=None,
        task='measure',
        seed=0,
        batch_number=128, # number of batches processed before updating
        batch_size=1,
        total_epochs=10,
        epoch_early_stop=9999,
        learning_rate=0.0001,  # default to 1e-4
        encoder_depth=0,
        clip_dropout=0.,
        transformer_type="standard",
        parallel_processes=1,
        sample_limit=1e10,
        preload_embeddings=False,
        run_id=None,
        config='config.json',
        split=(64,16,20),
        cache_video_tensors=False,
        cache_panecho_embeddings=False,
        end_to_end=True,
        panecho_trainable=True,
        transformer_trainable=True,
        load_transformer_path=None,
        load_panecho_path=None,
        load_strict=False,
        num_clips=16,
        clip_len=16,
        use_hdf5_index=False,
        label_path=None,
        embedding_path=None,
        video_base_path=None,
        video_subdir_format="{echo_id}_trim",
        max_videos_per_study=None,
        max_video_cache_gb=250,
        max_panecho_cache_gb=None,
        smoke_train=False,
        smoke_num_steps=2,
        debug_mem=False,
        amp=False,
        checkpoint_panecho=False,
        profile=False,
        profile_steps=20,
        profile_dir="profiles",
        timing_every=None,
        profile_summary=False,
        gpu_monitor=False,
        gpu_monitor_interval=10,
        ram_monitor=False,
        ram_monitor_interval=10,
        sharing_strategy="file_descriptor",
    ):
        """Initialize training/evaluation state and load config.

        Args:
            model_name (str|None): Name for the model run directory.
            dataset (str|None): Dataset key in the config file.
            task (str): Task key in the config file.
            seed (int): RNG seed for reproducibility.
            batch_number (int): Gradient accumulation steps.
            batch_size (int): Batch size (only 1 supported).
            total_epochs (int): Max epochs to train; -1 for eval-only.
            epoch_early_stop (int): Early stopping patience in epochs.
            learning_rate (float): Optimizer learning rate.
            encoder_depth (int): Number of transformer encoder layers.
            clip_dropout (float): Dropout probability for clip embeddings.
            transformer_type (str): Study-level transformer variant: ``"standard"``, ``"query"``, or ``"multiquery"``.
            parallel_processes (int): Number of dataloader workers.
            sample_limit (int): Limit number of samples.
            run_id (str|None): Optional run ID for reproducibility.
            config (str): Path to config JSON file.
            split (tuple[int, int, int]): Default train/val/test PID split.
            cache_video_tensors (bool): Cache raw video clips in memory (end-to-end only).
            cache_panecho_embeddings (bool): Cache PanEcho embeddings (end-to-end frozen) or precomputed embeddings (non end-to-end).
            end_to_end (bool): If True, train end-to-end with PanEcho backbone.
            panecho_trainable (bool): If True, fine-tune the PanEcho backbone.
            transformer_trainable (bool): If True, train the study-level transformer.
            load_transformer_path (str|None): Optional path to load transformer weights.
            load_panecho_path (str|None): Optional path to load PanEcho weights.
            load_strict (bool): If True, enforce strict loading for submodules.
            num_clips (int): Number of clips sampled per video.
            clip_len (int): Frames per clip.
            use_hdf5_index (bool): Use embedding HDF5s to locate video paths.
            video_base_path (str): Base path for raw study folders.
            video_subdir_format (str): Format for study folder under base path.
            max_videos_per_study (int|None): Optional cap on videos per study.
            max_video_cache_gb (float|None): Optional RAM cache cap for video tensors.
            max_panecho_cache_gb (float|None): Optional RAM cache cap for PanEcho embeddings.
            smoke_train (bool): If True, run a minimal smoke-training pass.
            smoke_num_steps (int): Number of batches per epoch for smoke training.
            debug_mem (bool): If True, print CUDA memory stats for first batch each epoch.
            amp (bool): If True, use autocast + GradScaler mixed precision.
            checkpoint_panecho (bool): If True, checkpoint PanEcho forward to save memory.
            profile (bool): If True, run PyTorch profiler for a short window.
            profile_steps (int): Number of steps to profile.
            profile_dir (str): Output directory for profiler traces.
            timing_every (int): Print timing stats every N batches.
            profile_summary (bool): If True, print a summary table after profiling.
            gpu_monitor (bool): If True, log GPU utilization periodically.
            gpu_monitor_interval (int): Seconds between GPU utilization logs.
            ram_monitor (bool): If True, log process RAM usage periodically.
            ram_monitor_interval (int): Seconds between RAM usage logs.
            sharing_strategy (str): torch.multiprocessing sharing strategy ("file_descriptor" or "file_system").
        """
        self.time = time.time()
        self.datetime = str(datetime.now()).replace(" ", "_")
        if run_id:
            self.run_id = run_id
        else:
            self.run_id = f"{self.datetime}_{uuid.uuid4()}"

        self.cache_video_tensors = cache_video_tensors
        self.cache_panecho_embeddings = cache_panecho_embeddings
        self.max_video_cache_gb = max_video_cache_gb
        self.max_panecho_cache_gb = max_panecho_cache_gb
        self._panecho_cache = OrderedDict()
        self._panecho_cache_bytes = 0

        if "TORCH_SHM_DIR" not in os.environ:
            tmp_base = os.environ.get("TMPDIR") or os.environ.get("TMP") or os.environ.get("TEMP")
            if tmp_base:
                os.environ["TORCH_SHM_DIR"] = os.path.join(tmp_base, "torch-shm")

        if self.sharing_strategy in ("file_descriptor", "file_system"):
            mp.set_sharing_strategy(self.sharing_strategy)
        else:
            print(f"WARNING: unknown sharing_strategy={self.sharing_strategy}; using file_descriptor")
            mp.set_sharing_strategy("file_descriptor")

        try:
            shm_stats = subprocess.check_output(["df", "-h", "/dev/shm"], text=True).strip().splitlines()
            shm_line = shm_stats[-1] if shm_stats else ""
        except Exception:
            shm_line = "unavailable"
        print(
            "preflight:",
            f"sharing_strategy={mp.get_sharing_strategy()}",
            f"TORCH_SHM_DIR={os.environ.get('TORCH_SHM_DIR', '')}",
            f"TMPDIR={os.environ.get('TMPDIR', '')}",
            f"/dev/shm={shm_line}",
        )
        print(
            "cache:",
            f"video_tensors={self.cache_video_tensors}",
            f"panecho_embeddings={self.cache_panecho_embeddings}",
            f"max_video_cache_gb={self.max_video_cache_gb}",
            f"max_panecho_cache_gb={self.max_panecho_cache_gb}",
            f"num_clips={self.num_clips}",
            f"clip_len={self.clip_len}",
            f"max_videos_per_study={self.max_videos_per_study}",
        )

        assert batch_size==1, "only batch_size=1 currently supported"
        print('main')
        args = {**locals()}
        # input is paired dict of strings named args
        start_time = time.time()

        print("random seed", seed, "\n")
        print("batch_number", batch_number, "\n")

        if total_epochs == -1:
            print("epoch lim missing. evaluating model")
        print("total_epochs", total_epochs, "\n")

        if epoch_early_stop == 9999:
            print("no early stop. defaulting to 10k epochs")
        print("epoch_early_stop", epoch_early_stop, "\n")

        print("learning_rate", learning_rate, "\n")

        self._cli_overrides = set()
        for arg in sys.argv[1:]:
            if not arg.startswith("--"):
                continue
            key = arg[2:]
            if "=" in key:
                key = key.split("=", 1)[0]
            self._cli_overrides.add(key)


        # 1. Check cuda
        for i in range(torch.cuda.device_count()):
            print(torch.cuda.get_device_properties(i).name)
            _ = torch.tensor(3).to('cuda:'+str(i)) # test CUDA device (sometimes crashes)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)          # or your desired device / local_rank
            torch.cuda.init()                 # explicitly initialize CUDA context
            _ = torch.empty(1, device="cuda") # tiny warmup alloc (optional but common)
        else:
            raise ValueError('No CUDA. Exiting.')
        
                
        # 2. Set random seeds 
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = False  # faster kernels, non-deterministic
        torch.backends.cudnn.benchmark = True       # auto-tune cuDNN for fixed input sizes
         
        # set model name 
        if not model_name:
            model_name = f'{task}_{self.run_id}'
        self.model_path = os.path.join('./trained_models', model_name)
        os.makedirs(self.model_path,exist_ok=True)

        self._load_config()
        self._set_loss()
        self.operation_configs = {}
        self.runtime_config = self._build_runtime_config()
        self._save_runtime_config()

    def _build_runtime_config(self):
        """Capture the resolved, stable configuration for this run."""
        return RuntimeConfig(
            run_id=self.run_id,
            datetime=self.datetime,
            model_path=self.model_path,
            model_name=self.model_name,
            dataset=self.dataset,
            task=self.task,
            seed=self.seed,
            batch_number=self.batch_number,
            batch_size=self.batch_size,
            split=tuple(self.split),
            total_epochs=self.total_epochs,
            epoch_early_stop=self.epoch_early_stop,
            learning_rate=self.learning_rate,
            encoder_depth=self.encoder_depth,
            clip_dropout=self.clip_dropout,
            transformer_type=self.transformer_type,
            parallel_processes=self.parallel_processes,
            sample_limit=self.sample_limit,
            config_path=self.config,
            end_to_end=self.end_to_end,
            panecho_trainable=self.panecho_trainable,
            transformer_trainable=self.transformer_trainable,
            load_transformer_path=self.load_transformer_path,
            load_panecho_path=self.load_panecho_path,
            load_strict=self.load_strict,
            use_hdf5_index=self.use_hdf5_index,
            label_path=self.label_path,
            embedding_path=self.embedding_path,
            video_base_path=self.video_base_path,
            video_subdir_format=self.video_subdir_format,
            max_videos_per_study=self.max_videos_per_study,
            num_clips=self.num_clips,
            clip_len=self.clip_len,
            cache_video_tensors=self.cache_video_tensors,
            cache_panecho_embeddings=self.cache_panecho_embeddings,
            max_video_cache_gb=self.max_video_cache_gb,
            max_panecho_cache_gb=self.max_panecho_cache_gb,
            amp=self.amp,
            checkpoint_panecho=self.checkpoint_panecho,
            profile=self.profile,
            profile_steps=self.profile_steps,
            profile_dir=self.profile_dir,
            gpu_monitor=self.gpu_monitor,
            gpu_monitor_interval=self.gpu_monitor_interval,
            ram_monitor=self.ram_monitor,
            ram_monitor_interval=self.ram_monitor_interval,
            sharing_strategy=self.sharing_strategy,
            task_labels=tuple(self.task_labels),
            loss_name=getattr(self.loss_fn, "__name__", self.loss_fn.__class__.__name__),
            cli_overrides=tuple(sorted(self._cli_overrides)),
        )

    def _write_json(self, path, payload):
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def _history_path(self, basename):
        history_dir = os.path.join(self.model_path, "config_history")
        os.makedirs(history_dir, exist_ok=True)
        stem, ext = os.path.splitext(basename)
        return os.path.join(history_dir, f"{stem}.{self.run_id}{ext}")

    def _save_runtime_config(self):
        """Persist the resolved run configuration to the model directory."""
        payload = asdict(self.runtime_config)
        self._write_json(os.path.join(self.model_path, "runtime_config.json"), payload)
        self._write_json(self._history_path("runtime_config.json"), payload)

    def _record_operation_config(self, op_name, cfg):
        """Persist the effective arguments for an operation invocation."""
        payload = asdict(cfg)
        setattr(self, f"last_{op_name}_config", cfg)
        self.operation_configs[op_name] = payload
        basename = f"{op_name}_config.json"
        self._write_json(os.path.join(self.model_path, basename), payload)
        self._write_json(self._history_path(basename), payload)
        return cfg

    def _start_gpu_monitor(self):
        return monitoring_ops.start_gpu_monitor(self)

    def _start_ram_monitor(self):
        return monitoring_ops.start_ram_monitor(self)

    def _load_submodule_weights(self, module, path, prefix=None):
        if path is None:
            return
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        if prefix:
            prefixed_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if prefixed_state:
                state = prefixed_state
        missing, unexpected = module.load_state_dict(state, strict=self.load_strict)
        if missing or unexpected:
            print(
                f"WARNING: load from {path} missing={len(missing)} unexpected={len(unexpected)}"
            )

    def _can_bootstrap_inference_checkpoint(self):
        """Return True when direct weight paths are sufficient to build a checkpoint."""
        return bool(self.load_transformer_path) or bool(self.load_panecho_path)

    def _get_checkpoint_metadata(self, path):
        """Read optional metadata from a checkpoint-like file."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        perf_log = ckpt.get("perf_log")
        input_norm_dict = ckpt.get("input_norm_dict")
        return perf_log, input_norm_dict

    def _apply_training_args_from_csv(self, train_args_path):
        """Apply key architecture settings from a saved training-args CSV."""
        if not os.path.isfile(train_args_path):
            return False

        train_args = pd.read_csv(train_args_path).to_dict(orient='records')[0]
        converters = {
            'encoder_depth': int,
            'clip_dropout': float,
        }
        direct_keys = ['encoder_depth', 'task_labels', 'clip_dropout', 'transformer_type']
        legacy_key_map = {
            'target_label_list': 'task_labels',
        }
        applied = False
        for source_key in direct_keys:
            if source_key not in train_args:
                continue
            target_key = source_key
            value = train_args[source_key]
            if target_key in converters and pd.notna(value):
                value = converters[target_key](value)
            if target_key in self._cli_overrides:
                continue
            if getattr(self, target_key, None) != value:
                print(f'WARNING: using {target_key}={value}, loaded from {train_args_path}')
                setattr(self, target_key, value)
                applied = True

        for source_key, target_key in legacy_key_map.items():
            if source_key not in train_args or target_key in self._cli_overrides:
                continue
            value = train_args[source_key]
            if target_key == 'task_labels' and isinstance(value, str):
                value = ast.literal_eval(value)
            if getattr(self, target_key, None) != value:
                print(f'WARNING: using {target_key}={value}, loaded from {train_args_path}')
                setattr(self, target_key, value)
                applied = True
        return applied

    def _apply_training_args_from_runtime_config(self, runtime_config_path):
        """Apply key architecture settings from a saved runtime config JSON."""
        if not os.path.isfile(runtime_config_path):
            return False

        with open(runtime_config_path, "r") as f:
            runtime_cfg = json.load(f)

        direct_keys = ['encoder_depth', 'task_labels', 'clip_dropout', 'transformer_type']
        applied = False
        for target_key in direct_keys:
            if target_key not in runtime_cfg or target_key in self._cli_overrides:
                continue
            value = runtime_cfg[target_key]
            if target_key == 'task_labels' and isinstance(value, list):
                value = list(value)
            if getattr(self, target_key, None) != value:
                print(f'WARNING: using {target_key}={value}, loaded from {runtime_config_path}')
                setattr(self, target_key, value)
                applied = True
        return applied

    def _maybe_apply_source_train_args(self):
        """Load architecture metadata from the target run directory or source checkpoints."""
        train_args_path = os.path.join(self.model_path, 'train_args.csv')
        if self._apply_training_args_from_csv(train_args_path):
            self.runtime_config = self._build_runtime_config()
            self._save_runtime_config()
            return
        runtime_config_path = os.path.join(self.model_path, 'runtime_config.json')
        if self._apply_training_args_from_runtime_config(runtime_config_path):
            self.runtime_config = self._build_runtime_config()
            self._save_runtime_config()
            return

        source_paths = [self.load_transformer_path, self.load_panecho_path]
        seen = set()
        for source_path in source_paths:
            if not source_path:
                continue
            source_dir = os.path.dirname(os.path.abspath(source_path))
            candidates = [
                os.path.join(source_dir, 'train_args.csv'),
                os.path.join(source_dir, 'runtime_config.json'),
            ]
            if source_dir in seen:
                continue
            seen.add(source_dir)
            for candidate in candidates:
                applied = False
                if candidate.endswith('.csv'):
                    applied = self._apply_training_args_from_csv(candidate)
                else:
                    applied = self._apply_training_args_from_runtime_config(candidate)
                if applied:
                    self.runtime_config = self._build_runtime_config()
                    self._save_runtime_config()
                    return

    def _bootstrap_inference_checkpoint(self, model, input_norm_dict=None):
        """Persist a synthetic best checkpoint from directly loaded weights."""
        if not self._can_bootstrap_inference_checkpoint():
            raise FileNotFoundError(
                f"No checkpoint found at {self.best_checkpoint_path}. "
                "Provide an existing model_name checkpoint or direct weight files."
            )

        metadata_paths = [self.load_transformer_path, self.load_panecho_path]
        perf_log = None
        for path in metadata_paths:
            if not path or not os.path.isfile(path):
                continue
            candidate_perf_log, candidate_input_norm = self._get_checkpoint_metadata(path)
            if perf_log is None and candidate_perf_log:
                perf_log = candidate_perf_log
            if input_norm_dict is None and candidate_input_norm is not None:
                input_norm_dict = candidate_input_norm

        if perf_log is None:
            perf_log = [[0, float("nan"), float("nan"), float(self.learning_rate), 0.0]]

        save_nn(
            model,
            self.best_checkpoint_path,
            perf_log,
            optimizer=None,
            scheduler=None,
            input_norm_dict=input_norm_dict,
        )
        print(f"bootstrapped inference checkpoint at {self.best_checkpoint_path}")
        return input_norm_dict

    def _load_inference_model(self):
        """Return an inference-ready model and normalization metadata."""
        model, _, _, _, input_norm_dict = self._setup_model()
        if os.path.isfile(self.best_checkpoint_path):
            best_model, _, _, _, input_norm_dict = load_model_and_random_state(
                self.best_checkpoint_path,
                model,
            )
            return best_model, input_norm_dict

        input_norm_dict = self._bootstrap_inference_checkpoint(
            model,
            input_norm_dict=input_norm_dict,
        )
        return model, input_norm_dict

    def _panecho_cache_get(self, eid):
        return data_ops.panecho_cache_get(self, eid)

    def _panecho_cache_put(self, eid, value):
        return data_ops.panecho_cache_put(self, eid, value)

    def _panecho_cache_clear(self):
        return data_ops.panecho_cache_clear(self)

    def _embedding_eids_from_path(self):
        return data_ops.embedding_eids_from_path(self)

    def _set_trainable_flags(self):
        return training_ops.set_trainable_flags(self)

    def _load_config(self):
        """Load dataset/task config and set instance attributes.

        Raises:
            ValueError: If dataset is not defined in the config.
            AssertionError: If task is not defined in the config.
        """
        with open(self.config,'r') as f:
            data = json.load(f)

        assert self.task in data['task'].keys(), f'task must be one of: {data["task"].keys()}; got "{self.task}"' 

        if self.dataset not in data['dataset'].keys():
            raise ValueError(f'dataset must be one of: {list(data["dataset"].keys())}; got \"{self.dataset}\"')
        for k,v in data['task'][self.task].items():
            if k in self._cli_overrides:
                continue
            if hasattr(self, k) and getattr(self, k) != v:
                print(f'WARNING: overriding init arg {k}={getattr(self, k)} with config value {v}')
            setattr(self,k,v)
        for k,v in data['dataset'][self.dataset].items():
            if k in self._cli_overrides:
                continue
            if hasattr(self, k) and getattr(self, k) != v:
                print(f'WARNING: overriding init arg {k}={getattr(self, k)} with config value {v}')
            setattr(self,k,v)

    def _setup_data(self, input_norm_dict=None, use_train_transforms=True):
        return data_ops.setup_data(self, input_norm_dict=input_norm_dict, use_train_transforms=use_train_transforms)

    def cache_embeddings(
        self,
        cache_root="embed/cache",
        cache_tag=None,
        num_shards=512,
        compression="lzf",
        dtype="float16",
        use_train_transforms=False,
        overwrite=False,
        max_eids=None,
        amp=False,
        seed=None,
    ):
        return data_ops.cache_embeddings(
            self,
            cache_root=cache_root,
            cache_tag=cache_tag,
            num_shards=num_shards,
            compression=compression,
            dtype=dtype,
            use_train_transforms=use_train_transforms,
            overwrite=overwrite,
            max_eids=max_eids,
            amp=amp,
            seed=seed,
        )

    # def _normalize_data(self):

    def _setup_model(self):
        return training_ops.setup_model(self)

    # def load_checkpoint(self, checkpoint):

    def train(self):
        return training_ops.train(self)

    def _run_training_loop(
        self,
        model,
        current_epoch,
        best_epoch,
        best_loss,
        input_norm_dict,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        smoke_steps=None,
        epoch_hook=None,
    ):
        return training_ops.run_training_loop(
            self,
            model,
            current_epoch,
            best_epoch,
            best_loss,
            input_norm_dict,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            smoke_steps=smoke_steps,
            epoch_hook=epoch_hook,
        )

    def train_ping_pong(
        self,
        total_epochs=10,
        start_with="transformer",
        switch_every=1,
        transformer_lr=None,
        panecho_lr=None,
    ):
        cfg = self._record_operation_config(
            "train_ping_pong",
            TrainPingPongConfig(
                total_epochs=total_epochs,
                start_with=start_with,
                switch_every=switch_every,
                transformer_lr=transformer_lr,
                panecho_lr=panecho_lr,
            ),
        )
        return training_ops.train_ping_pong(
            self,
            cfg,
        )
    def _evaluate(self, model, dataloader, fold, input_norm_dict=None):
        return evaluation_ops.evaluate_fold(self, model, dataloader, fold, input_norm_dict=input_norm_dict)

    def _bootstrap_metric_ci(
        self,
        y_true,
        y_pred,
        metric_fn,
        rng,
        n_bootstrap=1000,
        ci_percentiles=(2.5, 97.5),
        require_two_classes=False,
    ):
        return evaluation_ops.bootstrap_metric_ci(
            y_true,
            y_pred,
            metric_fn,
            rng,
            n_bootstrap=n_bootstrap,
            ci_percentiles=ci_percentiles,
            require_two_classes=require_two_classes,
        )

    def get_metrics(self, fold='test', dataset=None, n_bootstrap=1000, model_name=None):
        return evaluation_ops.get_metrics(self, fold=fold, dataset=dataset, n_bootstrap=n_bootstrap, model_name=model_name)
        
            
    def evaluate(self, split=None, folds=('train', 'val', 'test')):
        if isinstance(folds, str):
            folds = (folds,)
        cfg = self._record_operation_config(
            "evaluate",
            EvaluateConfig(
                split=split,
                folds=tuple(folds),
            ),
        )
        return evaluation_ops.evaluate(self, cfg)

    def _set_loss(self):
        """Set the loss function based on task type."""
        if self.task=='measure':
            self.loss_fn = utils.masked_mse_loss
        elif self.task in ['chd','fyler']:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def embed(self, embed_file=None, split=(0,0,100), pool_queries=True):
        cfg = self._record_operation_config(
            "embed",
            EmbedConfig(
                embed_file=embed_file,
                split=tuple(split),
                pool_queries=pool_queries,
            ),
        )
        return evaluation_ops.embed(self, cfg)

    def explain(
        self,
        explain_n=5,
        explain_mode='pred',
        explain_tasks = ('EF05','AR01'),
        split=(0,0,100)
    ):
        if isinstance(explain_tasks, str):
            explain_tasks_value = explain_tasks
        else:
            explain_tasks_value = tuple(explain_tasks)
        cfg = self._record_operation_config(
            "explain",
            ExplainConfig(
                explain_n=explain_n,
                explain_mode=explain_mode,
                explain_tasks=explain_tasks_value,
                split=tuple(split),
            ),
        )
        return evaluation_ops.explain(self, cfg)

    def save_log(self):
        return training_ops.save_log(self)
