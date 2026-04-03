"""Structured runtime and operation configuration."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class RuntimeConfig:
    """Resolved run configuration that stays stable across operations."""

    run_id: str
    datetime: str
    model_path: str
    model_name: Optional[str]
    dataset: Optional[str]
    task: str
    seed: int
    batch_number: int
    batch_size: int
    split: Tuple[int, int, int]
    total_epochs: int
    epoch_early_stop: int
    learning_rate: float
    encoder_depth: int
    clip_dropout: float
    transformer_type: str
    parallel_processes: int
    sample_limit: float
    config_path: str
    end_to_end: bool
    panecho_trainable: bool
    transformer_trainable: bool
    load_transformer_path: Optional[str]
    load_panecho_path: Optional[str]
    load_strict: bool
    use_hdf5_index: bool
    label_path: Optional[str]
    embedding_path: Optional[str]
    video_base_path: str
    video_subdir_format: str
    max_videos_per_study: Optional[int]
    num_clips: int
    clip_len: int
    cache_video_tensors: bool
    cache_panecho_embeddings: bool
    max_video_cache_gb: Optional[float]
    max_panecho_cache_gb: Optional[float]
    amp: bool
    checkpoint_panecho: bool
    profile: bool
    profile_steps: int
    profile_dir: str
    gpu_monitor: bool
    gpu_monitor_interval: int
    ram_monitor: bool
    ram_monitor_interval: int
    sharing_strategy: str
    task_labels: Tuple[str, ...]
    loss_name: str
    cli_overrides: Tuple[str, ...] = ()


@dataclass
class EvaluateConfig:
    """Arguments specific to an evaluation invocation."""

    split: Optional[Tuple[int, int, int]] = None
    folds: Tuple[str, ...] = ("train", "val", "test")


@dataclass
class EmbedConfig:
    """Arguments specific to an embedding export invocation."""

    embed_file: Optional[str] = None
    split: Tuple[int, int, int] = (0, 0, 100)
    pool_queries: bool = True


@dataclass
class ExplainConfig:
    """Arguments specific to an explanation invocation."""

    explain_n: int = 5
    explain_mode: str = "pred"
    explain_tasks: Tuple[str, ...] = ("EF05", "AR01")
    split: Tuple[int, int, int] = (0, 0, 100)


@dataclass
class TrainPingPongConfig:
    """Arguments specific to an alternating training invocation."""

    total_epochs: int = 10
    start_with: str = "transformer"
    switch_every: int = 1
    transformer_lr: Optional[float] = None
    panecho_lr: Optional[float] = None
