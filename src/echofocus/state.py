"""Structured runtime configuration and mutable run state."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RuntimeConfig:
    """User/config-driven inputs for a single EchoFocus run."""

    model_name: Optional[str]
    dataset: Optional[str]
    task: str
    seed: int
    split: Tuple[int, int, int]
    total_epochs: int
    epoch_early_stop: int
    learning_rate: float
    encoder_depth: int
    clip_dropout: float
    parallel_processes: int
    sample_limit: float
    config_path: str
    end_to_end: bool
    panecho_trainable: bool
    transformer_trainable: bool
    load_transformer_path: Optional[str]
    load_panecho_path: Optional[str]
    label_path: Optional[str]
    embedding_path: Optional[str]
    video_base_path: str
    video_subdir_format: str
    num_clips: int
    clip_len: int
    cache_video_tensors: bool
    cache_panecho_embeddings: bool
    amp: bool
    checkpoint_panecho: bool
    cli_overrides: Tuple[str, ...] = ()
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunState:
    """Mutable state produced while executing a single run."""

    run_id: str
    datetime: str
    model_path: str
    task_labels: List[str] = field(default_factory=list)
    input_norm_dict: Optional[Dict[str, Any]] = None
    loss_name: Optional[str] = None
    perf_log: List[Any] = field(default_factory=list)
