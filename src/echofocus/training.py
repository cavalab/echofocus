"""Training and model-setup helpers."""

import os
import time

import numpy as np
import torch
from tqdm import tqdm

from . import utils
from .checkpoints import load_model_and_random_state, run_model_on_dataloader, save_nn
from .models import CustomMultiQueryTransformer, CustomQueryTransformer, CustomTransformer, EchoFocusEndToEnd
from .monitoring import start_gpu_monitor, start_ram_monitor


def set_trainable_flags(self):
    """Apply trainable flags to model submodules."""
    if not hasattr(self, "model") or self.model is None:
        return
    if hasattr(self.model, "panecho"):
        for param in self.model.panecho.parameters():
            param.requires_grad = bool(self.panecho_trainable)
    if hasattr(self.model, "transformer"):
        for param in self.model.transformer.parameters():
            param.requires_grad = bool(self.transformer_trainable)


def setup_model(self, resume_training_state=True, load_saved_architecture=True):
    """Initialize model, optimizer, scheduler, and load checkpoints."""
    print('_setup_model...')
    self.last_checkpoint_path = os.path.join(self.model_path, 'last_checkpoint.pt')
    self.best_checkpoint_path = os.path.join(self.model_path, 'best_checkpoint.pt')
    self.log_path = os.path.join(self.model_path, 'train_losses.csv')

    if load_saved_architecture:
        self._maybe_apply_source_train_args()
    if resume_training_state:
        self.runtime_config = self._build_runtime_config()
        self._save_runtime_config()

    if self.end_to_end:
        self.model = EchoFocusEndToEnd(
            input_size=768,
            encoder_dim=768,
            n_encoder_layers=self.encoder_depth,
            output_size=len(self.task_labels),
            clip_dropout=self.clip_dropout,
            transformer_type=self.transformer_type,
            panecho_trainable=self.panecho_trainable,
            debug_mem=self.debug_mem,
            checkpoint_panecho=self.checkpoint_panecho,
        )
    else:
        if self.transformer_type == "standard":
            self.model = CustomTransformer(
                input_size=768,
                encoder_dim=768,
                n_encoder_layers=self.encoder_depth,
                output_size=len(self.task_labels),
                clip_dropout=self.clip_dropout,
                tf_combine="avg",
            )
        elif self.transformer_type == "query":
            self.model = CustomQueryTransformer(
                input_size=768,
                encoder_dim=768,
                n_encoder_layers=self.encoder_depth,
                output_size=len(self.task_labels),
                clip_dropout=self.clip_dropout,
            )
        elif self.transformer_type == "multiquery":
            self.model = CustomMultiQueryTransformer(
                input_size=768,
                encoder_dim=768,
                n_encoder_layers=self.encoder_depth,
                output_size=len(self.task_labels),
                clip_dropout=self.clip_dropout,
            )
        else:
            raise ValueError(
                f"transformer_type must be one of ('standard', 'query', 'multiquery'); got {self.transformer_type!r}"
            )
    self._set_trainable_flags()
    total_params = sum(p.numel() for p in self.model.parameters())
    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    print("model parameters:", f"total={total_params:,}", f"trainable={trainable_params:,}")

    if torch.cuda.is_available():
        self.model = self.model.to('cuda')
    elif self.amp:
        print("WARNING: amp=True requested but CUDA is not available; running in full precision on CPU.")

    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
    self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, factor=0.5)

    self.perf_log = []
    input_norm_dict = None
    if resume_training_state and os.path.isfile(self.last_checkpoint_path):
        print('loading lastcheckpoint')
        self.model, self.optimizer, self.scheduler, self.perf_log, input_norm_dict = load_model_and_random_state(
            self.last_checkpoint_path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        current_epoch = self.perf_log[-1][0]
        tmp = np.array(self.perf_log)
        best_epoch = tmp[np.argmin(tmp[:, 2]), 0]
        best_loss = tmp[np.argmin(tmp[:, 2]), 2]
    else:
        current_epoch = 0
        best_epoch = 0
        best_loss = 1e10

    if self.end_to_end:
        if self.load_panecho_path:
            self._load_submodule_weights(self.model.panecho, self.load_panecho_path, prefix="panecho.")
        if self.load_transformer_path:
            self._load_submodule_weights(self.model.transformer, self.load_transformer_path, prefix="transformer.")
    else:
        if self.load_transformer_path:
            self._load_submodule_weights(self.model, self.load_transformer_path, prefix="transformer.")

    return self.model, current_epoch, best_epoch, best_loss, input_norm_dict


def train(self):
    """Train the model and evaluate on train/val/test splits."""
    smoke_steps = None
    if self.smoke_train:
        if self.total_epochs < 1:
            self.total_epochs = 1
        else:
            self.total_epochs = min(self.total_epochs, 1)
        self.epoch_early_stop = 1
        self.sample_limit = min(self.sample_limit, 5000)
        smoke_steps = self.smoke_num_steps
        print("smoke_train enabled:", f"samples={self.sample_limit}, steps={smoke_steps}, epochs={self.total_epochs}")
    model, current_epoch, best_epoch, best_loss, input_norm_dict = self._setup_model()
    train_dataloader, val_dataloader, test_dataloader, input_norm_dict = self._setup_data(
        input_norm_dict,
        use_train_transforms=True,
    )
    self._run_training_loop(
        model,
        current_epoch,
        best_epoch,
        best_loss,
        input_norm_dict,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        smoke_steps=smoke_steps,
    )


def run_training_loop(
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
    """Run the main training loop with an optional per-epoch hook."""
    monitor_thread = None
    monitor_stop = None
    if self.gpu_monitor:
        self._gpu_status = ""
        monitor_thread, monitor_stop = start_gpu_monitor(self)
    ram_thread = None
    ram_stop = None
    if self.ram_monitor:
        self._ram_status = ""
        ram_thread, ram_stop = start_ram_monitor(self)
    prof = None
    prof_records = []
    profile_steps = int(self.profile_steps) if self.profile_steps is not None else 0
    profile_steps = max(0, profile_steps)
    if self.profile and profile_steps > 0:
        os.makedirs(self.profile_dir, exist_ok=True)
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.profile_dir),
        )
        prof.start()

    def _mem(tag):
        if not torch.cuda.is_available():
            return
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_alloc = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[mem] {tag}: alloc={alloc:.2f}G reserved={reserved:.2f}G max={max_alloc:.2f}G")

    def _backoff_lr(factor=0.5, min_lr=1e-7):
        new_lrs = []
        for param_group in self.optimizer.param_groups:
            current_lr = float(param_group["lr"])
            updated_lr = max(current_lr * factor, min_lr)
            param_group["lr"] = updated_lr
            new_lrs.append(updated_lr)
        return new_lrs

    global_step = 0
    prev_batch_end = None
    while (current_epoch < self.total_epochs) and (current_epoch - best_epoch < self.epoch_early_stop):
        print('beginning training loop')
        if epoch_hook is not None:
            epoch_hook(current_epoch)
            self._set_trainable_flags()
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.learning_rate
        if self.panecho_trainable and len(self._panecho_cache) > 0:
            print("clearing cached panecho embeddings (panecho_trainable=True)")
            self._panecho_cache_clear()

        self.model.train()
        epoch_start_time = time.time()
        train_loss_total = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {current_epoch}", total=len(train_dataloader))
        for batch_count, (embedding, correct_out, eid) in enumerate(pbar):
            postfix = []
            if self.gpu_monitor and self._gpu_status:
                postfix.append(f"gpu={self._gpu_status}")
            if self.ram_monitor and self._ram_status:
                postfix.append(f"ram={self._ram_status}")
            if postfix:
                pbar.set_postfix_str(" ".join(postfix))
            batch_start = time.time()
            data_wait = None
            if prev_batch_end is not None:
                data_wait = batch_start - prev_batch_end

            if torch.cuda.is_available():
                embedding = embedding.to('cuda')
                correct_out = correct_out.to('cuda')

            if self.debug_mem and batch_count == 0:
                _mem("before forward")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fwd_start = time.time()
            with torch.amp.autocast("cuda", enabled=self.amp):
                if self.end_to_end and self.cache_panecho_embeddings and not self.panecho_trainable:
                    eid_val = eid[0] if isinstance(eid, (list, tuple, np.ndarray)) else eid
                    try:
                        eid_key = int(eid_val)
                    except Exception:
                        eid_key = str(eid_val)
                    cached = self._panecho_cache_get(eid_key)
                    if cached is not None:
                        emb = cached.to(embedding.device)
                    else:
                        with torch.no_grad():
                            emb = self.model._panecho_embed(embedding)
                        self._panecho_cache_put(eid_key, emb.detach().cpu())
                    out = self.model.transformer(emb)
                else:
                    out = self.model(embedding)
                train_loss = self.loss_fn(out, correct_out)
            if not torch.isfinite(train_loss):
                reduced_lrs = _backoff_lr()
                self.model.zero_grad(set_to_none=True)
                print(
                    "WARNING: non-finite train loss detected; skipping batch and reducing lr to "
                    f"{reduced_lrs}"
                )
                prev_batch_end = time.time()
                continue
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_fwd_end = time.time()
            if self.debug_mem and batch_count == 0:
                _mem("after forward")
                _mem("after loss")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_bwd_start = time.time()
            if self.amp:
                self.scaler.scale(train_loss).backward()
            else:
                train_loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_bwd_end = time.time()
            if self.debug_mem and batch_count == 0:
                _mem("after backward")
            train_loss_total += train_loss.item()

            step_performed = False
            if (batch_count + 1) % self.batch_number == 0 or (batch_count + 1) == len(train_dataloader):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_step_start = time.time()
                if self.amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.model.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_step_end = time.time()
                step_performed = True

            if smoke_steps is not None and (batch_count + 1) >= smoke_steps:
                if (batch_count + 1) % self.batch_number != 0:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_step_start = time.time()
                    if self.amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.model.zero_grad()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_step_end = time.time()
                    step_performed = True
                break

            if prof is not None and global_step < profile_steps:
                prof.step()
            if prof is not None and global_step == profile_steps:
                if self.profile_summary:
                    prof_records.append(prof)
                prof.stop()
                prof = None

            if self.timing_every and (batch_count + 1) % int(self.timing_every) == 0:
                step_time = (t_step_end - t_step_start) if step_performed else 0.0
                total_time = time.time() - batch_start
                data_wait_s = data_wait if data_wait is not None else 0.0
                print(
                    f"[timing] batch={batch_count+1} data_wait={data_wait_s:.2f}s "
                    f"fwd={t_fwd_end - t_fwd_start:.2f}s bwd={t_bwd_end - t_bwd_start:.2f}s "
                    f"step={step_time:.2f}s total={total_time:.2f}s"
                )
            prev_batch_end = time.time()
            global_step += 1

        epoch_end_time = time.time()

        __, __, __, val_loss_total = run_model_on_dataloader(
            self.model,
            val_dataloader,
            self.loss_fn,
            amp=self.amp,
        )

        current_epoch = current_epoch + 1

        tmp_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        perf = {
            'epoch': current_epoch,
            'train loss': train_loss_total,
            'val loss': val_loss_total,
            'lr': tmp_lr,
            'epoch time': epoch_end_time - epoch_start_time,
        }
        self.perf_log.append(list(perf.values()))
        print(' '.join([f'{k}: {v}' for k, v in perf.items() if k in ['train loss', 'val loss', 'lr']]))
        self.save_log()

        self.scheduler.step(val_loss_total)
        save_nn(
            self.model,
            self.last_checkpoint_path,
            self.perf_log,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            input_norm_dict=input_norm_dict,
        )

        if val_loss_total < best_loss:
            save_nn(
                self.model,
                self.best_checkpoint_path,
                self.perf_log,
                optimizer=None,
                scheduler=None,
                input_norm_dict=input_norm_dict,
            )
            best_loss = val_loss_total
            best_epoch = current_epoch

        if current_epoch == self.total_epochs:
            print('current epoch = epoch limit, terminating')
        if current_epoch - best_epoch == self.epoch_early_stop:
            print('early stopping')

    if prof is not None:
        if self.profile_summary:
            prof_records.append(prof)
        prof.stop()
        prof = None

    if self.profile_summary and prof_records:
        try:
            summary = prof_records[-1].key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
                row_limit=50,
            )
            print("Profiler summary (top ops):")
            print(summary)
        except Exception as e:
            print(f"warning: failed to print profiler summary: {e}")

    if monitor_stop is not None:
        monitor_stop.set()
    if monitor_thread is not None:
        monitor_thread.join(timeout=1.0)
    if ram_stop is not None:
        ram_stop.set()
    if ram_thread is not None:
        ram_thread.join(timeout=1.0)

    utils.plot_training_progress(self.model_path, self.perf_log)
    print('Training Completed')

    best_checkpoint_path = os.path.join(self.model_path, 'best_checkpoint.pt')
    best_model, _, _, _, input_norm_dict = load_model_and_random_state(best_checkpoint_path, model)
    for dataloader, fold in zip((train_dataloader, val_dataloader, test_dataloader), ('train', 'val', 'test')):
        self._evaluate(best_model, dataloader, fold, input_norm_dict)
        self.get_metrics(fold=fold)


def train_ping_pong(self, cfg):
    """Alternate training between transformer-only and PanEcho-only phases."""
    if cfg.start_with not in ("transformer", "panecho"):
        raise ValueError("start_with must be 'transformer' or 'panecho'")
    smoke_steps = None
    if self.smoke_train:
        if cfg.total_epochs < 1:
            cfg.total_epochs = 1
        self.epoch_early_stop = 1
        self.sample_limit = min(self.sample_limit, 5000)
        smoke_steps = self.smoke_num_steps
        print("smoke_train enabled:", f"samples={self.sample_limit}, steps={smoke_steps}, epochs={cfg.total_epochs}")

    self.total_epochs = int(cfg.total_epochs)
    model, current_epoch, best_epoch, best_loss, input_norm_dict = self._setup_model()
    train_dataloader, val_dataloader, test_dataloader, input_norm_dict = self._setup_data(
        input_norm_dict,
        use_train_transforms=True,
    )

    orig_lr = self.learning_rate
    switch_every = max(1, int(cfg.switch_every))
    start_phase_idx = 0 if cfg.start_with == "transformer" else 1

    def _epoch_hook(epoch):
        phase_idx = ((epoch // switch_every) + start_phase_idx) % 2
        if phase_idx == 0:
            print(f"phase: transformer-only epoch {epoch}")
            self.panecho_trainable = False
            self.transformer_trainable = True
            self.learning_rate = cfg.transformer_lr if cfg.transformer_lr is not None else orig_lr
        else:
            print(f"phase: panecho-only epoch {epoch}")
            self.panecho_trainable = True
            self.transformer_trainable = False
            self.learning_rate = cfg.panecho_lr if cfg.panecho_lr is not None else orig_lr

    self._run_training_loop(
        model,
        current_epoch,
        best_epoch,
        best_loss,
        input_norm_dict,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        smoke_steps=smoke_steps,
        epoch_hook=_epoch_hook,
    )


def save_log(self):
    """Write the training loss log to CSV."""
    import csv

    with open(self.log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epochs_trained", "train_loss", "val_loss", "lr", "epoch_time"])
        writer.writerows(self.perf_log)
