"""Evaluation, embedding, and explanation helpers."""

import json
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, median_absolute_error, r2_score, roc_auc_score
from tqdm import tqdm

from . import utils
from .checkpoints import run_model_on_dataloader
from .integrated_gradients import integrated_gradients_video_level


def _run_in_eval_mode(self, fn, *args, **kwargs):
    """Run an inference helper with training flags disabled."""
    original_panecho_trainable = self.panecho_trainable
    original_transformer_trainable = self.transformer_trainable
    self.panecho_trainable = False
    self.transformer_trainable = False
    try:
        return fn(*args, **kwargs)
    finally:
        self.panecho_trainable = original_panecho_trainable
        self.transformer_trainable = original_transformer_trainable


def evaluate_fold(self, model, dataloader, fold, input_norm_dict=None):
    """Evaluate a model on a dataloader and write outputs."""
    if dataloader is None or len(dataloader) == 0:
        print('skipping', fold)
        return
    print(f'run model on {fold} set')
    model._echofocus_monitor_owner = self

    def _cache_hook(embedding, eid):
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
                emb = model._panecho_embed(embedding)
                self._panecho_cache_put(eid_key, emb.detach().cpu())
            return model.transformer(emb)
        return model(embedding)

    y_pred, y_true, eids, loss = run_model_on_dataloader(
        model,
        dataloader,
        self.loss_fn,
        amp=self.amp,
        cache_hook=_cache_hook,
    )
    y_true = np.array(y_true).squeeze()
    y_pred = np.array(y_pred)
    if self.task == 'measure':
        y_true = utils.un_normalize_output(y_true, self.task_labels, input_norm_dict)
        y_pred = utils.un_normalize_output(y_pred, self.task_labels, input_norm_dict)

    saveout_path = os.path.join(self.model_path, f'saveout_{fold}_{self.dataset}.csv')
    if isinstance(dataloader.dataset, torch.utils.data.Subset):
        data_df = dataloader.dataset.dataset.dataframe
    else:
        data_df = dataloader.dataset.dataframe
    pids = data_df.loc[eids]['pid'].values
    eids = np.array(eids)
    saveout_df = pd.DataFrame({'PID': pids, 'Echo_ID': eids})
    for i, _ in enumerate(self.task_labels):
        saveout_df[self.task_labels[i] + '_Correct'] = y_true[:, i]
        saveout_df[self.task_labels[i] + '_Predict'] = y_pred[:, i]
    print('writing', saveout_path)
    saveout_df.to_csv(saveout_path)

    if self.task == 'measure':
        utils.scatter_plots(self.model_path, self.dataset, fold, self.task_labels, y_true, y_pred)


def bootstrap_metric_ci(y_true, y_pred, metric_fn, rng, n_bootstrap=1000, ci_percentiles=(2.5, 97.5), require_two_classes=False):
    """Compute metric value and bootstrap confidence interval."""
    if y_true.size == 0:
        return None, None, None, 0
    if require_two_classes and np.unique(y_true).size < 2:
        return None, None, None, 0
    try:
        metric_value = float(metric_fn(y_true, y_pred))
    except ValueError:
        metric_value = None

    bootstrap_values = []
    n = y_true.shape[0]
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        yb_true = y_true[idx]
        yb_pred = y_pred[idx]
        if require_two_classes and np.unique(yb_true).size < 2:
            continue
        try:
            val = metric_fn(yb_true, yb_pred)
        except ValueError:
            continue
        if np.isfinite(val):
            bootstrap_values.append(float(val))

    if len(bootstrap_values) == 0:
        return metric_value, None, None, 0
    low, high = np.percentile(np.array(bootstrap_values), list(ci_percentiles))
    return metric_value, float(low), float(high), len(bootstrap_values)


def get_metrics(self, fold='test', dataset=None, n_bootstrap=1000, model_name=None):
    """Compute task metrics from saved predictions and write JSON output."""
    if model_name is not None:
        self.model_name = model_name
        self.model_path = os.path.join('./trained_models', model_name)

    if dataset is None:
        dataset = self.dataset
    saveout_path = os.path.join(self.model_path, f'saveout_{fold}_{dataset}.csv')
    if not os.path.exists(saveout_path):
        print(f'WARNING: saveout file not found, skipping metrics: {saveout_path}')
        return None

    df = pd.read_csv(saveout_path)
    rng = np.random.default_rng(self.seed)
    out = {
        'task': self.task,
        'dataset': dataset,
        'fold': fold,
        'n_bootstrap': int(n_bootstrap),
        'metrics': {},
    }

    y_true_all = []
    y_pred_all = []
    valid_labels = []

    for label in self.task_labels:
        true_col = f'{label}_Correct'
        pred_col = f'{label}_Predict'
        if true_col not in df.columns or pred_col not in df.columns:
            print(f'WARNING: missing columns for label {label}, skipping')
            continue

        y_true = pd.to_numeric(df[true_col], errors='coerce').to_numpy(dtype=float)
        y_pred = pd.to_numeric(df[pred_col], errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        label_metrics = {'n_samples': int(y_true.shape[0])}
        if self.task in ['chd', 'fyler']:
            if not np.all((y_true >= 0) & (y_true <= 1)):
                bad_vals = y_true[(y_true < 0) | (y_true > 1)]
                raise ValueError(
                    f"Label {label} has classification targets outside [0,1]. "
                    f"First bad values: {bad_vals[:10].tolist()}"
                )
            y_true_cls = y_true.astype(int)
            roc_val, roc_lo, roc_hi, roc_eff_n = bootstrap_metric_ci(
                y_true_cls, y_pred, roc_auc_score, rng=rng, n_bootstrap=n_bootstrap, require_two_classes=True
            )
            ap_val, ap_lo, ap_hi, ap_eff_n = bootstrap_metric_ci(
                y_true_cls, y_pred, average_precision_score, rng=rng, n_bootstrap=n_bootstrap, require_two_classes=True
            )
            label_metrics['roc_auc_score'] = {
                'value': roc_val,
                'ci_lower': roc_lo,
                'ci_upper': roc_hi,
                'n_bootstrap_effective': roc_eff_n,
            }
            label_metrics['average_precision'] = {
                'value': ap_val,
                'ci_lower': ap_lo,
                'ci_upper': ap_hi,
                'n_bootstrap_effective': ap_eff_n,
            }
            y_true_all.append(y_true_cls)
            y_pred_all.append(y_pred)
            valid_labels.append(label)
        elif self.task == 'measure':
            mae_val, mae_lo, mae_hi, mae_eff_n = bootstrap_metric_ci(
                y_true, y_pred, median_absolute_error, rng=rng, n_bootstrap=n_bootstrap
            )
            r2_val, r2_lo, r2_hi, r2_eff_n = bootstrap_metric_ci(
                y_true, y_pred, r2_score, rng=rng, n_bootstrap=n_bootstrap
            )
            label_metrics['median_absolute_error'] = {
                'value': mae_val,
                'ci_lower': mae_lo,
                'ci_upper': mae_hi,
                'n_bootstrap_effective': mae_eff_n,
            }
            label_metrics['r2_score'] = {
                'value': r2_val,
                'ci_lower': r2_lo,
                'ci_upper': r2_hi,
                'n_bootstrap_effective': r2_eff_n,
            }
        out['metrics'][label] = label_metrics

    metrics_path = os.path.join(self.model_path, f'metrics_{fold}_{dataset}.json')
    with open(metrics_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('writing', metrics_path)

    if self.task in ['chd', 'fyler'] and len(valid_labels) > 0:
        max_n = max(arr.shape[0] for arr in y_true_all)
        y_true_mat = np.full((max_n, len(valid_labels)), np.nan)
        y_pred_mat = np.full((max_n, len(valid_labels)), np.nan)
        for i, (yt, yp) in enumerate(zip(y_true_all, y_pred_all)):
            y_true_mat[: yt.shape[0], i] = yt
            y_pred_mat[: yp.shape[0], i] = yp
        utils.plot_roc_curves(self.model_path, dataset, fold, valid_labels, y_true_mat, y_pred_mat)
        utils.plot_pr_curves(self.model_path, dataset, fold, valid_labels, y_true_mat, y_pred_mat)
    return out


def evaluate(self, cfg):
    """Evaluate the best checkpoint on train/val/test splits."""
    def _run():
        original_split = self.split
        if cfg.split is not None:
            self.split = cfg.split
        monitor_thread = None
        monitor_stop = None
        if self.gpu_monitor:
            self._gpu_status = ""
            monitor_thread, monitor_stop = self._start_gpu_monitor()
        ram_thread = None
        ram_stop = None
        if self.ram_monitor:
            self._ram_status = ""
            ram_thread, ram_stop = self._start_ram_monitor()
        eval_start_time = time.time()
        try:
            best_model, input_norm_dict = self._load_inference_model()
            train_dl, val_dl, test_dl, input_norm_dict = self._setup_data(input_norm_dict, use_train_transforms=False)
            fold_map = {
                'train': train_dl,
                'val': val_dl,
                'test': test_dl,
            }
            selected_folds = [fold.lower() for fold in cfg.folds]
            invalid = [fold for fold in selected_folds if fold not in fold_map]
            if invalid:
                raise ValueError(f"folds must be chosen from ('train', 'val', 'test'); got {invalid}")

            for fold in selected_folds:
                dataloader = fold_map[fold]
                self._evaluate(best_model, dataloader, fold, input_norm_dict)
                self.get_metrics(fold=fold, dataset=self.dataset)
                print('eval time taken: ', time.time() - eval_start_time)
        finally:
            if monitor_stop is not None:
                monitor_stop.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=1.0)
            if ram_stop is not None:
                ram_stop.set()
            if ram_thread is not None:
                ram_thread.join(timeout=1.0)
            self.split = original_split

    return _run_in_eval_mode(self, _run)


def embed(self, cfg):
    """Generate and save embedding vectors for each study."""
    def _run():
        original_split = self.split
        self.split = cfg.split
        best_model, input_norm_dict = self._load_inference_model()
        _, _, dataloader, input_norm_dict = self._setup_data(input_norm_dict, use_train_transforms=False)

        if cfg.embed_file is None:
            resolved_embed_file = os.path.join(self.model_path, f'embeddings_{self.dataset}_{self.task}.h5')
        else:
            resolved_embed_file = cfg.embed_file
        try:
            with h5py.File(resolved_embed_file, 'w') as f:
                f.attrs['dataset'] = self.dataset
                f.attrs['model_name'] = self.model_name
                f.attrs['task'] = self.task
                pbar = tqdm(dataloader, total=len(dataloader.dataset))
                for embedding, correct_labels, eid in pbar:
                    if torch.cuda.is_available():
                        embedding = embedding.to("cuda")
                    with torch.no_grad():
                        f[str(eid)] = best_model.embed(embedding).cpu().numpy()
        finally:
            self.split = original_split
        print(f'saved embeddings to {resolved_embed_file}')

    return _run_in_eval_mode(self, _run)


def explain(self, cfg):
    """Generate integrated gradients explanations and save to CSV."""
    return _run_in_eval_mode(self, lambda: _explain_impl(self, cfg))


def _explain_impl(self, cfg):
    original_split = self.split
    self.split = cfg.split
    explain_tasks = cfg.explain_tasks
    if isinstance(explain_tasks, str):
        if explain_tasks.lower() == 'all':
            explain_tasks = self.task_labels
        else:
            explain_tasks = (explain_tasks,)
    print('explain_tasks:', explain_tasks)
    try:
        best_model, input_norm_dict = self._load_inference_model()
        _, _, test_dataloader, input_norm_dict = self._setup_data(input_norm_dict, use_train_transforms=False)

        print('run model on test set')
        return_model_outputs, return_correct_outputs, return_eids, loss = run_model_on_dataloader(
            best_model, test_dataloader, self.loss_fn, amp=self.amp
        )
        return_correct_output_np = np.array(return_correct_outputs).squeeze()
        y_true_test_norm = return_correct_output_np
        return_model_output_np = np.array(return_model_outputs)
        if self.task == 'measure':
            return_model_output_np = utils.un_normalize_output(return_model_output_np, self.task_labels, input_norm_dict)
            return_correct_output_np = utils.un_normalize_output(return_correct_output_np, self.task_labels, input_norm_dict)
        y_pred_test = return_model_output_np
        y_true_test = return_correct_output_np

        frames = []
        measure_maes = {'EF05': 0.0277, 'AR01': 0.13}
        for measure in explain_tasks:
            task_idx = self.task_labels.index(measure)
            y_trues = y_true_test[:, task_idx]
            y_trues_norm = y_true_test_norm[:, task_idx]
            y_preds = y_pred_test[:, task_idx]
            if self.task == 'measure':
                quantiles = [0.] + [np.nanquantile(y_trues, i) for i in [.2, .4, .6, .8, 1.]]
                test_errors = np.abs(y_trues - y_preds)
                sample_size = 10
            else:
                quantiles = [0.5, 1.]
                y_pred_top100 = np.sort(y_preds)[-100:][0]
                sample_size = 50
            print('quantiles for', measure, ':', quantiles)
            sample_idxs = np.arange(len(y_trues))
            for q_bot, q_top in zip(quantiles[:-1], quantiles[1:]):
                if self.task == 'measure':
                    mask = (
                        (~np.isnan(y_trues))
                        & (y_trues > q_bot)
                        & (y_trues <= q_top)
                        & (test_errors < measure_maes[measure])
                    )
                else:
                    mask = (
                        (~np.isnan(y_trues))
                        & (y_trues > q_bot)
                        & (y_trues <= q_top)
                        & (y_preds > y_pred_top100)
                    )
                sample_idxs_subset = sample_idxs[mask]
                if len(sample_idxs_subset) <= sample_size:
                    print('not enough samples (len subset:', len(sample_idxs_subset), ')', 'sample_size:', sample_size)
                    print('there are ', (~np.isnan(y_trues)).sum(), 'non-missing labels')
                    print('there are ', ((y_trues > q_bot) & (y_trues <= q_top)).sum(), f'samples in [{q_bot},{q_top}]')
                    if self.task != 'measure':
                        print('there are', (y_preds > y_pred_top100).sum(), 'predictions >', y_pred_top100)
                        print('try relaxing y_pred_top100 constraint')
                        mask = (~np.isnan(y_trues)) & (y_trues > q_bot) & (y_trues <= q_top)
                        sample_idxs_subset = sample_idxs[mask]
                    if len(sample_idxs_subset) <= sample_size:
                        print('didnt work, adjusting samples to', len(sample_idxs_subset))
                        sample_size = len(sample_idxs_subset)
                        print('new sample size:', sample_size)
                assert len(sample_idxs_subset) >= sample_size, "not enough samples per quantile"
                chosen_idxs = np.random.choice(sample_idxs_subset, size=sample_size, replace=False)
                for i in chosen_idxs:
                    sample = test_dataloader.dataset[i]
                    y_true = y_trues[i]
                    y_true_norm = y_trues_norm[i]
                    y_pred = y_preds[i]
                    if isinstance(test_dataloader.dataset, torch.utils.data.Subset):
                        study_filenames, echo_id = test_dataloader.dataset.dataset.get_filenames(i)
                    else:
                        study_filenames, echo_id = test_dataloader.dataset.get_filenames(i)
                    x_list, y, idx = sample
                    y_norm = y.cpu().numpy().T
                    if self.task == 'measure':
                        y = utils.un_normalize_output(y_norm, self.task_labels, input_norm_dict)
                    y = y.reshape(-1)
                    assert y[task_idx] == y_true
                    assert not np.isnan(y_true)

                    scores, attrs, obj, yhat = integrated_gradients_video_level(
                        best_model,
                        x_list,
                        mode=cfg.explain_mode,
                        loss='mae' if self.task == 'measure' else 'bce_logits',
                        y_true=y_true_norm,
                        task_type="regression" if self.task == 'measure' else 'classification',
                        task_idx=task_idx,
                        steps=64,
                    )
                    scores = scores.cpu().numpy()
                    cap = min(x_list.shape[0], cfg.explain_n)
                    ind = np.argpartition(scores, -cap)[-cap::-1]
                    top5scores = scores[ind]
                    top_filenames = study_filenames[ind]
                    if self.task != 'measure':
                        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
                    else:
                        loss = np.abs(y_true - y_pred)
                    result = dict(measure=measure, echo_id=echo_id, y_pred=y_pred, y_true=y_true, loss=loss)

                    for k in np.arange(cfg.explain_n):
                        if k >= len(top_filenames):
                            result[f'top_video_{k+1}'] = None
                            result[f'top_video_{k+1}_score'] = None
                        else:
                            result[f'top_video_{k+1}'] = top_filenames[k]
                            result[f'top_video_{k+1}_score'] = top5scores[k]
                    frames.append(result)
        df_explain = pd.DataFrame(frames)
        tv_cols = [c for c in df_explain.columns if 'top_video' in c and 'score' not in c]
        tvs_cols = [c for c in df_explain.columns if 'top_video' in c and 'score' in c]
        df_explain = df_explain[['echo_id', 'measure', 'y_true', 'y_pred', 'loss'] + tv_cols + tvs_cols]
        df_explain['dataset'] = self.dataset
        explain_file_name = f'explanation_test_{self.dataset}.explain_n-{cfg.explain_n}.mode-{cfg.explain_mode}.csv'
        df_explain.to_csv(os.path.join(self.model_path, explain_file_name))
        print('saved explanations to', os.path.join(self.model_path, explain_file_name))
    finally:
        self.split = original_split
