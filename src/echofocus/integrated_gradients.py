"""Integrated gradients utilities for video-level attributions."""

import torch
import torch.nn.functional as F

@torch.no_grad()
def _infer_device(model):
    """Infer the device of a model from its parameters.

    Args:
        model (torch.nn.Module): Model to inspect.

    Returns:
        torch.device: Device of the first parameter, or CPU if none exist.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _forward_from_video_tensor(model, x_tensor):
    """Run the model on stacked video embeddings.

    Args:
        model (torch.nn.Module): Model that expects a list of tensors.
        x_tensor (torch.Tensor): Stacked embeddings with shape (N, D).

    Returns:
        torch.Tensor: Output vector with shape (T,).
    """
    x_list = [x_tensor[i] for i in range(x_tensor.shape[0])]
    out = model(x_list)

    if out.ndim == 2 and out.shape[0] == 1:
        out = out[0]
    # out = out.reshape(-1)
    return out.reshape(-1)

def integrated_gradients_video_level(
    model,
    x_list,
    baseline=None,              # None -> zeros
    steps=64,

    # Choose how to form the scalar objective:
    mode="pred",                # "pred" or "loss" or "custom"
    task_type="regression",     # "regression" or "classification" (classification assumes logits)

    # For mode="pred":
    task_idx=None,              # int
    task_indices=None,          # list[int]
    task_weights=None,          # tensor/list length T

    # For mode="loss":
    y_true=None,                # scalar (if task_idx) or length T
    loss="mse",                 # regression: "mse"|"mae"; classification: "bce_logits"
    reduction="mean",           # "mean" or "sum" (for loss across tasks)
    pos_weight=None,            # optional tensor for BCEWithLogits, length T or scalar

    # For mode="custom":
    objective_fn=None,          # callable: objective_fn(yhat: (T,)) -> scalar

    # Attribution aggregation to per-video scores:
    aggregate="sum_abs",        # "sum"|"sum_abs"|"l2"
):
    """
    Unified Integrated Gradients for multitask regression or multitask classification.

    - Classification assumed to output logits (not softmax).
    - Regression assumed to output real-valued predictions.
    - Attributions are computed wrt input video embeddings (one row per video).

    Returns:
        video_scores: (N,) tensor
        attributions: (N, D) tensor
        objective_value: scalar tensor at true input
        yhat_at_x: (T,) tensor at true input
    """
    device = _infer_device(model)

    model_was_training = model.training
    model.eval()  # disables dropout etc.

    x = torch.stack([t.to(device) for t in x_list], dim=0)  # (N, D)

    if baseline is None:
        x0 = torch.zeros_like(x)
    else:
        if isinstance(baseline, list):
            x0 = torch.stack([t.to(device) for t in baseline], dim=0)
        else:
            x0 = baseline.to(device)
        if x0.shape != x.shape:
            raise ValueError(f"Baseline shape {x0.shape} must match input shape {x.shape}")

    # ---------------- objective builders ----------------
    def _pred_objective(yhat):
        yhat = yhat.reshape(-1)

        if task_weights is not None:
            w = torch.as_tensor(task_weights, device=device, dtype=yhat.dtype).reshape(-1)
            if w.numel() != yhat.numel():
                raise ValueError("task_weights must have length == number of tasks")
            return (yhat * w).sum()

        if task_idx is not None:
            return yhat[task_idx]

        if task_indices is not None:
            idx = torch.tensor(task_indices, device=device, dtype=torch.long)
            return yhat.index_select(0, idx).sum()

        # fallback
        return yhat.sum()

    def _loss_objective(yhat):
        yhat = yhat.reshape(-1)

        if y_true is None:
            raise ValueError("mode='loss' requires y_true")

        yt = torch.as_tensor(y_true, device=device, dtype=yhat.dtype).reshape(-1)

        # If task_idx is provided and y_true is scalar, treat as single-task loss
        if task_idx is not None and yt.numel() == 1:
            yhat_sel = yhat[task_idx].reshape(1)
            yt_sel = yt.reshape(1)
            w=None
        else:
            # Otherwise y_true must match task dimension
            if yt.numel() != yhat.numel():
                raise ValueError("y_true must be scalar with task_idx, or length == number of tasks")
            yhat_sel = yhat
            yt_sel = yt

            # Optionally restrict to task_indices
            if task_indices is not None:
                idx = torch.tensor(task_indices, device=device, dtype=torch.long)
                yhat_sel = yhat_sel.index_select(0, idx)
                yt_sel = yt_sel.index_select(0, idx)

            # Or weighted loss across tasks
            if task_weights is not None:
                w = torch.as_tensor(task_weights, device=device, dtype=yhat.dtype).reshape(-1)
                if w.numel() != yhat.numel():
                    raise ValueError("task_weights must have length == number of tasks")
                if task_indices is not None:
                    w = w.index_select(0, idx)
            else:
                w = None

        if task_type == "regression":
            if loss == "mse":
                per_task = (yhat_sel - yt_sel) ** 2
            elif loss == "mae":
                per_task = (yhat_sel - yt_sel).abs()
            else:
                raise ValueError("For regression, loss must be 'mse' or 'mae'")
        elif task_type == "classification":
            # BCE with logits: targets should be 0/1 (or probabilities)
            if loss not in ("bce_logits", "bcewithlogits"):
                raise ValueError("For classification, loss must be 'bce_logits' (BCEWithLogits)")
            # pos_weight can be scalar or vector of same length as selected tasks
            pw = None
            if pos_weight is not None:
                pw = torch.as_tensor(pos_weight, device=device, dtype=yhat.dtype).reshape(-1)
                if pw.numel() == 1:
                    pass
                elif pw.numel() != yhat_sel.numel():
                    raise ValueError("pos_weight must be scalar or match number of selected tasks")

            per_task = F.binary_cross_entropy_with_logits(
                yhat_sel, yt_sel, reduction="none", pos_weight=pw
            )
        else:
            raise ValueError("task_type must be 'regression' or 'classification'")

        if w is not None:
            per_task = per_task * w

        if reduction == "mean":
            return per_task.mean()
        elif reduction == "sum":
            return per_task.sum()
        else:
            raise ValueError("reduction must be 'mean' or 'sum'")

    def _objective(yhat):
        if mode == "pred":
            return _pred_objective(yhat)
        elif mode == "loss":
            return _loss_objective(yhat)
        elif mode == "custom":
            if objective_fn is None:
                raise ValueError("mode='custom' requires objective_fn")
            val = objective_fn(yhat)
            if not torch.is_tensor(val):
                val = torch.tensor(val, device=device, dtype=yhat.dtype)
            return val.reshape(())
        else:
            raise ValueError("mode must be 'pred', 'loss', or 'custom'")

    # ---------------- values at true input ----------------
    with torch.no_grad():
        yhat_at_x = _forward_from_video_tensor(model, x)     # (T,)
        objective_value = _objective(yhat_at_x).detach()     # scalar

    # ---------------- IG loop ----------------
    total_grads = torch.zeros_like(x)

    for s in range(1, steps + 1):
        alpha = float(s) / steps
        x_alpha = (x0 + alpha * (x - x0)).detach().requires_grad_(True)  # (N, D)

        yhat = _forward_from_video_tensor(model, x_alpha)  # (T,)
        obj = _objective(yhat)                             # scalar

        grads = torch.autograd.grad(
            outputs=obj,
            inputs=x_alpha,
            grad_outputs=torch.ones_like(obj),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]  # (N, D)

        total_grads += grads.detach()

    avg_grads = total_grads / steps
    attributions = (x - x0) * avg_grads  # (N, D)

    # Reduce to per-video score
    if aggregate == "sum":
        video_scores = attributions.sum(dim=2).sum(dim=1)
    elif aggregate == "sum_abs":
        video_scores = attributions.abs().sum(dim=2).sum(dim=1)
    elif aggregate == "l2":
        video_scores = torch.sqrt((attributions ** 2).sum(dim=2).sum(dim=1) + 1e-12)
    else:
        raise ValueError("aggregate must be one of: 'sum', 'sum_abs', 'l2'")

    if model_was_training:
        model.train()

    return video_scores, attributions, objective_value, yhat_at_x
