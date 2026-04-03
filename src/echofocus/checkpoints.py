"""Checkpoint and inference helpers shared across runtime flows."""

import numpy as np
import torch
from tqdm import tqdm

from . import utils


def save_nn(model, path, perf_log, optimizer=None, scheduler=None, input_norm_dict=None):
    """Save a model checkpoint to disk."""
    out_dict = {}

    out_dict["model_state_dict"] = model.state_dict()
    out_dict["perf_log"] = perf_log

    out_dict["numpy_random_state"] = np.random.get_state()
    out_dict["torch_random_state"] = torch.get_rng_state()
    out_dict["cuda_random_state"] = torch.cuda.get_rng_state()

    if optimizer is not None:
        out_dict["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        out_dict["scheduler_state_dict"] = scheduler.state_dict()

    if input_norm_dict is not None:
        out_dict["input_norm_dict"] = input_norm_dict
    torch.save(out_dict, path)


def load_model_and_random_state(path, model, optimizer=None, scheduler=None):
    """Load a checkpoint and restore model and RNG state."""
    import_dict = torch.load(path, weights_only=False)

    model.load_state_dict(import_dict["model_state_dict"])
    if "optimizer_state_dict" in import_dict.keys() and (optimizer is not None):
        optimizer.load_state_dict(import_dict["optimizer_state_dict"])
    else:
        print("warning no optimizer loaded")
    if "scheduler_state_dict" in import_dict.keys() and (scheduler is not None):
        scheduler.load_state_dict(import_dict["scheduler_state_dict"])
    else:
        print("warning no scheduler loaded")

    utils.load_random_state(import_dict)
    perf_log = import_dict["perf_log"]

    print("model loaded, epoch", perf_log[-1][0])

    if "input_norm_dict" in import_dict.keys():
        input_norm_dict = import_dict["input_norm_dict"]
        print("Loaded input_norm_dict")
    else:
        input_norm_dict = None
        print("input_norm_dict NOT loaded")

    return model, optimizer, scheduler, perf_log, input_norm_dict


def run_model_on_dataloader(model, dataloader, loss_func_pointer, amp=False, cache_hook=None):
    """Run inference on a dataloader and collect outputs."""
    return_model_outputs = []
    return_correct_outputs = []
    return_EIDs = []
    model.eval()

    loss = 0
    is_classification = isinstance(loss_func_pointer, torch.nn.BCEWithLogitsLoss)
    use_amp = amp and torch.cuda.is_available()
    pbar = tqdm(dataloader, total=len(dataloader.dataset), desc="Inference")
    monitor_owner = getattr(model, "_echofocus_monitor_owner", None)
    for embedding, correct_labels, eid in pbar:
        postfix = []
        if monitor_owner is not None:
            gpu_status = getattr(monitor_owner, "_gpu_status", "")
            ram_status = getattr(monitor_owner, "_ram_status", "")
            if gpu_status:
                postfix.append(f"gpu={gpu_status}")
            if ram_status:
                postfix.append(f"ram={ram_status}")
            if postfix:
                pbar.set_postfix_str(" ".join(postfix))
        if torch.cuda.is_available():
            embedding = embedding.to("cuda")
            correct_labels = correct_labels.to("cuda")

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp):
                if cache_hook is not None:
                    model_outputs = cache_hook(embedding, eid)
                else:
                    model_outputs = model(embedding)
            outputs_to_return = torch.sigmoid(model_outputs) if is_classification else model_outputs
            return_model_outputs.append(outputs_to_return.to("cpu"))
            return_correct_outputs.append(correct_labels.to("cpu"))
            return_EIDs.append(eid)
            loss += float(loss_func_pointer(model_outputs, correct_labels).to("cpu"))

    return return_model_outputs, return_correct_outputs, return_EIDs, loss
