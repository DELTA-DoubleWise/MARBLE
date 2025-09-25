"""
This file includes code adapted from the Icefall project:
https://github.com/k2-fsa/icefall

Original license: Apache 2.0
Substantial parts of the checkpoint loading and model averaging logic
were adapted from icefall/checkpoint.py, with modifications for Auden's model system.
"""

import glob
import logging
import shutil
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
from torch import Tensor

import torch
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from lhotse.dataset.sampling.base import CutSampler
LRSchedulerType = object

def save_checkpoint(
    filename: Path,
    model: Optional[nn.Module] = None,
    model_avg: Optional[nn.Module] = None,
    batch_idx_train: int = 0,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[Any] = None,
    rank: int = 0,
) -> None:
    """
    Save training state to a checkpoint file (only on rank 0).

    Args:
        filename: Path to save the checkpoint.
        model: Current training model (can be wrapped in DDP).
        model_avg: Averaged model weights (should be CPU-side).
        batch_idx_train: Current global step.
        optimizer: Optimizer to save.
        scheduler: LR scheduler to save.
        scaler: AMP GradScaler to save.
        sampler: Optional sampler (e.g., CutSampler).
        rank: Only saves if rank == 0 (for DDP).
    """
    if rank != 0:
        return

    logging.info(f"Saving checkpoint to {filename}")

    # Unwrap DDP if needed
    if isinstance(model, DDP):
        model = model.module
    if isinstance(model_avg, DDP):
        model_avg = model_avg.module

    def get_state_dict(m: Optional[nn.Module]) -> Optional[Dict[str, Tensor]]:
        if m is None:
            return None
        state = m.state_dict()
        return state

    checkpoint = {
        "model": get_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "grad_scaler": scaler.state_dict() if scaler else None,
        "sampler": sampler.state_dict() if sampler else None,
        "batch_idx_train": batch_idx_train,
    }

    if model_avg is not None:
        checkpoint["model_avg"] = get_state_dict(model_avg.to(torch.float32))

    torch.save(checkpoint, filename)



def update_averaged_model(
    average_period,
    batch_idx_train,
    model_cur: Union[nn.Module, DDP],
    model_avg: nn.Module,
) -> None:
    """Update the averaged model:
    model_avg = model_cur * (average_period / batch_idx_train)
      + model_avg * ((batch_idx_train - average_period) / batch_idx_train)

    Args:
      params:
        User defined parameters, e.g., epoch, loss.
      model_cur:
        The current model.
      model_avg:
        The averaged model to be updated.
    """
    weight_cur = average_period / batch_idx_train
    weight_avg = 1 - weight_cur

    if isinstance(model_cur, DDP):
        model_cur = model_cur.module

    cur = model_cur.state_dict()
    avg = model_avg.state_dict()

    average_state_dict(
        state_dict_1=avg,
        state_dict_2=cur,
        weight_1=weight_avg,
        weight_2=weight_cur,
    )


def average_checkpoints_with_averaged_model(
    filename_start: str,
    filename_end: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Tensor]:
    """Average model parameters over the range with given
    start model (excluded) and end model.

    Let start = batch_idx_train of model-start;
        end = batch_idx_train of model-end;
        interval = end - start.
    Then the average model over range from start (excluded) to end is
    (1) avg = (model_end * end - model_start * start) / interval.
    It can be written as
    (2) avg = model_end * weight_end + model_start * weight_start,
        where weight_end = end / interval,
              weight_start = -start / interval = 1 - weight_end.
    Since the terms `weight_end` and `weight_start` would be large
    if the model has been trained for lots of batches, which would cause
    overflow when multiplying the model parameters.
    To avoid this, we rewrite (2) as:
    (3) avg = (model_end + model_start * (weight_start / weight_end))
              * weight_end

    The model index could be epoch number or iteration number.

    Args:
      filename_start:
        Checkpoint filename of the start model. We assume it
        is saved by :func:`save_checkpoint`.
      filename_end:
        Checkpoint filename of the end model. We assume it
        is saved by :func:`save_checkpoint`.
      device:
        Move checkpoints to this device before averaging.
    """
    state_dict_start = torch.load(filename_start, map_location=device)
    state_dict_end = torch.load(filename_end, map_location=device)

    batch_idx_train_start = state_dict_start["batch_idx_train"]
    batch_idx_train_end = state_dict_end["batch_idx_train"]
    interval = batch_idx_train_end - batch_idx_train_start
    assert interval > 0, interval
    weight_end = batch_idx_train_end / interval
    weight_start = 1 - weight_end

    model_end = state_dict_end["model_avg"]
    model_start = state_dict_start["model_avg"]
    avg = model_end

    # scale the weight to avoid overflow
    average_state_dict(
        state_dict_1=avg,
        state_dict_2=model_start,
        weight_1=1.0,
        weight_2=weight_start / weight_end,
        scaling_factor=weight_end,
    )

    return avg


def average_state_dict(
    state_dict_1: Dict[str, Tensor],
    state_dict_2: Dict[str, Tensor],
    weight_1: float,
    weight_2: float,
    scaling_factor: float = 1.0,
) -> Dict[str, Tensor]:
    """Average two state_dict with given weights:
    state_dict_1 = (state_dict_1 * weight_1 + state_dict_2 * weight_2)
      * scaling_factor
    It is an in-place operation on state_dict_1 itself.
    """
    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()
    for k, v in state_dict_1.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())
    for k in uniqued_names:
        v = state_dict_1[k]
        if torch.is_floating_point(v):
            v *= weight_1
            v += state_dict_2[k].to(device=state_dict_1[k].device) * weight_2
            v *= scaling_factor


def find_checkpoints(out_dir: Path, iteration: int = 0) -> List[str]:
    """Find all available checkpoints in a directory.

    The checkpoint filenames have the form: `checkpoint-xxx.pt`
    where xxx is a numerical value.

    Assume you have the following checkpoints in the folder `foo`:

        - checkpoint-1.pt
        - checkpoint-20.pt
        - checkpoint-300.pt
        - checkpoint-4000.pt

    Case 1 (Return all checkpoints)::

      find_checkpoints(out_dir='foo')

    Case 2 (Return checkpoints newer than checkpoint-20.pt, i.e.,
    checkpoint-4000.pt, checkpoint-300.pt, and checkpoint-20.pt)

        find_checkpoints(out_dir='foo', iteration=20)

    Case 3 (Return checkpoints older than checkpoint-20.pt, i.e.,
    checkpoint-20.pt, checkpoint-1.pt)::

        find_checkpoints(out_dir='foo', iteration=-20)

    Args:
      out_dir:
        The directory where to search for checkpoints.
      iteration:
        If it is 0, return all available checkpoints.
        If it is positive, return the checkpoints whose iteration number is
        greater than or equal to `iteration`.
        If it is negative, return the checkpoints whose iteration number is
        less than or equal to `-iteration`.
    Returns:
      Return a list of checkpoint filenames, sorted in descending
      order by the numerical value in the filename.
    """
    checkpoints = list(glob.glob(f"{out_dir}/checkpoint-[0-9]*.pt"))
    pattern = re.compile(r"checkpoint-([0-9]+).pt")
    iter_checkpoints = []
    for c in checkpoints:
        result = pattern.search(c)
        if not result:
            logging.warn(f"Invalid checkpoint filename {c}")
            continue

        iter_checkpoints.append((int(result.group(1)), c))

    # iter_checkpoints is a list of tuples. Each tuple contains
    # two elements: (iteration_number, checkpoint-iteration_number.pt)

    iter_checkpoints = sorted(iter_checkpoints, reverse=True, key=lambda x: x[0])
    if iteration >= 0:
        ans = [ic[1] for ic in iter_checkpoints if ic[0] >= iteration]
    else:
        assert iter_checkpoints[0][0] >= -iteration
        ans = [ic[1] for ic in iter_checkpoints if ic[0] <= -iteration]

    return ans


def make_averaged_model_state_dict(exp_dir, iter, epoch, avg):
    if iter > 0:
        filenames = find_checkpoints(exp_dir, iteration=-iter)[
            : avg + 1
        ]
        if len(filenames) == 0:
            raise ValueError(
                f"No checkpoints found for"
                f" --iter {iter}, --avg {avg}"
            )
        elif len(filenames) < avg + 1:
            raise ValueError(
                f"Not enough checkpoints ({len(filenames)}) found for"
                f" --iter {iter}, --avg {avg}"
            )
        filename_start = filenames[-1]
        filename_end = filenames[0]
        logging.info(
            "Calculating the averaged model over iteration checkpoints"
            f" from {filename_start} (excluded) to {filename_end}"
        )
    else:
        assert avg > 0, avg
        start = epoch - avg
        assert start >= 1, start
        filename_start = f"{exp_dir}/epoch-{start}.pt"
        filename_end = f"{exp_dir}/epoch-{epoch}.pt"
        logging.info(
            f"Calculating the averaged model over epoch range from "
            f"{start} (excluded) to {epoch}"
        )

    state_dict = average_checkpoints_with_averaged_model(
        filename_start=filename_start,
        filename_end=filename_end,
    )
    
    return state_dict
        
        
def load_checkpoint(
    filename: Path,
    model: Optional[nn.Module] = None,
    model_avg: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRSchedulerType] = None,
    scaler: Optional[GradScaler] = None,
    sampler: Optional[CutSampler] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    TODO: document it
    """
    logging.info(f"Loading checkpoint from {filename}")
    checkpoint = torch.load(filename, map_location="cpu")

    if model is not None and "model" in checkpoint:
        if next(iter(checkpoint["model"])).startswith("module."):
            logging.info("Loading checkpoint saved by DDP")

            dst_state_dict = model.state_dict()
            src_state_dict = checkpoint["model"]
            for key in dst_state_dict.keys():
                src_key = "{}.{}".format("module", key)
                dst_state_dict[key] = src_state_dict.pop(src_key)
            assert len(src_state_dict) == 0
            model.load_state_dict(dst_state_dict, strict=strict)
        else:
            if next(iter(model.state_dict())).startswith("module."):
                model.module.load_state_dict(checkpoint["model"], strict=strict)
            else:
                model.load_state_dict(checkpoint["model"], strict=strict)

        checkpoint.pop("model")

    if model_avg is not None and "model_avg" in checkpoint:
        logging.info("Loading averaged model")
        model_avg.load_state_dict(checkpoint["model_avg"], strict=strict)
        checkpoint.pop("model_avg")

    def load(name, obj):
        s = checkpoint.get(name, None)
        if obj and s:
            obj.load_state_dict(s)
            checkpoint.pop(name)

    load("optimizer", optimizer)
    load("scheduler", scheduler)
    load("grad_scaler", scaler)
    load("sampler", sampler)

    return checkpoint


def remove_checkpoints(
    out_dir: Path,
    topk: int,
    rank: int = 0,
):
    """Remove checkpoints from the given directory.

    We assume that checkpoint filename has the form `checkpoint-xxx.pt`
    where xxx is a number, representing the number of processed batches
    when saving that checkpoint. We sort checkpoints by filename and keep
    only the `topk` checkpoints with the highest `xxx`.

    Args:
      out_dir:
        The directory containing checkpoints to be removed.
      topk:
        Number of checkpoints to keep.
      rank:
        If using DDP for training, it is the rank of the current node.
        Use 0 if no DDP is used for training.
    """
    assert topk >= 1, topk
    if rank != 0:
        return
    checkpoints = find_checkpoints(out_dir)

    if len(checkpoints) == 0:
        logging.warn(f"No checkpoints found in {out_dir}")
        return

    if len(checkpoints) <= topk:
        return

    to_remove = checkpoints[topk:]
    for c in to_remove:
        os.remove(c)
        
        
def resolve_checkpoint_filename(
    checkpoint_filename: str = None,
    iter: int = 0,
    epoch: int = 0,
    avg: int = 1):
    """
    Priority for checkpoint path:
        1. checkpoint_filename (explicit)
        2. iter/avg → checkpoint or averaged iter checkpoint
        3. epoch/avg → checkpoint or averaged epoch checkpoint
        4. fallback to pretrained.pt
    """
    if checkpoint_filename is not None:
        return checkpoint_filename
    elif iter > 0:
        if avg > 1:
            checkpoint_filename = f"averaged_iter{iter}_avg{avg}.pt"
        else:
            checkpoint_filename = f"checkpoint-{iter}.pt"
    elif epoch > 0:
        if avg > 1:
            checkpoint_filename = f"averaged_epoch{epoch}_avg{avg}.pt"
        else:
            checkpoint_filename = f"epoch-{epoch}.pt"
    else:
        checkpoint_filename = "pretrained.pt"

    logging.info(f"[Checkpoint] Resolved model checkpoint filename: {checkpoint_filename}") 
    return checkpoint_filename


def generate_and_save_averaged_model(exp_dir, iter=0, epoch=0 ,avg=1):
    if iter > 0:
        ckpt_path = Path(exp_dir) / f"averaged_iter{iter}_avg{avg}.pt"
    elif epoch > 0:
        ckpt_path = Path(exp_dir) / f"averaged_epoch{epoch}_avg{avg}.pt"    
    if not os.path.exists(ckpt_path):
        logging.warning(f"[Checkpoint] Checkpoint not found at {ckpt_path}. Attempting to generate averaged checkpoint.")
        state_dict = make_averaged_model_state_dict(exp_dir, iter, epoch, avg)
        logging.info(f"[Checkpoint] Saving generated averaged checkpoint to {ckpt_path}")
        torch.save(state_dict, ckpt_path)


def load_model_params(
    model: torch.nn.Module,
    ckpt_path: str,
    init_modules: Optional[List[str]] = None,
    strict: bool = True,
    map_location="cpu"
) -> None:
    """
    Load checkpoint into model.

    Args:
        model (nn.Module): The model to load parameters into.
        ckpt_path (str): Path to the checkpoint file.
        init_modules (List[str], optional): If provided, only load parameters with these prefixes.
        strict (bool): Whether to strictly enforce matching keys.
        map_location: torch.load map_location (default: "cpu")
    """
    def _load_state(model: torch.nn.Module, state_dict: dict, strict: bool):
        if strict:
            model.load_state_dict(state_dict, strict=True)
            logging.info("[Checkpoint] Loaded successfully with strict=True.")
        else:
            result = model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                logging.warning(f"[Checkpoint] Missing keys: {result.missing_keys}")
            if result.unexpected_keys:
                logging.warning(f"[Checkpoint] Unexpected keys: {result.unexpected_keys}")
    
    
    logging.info(f"[Checkpoint] Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=map_location,weights_only=False)

    # Handle nested structure (e.g., {"model": ..., "optimizer": ...})
    if isinstance(state_dict, dict):
        if "model_avg" in state_dict:
            state_dict = state_dict["model_avg"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

    # Strip "module." prefix from DDP-trained checkpoints
    if all(k.startswith("module.") for k in state_dict):
        logging.info("[Checkpoint] Detected DDP-style checkpoint. Stripping 'module.' prefix.")
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    # === Full load ===
    if not init_modules:
        _load_state(model, state_dict, strict)
        return

    # === Partial load by prefix ===
    dst_state_dict = model.state_dict()

    for prefix in init_modules:
        prefix = prefix.strip()
        logging.info(f"[Checkpoint] Loading module prefix: '{prefix}'")

        src_keys = [k for k in state_dict if k.startswith(prefix + ".")]
        dst_keys = [k for k in dst_state_dict if k.startswith(prefix + ".")]

        missing = set(dst_keys) - set(src_keys)
        extra = set(src_keys) - set(dst_keys)

        if missing:
            logging.warning(f"[Checkpoint] Missing keys for '{prefix}': {missing}")
        if extra:
            logging.warning(f"[Checkpoint] Extra keys in checkpoint for '{prefix}': {extra}")

        for k in src_keys:
            if k in dst_state_dict:
                dst_state_dict[k] = state_dict[k]

    # Load the updated state_dict
    model.load_state_dict(dst_state_dict, strict=False)
