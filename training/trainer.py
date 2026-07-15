# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
import copy
import contextlib
import gc
import json
from wat3r.utils.geometry import unproject_depth_map_to_point_map_bs
from wat3r.utils.static_mask import build_static_masks, depth_foreground_mask
from omegaconf import OmegaConf
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence
import torch.nn.functional as F
from wat3r.utils.pose_enc import pose_encoding_to_extri_fov, pose_fov_to_intri
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from training.train_utils.checkpoint import DDPCheckpointSaver
from training.train_utils.distributed import get_machine_local_and_dist_rank
from training.train_utils.freeze import freeze_modules
from training.train_utils.general import *
from training.train_utils.logging import setup_logging
from training.train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from training.train_utils.optimizer import construct_optimizers


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
            self,
            *,
            data: Dict[str, Any],
            model: Dict[str, Any],
            logging: Dict[str, Any],
            checkpoint: Dict[str, Any],
            max_epochs: int,
            mode: str = "train",
            device: str = "cuda",
            seed_value: int = 123,
            val_epoch_freq: int = 1,
            distributed: Dict[str, bool] = None,
            cuda: Dict[str, bool] = None,
            limit_train_batches: Optional[int] = None,
            limit_val_batches: Optional[int] = None,
            optim: Optional[Dict[str, Any]] = None,
            loss: Optional[Dict[str, Any]] = None,
            env_variables: Optional[Dict[str, Any]] = None,
            accum_steps: int = 1,
            debug: bool = False,
            **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value

        self.start_no_label = False

        if debug:
            self.save_no_label_data = True
        else:
            self.save_no_label_data = False

        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components(debug=debug)
        self.debug_mode = debug
        self._setup_dataloaders()

        # os.makedirs(self.temp_log_root, exist_ok=True)

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)

        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(
            backend=distributed_conf.backend,
            timeout=timedelta(minutes=distributed_conf.timeout_mins)
        )
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(
            model_state_dict, strict=self.checkpoint_conf.strict
        )
        # if self.use_ema:
        #     self.ema_model.load_state_dict(
        #         model_state_dict, strict=self.checkpoint_conf.strict
        #     )
        if self.rank == 0:
            logging.info(
                f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        is_training_checkpoint = any(
            key in checkpoint
            for key in ("steps", "optimizer", "scaler", "ema_models", "prev_epoch", "completed_epoch", "time_elapsed")
        )

        if is_training_checkpoint:
            if "epoch" in checkpoint:
                self.epoch = int(checkpoint["epoch"])
            elif "prev_epoch" in checkpoint:
                self.epoch = int(checkpoint["prev_epoch"]) + 1

            self.steps = checkpoint.get("steps", {"train": 0, "val": 0})
            self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

            if "optimizer" in checkpoint and hasattr(self, "optims"):
                optimizer_states = checkpoint["optimizer"]
                if isinstance(optimizer_states, list):
                    if len(optimizer_states) != len(self.optims):
                        raise ValueError(
                            f"Checkpoint has {len(optimizer_states)} optimizer states, "
                            f"but current training has {len(self.optims)} optimizers."
                        )
                    for optim, optim_state in zip(self.optims, optimizer_states):
                        optim.optimizer.load_state_dict(optim_state)
                else:
                    if len(self.optims) != 1:
                        raise ValueError(
                            "Checkpoint stores one optimizer state, but current training "
                            f"has {len(self.optims)} optimizers."
                        )
                    self.optims[0].optimizer.load_state_dict(optimizer_states)
                logging.info(f"Optimizer state loaded (rank {self.rank})")
            elif hasattr(self, "optims"):
                logging.info("Checkpoint has no optimizer state; optimizer starts fresh.")

            if self.optim_conf.amp.enabled and "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])
                logging.info(f"AMP scaler state loaded (rank {self.rank})")

        if self.use_ema and "ema_models" in checkpoint:
            missing_ema, unexpected_ema = self.ema_model.load_state_dict(checkpoint["ema_models"], strict=False)
            if self.rank == 0:
                logging.info(
                    f"EMA state loaded. Missing: {missing_ema or 'None'}. Unexpected: {unexpected_ema or 'None'}.")

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self, debug=False):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {'train': 0, 'val': 0}

        # Instantiate components from configs
        self.ema_conf = getattr(self.model_conf, "ema", None)
        self.use_ema = bool(getattr(self.model_conf, "enable_ema", False))

        model_conf_dict = OmegaConf.to_container(self.model_conf, resolve=True)

        # The current Wat3R model only receives architecture flags. Training-only
        # EMA options are handled by Trainer.
        model_conf_dict.pop('ema', None)
        model_conf_dict.pop('enable_ema', None)

        self.model_conf = OmegaConf.create(model_conf_dict)
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False, debug=debug)
        self.model = instantiate(self.model_conf, _recursive_=False)
        logging.info(f"EMA enabled: {self.use_ema}")
        if self.use_ema:
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.to(self.device)
            for p in self.ema_model.parameters():
                # print('p',p)
                p.requires_grad = False
            self.ema_use_buffers = bool(getattr(self.ema_conf, "use_buffers", False))
            self.ema_use_label = str(getattr(self.ema_conf, "use_label", 'depths'))
            self.ema_use_for_eval = bool(getattr(self.ema_conf, "use_for_eval", False))
            self.ema_base_momentum = float(getattr(self.ema_conf, "momentum", 0.9996))
            self.ema_momentum_schedule = str(
                getattr(self.ema_conf, "momentum_schedule", "constant"))
            self.ema_update_every = int(getattr(self.ema_conf, "update_every", 1))
            self.ema_start_unlabel = int(getattr(self.ema_conf, "start_unlabel", 1000))
            self.ema_start_ema = int(getattr(self.ema_conf, "start_ema", 500))
            self.enable_no_label_up = bool(getattr(self.ema_conf, "enable_no_label_up", False))
            self.enable_64 = bool(getattr(self.ema_conf, "enable_64", False))
            self.global_rotation = bool(getattr(self.ema_conf, "global_rotation", False))
            self.enable_static_mask = bool(getattr(self.ema_conf, "enable_static_mask", False))
            self.static_mask_visibility_tolerance = int(
                getattr(self.ema_conf, "static_mask_visibility_tolerance", 2)
            )
            self.static_mask_relative_depth_threshold = float(
                getattr(self.ema_conf, "static_mask_relative_depth_threshold", 0.05)
            )
            self.static_mask_boundary = int(getattr(self.ema_conf, "static_mask_boundary", 4))
            self.static_mask_use_foreground = bool(getattr(self.ema_conf, "static_mask_use_foreground", True))

            s_param_names = [n for n, _ in self.model.named_parameters()]
            t_param_names = [n for n, _ in self.ema_model.named_parameters()]
            assert s_param_names == t_param_names, f"EMA param names mismatch:\nS:{s_param_names}\nT:{t_param_names}"

            if self.ema_use_buffers:
                s_keys = list(self.model.state_dict().keys())
                t_keys = list(self.ema_model.state_dict().keys())
                assert s_keys == t_keys, f"EMA state_dict keys mismatch:\nS:{s_keys}\nT:{t_keys}"

            logging.info(
                f"EMA enabled. momentum={self.ema_base_momentum},\n"
                f"use_buffers={self.ema_use_buffers}, \n"
                f"use_for_eval={self.ema_use_for_eval}, \n"
                f"schedule={self.ema_momentum_schedule},\n"
                f"ema_update_every={self.ema_update_every},\n"
                f"ema_start_unlabel={self.ema_start_unlabel},\n"
                f"ema_start_ema={self.ema_start_ema},\n"
                f"enable_no_label_up={self.enable_no_label_up},\n"
                f"enable_64={self.enable_64},\n"
                f"global_rotation={self.global_rotation},\n"
                f"enable_static_mask={self.enable_static_mask},\n"
                f"static_mask_visibility_tolerance={self.static_mask_visibility_tolerance},\n"
                f"static_mask_relative_depth_threshold={self.static_mask_relative_depth_threshold},\n"
                f"static_mask_boundary={self.static_mask_boundary},\n"
                f"static_mask_use_foreground={self.static_mask_use_foreground},\n"

            )
        else:
            self.ema_model = None
        self.loss = instantiate(self.loss_conf, _recursive_=False, device=self.device)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(
                f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(
                f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}"
            )

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    @torch.no_grad()
    def _ema_update(self, student_module: nn.Module, teacher_module: nn.Module, momentum: float,
                    use_buffers: bool = False):
        s_named = dict(student_module.named_parameters())
        t_named = dict(teacher_module.named_parameters())
        for name, t_param in t_named.items():
            s_param = s_named[name]
            # print(f'{name} t_param.data.dtype',t_param.data.dtype)
            if t_param.data.dtype.is_floating_point:
                t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
            else:
                logging.debug(f"Skip non-floating EMA parameter: {name}, dtype={t_param.data.dtype}")


        if use_buffers:
            raise NotImplementedError
            # s_state = student_module.state_dict()
            # t_state = teacher_module.state_dict()
            # for k, t_buf in t_state.items():
            #     s_buf = s_state[k]
            #     if hasattr(t_buf, "dtype") and t_buf.dtype.is_floating_point:
            #         t_buf.copy_(momentum * t_buf + (1.0 - momentum) * s_buf)

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(
                self.data_conf.get('val', None), _recursive_=False
            )
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _broadcast_model(self, module: nn.Module):
        for p in module.state_dict().values():
            if torch.is_tensor(p):
                dist.broadcast(p, src=0)

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )
        # if self.ema_model is not None:
        #     self.ema_model = nn.parallel.DistributedDataParallel(
        #         self.ema_model,
        #         device_ids=[self.local_rank] if device == "cuda" else [],
        #         **ddp_options,
        #     )
        if self.use_ema:
            self._broadcast_model(self.ema_model)

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                    self.checkpoint_conf.save_freq > 0
                    and int(epoch) % self.checkpoint_conf.save_freq == 0
                    and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "epoch": int(epoch) + 1,
            "completed_epoch": epoch,
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
        }

        if self.mode != "val" and hasattr(self, "optims") and self.optims is not None:
            checkpoint_content["optimizer"] = [optim.optimizer.state_dict() for optim in self.optims]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module
        else:
            model = self.model

        ema_state = None
        if self.use_ema and (self.ema_model is not None):
            ema_state = self.ema_model.state_dict()

        saver.save_checkpoint(
            model=model,
            ema_models=ema_state,
            skip_saving_parameters=[],
            **checkpoint_content,
        )

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)

            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            if self.use_ema and self.start_no_label:
                dataloader.batch_sampler.use_no_label = True

            self.train_epoch(dataloader)

            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()

            self.epoch += 1

        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'val'

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }

        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = (
            iters_per_epoch
            if self.limit_val_batches is None
            else self.limit_val_batches
        )

        for data_iter, batch in enumerate(val_loader):
            if data_iter > limit_val_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            if 'depths' in batch.keys():
                with torch.cuda.amp.autocast(enabled=False):
                    batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16

            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                        enabled=False,
                        dtype=amp_type,
                ):
                    val_loss_dict = self._step(
                        batch, self.model, phase, loss_meters
                    )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = 'train'

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {
            name: AverageMeter(name, self.device, ":.4f") for name in loss_names
        }

        for config in self.gradient_clipper.configs:
            param_names = ",".join(config['module_names'])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")

        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = (
            iters_per_epoch
            if self.limit_train_batches is None
            else self.limit_train_batches
        )
        # if self.use_ema:
        #     train_iter = iter(train_loader)
        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)
        # data_iter_indeed = 0
        for data_iter, batch in enumerate(train_loader):
            # data_iter = data_iter_indeed
            if self.use_ema:
                # print("self.steps['train']",self.steps['train'])
                # print('self.ema_start_step',self.ema_start_step)
                if self.steps['train'] < self.ema_start_unlabel:
                    # while ('extrinsics' not in batch):
                    #     batch = next(train_iter)
                    # if 'extrinsics' not in batch:
                    #     continue
                    assert 'extrinsics' in batch
                elif self.steps['train'] == self.ema_start_unlabel:
                    train_loader.batch_sampler.use_no_label = True
                    self.start_no_label = True
                # if self.steps['train'] == 30:
                #     train_loader.dataset.base_dataset.datasets[0].min_teacher_num_images= 24

            # data_iter_indeed += 1
            # if data_iter<=270:
            #     continue

            if data_iter > limit_train_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            if 'extrinsics' in batch:
                with torch.cuda.amp.autocast(enabled=False):
                    batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps == 1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            t_temp = 1.0
            if self.use_ema and self.enable_no_label_up:
                max_step = self.ema_start_unlabel * 2  # 20
                now_step = self.steps[phase]
                t_temp *= (now_step - self.ema_start_unlabel) / self.ema_start_unlabel
                t_temp = max(0., t_temp)
                t_temp = min(1., t_temp)

            self._run_steps_on_batch_chunks(
                chunked_batches, phase, loss_meters, t_temp=t_temp
            )

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs

            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(
                    f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                )

            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_"
                                if len(self.optims) > 1
                                else (
                                    "" + f"{j}_"
                                    if len(optim.optimizer.param_groups) > 1
                                    else ""
                                )
                            )
                            self.tb_writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )
                self.tb_writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                )
            # if dist.get_rank() == 0:
            # for name, p in self.model.named_parameters():
            #         if p.grad is None:
            #             print(f"{name}: grad is None")
            #         else:
            #             print(f"{name}: grad_norm = {p.grad.norm().item():.4f}")
            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            # Optimizer step
            for optim in self.optims:
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            if self.use_ema:
                global_step = self.steps['train']  # 
                student_mod = self.model.module if isinstance(self.model,
                                                              nn.parallel.DistributedDataParallel) else self.model

                if global_step == self.ema_start_ema:
                    self.ema_model.to(self.device)
                    m = 0
                    self._ema_update(student_mod, self.ema_model, momentum=m, use_buffers=self.ema_use_buffers)
                    # self._broadcast_model(self.ema_model)
                    if global_step % self.logging_conf.log_freq == 0 and self.rank == 0:
                        self.tb_writer.log("EMA/momentum", m, global_step)
                elif (global_step > self.ema_start_ema) and (global_step % self.ema_update_every == 0):
                    self.ema_model.to(self.device)
                    m = self._ema_get_momentum()
                    self._ema_update(student_mod, self.ema_model, momentum=m, use_buffers=self.ema_use_buffers)
                    # self._broadcast_model(self.ema_model)
                    if global_step % self.logging_conf.log_freq == 0 and self.rank == 0:
                        self.tb_writer.log("EMA/momentum", m, global_step)
                elif (global_step < self.ema_start_ema):
                    self.ema_model.to('cpu')

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _ema_get_momentum(self) -> float:
        """
        :constant / ramp / cosine
         self.where (0->1) 
        """
        m0 = self.ema_base_momentum
        sched = self.ema_momentum_schedule
        w = min(max(self.where, 0.0), 1.0)

        if sched == "constant":
            return m0
        elif sched == "ramp":
            return 1.0 - (1.0 - m0) * w
        elif sched == "cosine":
            m_start = min(0.9, m0 - 0.05)
            return m0 - (m0 - m_start) * (0.5 * (1 + math.cos(math.pi * w)))
        else:
            logging.warning(f"Unknown EMA schedule '{sched}', fallback to constant.")
            return m0

    def _run_steps_on_batch_chunks(
            self,
            chunked_batches: List[Any],
            phase: str,
            loss_meters: Dict[str, AverageMeter],
            t_temp: float,
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """

        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16

        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )

            with ddp_context:
                with torch.cuda.amp.autocast(
                        enabled=self.optim_conf.amp.enabled,
                        dtype=amp_type,
                ):
                    if not self.use_ema:
                        loss_dict = self._step(
                            chunked_batch, self.model, phase, loss_meters
                        )
                    else:
                        loss_dict = self._step_ssl(
                            chunked_batch,
                            self.model,
                            self.ema_model,
                            phase,
                            loss_meters,
                            t_temp=t_temp,
                        )

                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                # print('chunked_batch',chunked_batch.keys())
                batch_size = chunked_batch["ids"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)

    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images", "depths", "extrinsics", "intrinsics",
            "cam_points", "world_points", "point_masks"
        ]
        string_keys = ["seq_name"]

        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor,
                                                torch.flip(original_tensor, dims=[1])],
                                               dim=0)

        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2

        return batch

    def _process_batch(self, batch: Mapping):
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)

        # Normalize camera extrinsics and points. The function returns new tensors.
        normalized_extrinsics, _, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )
        # print('normalized_extrinsics', normalized_extrinsics.dtype)
        # print('normalized_world_points', normalized_world_points.dtype)
        # Replace the original values in the batch with the normalized ones.
        batch["extrinsics"] = normalized_extrinsics
        batch["world_points"] = normalized_world_points
        batch["depths"] = normalized_depths

        return batch

    def _process_extrinsic(self, extrinsic, R_matrixs):
        R_old = extrinsic[..., :3, :3]  # (6, 2, 3, 3)
        # print('R_old',R_old.dtype)
        t_old = extrinsic[..., :3, 3:]  # (6, 2, 3, 1)
        # print('t_old',t_old.dtype)
        R_new = R_matrixs
        # print('R_new',R_new.dtype)
        R_rot = R_new @ R_old  # (6, 2, 3, 3)
        # print('R_rot',R_rot.dtype)

        t_rot = R_new @ t_old  # (6, 2, 3, 1)
        # print('t_rot',t_rot.dtype)

        extrinsic_rot = torch.cat([R_rot, t_rot], dim=-1)  # (6, 2, 3, 4)
        # print('extrinsic_rot',extrinsic_rot.dtype)
        return extrinsic_rot

    def _process_fov(self,
                     fov_h: torch.Tensor,
                     fov_w: torch.Tensor,
                     R_matrixs: torch.Tensor):
        """
        fov_h, fov_w: (B, S)
        R_matrixs:    (B, S, 3, 3)

        :
            fov_h_rot, fov_w_rot: ""/
        """
        swap_mask = (R_matrixs[..., 0, 1].abs() > 0.5) | (R_matrixs[..., 1, 0].abs() > 0.5)

        fov_h_rot = torch.where(swap_mask, fov_w, fov_h)
        fov_w_rot = torch.where(swap_mask, fov_h, fov_w)

        return fov_h_rot, fov_w_rot

    def _process_depth(self, depth: torch.Tensor, depth_conf: torch.Tensor, rotate_label: torch.Tensor):
        """
        depth: (B, S, H, W, 1)
        rotate_label: (B, S) , {0, 90, 180, 270}
        """
        B, S, H, W, _ = depth.shape
        depth_2d = depth.clone()[..., 0]  # (B, S, H, W)
        depth_conf_2d = depth_conf.clone()

        depth_rot = torch.empty_like(depth_2d)
        depth_conf_rot = torch.empty_like(depth_conf_2d)

        for theta, k in zip([0, 90, 180, 270], [0, 1, 2, 3]):
            mask = (rotate_label == theta)
            if not mask.any():
                continue

            d = depth_2d[mask]  # (N, H, W)
            d_conf = depth_conf_2d[mask]

            if k == 0:
                d_rot = d
                d_conf_rot = d_conf
            else:
                d_rot = torch.rot90(d, k=k, dims=(-2, -1))
                d_conf_rot = torch.rot90(d_conf, k=k, dims=(-2, -1))
            d_rot = d_rot.unsqueeze(1)
            d_conf_rot = d_conf_rot.unsqueeze(1)

            if d_rot.shape[-2:] != (H, W):
                d_rot = F.interpolate(d_rot, size=(H, W), mode='bilinear', align_corners=False)
                d_conf_rot = F.interpolate(d_conf_rot, size=(H, W), mode='bilinear', align_corners=False)

            d_rot = d_rot.squeeze(1)
            d_conf_rot = d_conf_rot.squeeze(1)
            # print('theta,k', theta, k)
            # print('d_rot', d_rot.shape)
            # print('d_conf_rot', d_conf_rot.shape)
            # theta,k 0 0
            # d_rot torch.Size([2, 308, 518])
            # theta,k 90 1
            # d_rot torch.Size([1, 518, 308])
            # class_mode normal
            # theta,k 180 2
            # d_rot torch.Size([1, 308, 518])
            # theta,k 270 3
            # d_rot torch.Size([3, 518, 308])

            depth_rot[mask] = d_rot
            depth_conf_rot[mask] = d_conf_rot

        return depth_rot.unsqueeze(-1), depth_conf_rot  # (B, S, H, W, 1)

    def _process_depth_global(self, depth: torch.Tensor, depth_conf: torch.Tensor, rotate_label: int):
        """
        depth: (B, S, H, W, 1)
        rotate_label: (B, S) , {0, 90, 180, 270}
        """
        # B, S, H, W, _ = depth.shape
        depth_rot = torch.rot90(depth, k=rotate_label // 90, dims=(2, 3))
        depth_conf = torch.rot90(depth_conf, k=rotate_label // 90, dims=(2, 3))

        # depth_conf_2d = depth_conf

        return depth_rot, depth_conf  # (B, S, H, W, 1)

    def _build_teacher_static_masks(self, teacher_out: Mapping[str, torch.Tensor]) -> torch.Tensor:
        depth = teacher_out["depth"].clone()
        if depth.ndim == 5 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)

        candidate_masks = None
        if self.static_mask_use_foreground:
            candidate_masks = depth_foreground_mask(depth)

        static_masks = build_static_masks(
            depth=depth,
            extrinsics=teacher_out["extrinsic"],
            intrinsics=teacher_out["intrinsic"],
            candidate_masks=candidate_masks,
            visibility_tolerance=self.static_mask_visibility_tolerance,
            relative_depth_threshold=self.static_mask_relative_depth_threshold,
            boundary=self.static_mask_boundary,
        )
        return static_masks

    def _process_rotation(self, batch, teacher_out, image_size):
        R_matrixs = batch['R_matrixs'].clone()

        # print('teacher_extrinsic',teacher_extrinsic.dtype)
        # print('teacher_fov_h',teacher_fov_h.dtype)
        # print('extrinsic', teacher_extrinsic.shape)
        # print('teacher_fov_h', teacher_fov_h.shape)
        # print('teacher_fov_w', teacher_fov_w.shape)
        # print('R_matrixs', R_matrixs.shape)
        teacher_depth = teacher_out["depth"].clone()
        teacher_depth_conf = teacher_out["depth_conf"].clone()
        rotate_label = batch['thetas'].clone()
        if self.global_rotation:
            # print('batch', batch.keys())
            global_thetas = batch['global_thetas']
            global_R_matrixs = batch['global_R_matrixs'].clone()
        if not self.enable_64:
            teacher_extrinsic, teacher_fov_h, teacher_fov_w = pose_encoding_to_extri_fov(
                teacher_out["pose_enc"].clone())
            student_extrinsic = self._process_extrinsic(extrinsic=teacher_extrinsic, R_matrixs=R_matrixs)
            student_fov_h, student_fov_w = self._process_fov(fov_h=teacher_fov_h, fov_w=teacher_fov_w,
                                                             R_matrixs=R_matrixs)

            if self.global_rotation:
                student_extrinsic = self._process_extrinsic(extrinsic=student_extrinsic, R_matrixs=global_R_matrixs)
                student_fov_h, student_fov_w = self._process_fov(fov_h=student_fov_h, fov_w=student_fov_w,
                                                                 R_matrixs=global_R_matrixs)
            student_intrinsic = pose_fov_to_intri(fov_h=student_fov_h, fov_w=student_fov_w, image_size_hw=image_size)
        else:
            with torch.autocast('cuda', dtype=torch.float64):  # 
                teacher_extrinsic, teacher_fov_h, teacher_fov_w = pose_encoding_to_extri_fov(
                    teacher_out["pose_enc"].clone())
                # print('teacher_extrinsic',teacher_extrinsic.dtype)
                student_extrinsic = self._process_extrinsic(extrinsic=teacher_extrinsic, R_matrixs=R_matrixs)
                student_fov_h, student_fov_w = self._process_fov(fov_h=teacher_fov_h, fov_w=teacher_fov_w,
                                                                 R_matrixs=R_matrixs)

                if self.global_rotation:
                    student_extrinsic = self._process_extrinsic(extrinsic=student_extrinsic, R_matrixs=global_R_matrixs)
                    student_fov_h, student_fov_w = self._process_fov(fov_h=student_fov_h, fov_w=student_fov_w,
                                                                     R_matrixs=global_R_matrixs)

                student_intrinsic = pose_fov_to_intri(fov_h=student_fov_h, fov_w=student_fov_w,
                                                      image_size_hw=image_size)
        # print('student_intrinsic',student_intrinsic.dtype)
        # print('student_extrinsic',student_extrinsic.dtype)
        # print('teacher_extrinsic',teacher_extrinsic.dtype)

        # print('R_matrixs', R_matrixs.dtype)
        # print('student_extrinsic', student_extrinsic.dtype)
        # print('student_extrinsic', student_extrinsic.shape)

        # print('student_fov_h', student_fov_h.shape)
        # print('student_fov_w', student_fov_w.shape)
        # extrinsic torch.Size([2, 5, 3, 4])
        # teacher_fov_h torch.Size([2, 5])
        # teacher_fov_w torch.Size([2, 5])
        # R_matrixs torch.Size([2, 5, 3, 3])
        # student_extrinsic torch.Size([2, 5, 3, 4])
        # student_fov_h torch.Size([2, 5])
        # student_fov_w torch.Size([2, 5])
        student_depth, student_depth_conf = self._process_depth(depth=teacher_depth, depth_conf=teacher_depth_conf,
                                                                rotate_label=rotate_label)

        if self.global_rotation:
            student_depth, student_depth_conf = self._process_depth_global(depth=student_depth,
                                                                           depth_conf=student_depth_conf,
                                                                           rotate_label=global_thetas)

        teacher_out['depth'] = student_depth
        teacher_out['depth_conf'] = student_depth_conf
        teacher_out['intrinsic'] = student_intrinsic
        teacher_out['extrinsic'] = student_extrinsic
        return teacher_out

    def _process_teacher_out(self, batch, teacher_out):
        # # print('teacher_out', teacher_out.keys())
        # pose_enc=teacher_out['pose_enc']
        # # print('pose_enc',pose_enc.shape)
        # depth=teacher_out['depth']
        # # print('depth',depth.shape)
        # depth_conf=teacher_out['depth_conf']
        # # print('depth_conf',depth_conf.shape)
        # ids=batch['ids']
        # # print('idx',ids)
        teacher_ids = batch['teacher_ids']
        # print('teacher_ids',teacher_ids)
        mapping = batch['mapping']
        # print('mapping',mapping)
        # print('mapping',mapping.shape)

        B, S = mapping.shape
        mapped_out = {}
        for k in teacher_out.keys():
            v = teacher_out[k]
            if v.dim() == 3:
                # [B, T, C] -> [B, N, C]
                index = mapping.unsqueeze(-1).expand(-1, -1, v.size(-1))  # [B, N, C]
                mapped_v = torch.gather(v, dim=1, index=index)  # [B, N, C]

            elif v.dim() == 5:
                # [B, T, H, W, C] -> [B, N, H, W, C]
                Bv, Tv, H, W, C = v.shape
                assert Tv == teacher_ids.size(1)
                index = mapping.view(B, S, 1, 1, 1).expand(-1, -1, H, W, C)  # [B, N, H, W, C]
                mapped_v = torch.gather(v, dim=1, index=index)  # [B, N, H, W, C]

            elif v.dim() == 4:
                # [B, T, H, W] -> [B, N, H, W]
                Bv, Tv, H, W = v.shape
                assert Tv == teacher_ids.size(1)
                index = mapping.view(B, S, 1, 1).expand(-1, -1, H, W)  # [B, N, H, W]
                mapped_v = torch.gather(v, dim=1, index=index)  # [B, N, H, W]

            else:
                raise ValueError(f"Unexpected dim for {k}: {v.shape}")
            # print('k',k,'mapped_v',mapped_v.shape)
            mapped_out[k] = mapped_v
        return mapped_out

    def _process_teacher_batch_with_rotation(self, batch, teacher_out, image_size, pick=True):
        assert 'point_masks' not in batch
        new_teacher_out = {}
        if pick:
            teacher_out = self._process_teacher_out(batch=batch, teacher_out=teacher_out)

        teacher_out = self._process_rotation(batch=batch, teacher_out=teacher_out, image_size=image_size)
        if self.enable_static_mask:
            teacher_out["static_masks"] = self._build_teacher_static_masks(teacher_out)
        new_t_out = self.make_new_teacher_out(teacher_out=teacher_out)
        new_teacher_out.update(new_t_out)

        return new_teacher_out

    def make_new_teacher_out(self, teacher_out):
        extrinsic = teacher_out['extrinsic'].clone()
        intrinsic = teacher_out['intrinsic'].clone()
        depth_map = teacher_out['depth'].clone()
        assert 'point_masks' not in teacher_out
        depth_conf = teacher_out["depth_conf"].clone()
        point_masks = torch.ones_like(depth_conf, dtype=torch.bool)

        if self.enable_64:
            with torch.autocast('cuda', dtype=torch.float64):
                world_points_from_depth, cam_points, valid_depth = unproject_depth_map_to_point_map_bs(
                    depth_map, extrinsic, intrinsic
                )
        else:
            world_points_from_depth, cam_points, valid_depth = unproject_depth_map_to_point_map_bs(
                depth_map, extrinsic, intrinsic
            )
        if "static_masks" in teacher_out:
            static_masks = teacher_out["static_masks"].clone().bool() & valid_depth

        normalized_extrinsics, _, normalized_world_points, normalized_depths = \
            normalize_camera_extrinsics_and_points_batch(
                extrinsics=extrinsic,
                cam_points=cam_points,
                world_points=world_points_from_depth,
                depths=depth_map.squeeze(-1),
                point_masks=point_masks,
                scale_by_points=False,
            )

        if self.enable_64:
            normalized_extrinsics = normalized_extrinsics.to(torch.float32)
            normalized_world_points = normalized_world_points.to(torch.float32)

        teacher_labels = {
            "extrinsics": normalized_extrinsics,
            "intrinsics": intrinsic,
            "depth": normalized_depths.unsqueeze(-1),
            "point_masks": point_masks,
            "world_points": normalized_world_points,
        }
        if "static_masks" in teacher_out:
            teacher_labels["static_masks"] = static_masks
        return teacher_labels

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.

        Returns:
            A dictionary containing the computed losses.
        """
        y_hat = model(images=batch["images"])
        # save_iter = 1000
        # if self.steps[phase] % save_iter == 0:
        #     self.save_no_label_data = True
        #     self._save_tensor_while_training(phase, t_out=None, s_out=y_hat, batch=batch, save_name='every_kilo')

        # Loss computation
        loss_dict = self.loss(y_hat, batch)

        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def rotate_tensor(self, images, angle):
        k = angle // 90
        return torch.rot90(images, k=k, dims=(3, 4))

    def _process_batch_global_rotation(self, batch):
        ANGLES = [0, 90, 180, 270]
        R_LOOKUP = {
            0: torch.tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=torch.float32),
            90: torch.tensor([
                [0, 1, 0],
                [-1, 0, 0],
                [0, 0, 1]
            ], dtype=torch.float32),
            180: torch.tensor([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ], dtype=torch.float32),
            270: torch.tensor([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ], dtype=torch.float32),
        }
        theta = random.choice(ANGLES)
        images = batch['images']
        B, S, _, _, _ = images.shape
        device = images.device
        batch['images'] = self.rotate_tensor(images, theta)
        R_matrix = R_LOOKUP[theta]
        batch['global_R_matrixs'] = R_matrix.unsqueeze(0).unsqueeze(0).expand(B, S, 3, 3).to(device)
        batch['global_thetas'] = theta
        return batch

    def _step_ssl(self, batch, student_model: nn.Module, teacher_model: nn.Module,
                  phase: str, loss_meters: dict, t_temp=1.):
        assert self.use_ema is True, f'self.use_ema = {self.use_ema}'
        teacher_model.eval()
        t_out = None

        if self.global_rotation and self.ema_use_label not in batch:
            batch = self._process_batch_global_rotation(batch)

        with torch.no_grad():
            image_size = batch['images'].shape[-2:]
            if self.ema_use_label not in batch:
                t_out = teacher_model(images=batch["teacher_images"], frames_chunk_size=4, need_point=False)

                if 'world_points' in t_out:
                    t_out.pop('world_points')
                    t_out.pop('world_points_conf')

                if 'pose_enc_list' in t_out:
                    t_out.pop('pose_enc_list')
                    t_out.pop('images')

                torch.cuda.empty_cache()
                # if not dist.is_initialized() or dist.get_rank() == 0:
                #     if self.save_no_label_data:
                #         self._save_teacher_tensor_while_training(phase, t_out, save_name='teacher_orin')

                assert 'depths' not in batch
                assert 'point_masks' not in batch

                with torch.cuda.amp.autocast(enabled=False):
                    t_out = self._process_teacher_batch_with_rotation(
                        batch=batch,
                        teacher_out=t_out,
                        image_size=image_size,
                    )

        s_out = student_model(images=batch["images"])

        # if not dist.is_initialized() or dist.get_rank() == 0:
        #     save_iter = 1000
        #
        #     if self.save_no_label_data and t_out is not None:
        #         self._save_tensor_while_training(phase, t_out, s_out, batch, save_name='teacher')
        #         self.save_no_label_data = False
        #     if self.steps[phase] % save_iter == 0:
        #         self.save_no_label_data = True
        #         self._save_tensor_while_training(phase, t_out, s_out, batch, save_name='every_kilo')

        if self.ema_use_label not in batch:
            del batch
            batch = {}
        else:
            batch.pop('images')

        torch.cuda.empty_cache()

        with torch.cuda.amp.autocast(enabled=False):
            loss_dict = self.loss(s_out, batch, teacher_out=t_out, t_temp=t_temp)

        log_data = {**s_out, **loss_dict, **batch}
        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])
        self.steps[phase] += 1
        return loss_dict

    def _save_tensor_while_training(self, phase, t_out, s_out, batch, save_name):
        save_dir = f'{self.logging_conf.log_dir}/debug_tensor/'
        os.makedirs(save_dir, exist_ok=True)

        base_name = f"step_at_{self.steps[phase]}_{save_name}_save"  # 
        idx = 0
        while True:
            save_path = os.path.join(save_dir, f"{base_name}_{idx}.pt")
            if not os.path.exists(save_path):
                break
            idx += 1

        def dict_to_cpu(d):
            out = {}
            for k, v in d.items():
                if torch.is_tensor(v):
                    out[k] = v.detach().cpu()
                elif isinstance(v, dict):
                    out[k] = dict_to_cpu(v)
                else:
                    out[k] = v
            return out

        torch.save({
            "t_out": t_out,
            "predictions": dict_to_cpu(s_out),
            "batch": dict_to_cpu(batch),
        }, save_path)

    def _save_teacher_tensor_while_training(self, phase, t_out, save_name):
        save_dir = f'{self.logging_conf.log_dir}/debug_tensor/'
        os.makedirs(save_dir, exist_ok=True)

        base_name = f"step_at_{self.steps[phase]}_{save_name}_save"  # 
        idx = 0
        while True:
            save_path = os.path.join(save_dir, f"{base_name}_{idx}.pt")
            if not os.path.exists(save_path):
                break
            idx += 1

        def dict_to_cpu(d):
            out = {}
            for k, v in d.items():
                if torch.is_tensor(v):
                    out[k] = v.detach().cpu()
                elif isinstance(v, dict):
                    out[k] = dict_to_cpu(v)
                else:
                    out[k] = v
            return out

        torch.save({
            "t_out": t_out,
        }, save_path)

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        if 'depth' in data:
            batch_size = data['depth'].shape[0]
        else:
            batch_size = data['points'].shape[0]

        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.tb_writer.log(f"Values/{phase}/{key}", value, step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""
        if not (
                self.logging_conf.log_visuals
                and (phase in self.logging_conf.log_visual_frequency)
                and self.logging_conf.log_visual_frequency[phase] > 0
                and (step % self.logging_conf.log_visual_frequency[phase] == 0)
                and (self.logging_conf.visuals_keys_to_log is not None)
        ):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase][
                "keys_to_log"
            ]
            assert (
                    len(keys_to_log) > 0
            ), "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase][
                "modality"
            ]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            name = f"Visuals/{phase}"

            visuals_to_log = torchvision.utils.make_grid(
                [
                    torchvision.utils.make_grid(
                        batch[key][0],  # Ensure batch[key][0] is tensor and has at least 3 dimensions
                        nrow=self.logging_conf.visuals_per_batch_to_log,
                    )
                    for key in keys_to_log if key in batch and batch[key][0].dim() >= 3
                ],
                nrow=1,
            ).clamp(-1, 1)

            visuals_to_log = visuals_to_log.cpu()
            if visuals_to_log.dtype == torch.bfloat16:
                visuals_to_log = visuals_to_log.to(torch.float16)
            visuals_to_log = visuals_to_log.numpy()

            self.tb_writer.log_visuals(
                name, visuals_to_log, step, self.logging_conf.video_logging_fps
            )


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]


def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return (
            isinstance(data, Sequence)
            and not isinstance(data, str)
            and len(data) > 0
            and isinstance(data[0], (str, int, float, bool))
    )


def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {
            key: get_chunk_from_data(value, chunk_id, num_chunks)
            for key, value in data.items()
        }
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
