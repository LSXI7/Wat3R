# train_utils/wandb_logger.py
import atexit
import logging
import os
import uuid
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import wandb
import dataclasses

try:
    from omegaconf import OmegaConf
    _HAVE_OMEGA = True
except Exception:
    _HAVE_OMEGA = False

from .distributed import get_machine_local_and_dist_rank


def _to_number(x: Any):
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().item()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return float(x)
    except Exception:
        return x


def _to_uint8_image_hwc(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().numpy()
    if x.ndim == 3 and x.shape[0] in (1, 3, 4):  # CHW -> HWC
        x = np.transpose(x, (1, 2, 0))
    x = np.asarray(x)
    if x.dtype == np.uint8:
        return x
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    if x_max <= 1.0 and x_min >= -1.0:
        x = (x + 1.0) / 2.0 if x_min < 0.0 else x
        x = np.clip(x, 0.0, 1.0) * 255.0
    else:
        x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def _to_video_thwc_uint8(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().contiguous().numpy()
    arr = np.asarray(x)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D video, got {arr.shape} (ndim={arr.ndim})")
    # (T,C,H,W) -> (T,H,W,C)
    if arr.shape[1] in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.dtype != np.uint8:
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
        if vmax <= 1.0 and vmin >= -1.0:
            arr = (arr + 1.0) / 2.0 if vmin < 0.0 else arr
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    return arr


def _is_primitive(x):
    return isinstance(x, (str, int, float, bool, type(None)))


def _jsonable(obj: Any, _depth=0):
    """Convert nested objects to W&B-safe Python values."""
    if _depth > 50:
        return str(obj)

    if _is_primitive(obj):
        return obj

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (torch.device, torch.dtype)):
        return str(obj)

    # numpy
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        try:
            return f"ndarray(shape={obj.shape}, dtype={obj.dtype})"
        except Exception:
            return "ndarray"

    # dataclass
    if dataclasses.is_dataclass(obj):
        try:
            asd = dataclasses.asdict(obj)
            return {k: _jsonable(v, _depth + 1) for k, v in asd.items()}
        except Exception:
            return str(obj)

    # OmegaConf
    if _HAVE_OMEGA and isinstance(obj, (OmegaConf.__class__,)):
        pass
    if _HAVE_OMEGA and ("omegaconf" in str(type(obj))):
        try:
            plain = OmegaConf.to_container(obj, resolve=True, enum_to_str=True)
            return _jsonable(plain, _depth + 1)
        except Exception:
            return str(obj)

    # Mapping
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            try:
                key = str(k)
            except Exception:
                key = repr(k)
            out[key] = _jsonable(v, _depth + 1)
        return out

    # Sequence
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v, _depth + 1) for v in obj]

    try:
        return str(obj)
    except Exception:
        return repr(obj)


class WandbLogger:
    """W&B adapter with the same interface used by the TensorBoard logger."""

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: Optional[str] = None,
        summary_writer_method: Any = None,
        project: str = "my-project",
        name: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        tags: Optional[list] = None,
        mode: str = "online",                    # "online" | "offline" | "disabled"
        config: Optional[Dict[str, Any]] = None,
        debug:bool = False,
        **kwargs: Any,
    ) -> None:
        self._path = path
        _, self._rank = get_machine_local_and_dist_rank()
        self._run = None
        self._enabled = (mode != "disabled") and (self._rank == 0)
        if debug:
            mode = "offline"

        if self._enabled:
            os.makedirs(path, exist_ok=True)
            os.environ.setdefault("WANDB_SILENT", "true")
            for c in "/\\#?%:":
                project = project.replace(c, "_")

            self._run = wandb.init(
                project=project,
                name=name or str(uuid.uuid4()),
                entity=entity,
                group=group,
                tags=tags,
                dir=path,
                mode=mode,
                reinit=False,
            )
            logging.info(f"W&B run initialized at dir: {path} (mode={mode})")

            if config is not None:
                try:
                    safe_cfg = _jsonable(config)
                    wandb.config.update(safe_cfg, allow_val_change=True)
                except Exception as e:
                    logging.warning(f"WandbLogger: failed to update config safely: {e}")
        else:
            logging.debug(f"W&B disabled on rank {self._rank} (mode={mode}).")

        atexit.register(self.close)

    @property
    def writer(self):
        return self._run

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        pass

    def close(self) -> None:
        if self._run is not None:
            try:
                wandb.finish()
            except Exception:
                pass
            finally:
                self._run = None

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        if not self._enabled or self._run is None:
            return
        data = {k: _to_number(v) for k, v in payload.items()}
        data["global_step"] = step
        wandb.log(data, step=step)

    def log(self, name: str, data: Any, step: int) -> None:
        if not self._enabled or self._run is None:
            return
        wandb.log({name: _to_number(data)}, step=step)

    def log_visuals(
        self,
        name: str,
        data: Union[torch.Tensor, np.ndarray, Any],
        step: int,
        fps: int = 4,
    ) -> None:
        if not self._enabled or self._run is None:
            return
        if isinstance(data, (torch.Tensor, np.ndarray)):
            if getattr(data, "ndim", 0) == 3:
                img = _to_uint8_image_hwc(data)
                wandb.log({name: wandb.Image(img)}, step=step)
                return
            if getattr(data, "ndim", 0) in (4, 5):
                try:
                    vid = _to_video_thwc_uint8(data)
                    wandb.log({name: wandb.Video(vid, fps=fps, format="mp4")}, step=step)
                    return
                except Exception as e:
                    logging.warning(f"WandbLogger: video convert failed: {e}; fallback to first frame.")
                    try:
                        first = data[0] if data.ndim == 4 else data[0, 0]
                        img = _to_uint8_image_hwc(first)
                        wandb.log({name: wandb.Image(img)}, step=step)
                        return
                    except Exception:
                        pass
        try:
            wandb.log({name: data}, step=step)
        except Exception:
            logging.debug("WandbLogger: unsupported visual payload; skipped.")
