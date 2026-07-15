import argparse
import os
import shutil
import sys
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf


TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train Wat3R with a YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default="wat3r_training",
        help="Config name, with or without the .yaml suffix.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.chdir(TRAINING_DIR)

    config_name = args.config[:-5] if args.config.endswith(".yaml") else args.config
    config_dir = Path("config")

    with initialize(version_base=None, config_path=str(config_dir)):
        cfg = compose(config_name=config_name)

    save_root = Path("debug/result") if args.debug else Path("result")
    if args.debug:
        cfg.log_save_dir = os.path.join("debug", cfg.log_save_dir)
        cfg.model.ema.start_unlabel = 10
        cfg.model.ema.start_ema = 10
        cfg.num_workers = 1
        cfg.data.train.num_workers = 1
        cfg.data.val.num_workers = 1

    config_copy_dir = save_root / cfg.log_save_dir
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        save_path = config_copy_dir / "config_copy.yaml"
        if save_path.exists() and not args.debug:
            raise RuntimeError(f"Existing training config would be overwritten: {save_path}")
        config_copy_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, save_path)
        shutil.copy(config_dir / f"{config_name}.yaml", config_copy_dir)

    trainer = Trainer(**cfg, debug=args.debug)
    trainer.run()


if __name__ == "__main__":
    main()
