from pathlib import Path


DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"


def default_dataset_path(dataset_name):
    roots = {
        'squid': DATASETS_DIR / 'SQUID',
        'flsea_stereo': DATASETS_DIR / 'flsea_stereo',
        'flsea_vi': DATASETS_DIR / 'FLSea_VI',
        'seathru': DATASETS_DIR / 'seathru',
        'seathru_full': DATASETS_DIR / 'seathru',
        'flsea_stereo_full': DATASETS_DIR / 'flsea_stereo',
    }
    return roots[dataset_name]


def get_data_loader(dataset_name, dataset_path=None, skip=0):
    dataset_path = Path(dataset_path) if dataset_path else default_dataset_path(dataset_name)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    if dataset_name == 'squid':
        from .datasets.squid import UnderwaterDepthDataset
    elif dataset_name == 'flsea_stereo':
        from .datasets.flsea_stereo import UnderwaterDepthDataset
    elif dataset_name == 'flsea_vi':
        from .datasets.flsea_vi import UnderwaterDepthDataset
    elif dataset_name == 'seathru':
        from .datasets.seathru import UnderwaterDepthDataset
    elif dataset_name == 'seathru_full':
        from .datasets.seathru_all import UnderwaterDepthDataset
    elif dataset_name == 'flsea_stereo_full':
        from .datasets.flsea_stereo_full import UnderwaterDepthDataset
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset_name.endswith('_full'):
        dataset = UnderwaterDepthDataset(gt_root=str(dataset_path), skip=skip)
    else:
        dataset = UnderwaterDepthDataset(gt_root=str(dataset_path))
    return dataset.create_dataloader()
