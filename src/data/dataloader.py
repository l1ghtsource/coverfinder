from torch.utils.data import DataLoader
from typing import Dict, Literal
from .dataset import CoverDataset


def cover_dataloader(
    data_path: str,
    file_ext: str,
    dataset_path: str,
    data_split: Literal['train', 'val', 'test'],
    debug: bool,
    max_len: int,
    batch_size: int,
    **config: Dict,
) -> DataLoader:
    return DataLoader(
        CoverDataset(data_path, file_ext, dataset_path, data_split, debug, max_len=max_len),
        batch_size=batch_size if max_len > 0 else 1,
        num_workers=config['num_workers'],
        shuffle=config['shuffle'],
        drop_last=config['drop_last'],
    )
