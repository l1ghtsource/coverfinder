import os
import random
import numpy as np
import pandas as pd
import torch
import torchaudio.transforms as T
from collections import deque
from typing import List, Literal, Optional, Tuple, TypedDict
from torch.utils.data import Dataset


class ValDict(TypedDict):
    anchor_id: int
    f_t: torch.Tensor
    f_c: torch.Tensor


class BatchDict(TypedDict):
    anchor_id: int
    anchor: torch.Tensor
    anchor_label: torch.Tensor
    positive_id: int
    positive: torch.Tensor
    negative_id: int
    negative: torch.Tensor


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


def adjust_contrast(cqt, contrast_factor):
    mean = cqt.mean()
    return torch.clamp((cqt - mean) * contrast_factor + mean, 0, 1)


class CoverDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        file_ext: str,
        dataset_path: str,
        data_split: Literal['train', 'val', 'test'],
        debug: bool,
        max_len: int,
        cache_size: int = 1000,
        num_candidates: int = 20,
        augment: bool = True
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.file_ext = file_ext
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.debug = debug
        self.max_len = max_len
        self.cache_size = cache_size
        self.num_candidates = num_candidates
        self._load_data()

        self.embedding_cache = {}
        self.cache_queue = deque(maxlen=cache_size)

        self.mining_stats = {
            'semi_hard_count': 0,
            'hard_count': 0,
            'random_count': 0
        }

        self.augment = augment
        self.augmentation_pipeline = Compose([
            T.FrequencyMasking(freq_mask_param=15),
            T.TimeMasking(time_mask_param=35),
            T.Vol(0.5)
        ])
        self.add_noise = lambda x: x + 0.005 * torch.randn_like(x)

    def update_embedding_cache(self, track_id: int, embedding: torch.Tensor):
        self.embedding_cache[track_id] = embedding.detach()
        self.cache_queue.append(track_id)
        if len(self.cache_queue) >= self.cache_size:
            old_id = self.cache_queue[0]
            if old_id in self.embedding_cache:
                del self.embedding_cache[old_id]

    def _select_negative(
            self, anchor_id: int, anchor_emb: torch.Tensor, positive_emb: torch.Tensor, versions: List[int]) -> int:
        if len(self.embedding_cache) < self.num_candidates:
            return self._random_negative(versions)

        candidates = random.sample(
            list(self.embedding_cache.keys()),
            min(self.num_candidates, len(self.embedding_cache)))
        candidates = [c for c in candidates if c not in versions]

        if not candidates:
            return self._random_negative(versions)

        anchor_emb = anchor_emb.unsqueeze(0)
        positive_emb = positive_emb.unsqueeze(0)

        neg_embeddings = torch.stack([self.embedding_cache[c] for c in candidates])
        pos_dist = torch.pairwise_distance(anchor_emb, positive_emb)
        neg_dists = torch.pairwise_distance(anchor_emb, neg_embeddings)

        semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + self.current_margin)
        hard_mask = neg_dists < pos_dist

        if semi_hard_mask.any():
            self.mining_stats['semi_hard_count'] += 1
            semi_hard_indices = torch.where(semi_hard_mask)[0]
            selected_idx = semi_hard_indices[torch.randint(0, len(semi_hard_indices), (1,))]
            return candidates[selected_idx]
        elif hard_mask.any():
            self.mining_stats['hard_count'] += 1
            hard_indices = torch.where(hard_mask)[0]
            selected_idx = hard_indices[torch.randint(0, len(hard_indices), (1,))]
            return candidates[selected_idx]
        else:
            self.mining_stats['random_count'] += 1
            return candidates[torch.randint(0, len(candidates), (1,))]

    def _random_negative(self, versions: List[int]) -> int:
        while True:
            neg_id = random.choice(self.track_ids)
            if neg_id not in versions:
                return neg_id

    def _triplet_sampling(self, track_id: int, clique_id: int, anchor_emb: Optional[torch.Tensor] = None) -> Tuple[int, int]:
        versions = self.versions.loc[clique_id, 'versions']
        pos_list = np.setdiff1d(versions, track_id)
        pos_id = np.random.choice(pos_list, 1)[0]

        if anchor_emb is not None and pos_id in self.embedding_cache:
            pos_emb = self.embedding_cache[pos_id]
            neg_id = self._select_negative(track_id, anchor_emb, pos_emb, versions)
        else:
            neg_id = self._random_negative(versions)

        return (pos_id, neg_id)

    def get_mining_stats(self):
        total = sum(self.mining_stats.values())
        if total == 0:
            return self.mining_stats
        return {k: v / total for k, v in self.mining_stats.items()}

    def _make_file_path(self, track_id, file_ext):
        a = track_id % 10
        b = track_id // 10 % 10
        c = track_id // 100 % 10
        return os.path.join(str(c), str(b), str(a), f'{track_id}.{file_ext}')

    def __len__(self) -> int:
        return len(self.track_ids)

    def __getitem__(self, index: int) -> BatchDict:
        track_id = self.track_ids[index]
        anchor_cqt = self._load_cqt(track_id)

        if self.data_split == 'train':
            clique_id = self.version2clique.loc[track_id, 'clique']
            anchor_emb = self.embedding_cache.get(track_id)
            pos_id, neg_id = self._triplet_sampling(track_id, clique_id, anchor_emb)
            positive_cqt = self._load_cqt(pos_id)
            negative_cqt = self._load_cqt(neg_id)
        else:
            clique_id = -1
            pos_id = torch.empty(0)
            positive_cqt = torch.empty(0)
            neg_id = torch.empty(0)
            negative_cqt = torch.empty(0)

        return dict(
            anchor_id=track_id,
            anchor=anchor_cqt,
            anchor_label=torch.tensor(clique_id, dtype=torch.float),
            positive_id=pos_id,
            positive=positive_cqt,
            negative_id=neg_id,
            negative=negative_cqt,
        )

    def _load_data(self) -> None:
        if self.data_split in ['train', 'val']:
            cliques_subset = np.load(os.path.join(self.data_path, '{}_cliques.npy'.format(self.data_split)))
            self.versions = pd.read_csv(
                os.path.join(self.data_path, 'cliques2versions.tsv'), sep='\t', converters={'versions': eval}
            )
            self.versions = self.versions[self.versions['clique'].isin(set(cliques_subset))]
            mapping = {}
            for k, clique in enumerate(sorted(cliques_subset)):
                mapping[clique] = k
            self.versions['clique'] = self.versions['clique'].map(lambda x: mapping[x])
            self.versions.set_index('clique', inplace=True)
            self.version2clique = pd.DataFrame(
                [{'version': version, 'clique': clique} for clique, row in self.versions.iterrows()
                 for version in row['versions']]).set_index('version')
            self.track_ids = self.version2clique.index.to_list()
        else:
            self.track_ids = np.load(os.path.join(self.data_path, '{}_ids.npy'.format(self.data_split)))

    def _load_cqt(self, track_id: str) -> torch.Tensor:
        filename = os.path.join(self.dataset_path, self._make_file_path(track_id, self.file_ext))
        cqt_spectrogram = np.load(filename)
        cqt_spectrogram = torch.from_numpy(cqt_spectrogram)

        if self.augment and self.data_split == 'train':
            cqt_spectrogram = self.augmentation_pipeline(cqt_spectrogram)
            if random.random() < 0.3:
                cqt_spectrogram = self.add_noise(cqt_spectrogram)

        return cqt_spectrogram
