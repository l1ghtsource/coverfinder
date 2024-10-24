import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

from .models.conformer import Conformer, ConformerV2
from .models.transformer import MusicTransformer
from .data.dataloader import cover_dataloader


def _inference(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    output_path: str
) -> None:
    model.eval()
    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Inference'):
            anchor = batch['anchor'].to(device)
            track_ids = batch['anchor_id']

            embeddings = model(anchor)
            all_embeddings.append(embeddings)
            all_ids.extend(track_ids)

    embeddings = torch.cat(all_embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

    similarity = torch.mm(normalized_embeddings, normalized_embeddings.t())

    _, indices = similarity.topk(101, dim=1)
    indices = indices[:, 1:]

    with open(output_path, 'w') as f:
        for i, track_id in enumerate(all_ids):
            similar_tracks = [str(all_ids[idx].item()) for idx in indices[i].tolist()]
            f.write(f"{track_id.item()} {' '.join(similar_tracks)}\n")


def inference(config: Dict, checkpoint_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if config['model']['name'] == 'CONFORMER':
        model = Conformer(**config['model']).to(device)
    elif config['model']['name'] == 'CONFORMERV2':
        model = ConformerV2(**config['model']).to(device)
    else:
        model = MusicTransformer(**config['model']).to(device)

    print('\nLoading model for inference...')
    model.load_state_dict(torch.load(checkpoint_path))

    test_loader = cover_dataloader(
        data_path=config['data']['base_path'],
        file_ext='npy',
        dataset_path=config['data']['test_path'],
        data_split='test',
        max_len=config['training']['max_len'],
        batch_size=config['training']['batch_size'],
        **config['dataloader_val']
    )

    _inference(model, test_loader, device, 'submission.csv')
    print('\nInference completed. Submission file created.')
