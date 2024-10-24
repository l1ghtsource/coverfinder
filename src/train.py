import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from typing import Dict

from .models.modules import TripletLoss
from .models.conformer import Conformer, ConformerV2
from .models.transformer import MusicTransformer
from .data.dataloader import cover_dataloader
from .utils.metrics import calculate_ndcg


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc='Training'):
        anchor = batch['anchor'].to(device)
        positive = batch['positive'].to(device)
        negative = batch['negative'].to(device)

        optimizer.zero_grad()

        anchor_emb = model(anchor)
        positive_emb = model(positive)
        negative_emb = model(negative)

        for idx, track_id in enumerate(batch['anchor_id']):
            dataloader.dataset.update_embedding_cache(track_id, anchor_emb[idx])

        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        mining_stats = dataloader.dataset.get_mining_stats()
        wandb.log({
            'train_loss': loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'current_margin': criterion.current_margin,
            'semi_hard_ratio': mining_stats['semi_hard_count'],
            'hard_ratio': mining_stats['hard_count'],
            'random_ratio': mining_stats['random_count']
        })

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch in tqdm(dataloader, desc='Validation'):
        anchor = batch['anchor'].to(device)
        labels = batch['anchor_label'].to(device)

        embeddings = model(anchor)
        all_embeddings.append(embeddings)
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings)
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    labels = torch.cat(all_labels)

    similarity = torch.mm(normalized_embeddings, normalized_embeddings.t())

    ndcg = calculate_ndcg(similarity, labels, k=100)
    return ndcg


def train(config: Dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    wandb.init(project=config['wandb']['project'], config=config)

    if config['model']['name'] == 'CONFORMER':
        model = Conformer(**config['model']).to(device)
    elif config['model']['name'] == 'CONFORMERV2':
        model = ConformerV2(**config['model']).to(device)
    else:
        model = MusicTransformer(**config['model']).to(device)

    train_loader = cover_dataloader(
        data_path=config['data']['base_path'],
        file_ext='npy',
        dataset_path=config['data']['train_path'],
        data_split='train',
        max_len=config['training']['max_len'],
        batch_size=config['training']['batch_size'],
        **config['dataloader']
    )

    val_size = int(0.1 * len(train_loader.dataset))
    train_size = len(train_loader.dataset) - val_size

    _, val_dataset = random_split(train_loader.dataset, [train_size, val_size])

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        **config['dataloader_val']
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # scheduler = CosineAnnealingLR(
    #     optimizer,
    #     T_max=config['training']['num_epochs'],
    #     eta_min=config['training']['learning_rate'] * 0.01
    # )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=config['training']['num_epochs']
    )

    criterion = TripletLoss(
        margin_min=config['loss']['margin_min'],
        margin_max=config['loss']['margin_max'],
        adjust_margin=config['loss']['adjust_margin'],
        hard_negative_ratio=config['loss']['hard_negative_ratio']
    )

    best_ndcg = 0
    for epoch in range(config['training']['num_epochs']):
        print(f'\nEpoch {epoch + 1}/{config['training']['num_epochs']}')

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f'Training Loss: {train_loss:.4f}')

        scheduler.step()

        val_ndcg = validate(model, val_loader, device)
        print(f'Validation NDCG@100: {val_ndcg:.4f}')
        wandb.log({'val_ndcg': val_ndcg})

        if val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            print(f'New best NDCG! Saving model...')
            torch.save(model.state_dict(), 'best_model.pth')

    wandb.finish()
    print(f'\nTraining completed. Best validation NDCG: {best_ndcg:.4f}')
