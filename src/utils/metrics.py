import torch
import torch.nn.functional as F


def calculate_ndcg(
    similarity: torch.Tensor,
    labels: torch.Tensor,
    k: int = 100
) -> float:
    device = similarity.device

    _, indices = similarity.topk(k + 1, dim=1)
    indices = indices[:, 1:]

    relevant = (labels.unsqueeze(1) == labels[indices]).float()

    position_discount = 1.0 / torch.log2(torch.arange(2, k + 2, dtype=torch.float, device=device))
    dcg = (relevant * position_discount.unsqueeze(0)).sum(dim=1)

    ideal_relevant = relevant.sort(descending=True)[0]
    idcg = (ideal_relevant * position_discount.unsqueeze(0)).sum(dim=1)

    ndcg = (dcg / (idcg + 1e-8)).mean().item()
    return ndcg
