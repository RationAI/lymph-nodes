import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from lymph_nodes.data.datasets import LymphNodesPredict


URIS = [
    "mlflow-artifacts:/68/e6e0d31541004d0fa972f3e04e387f68/artifacts/DAB Lymph Nodes with Epytelium - train"
]


def calculate_mean_std() -> None:
    dataset = LymphNodesPredict(URIS)
    sampler = RandomSampler(dataset, num_samples=10000)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler)

    means = []
    stds = []

    for x, _ in tqdm(dataloader):
        x = x.float()
        means.append(x.mean((0, 2, 3)))
        stds.append(x.std((0, 2, 3)))

    mean = torch.stack(means).mean(0)
    std = torch.stack(stds).mean(0)

    print(f"Mean: {mean}")
    print(f"Std: {std}")
