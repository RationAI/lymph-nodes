import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lymph_nodes.data.datasets import LymphNodesPredict


URIS = [
    "mlflow-artifacts:/68/b817e33dfc6e4f16a68590a42f6d9e6f/artifacts/DAB Lymph Nodes with Epytelium - train"
]


def main() -> None:
    dataset = LymphNodesPredict(URIS)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=8)

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


if __name__ == "__main__":
    main()
