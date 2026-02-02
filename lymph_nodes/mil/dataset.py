# will be removed in future versions
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MnistBags(Dataset):
    def __init__(
        self,
        root,
        target_number,
        mean_bag_length,
        var_bag_length,
        num_bag,
        train=True,
        seed=42,
    ):
        self.root = root
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train
        self.r = np.random.RandomState(seed)

        self.mnist = datasets.MNIST(
            root=self.root,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )

        self.target_indices = np.where(self.mnist.targets == target_number)[0]
        self.non_target_indices = np.where(self.mnist.targets != target_number)[0]
        self.bags_list, self.labels_list = self._create_bags()

    def _create_bags(self):
        bags_list = []
        labels_list = []
        for _ in range(self.num_bag):
            bag_length = int(
                self.r.normal(self.mean_bag_length, self.var_bag_length, 1)
            )
            if bag_length < 1:
                bag_length = 1

            label = 1 if self.r.rand() > 0.5 else 0
            indices = []

            if label == 1:
                indices.append(self.r.choice(self.target_indices))
                indices.extend(self.r.choice(self.non_target_indices, bag_length - 1))
            else:
                indices.extend(self.r.choice(self.non_target_indices, bag_length))

            self.r.shuffle(indices)
            bag_images = torch.stack([self.mnist[i][0] for i in indices])
            bags_list.append(bag_images)
            labels_list.append(label)
        return bags_list, labels_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        return self.bags_list[index], torch.tensor(
            [self.labels_list[index]], dtype=torch.float32
        )
