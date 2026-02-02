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

        # --- COMMENT: MNIST SPECIFIC ---
        # Later, instead of loading MNIST images, we will load patches from WSIs.
        self.mnist = datasets.MNIST(
            root=self.root,
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        #   We pre-calculate which images are '9's (targets) and which are not.
        self.target_indices = np.where(self.mnist.targets == target_number)[0]
        self.non_target_indices = np.where(self.mnist.targets != target_number)[0]

        # We generate the bags immediately upon initialization
        self.bags_list, self.labels_list = self._create_bags()

    def _create_bags(self):
        bags_list = []
        labels_list = []
        for _ in range(self.num_bag):
            # 1. Random bag size (simulates different tissue sizes per slide)
            bag_length = int(
                self.r.normal(self.mean_bag_length, self.var_bag_length, 1)
            )
            if bag_length < 1:
                bag_length = 1
            # 2. Decide label: Is this bag Positive (contains tumor/9) or Negative?
            label = 1 if self.r.rand() > 0.5 else 0
            indices = []

            if label == 1:
                # POSITIVE BAG CONCEPT:
                # Must contain at least one positive instance
                indices.append(self.r.choice(self.target_indices))
                indices.extend(self.r.choice(self.non_target_indices, bag_length - 1))
            else:
                # NEGATIVE BAG CONCEPT:
                # Contains ONLY noise/background (no tumor)
                indices.extend(self.r.choice(self.non_target_indices, bag_length))

            self.r.shuffle(indices)

            # --- DATA FORMAT ---
            # Current Output: Tensor of shape (Bag_Length, 1, 28, 28) -> Raw Images
            # Future WSI Output: Tensor of shape (Bag_Length, 1024) -> Pre-computed Features
            bag_images = torch.stack([self.mnist[i][0] for i in indices])
            bags_list.append(bag_images)
            labels_list.append(label)
        return bags_list, labels_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        # Returns: (Bag of instances, Bag Label)
        return self.bags_list[index], torch.tensor(
            [self.labels_list[index]], dtype=torch.float32
        )
