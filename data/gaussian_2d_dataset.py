import numpy as np
import torch
from torch.utils.data import Dataset

class Gaussian2DClassificationDataset(Dataset):
    def __init__(self, split="train", n_classes=4, points_per_class=1000, center_box=(-1, 1),
                 std=0.25, split_ratio=0.8, seed=42):
        """
        PyTorch-style Dataset for 2D Gaussian blob classification.

        Args:
            split (str): 'train' or 'test' set.
            n_classes (int): Number of distinct classes.
            points_per_class (int): Number of samples per class.
            center_box (tuple): Range for random class centers.
            std (float): Standard deviation of each Gaussian cluster.
            split_ratio (float): Proportion of training data.
            seed (int): Random seed.
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"
        self.split = split

        np.random.seed(seed)
        centers = np.random.uniform(center_box[0], center_box[1], size=(n_classes, 2))
        X = []
        y = []

        for class_id, center in enumerate(centers):
            points = center + std * np.random.randn(points_per_class, 2)
            X.append(points)
            y.append(np.full(points_per_class, class_id))

        X = np.vstack(X)
        y = np.concatenate(y)

        # Shuffle and split
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        split_index = int(split_ratio * len(X))
        if split == "train":
            self.X = torch.tensor(X[:split_index], dtype=torch.float32)
            self.y = torch.tensor(y[:split_index], dtype=torch.long)
        else:
            self.X = torch.tensor(X[split_index:], dtype=torch.float32)
            self.y = torch.tensor(y[split_index:], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
