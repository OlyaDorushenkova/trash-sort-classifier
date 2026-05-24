from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl


class TrashDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/raw",
        batch_size: int = 32,
        num_workers: int = 0,
        image_size: int = 224,
        seed: int = 42,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.seed = seed

        self.train_transform = None
        self.eval_transform = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Split dataset into train/val/test.
        """

        data_path = Path(self.data_dir)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {data_path}"
            )

        self.train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        full_dataset = datasets.ImageFolder(
            root=self.data_dir
        )

        indices = list(range(len(full_dataset)))

        train_idx, temp_idx = train_test_split(
            indices,
            test_size=0.3,
            random_state=self.seed,
            stratify=full_dataset.targets,
        )

        temp_targets = [
            full_dataset.targets[i]
            for i in temp_idx
        ]

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=self.seed,
            stratify=temp_targets,
        )

        # datasets with transforms
        train_base = datasets.ImageFolder(
            self.data_dir,
            transform=self.train_transform,
        )

        eval_base = datasets.ImageFolder(
            self.data_dir,
            transform=self.eval_transform,
        )

        self.train_dataset = Subset(
            train_base,
            train_idx,
        )

        self.val_dataset = Subset(
            eval_base,
            val_idx,
        )

        self.test_dataset = Subset(
            eval_base,
            test_idx,
        )

        self.num_classes = len(full_dataset.classes)
        self.classes = full_dataset.classes

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )