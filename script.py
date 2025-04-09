import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd
import warnings
from PIL import Image
import h5py
from torch.utils.data import Dataset, DataLoader
from train import train

from model_architecture.cnn import CNNModel

warnings.filterwarnings('ignore')
print("All modules have been imported (PyTorch version)")

trainLabels = pd.read_csv("./trainLabels.csv")

# Load HDF5 labels
with h5py.File("./datasets/dataset.h5", "r") as f:
    train_labels = f["train"]["labels"][:]
    val_labels = f["val"]["labels"][:]

# Define transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
print("Data augmentation and preprocessing pipeline updated for PyTorch!")

class HDF5Dataset(Dataset):
    def __init__(self, h5_path, split="train", transform=None):
        super().__init__()
        self.h5_path = h5_path
        self.split = split
        self.transform = transform
        self.h5_file = None  # Will be lazily initialized per worker

    def _init_file(self):
        self.h5_file = h5py.File(self.h5_path, "r")
        self.images = self.h5_file[self.split]["images"]
        self.labels = self.h5_file[self.split]["labels"]
        self.names = self.h5_file[self.split]["names"]

    def __len__(self):
        if self.h5_file is None:
            with h5py.File(self.h5_path, "r") as f:
                return len(f[self.split]["labels"])
        return len(self.labels)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self._init_file()

        image = self.images[idx]  # shape (224, 224, 3)
        label = self.labels[idx]

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        return image, int(label)

    def close(self):
        if self.h5_file:
            self.h5_file.close()
            self.h5_file = None

# Create dataset instances
train_dataset = HDF5Dataset("./datasets/dataset.h5", split="train", transform=train_transforms)
val_dataset = HDF5Dataset("./datasets/dataset.h5", split="val",   transform=test_transforms)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print("PyTorch DataLoaders are ready!")

# Simple CNN Model

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} | Num GPUs Available: {torch.cuda.device_count()}")

model = CNNModel().to(device)
print(f"Model is on: {next(model.parameters()).device}")

# ----------- MAIN -----------
if __name__ == "__main__":
    train(model= model,num_epochs=15,train_loader=train_loader, val_loader=val_loader)
