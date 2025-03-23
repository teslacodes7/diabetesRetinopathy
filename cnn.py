import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# AMP for mixed precision training
from torch.amp import autocast, GradScaler

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()

# ======== STEP 1: Load Dataset from CSV ========
class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0] + '.jpeg')
        image = Image.open(img_name).convert('RGB')
        label = int(self.labels.iloc[idx, 1])  # Convert label to integer
        
        if self.transform:
            image = self.transform(image)

        return image, label

# ======== STEP 2: Define Data Transformations ========
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======== STEP 3: Load Data ========
data_path = "../../../Downloads/train"  # Change this to your image directory
csv_path = "trainLabels.csv"  # Change this to your CSV file

dataset = DRDataset(csv_path, data_path, transform)

# ======== STEP 4: Handle Class Imbalance ========
labels = dataset.labels.iloc[:, 1].tolist()  # Extract labels
class_counts = Counter(labels)
total_samples = len(dataset)
class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

sample_weights = [class_weights[label] for label in labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# ======== STEP 5: Define CNN Model ========
class CNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Compute the flattened dimension dynamically
        self.flatten_dim = 128 * (128 // 8) * (128 // 8)
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model
model = CNNModel(num_classes=5).to(device)

# ======== STEP 6: Define Loss and Optimizer ========
class_weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_counts))], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# ======== STEP 7: Create DataLoader with WeightedRandomSampler ========
dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

# ======== STEP 8: Train the Model with Additional Metrics ========
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        for images, labels in tqdm(dataloader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

        # Compute additional metrics
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        # Print results
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

        # Update learning rate scheduler
        scheduler.step(epoch_loss)

    print("Training Complete")

# Run Training
if __name__ == "__main__":
    print("Starting training...")
    train_model(model, dataloader, criterion, optimizer, num_epochs=40)

    print("Saving model...")
    torch.save(model.state_dict(), "dr_cnn.pth")
    print("CNN Model saved successfully!")  
