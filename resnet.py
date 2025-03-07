import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ======== STEP 1: Load Dataset from CSV ========
class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0] + ".jpeg")
        image = Image.open(img_name).convert('RGB')
        label = int(self.labels.iloc[idx, 1])  # Convert label to integer
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ======== STEP 2: Define Data Transformations ========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ======== STEP 3: Load Data ========
data_path = "./train"  # Change this to your image directory
csv_path = "trainLabels.csv"  # Change this to your CSV file

# Create Dataset & DataLoader
dataset = DRDataset(csv_path, data_path, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ======== STEP 4: Load Pretrained ResNet Model ========
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 5 classes for DR stages
model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ======== STEP 5: Define Loss and Optimizer ========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ======== STEP 6: Train the Model ========
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    print("Training Complete")

# Run Training
train_model(model, dataloader, criterion, optimizer, num_epochs=10)

# ======== STEP 7: Save the Model ========
torch.save(model.state_dict(), "dr_resnet50.pth")
print("Model saved successfully!")
