import torch
import torch.nn as nn
class CNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Dynamic reshape
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize model
model = CNNModel(num_classes=5).to(device)
# Load the model


# Load saved model weights
model.load_state_dict(torch.load("dr_cnn.pth", map_location=device))  # Change "model.pth" to your actual file name
model.eval()  # Set model to evaluation mode


import torchvision.transforms as transforms
from PIL import Image

# Load and preprocess an image
image_path =  "../../../Downloads/train/217_left.jpeg"  # Change this to your test image
image = Image.open(image_path)

# Define the same transformations as during training
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Adjust size based on your model
    transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

# Make a prediction
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

print(f"Predicted class: {predicted_class}")
