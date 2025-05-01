# pneumonia_detection_torch.py
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# GPU Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom Dataset class
class PneumoniaDataset(Dataset):
    def __init__(self, image_dir, label, transform=None, max_samples=None):
        self.image_dir = image_dir
        self.label = label
        self.transform = transform
        self.image_names = os.listdir(image_dir)[:max_samples]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, self.label

# CNN Model
class PneumoniaCNN(nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Training Function
def train_model():
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    pneumonia_dataset = PneumoniaDataset("dataset/Bacterial_Pnemonia", 1, transform)
    normal_dataset = PneumoniaDataset("dataset/Normal", 0, transform)

    full_dataset = torch.utils.data.ConcatDataset([pneumonia_dataset, normal_dataset])

    train_size = int(0.75 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = PneumoniaCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0
    history = {"train_acc": [], "train_loss": []}

    for epoch in range(20):
        model.train()
        correct = 0
        total = 0
        running_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            running_loss += loss.item()

        acc = correct / total
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Accuracy: {acc:.4f}, Loss: {avg_loss:.4f}")
        history["train_acc"].append(acc)
        history["train_loss"].append(avg_loss)

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "pneumonia_model.pth")

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.legend()
    plt.savefig("training_history.png")
    plt.close()

# Inference Function
def predict_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    model = PneumoniaCNN().to(device)
    model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
    model.eval()

    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image).item()
        print(f"Pneumonia probability: {pred:.2%}")

if __name__ == "__main__":
    # train_model()
    predict_image("dataset//Normal//Normal_1.JPG")
