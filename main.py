# main.py
import io
import cv2
import torch
import uvicorn
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
from torch import nn

# FastAPI app
app = FastAPI(title="Pneumonia Detection API")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor()
])

# Define CNN model (same architecture as before)
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

# Load model once on startup
model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
model.eval()

@app.get("/")
def root():
    return {"message": "Pneumonia Detection API is running."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(image).item()
            return {
                "pneumonia_probability": round(pred, 4),
                "prediction": "Pneumonia" if pred > 0.5 else "Normal"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run: uvicorn main:app --reload
