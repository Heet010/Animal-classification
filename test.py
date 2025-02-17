import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image
from model import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(model, image_path, checkpoint_path=r"C:\Users\bhalani\Animal_classification\Animal-classification\checkpoints\best_model_epoch_20.pth"):
    
    print(f"Loading model from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    class_names = ['Cat', 'Dog']

    # Image preprocessing for test image as done in training and validation
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # This will load and preprocess the image
    image = Image.open(image_path)
    image_tensor = preprocess(image).unsqueeze(0).to(device)  


    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)  
        confidence, pred = torch.max(probs, dim=1)

    # Output predicted class and confidence score
    predicted_class = class_names[pred.item()]
    confidence_score = confidence.item() * 100  
    print(f"Predicted: {predicted_class}, Confidence: {confidence_score:.2f}%")

    return predicted_class, confidence_score

model = ResNet18(num_classes=2).to(device)

#Input any random image from test set to predict class and confidence score
predict_image(model, r"C:\Users\bhalani\Animal_classification\Dataset\Dataset\test\cat\cat-323262_640.jpg")