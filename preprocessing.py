import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score

# This block will apply some preprocess on images of train, val and test dataset
def preprocessing():
    transform = {
        "train": transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((64, 64)),  # Resize to 64x64 as per model need
            transforms.RandomCrop(64, padding=4),  # Random crop with padding
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # This will change brightness of the image
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.5], std=[0.5])  # This will help to normalize the image
        ]),
        
        "val": transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ]),
        
        "test": transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    }

    # Location of the dataset and sub directories like train, test and val directory
    data_dir = r"C:\Users\bhalani\Animal_classification\Dataset\Dataset"  
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"
    test_dir = f"{data_dir}/test"

    # This will Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform["train"])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform["val"])
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform["test"])

    # Define the batch size (As dataset small batch size 8 is taken)
    batch_size = 8

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Class-to-Index Mapping:", train_dataset.class_to_idx)
    
if __name__ == "__main__":
    preprocessing()