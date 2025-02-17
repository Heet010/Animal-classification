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
from model import ResNet18

# Training Function
def train_model(model, trainloader, valloader, num_epochs, save_dir=r'C:\Users\bhalani\Animal_classification\Animal-classification\checkpoints'):
    train_losses = []
    val_losses = []
    val_f1_scores = []
    val_accuracies = []

    os.makedirs(save_dir, exist_ok=True)

    # Initialize best loss to a large value
    best_val_loss = float('inf')
    best_model_path = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(trainloader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(valloader)
        val_losses.append(val_loss)

        # This will Compute F1 Score & Accuracy with the help of scikit-learn's predifined functions
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_acc = accuracy_score(all_labels, all_preds)

        val_f1_scores.append(val_f1)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # This will save the model every time if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at: {best_model_path}")
    
    # This method will plot training loss and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")
    loss_plot_path = os.path.join(r'C:\Users\bhalani\Animal_classification\Animal-classification\Result_plots', 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Loss plot saved at: {loss_plot_path}")

    # This method will plot Validation Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Val Accuracy", marker="o", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    acc_plot_path = os.path.join(r'C:\Users\bhalani\Animal_classification\Animal-classification\Result_plots', 'val_accuracy_curve.png')
    plt.savefig(acc_plot_path)
    plt.close()
    print(f"Validation accuracy plot saved at: {acc_plot_path}")

    # This method will plot Validation F1 Score
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), val_f1_scores, label="Val F1 Score", marker="o", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.title("Validation F1 Score")
    f1_plot_path = os.path.join(r'C:\Users\bhalani\Animal_classification\Animal-classification\Result_plots', 'val_f1_curve.png')
    plt.savefig(f1_plot_path)
    plt.close()
    print(f"Validation F1 Score plot saved at: {f1_plot_path}")

    return best_model_path
        

    
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
    }

    # Location of the dataset and sub directories like train, test and val directory
    data_dir = r'C:\Users\bhalani\Animal_classification\Dataset\Dataset'  
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"

    # This will Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform["train"])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform["val"])

    # Define the batch size (As dataset small batch size 8 is taken)
    batch_size = 8

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
if __name__ == "__main__":
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
    }

    # Location of the dataset and sub directories like train, test and val directory
    data_dir = r'C:\Users\bhalani\Animal_classification\Dataset\Dataset'  
    train_dir = f"{data_dir}/train"
    val_dir = f"{data_dir}/val"

    # This will Load datasets using ImageFolder
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform["train"])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform["val"])

    # Define the batch size (As dataset small batch size 8 is taken)
    batch_size = 8

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = ResNet18(num_classes=2).to(device)

    # Loss function & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_model_path = train_model(model, trainloader, valloader, num_epochs=20) # This will start model training add number of epochs as required.

    # This will print the absolute path of the saved best model after training
    if best_model_path:
        print(f"Best model checkpoint saved at: {os.path.abspath(best_model_path)}")
        