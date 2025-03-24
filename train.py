from models.sinemodel import SineNet, compute_q_lla
from data.utils import generate_sine_data
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np 
import matplotlib.pyplot as plt
from plots import plot_model
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_sine(model, data, learning_rate = 1e-3, weight_decay = 1e-2, num_epochs = 10000, save_path="results/models/sine_net.pth"):
    criterion = nn.MSELoss()  # MSE loss for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Generate training data
    x_train, y_train = data

    # Train the model
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Final loss
    print(f"Final Loss: {loss.item():.6f}")
     # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the trained model
    torch.save(model.state_dict(), save_path)

def train_classifier(model, data, save_path, learning_rate=1e-3, weight_decay=1e-4, num_epochs=10):
    print(f"Using device: {DEVICE}")
    model.to(DEVICE)

    # Loss function for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in data:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_loss += loss.item()

            # Compute training accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Print stats every epoch
        avg_loss = total_loss / len(data)
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")

    print(f"Final Loss: {avg_loss:.6f}, Final Accuracy: {accuracy:.2f}%")

    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the trained model
    torch.save(model.state_dict(), save_path)