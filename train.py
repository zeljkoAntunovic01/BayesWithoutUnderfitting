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

def train_classifier(model, train_data, val_data=None, save_path=None, learning_rate=1e-3, weight_decay=1e-4, num_epochs=10):
    print(f"Using device: {DEVICE}")
    model.to(DEVICE)

    # Loss function for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_data:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            total_train_loss += loss.item()

            # Compute training accuracy
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Print stats every epoch
        avg_train_loss = total_train_loss / len(train_data)
        train_accuracy = 100.0 * train_correct / train_total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.6f}, Accuracy: {train_accuracy:.2f}%")

        # 🔎 Validation Evaluation
        if val_data is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for val_images, val_labels in val_data:
                    val_images, val_labels = val_images.to(DEVICE), val_labels.to(DEVICE)
                    val_outputs = model(val_images)
                    loss = criterion(val_outputs, val_labels)

                    val_loss += loss.item()
                    _, val_pred = val_outputs.max(1)
                    val_total += val_labels.size(0)
                    val_correct += val_pred.eq(val_labels).sum().item()

            avg_val_loss = val_loss / len(val_data)
            val_acc = 100.0 * val_correct / val_total
            model.train()
        print(f"Epoch {epoch+1}/{num_epochs}:\nTrain Loss: {avg_train_loss:.6f}, Train Accuracy: {train_accuracy:.2f}%\nVal Loss: {avg_val_loss:.6f}, Val Accuracy: {val_acc:.2f}%")

    print(f"Final Loss: {avg_train_loss:.6f}, Final Train Accuracy: {train_accuracy:.2f}%\nFinal Val Loss: {avg_val_loss:.6f}, Final Val Accuracy: {val_acc:.2f}%")

    if (save_path):
        # Ensure the save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the trained model
        torch.save(model.state_dict(), save_path)