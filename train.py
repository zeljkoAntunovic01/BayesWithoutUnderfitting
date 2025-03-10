from model import SineNet, compute_q_lla
from data import generate_sine_data
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np 
import matplotlib.pyplot as plt
from plots import plot_model
import os

def train(model, data, learning_rate = 1e-3, weight_decay = 1e-2, num_epochs = 10000, save_path="models/sine_net.pth"):
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

    return model

if __name__ == "__main__":
    model = SineNet()
    train(model=model)
    x_train, y_train = generate_sine_data(100)
    plot_model(model=model, x_train=x_train, y_train=y_train)
    
    # Compute q_LLA posterior
    theta_map, covariance = compute_q_lla(model, x_train)
    print("MAP Parameters:", theta_map)
    print("Posterior Covariance Shape:", covariance.shape)

