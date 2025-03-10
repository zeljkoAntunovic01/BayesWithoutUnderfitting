import torch
import numpy as np
import matplotlib.pyplot as plt

# Generate sine wave dataset
def generate_sine_data(n_samples=100, noise_std=0.1):
    x = torch.linspace(-2 * np.pi, 2 * np.pi, n_samples).unsqueeze(1)  # Shape: (n_samples, 1)
    y = torch.sin(x) + noise_std * torch.randn_like(x)  # Adding noise
    return x, y

def main():
    # Generate and visualize the data
    x_train, y_train = generate_sine_data(100)
    plt.scatter(x_train.numpy(), y_train.numpy(), label="Training Data")
    plt.plot(x_train.numpy(), np.sin(x_train.numpy()), color="red", label="True Function")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
