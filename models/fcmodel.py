import torch.nn as nn
import torch.nn.functional as F 

class FC_2D_Net(nn.Module):
    def __init__(self, hidden_units=64, n_classes=4):
        super(FC_2D_Net, self).__init__()

        # Fully Connected Layers
        self.fc1 = nn.Linear(2, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Raw logits
        return x