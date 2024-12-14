import torch
import torch.nn as nn
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)  # Increase neurons
        self.fc2 = nn.Linear(16, 32)  # Add more capacity
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class EnhancedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EnhancedDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # Increase first layer size
        self.bn1 = nn.LayerNorm(128)         # Add Layer Normalization for stability
        
        self.fc2 = nn.Linear(128, 256)       # Increase intermediate layer size
        self.bn2 = nn.LayerNorm(256)
        
        self.fc3 = nn.Linear(256, 128)       # Adjust layer sizes to match capacity
        self.bn3 = nn.LayerNorm(128)
        
        self.fc4 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))  # LayerNorm + ReLU
        x = torch.relu(self.bn2(self.fc2(x)))  # LayerNorm + ReLU
        x = torch.relu(self.bn3(self.fc3(x)))  # LayerNorm + ReLU
        return self.fc4(x)  # Raw Q-values for each action