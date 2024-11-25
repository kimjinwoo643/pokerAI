import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from players import DEEPQPlayer
from KuhnPoker import KuhnPoker  # Assuming KuhnPoker is already implemented

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
learning_rate = 1e-3
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.1
replay_buffer_size = 10000
batch_size = 64
target_update_frequency = 10

input_dim = 8  # Adjust based on the actual state representation
output_dim = 4  # Actions: "check", "raise", "fold", "call"
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)

target_net.load_state_dict(policy_net.state_dict())  # Synchronize networks
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = deque(maxlen=replay_buffer_size)

# Utility to create a mask for legal actions
def get_action_mask(env):
    """
    Creates a binary mask for legal actions.
    """
    legal_actions = env.get_legal_actions()
    action_space = ["check", "raise", "fold", "call"]  # Define the full action space
    return [1 if action in legal_actions else 0 for action in action_space]

# Store transitions in replay buffer
def store_transition(buffer, transition):
    buffer.append(transition)

# Sample a batch from the replay buffer
def sample_batch(buffer):
    batch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(next_states, dtype=torch.float32),
        torch.tensor(dones, dtype=torch.float32),
    )

# Training loop for DQN
def train_dqn():
    if len(replay_buffer) < batch_size:
        return  # Skip training if not enough data

    states, actions, rewards, next_states, dones = sample_batch(replay_buffer)

    # Compute Q(s, a) using the policy network
    q_values = policy_net(states).gather(1, actions.unsqueeze(1))

    # Compute max Q(s', a') from the target network
    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q_values = next_q_values.max(1)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    # Compute the loss and update the network
    loss = nn.MSELoss()(q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ε-greedy action selection with dynamic action masking
def select_action(state, epsilon, env):
    action_mask = get_action_mask(env)
    legal_indices = [i for i, mask in enumerate(action_mask) if mask == 1]

    if random.random() < epsilon:
        # Random action from the legal actions
        return random.choice(legal_indices)

    # Predict Q-values and mask illegal actions
    with torch.no_grad():
        q_values = policy_net(torch.tensor(state, dtype=torch.float32))
        masked_q_values = q_values + (torch.tensor(action_mask) - 1) * 1e9  # Mask invalid actions with a large negative value
        return torch.argmax(masked_q_values).item()

# Update target network
def update_target_network():
    target_net.load_state_dict(policy_net.state_dict())



