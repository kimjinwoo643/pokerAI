import torch
import torch.optim as optim
import random
from KuhnPoker import KuhnPoker
from players import RandomPlayer, DEEPQPlayer, HumanPlayer, PassivePlayer  # Import player classes
from DQN import DQN  # Import the DQN architecture

# Parameters (must match the trained model)
input_dim = 8  # Adjust based on your state vector size
output_dim = 4  # Actions: "check", "raise", "fold", "call"
learning_rate = 1e-3
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.1
replay_buffer_size = 10000
batch_size = 64
target_update_frequency = 10

# Initialize the DQN and optimizer
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())  # Sync the networks
target_net.eval()  # Set target network to eval mode

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = []

# Initialize the environment
env = KuhnPoker()

# Utility to store transitions in replay buffer
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
    loss = torch.nn.functional.mse_loss(q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Update target network
def update_target_network():
    target_net.load_state_dict(policy_net.state_dict())

# Main training loop
def train_model(player_class, num_episodes=1000):
    global epsilon

    # Instantiate the AI player (DEEPQPlayer)
    ai_player = DEEPQPlayer(name="AI_Player", policy_net=policy_net, epsilon=epsilon)

    # Instantiate the opponent player (RandomPlayer, HumanPlayer, etc.)
    opponent_player = player_class(name="Opponent_Player")

    for episode in range(num_episodes):
        env.reset_round()
        env.deal_cards()
        done = False
        state = env.get_state_vector()

        while not done:
            # Determine the current player
            current_player = ai_player if env.current_player == 0 else opponent_player
            legal_actions = env.get_legal_actions()
            card = env.player_cards[env.current_player]

            # AI player chooses an action
            if isinstance(current_player, DEEPQPlayer):
                action = current_player.choose_action(legal_actions, card, state)
            else:
                action = current_player.choose_action(legal_actions, card, state)

            # Perform the action in the environment
            next_state, reward, done = env.step(action)

            # Store transition in replay buffer
            action_index = ["check", "raise", "fold", "call"].index(action)
            store_transition(replay_buffer, (state, action_index, reward, next_state, done))

            # Update state
            state = next_state

            # Train the model
            train_dqn()

        # Update the target network periodically
        if episode % target_update_frequency == 0:
            update_target_network()

        # Reduce epsilon (exploration rate)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon {epsilon:.2f}")
    torch.save(policy_net.state_dict(), "policy_net.pth")
    print("Model saved successfully!")

if __name__ == "__main__":
    train_model(PassivePlayer, num_episodes=10000)
