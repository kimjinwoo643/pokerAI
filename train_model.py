import torch
import torch.nn as nn
import torch.optim as optim
from DQN import DQN
import random
from KuhnPoker import KuhnPoker
from players import RandomPlayer, DEEPQPlayer, HumanPlayer, PassivePlayer, RaisePlayer  # Import player classes



# Parameters (must match the trained model)
input_dim = 12  # Adjust based on your state vector size
output_dim = 4  # Actions: "check", "raise", "fold", "call"
learning_rate = 1e-3
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.999
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
# Main training loop
def train_model(opponent_classes, num_episodes=1000, hands_per_episode=20):
    global epsilon
    # Instantiate the AI player (DEEPQPlayer)
    ai_player = DEEPQPlayer(name="AI_Player", policy_net=policy_net, epsilon=epsilon)

    for episode in range(num_episodes):
        # Randomly select an opponent class
        opponent_class = random.choice(opponent_classes)
        opponent_player = opponent_class(name="Opponent_Player")

        cumulative_reward = 0  # Track cumulative reward for the episode
        episode_transitions = []  # Store transitions for the episode

        for hand in range(hands_per_episode):  # Play multiple hands per episode
            env.reset_round()
            env.deal_cards()
            done = False
            state = env.get_state_vector()

            while not done:
                # Determine the current player
                current_player = ai_player if env.current_player == 0 else opponent_player
                legal_actions = env.get_legal_actions()
                card = env.player_cards[env.current_player]

                action = current_player.choose_action(legal_actions, card, state)

                # Perform the action in the environment
                next_state, reward, done = env.step(action)

                # Store the transition for the AI player
                if isinstance(current_player, DEEPQPlayer):
                    action_index = ["check", "raise", "fold", "call"].index(action)
                    episode_transitions.append((state, action_index, reward, next_state, done))

                # Update state and cumulative reward
                state = next_state
                if isinstance(current_player, DEEPQPlayer):
                    cumulative_reward += reward

        # Adjust the rewards for all transitions in the episode
        for i, (state, action_index, reward, next_state, done) in enumerate(episode_transitions):
            # Scale the reward by the episode's cumulative performance
            adjusted_reward = cumulative_reward
            store_transition(replay_buffer, (state, action_index, adjusted_reward, next_state, done))

        # Train the model after every episode
        train_dqn()

        # Update the target network periodically
        if episode % target_update_frequency == 0:
            update_target_network()

        # Reduce epsilon (exploration rate)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon {epsilon:.2f}, Cumulative Reward: {cumulative_reward:.2f}")

    # Save the trained model after training
    torch.save(policy_net.state_dict(), "policy_net.pth")
    print("Model saved successfully!")

# Main training loop for self-play
def train_self_play(num_episodes=1000, hands_per_episode=20):
    global epsilon
    policy_net.load_state_dict(torch.load("policy_net.pth"))
    # Instantiate two AI players
    ai_player = DEEPQPlayer(name="AI_Player", policy_net=policy_net, epsilon=epsilon)
    opponent_player = DEEPQPlayer(name="Opponent_Player", policy_net=policy_net, epsilon=epsilon)

    for episode in range(num_episodes):
        cumulative_reward = 0  # Track cumulative reward for the primary player
        episode_transitions = []  # Store transitions for training

        for hand in range(hands_per_episode):  # Play multiple hands per episode
            env.reset_round()
            env.deal_cards()
            done = False
            state = env.get_state_vector()

            while not done:
                # Determine the current player (alternates between AI and opponent)
                current_player = ai_player if env.current_player == 0 else opponent_player
                legal_actions = env.get_legal_actions()
                card = env.player_cards[env.current_player]

                # Current player chooses an action
                action = current_player.choose_action(legal_actions, card, state)

                # Perform the action in the environment
                next_state, reward, done = env.step(action)

                # Store transition only for the primary AI player
                if current_player is ai_player:
                    action_index = ["check", "raise", "fold", "call"].index(action)
                    episode_transitions.append((state, action_index, reward, next_state, done))

                # Update state and cumulative reward for AI player
                state = next_state
                if current_player is ai_player:
                    cumulative_reward += reward

        # Adjust rewards for all transitions in the episode
        for i, (state, action_index, reward, next_state, done) in enumerate(episode_transitions):
            # Scale the reward by the episode's cumulative performance
            adjusted_reward = cumulative_reward
            store_transition(replay_buffer, (state, action_index, adjusted_reward, next_state, done))

        # Train the model after each episode
        train_dqn()

        # Update the target network periodically
        if episode % target_update_frequency == 0:
            update_target_network()

        # Update the opponent's policy periodically to ensure dynamic self-play
        if episode % 50 == 0:  # Update opponent every 50 episodes
            opponent_player.policy_net.load_state_dict(policy_net.state_dict())
            print(f"Opponent policy updated at episode {episode}.")

        # Reduce epsilon (exploration rate)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon {epsilon:.2f}, Cumulative Reward: {cumulative_reward:.2f}")

    # Save the trained model after training
    torch.save(policy_net.state_dict(), "policy_net_self_play.pth")
    print("Model saved successfully!")


if __name__ == "__main__":
    # Train against multiple types of players
    train_model([PassivePlayer], num_episodes=10000)
    train_self_play(num_episodes=10000)
