import itertools
import torch
import torch.optim as optim
from KuhnPoker import KuhnPoker
from players import DEEPQPlayer, PassivePlayer
from DQN import DQN
import random

# Define hyperparameter search space
hyperparams = {
    "learning_rate": [1e-3],
    "gamma": [.85, 0.9],
    "epsilon_decay": [0.995],
    "batch_size": [128, 160],
    "target_update_frequency": [10, 15],
}

# Function to evaluate a single combination of hyperparameters
def evaluate_hyperparams(params, num_episodes=500, hands_per_episode=20):
    # Extract parameters
    learning_rate = params["learning_rate"]
    gamma = params["gamma"]
    epsilon_decay = params["epsilon_decay"]
    batch_size = params["batch_size"]
    target_update_frequency = params["target_update_frequency"]

    # Initialize components
    input_dim = 12  # Adjust based on state vector size
    output_dim = 4  # Actions: "check", "raise", "fold", "call"
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = []
    epsilon = 1.0
    epsilon_min = 0.1
    cumulative_rewards = []

    env = KuhnPoker()

    # Utility functions
    # Update epsilon based on recent performance
    def update_epsilon(epsilon, recent_rewards, threshold=10, decay=0.995):
        avg_reward = sum(recent_rewards[-threshold:]) / threshold
        if avg_reward > 0:  # Encourage exploration only if recent performance is improving
            epsilon *= decay
        return max(epsilon, epsilon_min)

    def store_transition(buffer, transition):
        buffer.append(transition)
        if len(buffer) > 10000:  # Replay buffer size
            buffer.pop(0)

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

    def train_dqn():
        if len(replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = sample_batch(replay_buffer)
        q_values = policy_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
        loss = torch.nn.functional.mse_loss(q_values.squeeze(), target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def update_target_network():
        target_net.load_state_dict(policy_net.state_dict())

    # Training loop
    for episode in range(num_episodes):
        cumulative_reward = 0
        for hand in range(hands_per_episode):
            env.reset_round()
            env.deal_cards()
            ai_player = DEEPQPlayer(name="AI_Player", policy_net=policy_net, epsilon=epsilon)
            opponent_player = PassivePlayer(name="Opponent_Player")

            done = False
            state = env.get_state_vector()
            while not done:
                current_player = ai_player if env.current_player == 0 else opponent_player
                action = current_player.choose_action(env.get_legal_actions(), env.player_cards[env.current_player], state)
                next_state, reward, done = env.step(action)

                if current_player is ai_player:
                    action_index = ["check", "raise", "fold", "call"].index(action)
                    store_transition(replay_buffer, (state, action_index, reward, next_state, done))
                    cumulative_reward += reward
                state = next_state
            if env.stacks[0] <= 0 or env.stacks[1] <= 0:
                if env.stacks[0] <= 0:
                    cumulative_reward-=10
                elif env.stacks[1]<=0:
                    cumulative_reward+=10
                break

        cumulative_rewards.append(cumulative_reward)
        train_dqn()

        if episode % target_update_frequency == 0:
            update_target_network()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    return sum(cumulative_rewards) / num_episodes  # Average reward

# Grid search over hyperparameters
def grid_search_hyperparams():
    best_params = None
    best_avg_reward = float("-inf")
    param_combinations = list(itertools.product(*hyperparams.values()))
    keys = list(hyperparams.keys())

    for combination in param_combinations:
        params = dict(zip(keys, combination))
        print(f"Evaluating parameters: {params}")
        avg_reward = evaluate_hyperparams(params)
        print(f"Average reward: {avg_reward}")

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_params = params

    return best_params, best_avg_reward

if __name__ == "__main__":
    best_params, best_avg_reward = grid_search_hyperparams()
    print(f"Best Parameters: {best_params}")
    print(f"Best Average Reward: {best_avg_reward}")
