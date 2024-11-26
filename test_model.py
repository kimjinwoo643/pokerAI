import torch
from KuhnPoker import KuhnPoker
from players import RandomPlayer, DEEPQPlayer, PassivePlayer, HumanPlayer, RaisePlayer
from DQN import DQN  # Import the DQN architecture

input_dim = 12  # Adjust based on your state vector size
output_dim = 4  # Actions: "check", "raise", "fold", "call"

# Load the trained model
policy_net = DQN(input_dim, output_dim)
policy_net.load_state_dict(torch.load("policy_net_self_play.pth"))
policy_net.eval()  # Set to evaluation mode

# Initialize the players
ai_player = DEEPQPlayer(name="AI_Player", policy_net=policy_net, epsilon=0.0)  # No exploration during testing
# ai_player = PassivePlayer(name="Raise Player")
player_2 = RaisePlayer(name="Player")

# Initialize the environment
env = KuhnPoker()

# Function to play a single game
def play_game():
    env.reset_round()
    env.deal_cards()

    done = False
    state = env.get_state_vector()

    while not done:
        # Determine the current player
        current_player = ai_player if env.current_player == 0 else player_2
        legal_actions = env.get_legal_actions()
        card = env.player_cards[env.current_player]
        # print(env.bet_sizes)
        # AI player chooses an action
        action = current_player.choose_action(legal_actions, card, state)
        print(f"{current_player.name} chooses to {action}.")

        # Perform the action in the environment
        next_state, reward, done = env.step(action)

        # Update state
        state = next_state


# Function to evaluate the model over multiple hands
def evaluate_model(num_games=100):
    ai_wins = 0
    random_wins = 0
    for j in range(num_games):
        for round_number in range(20):
            print(f"\n--- Round {round_number} ---")
            play_game()
            # print(env.stacks[0], env.stacks[1])
            print(f"AI card: {env.player_cards[0]}, Player: {env.player_cards[1]}")
            print(f"Stacks after round: AI: {env.stacks[0]}, Player: {env.stacks[1]}")
            if env.stacks[0] <= 0 or env.stacks[1] <= 0:
                break
        if env.stacks[0] > env.stacks[1]:
            ai_wins += 1
            winner = "AI Player"
        else:
            random_wins += 1
            winner = "Player 2"

        print(f"Game Over! Winner: {winner}")
        env.stacks[0] = 20
        env.stacks[1] = 20

    print(f"\nAfter {num_games} games:")
    print(f"AI Wins: {ai_wins}, Player 2 Wins: {random_wins}")
    print(f"AI Win Rate: {ai_wins / num_games * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model(num_games=1000)  # Evaluate the model against a RandomPlayer
