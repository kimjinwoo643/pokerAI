import random
import torch
class Player:
    def __init__(self, name):
        self.name = name

    def choose_action(self, legal_actions, card, game_state):
        raise NotImplementedError("Subclasses should implement this!")
class RandomPlayer(Player):
    def choose_action(self, legal_actions, card, game_state):
        """
        Choose a random action from the list of legal actions.

        :param legal_actions: List of actions the player can take.
        :param card: The player's card (for future strategy use, but not used here).
        :param game_state: The current game state (for future strategy use, but not used here).
        :return: A randomly chosen action.
        """
        if "fold" in legal_actions:
            legal_actions.remove("fold")
        return random.choice(legal_actions)

class RaisePlayer(Player):
    def choose_action(self, legal_actions, card, game_state):
        if "raise" in legal_actions:
            return "raise"
        if "call" in legal_actions:
            return "call"
        else:
            return "check"
class FoldPlayer(Player):
    def choose_action(self, legal_actions, card, game_state):
        return "fold"
class PassivePlayer(Player):
    def choose_action(self, legal_actions, card, game_state):
        if card == 1:
            return "fold"
        elif card == 2:
            if "check" in legal_actions:
                return "check"
            else:
                return "fold"
        else:
            if "raise" in legal_actions:
                return "raise"
            if "call" in legal_actions:
                return "call"
            return "fold"
class ScaredPlayer(Player):
    def choose_action(self, legal_actions, card, game_state):
        if card == 1:
            if "check" in legal_actions:
                return "check"
            else:
                return "fold"
        elif card == 2:
            if "check" in legal_actions:
                return "check"
            else:
                return "fold"
        else:
            if "raise" in legal_actions:
                return "raise"
            if "call" in legal_actions:
                return "call"
            return "fold"

class AggressivePlayer(Player):
    def choose_action(self, legal_actions, card, game_state):
        if card == 1:
            if "raise" in legal_actions:
                return "raise"
            else:
                return "fold"
        else:
            if "raise" in legal_actions:
                return "raise"
            elif "call" in legal_actions:
                return "call"
        return "check"
class HumanPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def choose_action(self, legal_actions, card, game_state):
        print(f"\n{self.name}, your card is {card}.")
        print(f"Legal actions: {', '.join(legal_actions)}")

        while True:
            action = input("Choose your action: ").strip().lower()
            if action in legal_actions:
                print()
                return action
            else:
                print(f"Invalid action. Please choose from: {', '.join(legal_actions)}")

class DEEPQPlayer:
    def __init__(self, name, policy_net, epsilon=0.1):
        """
        Initialize the DEEPQPlayer.

        :param name: The name of the player.
        :param policy_net: The trained DQN model used to decide actions.
        :param epsilon: The exploration rate for epsilon-greedy action selection.
        """
        self.name = name
        self.policy_net = policy_net
        self.epsilon = epsilon

    def choose_action(self, legal_actions, card, game_state):
        """
        Choose an action using the trained DQN model.

        :param legal_actions: List of actions the player can take.
        :param card: The player's card (used as part of the state).
        :param game_state: The current game state (used as part of the state).
        :return: The chosen action.
        """
        state_vector = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0)

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Choose a random legal action
            return random.choice(legal_actions)

        # Get Q-values from the policy network
        with torch.no_grad():
            q_values = self.policy_net(state_vector)

        # Map Q-values to legal actions
        legal_action_indices = [i for i, action in enumerate(["check", "raise", "fold", "call"]) if
                                action in legal_actions]
        legal_q_values = q_values[0, legal_action_indices]

        # Choose the action with the highest Q-value among legal actions
        best_action_index = legal_action_indices[torch.argmax(legal_q_values).item()]

        return ["check", "raise", "fold", "call"][best_action_index]



