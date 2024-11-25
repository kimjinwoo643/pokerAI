import random
from collections import defaultdict
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
        return random.choice(legal_actions)
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


class CFRPlayer:
    def __init__(self, name):
        self.name = name
        self.regret_sum = defaultdict(float)  # Tracks regret values for actions
        self.strategy_sum = defaultdict(float)  # Tracks cumulative strategies for averaging
        self.actions = ["check", "bet", "call", "fold", "raise"]

    def get_strategy(self, legal_actions):
        """
        Calculate the strategy for the current information set based on regrets.
        """
        strategy = {action: max(self.regret_sum[action], 0) for action in legal_actions}
        normalizing_sum = sum(strategy.values())

        if normalizing_sum > 0:
            for action in legal_actions:
                strategy[action] /= normalizing_sum
        else:
            # If all regrets are zero, choose uniformly among legal actions
            strategy = {action: 1 / len(legal_actions) for action in legal_actions}

        # Add to cumulative strategy sum
        for action in legal_actions:
            self.strategy_sum[action] += strategy[action]

        return strategy

    def choose_action(self, legal_actions, card, game_state):
        """
        Choose an action based on the current strategy.
        """
        strategy = self.get_strategy(legal_actions)
        actions, probabilities = zip(*strategy.items())
        return random.choices(actions, probabilities)[0]

    def update_regret(self, action_history, terminal_utility, player_index, game):
        """
        Update regrets based on the outcome of the game.
        """
        legal_actions = game.get_legal_actions()
        action_utility = {}

        # Compute counterfactual utilities for each legal action
        for action in legal_actions:
            modified_game = game.clone()  # Clone the game to simulate this action
            modified_game.play_action(action)
            if modified_game.is_terminal():
                utility = modified_game.get_payoff()[player_index]
            else:
                opponent = CFRPlayer("Opponent")
                utility = opponent.get_expected_utility(modified_game, 1 - player_index)
            action_utility[action] = utility

        # Calculate regret for each action
        for action in legal_actions:
            regret = action_utility[action] - terminal_utility
            self.regret_sum[action] += regret

    def get_expected_utility(self, game, player_index):
        """
        Compute the expected utility for the current game state and player index.
        """
        if game.is_terminal():
            return game.get_payoff()[player_index]

        legal_actions = game.get_legal_actions()
        strategy = self.get_strategy(legal_actions)
        utility = 0

        for action, prob in strategy.items():
            modified_game = game.clone()
            modified_game.play_action(action)
            utility += prob * self.get_expected_utility(modified_game, 1 - player_index)

        return utility

    def get_average_strategy(self):
        """
        Compute the average strategy based on cumulative strategy sums.
        """
        normalizing_sum = sum(self.strategy_sum.values())
        if normalizing_sum > 0:
            return {action: self.strategy_sum[action] / normalizing_sum for action in self.actions}
        else:
            return {action: 1 / len(self.actions) for action in self.actions}

