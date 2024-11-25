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
    def __init__(self, cfr_trainer, name):
        self.cfr_trainer = cfr_trainer
        self.name = name

    def choose_action(self, legal_actions, card, game_state):
        card = card
        history = game_state.history
        info_set = self.cfr_trainer.get_info_set(card, history)

        strategy = self.cfr_trainer.get_average_strategy(info_set)
        legal_actions = legal_actions

        if not legal_actions:
            return None

        # Choose action with highest strategy probability
        best_action = max(legal_actions, key=lambda action: strategy[self.cfr_trainer.action_space.index(action)])

        return best_action
