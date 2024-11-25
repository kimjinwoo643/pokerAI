import random
import numpy as np
import copy
from KuhnPoker import KuhnPoker


class KuhnPokerCFR:
    def __init__(self):
        self.info_sets = {}
        self.action_space = ["check", "raise", "call", "fold"]

    def get_info_set(self, card, history):
        return f"{card}:{history}"

    def get_strategy(self, info_set):
        if info_set not in self.info_sets:
            self.info_sets[info_set] = {
                "strategy": np.ones(len(self.action_space)) / len(self.action_space),
                "regret_sum": np.zeros(len(self.action_space)),
                "strategy_sum": np.zeros(len(self.action_space))
            }
        return self.info_sets[info_set]

    def get_average_strategy(self, info_set):
        info_set_data = self.get_strategy(info_set)
        strategy = info_set_data["strategy_sum"] / np.sum(info_set_data["strategy_sum"]) if np.sum(
            info_set_data["strategy_sum"]) > 0 else info_set_data["strategy"]
        return strategy

    def train(self, iterations=50000):
        total_utility = 0
        for _ in range(iterations):
            utility = self.cfr(0)
            total_utility += utility
        return total_utility / iterations

    def cfr(self, player):
        game = KuhnPoker()
        game.deal_cards()
        return self._cfr(game, player, 1.0)

    def _cfr(self, game, player, reach_prob):
        if game.is_terminal():
            return self.terminal_utility(game, player)

        card = game.player_cards[player]
        history = game.history
        info_set = self.get_info_set(card, history)

        strategy_data = self.get_strategy(info_set)
        legal_actions = game.get_legal_actions()
        strategy = strategy_data["strategy"]

        # Compute action utilities
        utilities = np.zeros(len(self.action_space))
        node_utility = 0

        for i, action in enumerate(self.action_space):
            if action not in legal_actions:
                utilities[i] = 0
                continue

            next_game = copy.deepcopy(game)
            next_game.play_action(action)
            utility = -self._cfr(next_game, 1 - player, reach_prob * strategy[i])

            utilities[i] = utility
            node_utility += strategy[i] * utility

        # Update regrets
        for i in range(len(self.action_space)):
            regret = utilities[i] - node_utility
            strategy_data["regret_sum"][i] += reach_prob * regret
            strategy_data["strategy_sum"][i] += reach_prob * strategy[i]

        # Update strategy based on cumulative regret
        strategy_data["strategy"] = np.maximum(strategy_data["regret_sum"], 0)
        strategy_sum = np.sum(strategy_data["strategy"])
        strategy_data["strategy"] = strategy_data["strategy"] / strategy_sum if strategy_sum > 0 else np.ones(
            len(self.action_space)) / len(self.action_space)

        return node_utility

    def terminal_utility(self, game, player):
        payoffs = game.get_payoff()
        return payoffs[player] - payoffs[1 - player]


class KuhnPokerPlayer:
    def __init__(self, cfr_trainer):
        self.cfr_trainer = cfr_trainer

    def get_best_action(self, game):
        legal_actions = game.get_legal_actions()

        # If no legal actions, return None
        if not legal_actions:
            return None

        # If only one legal action, return it
        if len(legal_actions) == 1:
            return legal_actions[0]

        card = game.player_cards[game.current_player]
        history = game.history
        info_set = self.cfr_trainer.get_info_set(card, history)

        strategy = self.cfr_trainer.get_average_strategy(info_set)

        # Choose action with highest strategy probability among legal actions
        best_action = max(legal_actions, key=lambda action:
        strategy[self.cfr_trainer.action_space.index(action)]
                          )

        return best_action


# Demonstration
def play_game(cfr_trainer):
    game = KuhnPoker()
    game.deal_cards()
    player = KuhnPokerPlayer(cfr_trainer)

    while not game.is_terminal():
        action = player.get_best_action(game)
        game.play_action(action)

    return game.get_payoff()


# Train and play
cfr_trainer = KuhnPokerCFR()
average_utility = cfr_trainer.train(iterations=50000)
print(f"Average Utility: {average_utility}")

# Play multiple games
num_games = 1
total_player_0_winnings = 0

for _ in range(num_games):
    final_stacks = play_game(cfr_trainer)
    total_player_0_winnings += final_stacks[0]

print(f"Average Player 0 Winnings over {num_games} games: {total_player_0_winnings / num_games}")