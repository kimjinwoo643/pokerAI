import random
from KuhnPoker import KuhnPoker
import numpy as np
import copy

class KuhnCFR:
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
        utilities = []
        for _ in range(iterations):
            utility = self.cfr(0)
            total_utility += utility
            utilities.append(utility)

        print("Utility statistics:")
        print(f"Min Utility: {min(utilities)}")
        print(f"Max Utility: {max(utilities)}")
        print(f"Mean Utility: {np.mean(utilities)}")
        print(f"Median Utility: {np.median(utilities)}")

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

    def print_strategies(self):
        print("Learned Strategies:")
        for info_set, data in self.info_sets.items():
            avg_strategy = data["strategy_sum"] / np.sum(data["strategy_sum"]) if np.sum(data["strategy_sum"]) > 0 else \
            data["strategy"]
            print(f"Info Set: {info_set}")
            for action, prob in zip(self.action_space, avg_strategy):
                print(f"  {action}: {prob:.4f}")