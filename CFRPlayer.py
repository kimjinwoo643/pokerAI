import random
from collections import defaultdict

class CFRPlayer:
    def __init__(self):
        self.strategy = defaultdict(lambda: [0.5, 0.5])  # Default equal probability for actions
        self.regret_sum = defaultdict(lambda: [0, 0])   # Sum of regrets for actions
        self.strategy_sum = defaultdict(lambda: [0, 0])  # Sum of strategies over iterations

    def get_strategy(self, infoset):
        regrets = self.regret_sum[infoset]
        normalizing_sum = sum(max(0, r) for r in regrets)
        if normalizing_sum > 0:
            strategy = [max(0, r) / normalizing_sum for r in regrets]
        else:
            strategy = [0.5, 0.5]  # Default to equal probability
        self.strategy[infoset] = strategy
        return strategy

    def update_strategy_sum(self, infoset, strategy):
        for i in range(len(strategy)):
            self.strategy_sum[infoset][i] += strategy[i]

    def get_average_strategy(self, infoset):
        normalizing_sum = sum(self.strategy_sum[infoset])
        if normalizing_sum > 0:
            return [x / normalizing_sum for x in self.strategy_sum[infoset]]
        else:
            return [0.5, 0.5]  # Default to equal probability
