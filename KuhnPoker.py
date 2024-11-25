import random
class KuhnPoker:
    def __init__(self):
        self.deck = [1, 2, 3]
        self.player_cards = []
        self.history = ""
        self.pot = 0
        self.current_player = 0

    def deal_cards(self):
        self.deck = random.sample(self.deck, len(self.deck))
        self.player_cards = [self.deck.pop(), self.deck.pop()]

    def get_actions(self):
        if self.history == "":
            return ["bet", "check"]
        elif self.history == "bet":
            return ["call", "fold"]
        elif self.history == "check":
            return ["bet", "check"]
        elif self.history == "check-bet":
            return ["call", "fold"]

    def is_terminal(self):
        return self.history in ["bet-call", "bet-fold", "check-bet-call", "check-bet-fold", "check-check"]

    def get_payoff(self):
        if self.history == "bet-fold":
            return 1 if self.current_player == 0 else -1
        elif self.history == "check-bet-fold":
            return 1 if self.current_player == 1 else -1
        elif self.history in ["bet-call", "check-bet-call", "check-check"]:
            if self.player_cards[0] > self.player_cards[1]:
                return 2 if self.history == "bet-call" else 1
            else:
                return -2 if self.history == "bet-call" else -1

    def play(self, action):
        if self.history:
            self.history += "-"
        self.history += action
        if action == "bet":
            self.pot += 1
        elif action == "call":
            self.pot += 1
        self.current_player = 1 - self.current_player

    def reset(self):
        self.deck = [1, 2, 3]
        self.player_cards = []
        self.history = ""
        self.pot = 0
        self.current_player = 0