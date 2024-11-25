import random

class KuhnPoker:
    def __init__(self):
        self.turn = 1
        self.deck = [1, 2, 3]
        self.pot = 0
        self.history = ""
        self.current_player = 0
        self.player_cards = []
        self.bet_sizes = [0, 0]  # Tracks the current bet size for each player
        self.max_bet = 2
        self.stacks = [20, 20]  # Each player starts with 10 chips

    def deal_cards(self):
        random.shuffle(self.deck)
        self.player_cards = [self.deck.pop(), self.deck.pop()]

    def reset_round(self):
        """Reset the game state for a new round."""
        self.turn = 1 - self.turn
        self.pot = 0
        self.history = ""
        self.current_player = self.turn
        self.bet_sizes = [0, 0]
        self.deck = [1, 2, 3]
        self.force_initial_bets()  # Force each player to bet 1 chip at the start of the round

    def force_initial_bets(self):
        """Force both players to bet 1 chip at the start of the round."""
        for player in range(2):
            self.stacks[player] -= 1
            self.pot += 1
            self.bet_sizes[player] = 1
        self.history = "bet-bet"

    def get_legal_actions(self):
        current_bet = self.bet_sizes[1 - self.current_player]
        player_stack = self.stacks[self.current_player]  # Current player's stack
        opponent_stack = self.stacks[1 - self.current_player]  # Opponent's stack

        if current_bet == 1:
            actions = ["check", "fold"]
            if player_stack > 0:
                actions.append("raise")
            return actions

        # If the opponent has bet 1, the player can call, fold, or raise (if they have enough chips)
        if current_bet == 2:
            if player_stack > 0:  # Player can fold, call, or raise if they have enough chips
                actions = ["call", "fold"]
                if player_stack >= 1 and opponent_stack >= 1:  # If they have enough chips for a raise
                    actions.append("raise")
                return actions
            else:  # If no chips, the player can only fold
                return ["fold"]

        # If the opponent has bet 2, the player can only call or fold
        elif current_bet == 3:
            return ["call", "fold"]

        return []

    def play_action(self, action):
        if action == "check":
            if self.history == "":
                self.history = "check"
            else:
                self.history += "-check"
        elif action == "raise":
            self.pot += 1
            self.bet_sizes[self.current_player] += 1
            self.stacks[self.current_player] -= 1
            self.history += "-raise"
        elif action == "call":
            self.pot += self.bet_sizes[1 - self.current_player] - self.bet_sizes[self.current_player]
            self.stacks[self.current_player] -= self.bet_sizes[1 - self.current_player] - self.bet_sizes[self.current_player]
            self.bet_sizes[self.current_player] = self.bet_sizes[1 - self.current_player]
            self.history += "-call"
        elif action == "fold":
            self.history += "-fold"
        self.current_player = 1 - self.current_player

    def is_terminal(self):
        return "fold" in self.history or self.bet_sizes == [2, 2] or self.history.endswith("check-check") or self.history.endswith("call")

    def get_payoff(self):
        if "fold" in self.history:
            winner = 1 - self.current_player
            self.stacks[winner] += self.pot
        else:
            winner = 0 if self.player_cards[0] > self.player_cards[1] else 1
            self.stacks[winner] += self.pot

        return self.stacks
