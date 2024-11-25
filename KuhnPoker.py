import random
class KuhnPoker:
    def __init__(self):
        self.deck = [1, 2, 3]
        self.pot = 0
        self.history = ""
        self.current_player = 0
        self.player_cards = []
        self.bet_sizes = [0, 0]  # Tracks the current bet size for each player
        self.max_bet = 2
        self.stacks = [10, 10]  # Each player starts with 10 chips

    def deal_cards(self):
        random.shuffle(self.deck)
        self.player_cards = [self.deck.pop(), self.deck.pop()]

    def reset_round(self):
        """Reset the game state for a new round."""
        self.pot = 0
        self.history = ""
        self.current_player = 0
        self.bet_sizes = [0, 0]
        self.deck = [1, 2, 3]

    def get_legal_actions(self):
        current_bet = self.bet_sizes[1 - self.current_player]
        player_stack = self.stacks[self.current_player]  # Current player's stack
        opponent_stack = self.stacks[1 - self.current_player]  # Opponent's stack

        # If the opponent hasn't bet yet, the player can check or bet
        if current_bet == 0:
            if player_stack > 0:  # Player can bet or check
                return ["check", "bet"]
            else:  # If no chips, only check
                return ["check"]

        # If the opponent has bet 1, the player can call, fold, or raise (if they have enough chips)
        elif current_bet == 1:
            if player_stack > 0:  # Player can fold, call, or raise if they have enough chips
                actions = ["call", "fold"]
                if self.history.endswith("raise"):  # No more raises after a raise has been called
                    return ["call", "fold"]
                if player_stack >= 1 and opponent_stack >= 2:  # If they have enough chips for a raise
                    actions.append("raise")
                return actions
            else:  # If no chips, the player can only fold
                return ["fold"]

        # If the opponent has bet 2, the player can only call or fold
        elif current_bet == 2:
            if player_stack > 0:  # Player can fold or call if they have chips
                return ["call", "fold"]
            else:  # If no chips, the player can only fold
                return ["fold"]

        return []

    def play_action(self, action):
        if action == "check":
            if self.history == "":
                self.history = "check"
            else:
                self.history += "-check"
        elif action == "bet":
            self.bet_sizes[self.current_player] += 1
            self.pot += 1
            self.stacks[self.current_player] -= 1
            self.history += "-bet"
        elif action == "raise":
            self.pot += 1
            self.bet_sizes[self.current_player] = 2
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