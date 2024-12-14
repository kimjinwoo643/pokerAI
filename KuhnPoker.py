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
        self.opponent_last_action = None

    def get_state_vector(self):
        """
        Converts the current game state into a numerical vector representation.
        """
        # Encode player's card (one-hot encoding for simplicity)
        card_encoding = [0, 0, 0]
        card_encoding[self.player_cards[self.current_player] - 1] = 1

        # Normalize stacks and bet sizes (optional, but helps with NN training)
        normalized_stacks = [stack / 20 for stack in self.stacks]
        normalized_bets = [bet / self.max_bet for bet in self.bet_sizes]

        action_space = ["check", "raise", "fold", "call"]
        current_player_flag = [1 if self.current_player == 0 else 0]
        opponent_last_action_encoding = [0] * len(action_space)
        if self.opponent_last_action in action_space:
            opponent_last_action_encoding[action_space.index(self.opponent_last_action)] = 1
        opponent_checked = [opponent_last_action_encoding[0]]
        # Flatten and return the state vector
        return (
                card_encoding
                + normalized_stacks
                + normalized_bets
                + current_player_flag
                + opponent_checked
        )

    def step(self, action, perform_action=True):
        prev_player = self.current_player
        if perform_action:
            legal_actions = self.get_legal_actions()

            if action not in legal_actions:
                return self.get_state_vector(), 0, self.is_terminal()
            
            # Upon playing an action, the action gets added to the history and the player switches
            self.play_action(action)

        done = self.is_terminal()
        reward = 0

        if done:
            # Game is over, calculate payoff (account for folding when having a higher card)
            # Doesnt matter which player, since only AI player can use the reward
            difference_in_cards = self.player_cards[self.current_player] - self.player_cards[1 - self.current_player]
            if "fold" in self.history:
                winner = self.current_player
                reward = self.bet_sizes[1 - prev_player] if winner == prev_player else -self.bet_sizes[prev_player]
                # Increase reward when a good fold, decrease reward when a bad fold
                reward += difference_in_cards if perform_action else 0
                if perform_action:
                    self.stacks[winner] += self.pot
            else:
                winner = 0 if self.player_cards[0] > self.player_cards[1] else 1
                reward = self.bet_sizes[1 - prev_player] if winner == prev_player else -self.bet_sizes[prev_player]
                if perform_action:
                    self.stacks[winner] += self.pot

        # Get the new state vector
        state = self.get_state_vector()

        return state, reward, done

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
            if player_stack > 0 and opponent_stack > 0:
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
        elif max(self.bet_sizes) == 3:
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
            self.bet_sizes[self.current_player] += max(self.bet_sizes)
            self.stacks[self.current_player] -= 1
            self.history += "-raise"
        elif action == "call":
            self.pot += self.bet_sizes[1 - self.current_player] - self.bet_sizes[self.current_player]
            self.stacks[self.current_player] -= self.bet_sizes[1 - self.current_player] - self.bet_sizes[self.current_player]
            self.bet_sizes[self.current_player] = self.bet_sizes[1 - self.current_player]
            self.history += "-call"
        elif action == "fold":
            self.history += "-fold"
        self.opponent_last_action = action
        self.current_player = 1 - self.current_player

    def is_terminal(self):
        return "fold" in self.history or self.bet_sizes == [3, 3] or self.history.endswith("check-check") or self.history.endswith("call")

    def get_payoff(self):
        if "fold" in self.history:
            winner = 1 - self.current_player
            self.stacks[winner] += self.pot
        else:
            winner = 0 if self.player_cards[0] > self.player_cards[1] else 1
            self.stacks[winner] += self.pot

        return self.stacks
