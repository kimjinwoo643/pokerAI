import KuhnPoker
import tkinter as tk
class KuhnPokerGUI:
    def __init__(self, root):
        self.game = KuhnPoker.KuhnPoker()
        self.game.deal_cards()

        # Setup GUI
        self.root = root
        self.root.title("Kuhn Poker")

        self.info_label = tk.Label(root, text="Player 0's Turn", font=("Arial", 16))
        self.info_label.pack(pady=10)

        self.action_label = tk.Label(root, text="Actions: bet, check", font=("Arial", 14))
        self.action_label.pack(pady=10)

        self.history_label = tk.Label(root, text="History: ", font=("Arial", 14))
        self.history_label.pack(pady=10)

        self.pot_label = tk.Label(root, text="Pot: 0", font=("Arial", 14))
        self.pot_label.pack(pady=10)

        self.cards_label = tk.Label(root, text="Your card: ?", font=("Arial", 14))
        self.cards_label.pack(pady=10)

        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack()

        self.bet_button = tk.Button(self.buttons_frame, text="Bet", command=lambda: self.play("bet"))
        self.bet_button.pack(side=tk.LEFT, padx=10)

        self.check_button = tk.Button(self.buttons_frame, text="Check", command=lambda: self.play("check"))
        self.check_button.pack(side=tk.LEFT, padx=10)

        self.call_button = tk.Button(self.buttons_frame, text="Call", command=lambda: self.play("call"))
        self.call_button.pack(side=tk.LEFT, padx=10)

        self.fold_button = tk.Button(self.buttons_frame, text="Fold", command=lambda: self.play("fold"))
        self.fold_button.pack(side=tk.LEFT, padx=10)

        self.reset_button = tk.Button(root, text="New Game", command=self.reset_game)
        self.reset_button.pack(pady=10)

        self.update_ui()

    def play(self, action):
        if action in self.game.get_actions():
            self.game.play(action)
            self.update_ui()
            if self.game.is_terminal():
                self.end_game()

    def update_ui(self):
        self.action_label.config(text=f"Actions: {', '.join(self.game.get_actions())}")
        self.history_label.config(text=f"History: {self.game.history}")
        self.pot_label.config(text=f"Pot: {self.game.pot}")
        player_card = self.game.player_cards[self.game.current_player]
        self.cards_label.config(text=f"Your card: {player_card}")
        self.info_label.config(text=f"Player {self.game.current_player}'s Turn")

        if self.game.is_terminal():
            self.end_game()

    def end_game(self):
        payoff = self.game.get_payoff()
        result_text = f"Game Over! Player {self.game.current_player} wins {abs(payoff)}" \
                      f" point{'s' if abs(payoff) > 1 else ''}!" if payoff > 0 else \
            f"Game Over! Player {1 - self.game.current_player} wins {abs(payoff)} point{'s' if abs(payoff) > 1 else ''}!"
        self.info_label.config(text=result_text)
        self.disable_buttons()

    def disable_buttons(self):
        self.bet_button.config(state=tk.DISABLED)
        self.check_button.config(state=tk.DISABLED)
        self.call_button.config(state=tk.DISABLED)
        self.fold_button.config(state=tk.DISABLED)

    def reset_game(self):
        self.game.reset()
        self.game.deal_cards()
        self.bet_button.config(state=tk.NORMAL)
        self.check_button.config(state=tk.NORMAL)
        self.call_button.config(state=tk.NORMAL)
        self.fold_button.config(state=tk.NORMAL)
        self.update_ui()
