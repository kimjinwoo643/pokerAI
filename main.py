import tkinter as tk
import DQN
from KuhnPoker import KuhnPoker
from players import Player, RandomPlayer, HumanPlayer, DEEPQPlayer

def simulate_game(player1, player2, rounds=20):
    game = KuhnPoker()
    players = [player1, player2]

    for round_number in range(1, rounds + 1):
        print(f"\n--- Round {round_number} ---")
        game.reset_round()
        game.deal_cards()

        while not game.is_terminal():
            current_player = players[game.current_player]
            card = game.player_cards[game.current_player]
            legal_actions = game.get_legal_actions()

            # Display the current game state for debugging
            #print(f"\n{current_player.name}'s turn. Their card: {card}")
            action = current_player.choose_action(legal_actions, card, game)
            print(f"{current_player.name} chooses to {action}.")
            game.play_action(action)

        stacks = game.get_payoff()
        print(f"Player 1 card: {game.player_cards[0]}, Player 2 card: {game.player_cards[1]}")
        print(f"Stacks after round: Player 1: {stacks[0]}, Player 2: {stacks[1]}")

        if stacks[0] <= 0 or stacks[1] <= 0:
            print("\nA player's stack has reached 0. Ending the game early.")
            break

    print("\n--- Final Results ---")
    print(f"Player 1 Stack: {game.stacks[0]}")
    print(f"Player 2 Stack: {game.stacks[1]}")
    print(f"Winner: {'Player 1' if game.stacks[0] > game.stacks[1] else 'Player 2'}")
    return 0 if game.stacks[0] > game.stacks[1] else 1


if __name__ == "__main__":
    player2 = RandomPlayer("Random Player 1")
    player1 = DEEPQPlayer("AI Player 2")
    # # simulate_game(player1, player2)
    # iters = 1000
    # winrate = [simulate_game(player1, player2) for i in range(iters)].count(1) / float(iters)
    # print(f"The AI had a {winrate*100}% winrate")
