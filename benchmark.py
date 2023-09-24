import torch
from tokens import START
from board_ops import check_winner, board_full, get_valid_moves
import numpy as np
from setup import load_from_checkpoint, device
import random


model = load_from_checkpoint()
model.eval()
model.to(device)

with torch.no_grad():
    counts = {"player_2": 0, "player_1": 0, "draw": 0, "invalid": 0}
    for _ in range(1000):
        board = np.zeros((3, 3), dtype=int)
        player = 1
        winner = None
        moves = [START]
        while winner is None and not board_full(board):
            if player == 1:
                x = torch.tensor(moves, dtype=torch.long, device=device)[None, ...]
                y = model.generate(x, max_new_tokens=1, temperature=1.0, top_k=3)
                y = y[0][-1].item()

                if y not in set(range(9)) or y in moves:
                    print(f"invalid move: {y} moves: {moves}")
                    winner = None
                    break

                i, j = divmod(y, 3)
            else:
                i, j = random.choice(get_valid_moves(board))

            moves.append(i * 3 + j)
            board[i][j] = player
            player *= -1
            winner = check_winner(board)

        if winner == 1:
            counts["player_1"] += 1
        elif winner == -1:
            counts["player_2"] += 1
        elif board_full(board):
            counts["draw"] += 1
        else:
            counts["invalid"] += 1

    print(counts)
    print(counts["player_1"] / sum(counts.values()))
