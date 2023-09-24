import numpy as np
import copy
from collections import defaultdict

from tokens import PAD, SEQ_LENGTH, START
from board_ops import check_winner, board_full, optimal_moves, get_valid_moves


def all_trajectories(board, seq, player):
    winner = check_winner(board)
    if winner is not None or board_full(board):
        return [(board.copy(), copy.copy(seq), winner)]
    trajectories = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = player
                seq.append((i * 3 + j))
                trajectories += all_trajectories(board, seq, -player)
                board[i][j] = 0
                seq.pop()
    return trajectories


def all_optimal_trajectories(board, seq, player, optimal_player={1}):
    winner = check_winner(board)
    if winner is not None or board_full(board):
        return [(board.copy(), copy.copy(seq), winner)]
    trajectories = []
    moves = (
        optimal_moves(board, player)
        if player in optimal_player
        else get_valid_moves(board)
    )
    if not moves:
        moves = get_valid_moves(board)
    for i, j in moves:
        board[i][j] = player
        seq.append((i * 3 + j))
        trajectories += all_optimal_trajectories(board, seq, -player)
        board[i][j] = 0
        seq.pop()
    return trajectories


def seq_to_board(seq):
    board = np.zeros((3, 3), dtype=int)
    player = 1
    for s in seq:
        i, j = s // 3, s % 3
        board[i][j] = player
        player = -player
    return board


def save_data(trajectories):
    outcomes = defaultdict(int)
    for b, s, w in trajectories:
        outcomes[w] += 1

    print(outcomes)

    data = np.full((len(trajectories), SEQ_LENGTH), PAD, dtype=np.int16)
    for i in range(len(trajectories)):
        row = [START] + trajectories[i][1]
        if i < 10:
            print(row)
        data[i, : len(row)] = row

    np.random.shuffle(data)

    np.save("data/train.npy", data)


if __name__ == "__main__":
    board = np.zeros((3, 3), dtype=int)
    # trajectories = all_trajectories(board, [], 1)
    # trajectories = all_optimal_trajectories(board, [], 1, {-1, 1})
    trajectories = all_optimal_trajectories(board, [], 1)
    print(len(trajectories))
    save_data(trajectories)
