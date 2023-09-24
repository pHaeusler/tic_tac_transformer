import torch
from tokens import PAD
import numpy as np


def winning_moves(board, player):
    moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                board[i][j] = player
                if check_winner(board) == player:
                    moves.append((i, j))
                board[i][j] = 0
    return moves


def board_full(board):
    return 0 not in board


def check_winner(board):
    for player in [-1, 1]:
        for i in range(3):
            if all(board[i, :] == player) or all(board[:, i] == player):
                return player
        if all(np.diag(board) == player) or all(np.diag(np.fliplr(board)) == player):
            return player
    return None


def optimal_moves(board, player):
    # Winning moves
    moves = winning_moves(board, player)
    if moves:
        return moves

    # Blocking moves
    moves = winning_moves(board, -player)
    if moves:
        return moves

    # Center move
    if board[1][1] == 0:
        return [(1, 1)]

    # Corner moves
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    return [corner for corner in corners if board[corner[0]][corner[1]] == 0]


def get_valid_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]


def batch_board_full(boards):
    return ~(boards == 0).any(dim=2).any(dim=1)


def batch_check_winner(boards):
    players = torch.tensor([-1, 1], device=boards.device)

    # Sum along dimensions
    row_sums = boards.sum(dim=2)
    col_sums = boards.sum(dim=1)
    diag1_sums = boards[:, torch.arange(3), torch.arange(3)].sum(dim=1)
    diag2_sums = boards[:, torch.arange(3), torch.arange(2, -1, -1)].sum(dim=1)

    # Check for winning conditions
    wins = torch.cat(
        [row_sums, col_sums, diag1_sums.unsqueeze(1), diag2_sums.unsqueeze(1)], dim=1
    )

    matches = (wins.unsqueeze(-1) == (3 * players)).any(dim=1)

    # Get winners based on conditions
    winners = torch.where(
        matches[:, 1], players[1], torch.where(matches[:, 0], players[0], 0)
    )

    return winners


def batch_detect_illegal_moves(batch_seq, pad_token=PAD):
    batch_size, seq_len = batch_seq.shape

    # Create masks for valid moves (0-8) and pad tokens
    valid_moves = (batch_seq >= 0) & (batch_seq < 9)
    pad_mask = batch_seq == pad_token

    # Ensure all tokens in sequences are either valid moves or PAD
    valid_seq = valid_moves | pad_mask

    # Create safe indices to avoid out-of-bounds during scatter
    safe_batch_seq = torch.where(valid_moves, batch_seq, torch.zeros_like(batch_seq))

    # Convert sequences to one-hot encoding to detect repeated moves
    one_hot_moves = torch.zeros(
        batch_size, seq_len, 9, dtype=torch.float, device=batch_seq.device
    )
    one_hot_moves.scatter_(
        2, safe_batch_seq.unsqueeze(-1), valid_moves.float().unsqueeze(-1)
    )

    # Sum along the sequence length dimension to count occurrences of each move
    move_counts = one_hot_moves.sum(dim=1)

    # Moves that are repeated (count > 1) are illegal
    repeated_moves = move_counts > 1

    # Also, ensure all tokens before PAD tokens in a sequence are valid moves
    no_invalid_before_pad = ~pad_mask | valid_moves.cumsum(dim=1).bool()

    # Final validity check for each sequence in the batch
    valid_sequences = (
        valid_seq.all(dim=1)
        & ~repeated_moves.any(dim=1)
        & no_invalid_before_pad.all(dim=1)
    )

    return ~valid_sequences


def batch_seq_to_board(batch_seq, pad_token=PAD):
    batch_size, seq_len = batch_seq.shape
    one_hot = torch.zeros(batch_size, seq_len, 9, device=batch_seq.device)

    # Mask to identify non-padding tokens
    valid_mask = batch_seq != pad_token

    # Mask to identify valid move tokens
    move_mask = (batch_seq >= 0) & (batch_seq <= 8)

    # Combine masks to identify valid non-padding move tokens
    combined_mask = valid_mask & move_mask

    # Adjust the sequence length based on non-padding tokens
    valid_lengths = combined_mask.sum(dim=1)

    # Create one-hot encoded tensor for each valid sequence entry in the batch
    br = torch.arange(batch_size, device=batch_seq.device).unsqueeze(1)
    sr = torch.arange(seq_len, device=batch_seq.device)

    safe_batch_seq = torch.where(combined_mask, batch_seq, torch.zeros_like(batch_seq))

    one_hot[
        br,
        sr,
        safe_batch_seq,
    ] = combined_mask.float()

    # Generate player matrix [-1, 1, -1, 1, ...]
    max_players = torch.tensor([1, -1] * (seq_len // 2 + 1), device=batch_seq.device)
    players = [max_players[:seq_len].view(seq_len, 1) for _ in valid_lengths]
    players_padded = torch.nn.utils.rnn.pad_sequence(players, batch_first=True)

    # Matrix multiplication
    one_hot = one_hot.float()
    players_padded = players_padded.float()
    batch_board = torch.bmm(one_hot.transpose(1, 2), players_padded).squeeze(2)

    return batch_board.view(batch_size, 3, 3)


def batch_next_valid_move_from_seq(batch_seq):
    all_moves = torch.arange(9, device=batch_seq.device)
    available_moves = [set(all_moves.tolist()) - set(seq.tolist()) for seq in batch_seq]
    next_moves = [
        list(moves)[torch.randint(0, len(moves), (1,)).item()] if moves else -1
        for moves in available_moves
    ]
    return torch.tensor(next_moves, device=batch_seq.device)
