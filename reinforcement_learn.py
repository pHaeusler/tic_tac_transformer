import torch
from torch.distributions import Categorical
from torch.optim import AdamW
import wandb
from tokens import PAD, SEQ_LENGTH, START
from board_ops import (
    batch_check_winner,
    batch_next_valid_move_from_seq,
    batch_board_full,
    batch_seq_to_board,
    batch_detect_illegal_moves,
)
from setup import load_from_checkpoint, device, save_checkpoint

NUM_EPOCHS = 4000
BATCH_SIZE = 1024
LR = 1e-5
WANDB = True

if WANDB:
    run = wandb.init(
        project="ttt",
        config={
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
        },
    )

model = load_from_checkpoint()
model.eval()
model.to(device)


def compute_rewards(sequences):
    illegal = batch_detect_illegal_moves(sequences[:, 1:])
    valid_sequences = ~illegal
    rewards = torch.full(
        (sequences.shape[0],), 0.5, device=sequences.device, dtype=torch.float
    )
    rewards[illegal] = 0
    if torch.any(valid_sequences):
        boards = batch_seq_to_board(sequences[valid_sequences, 1:])
        winners = torch.zeros(
            sequences.shape[0], dtype=torch.long, device=sequences.device
        )
        full = torch.zeros(
            sequences.shape[0], dtype=torch.bool, device=sequences.device
        )
        winners[valid_sequences] = batch_check_winner(boards)
        full[valid_sequences] = batch_board_full(boards)
        rewards[winners == 1] = 1
        rewards[winners == -1] = 0
        # incomplete games (due to an invalid move)
        incomplete = winners == 0 & ~full
        # print(incomplete)
        rewards[incomplete] = -0.5
    return rewards


optimizer = AdamW(model.parameters(), lr=LR)
input_ids = torch.tensor([START], dtype=torch.long, device=device)[None, ...]

for epoch in range(NUM_EPOCHS):
    model.train()

    output_ids = torch.full(
        (BATCH_SIZE, SEQ_LENGTH), PAD, dtype=torch.long, device=device
    )
    output_ids[:, : input_ids.shape[1]] = input_ids

    log_probs_accumulated = torch.zeros((BATCH_SIZE, 1), device=device)
    entropy_accumulated = torch.zeros((BATCH_SIZE, 1), device=device)

    # keep track of which games (within the batch) have completed
    # when a game is complete there is an PAD token
    # we must stop accumulating for that story
    active_mask = torch.ones(BATCH_SIZE, dtype=torch.bool, device=device)

    for i in range(input_ids.shape[1], SEQ_LENGTH):
        # player 2 - RANDOM MOVE
        if i % 2 == 0:
            next_moves = batch_next_valid_move_from_seq(output_ids[active_mask, 1:])
            output_ids[active_mask, i] = next_moves
            assert torch.all(output_ids[active_mask, i] != PAD)
            illegal = batch_detect_illegal_moves(output_ids[active_mask, 1:])
            assert not torch.any(illegal)

        # player 1 - AI MOVE
        else:
            prompt = output_ids[:, :i].clone()
            logits = model(prompt)[0]

            # Only consider logits of active sequences
            logits_active = logits[active_mask]
            if logits_active.shape[0] == 0:
                # All sequences are finished
                break

            probs = torch.nn.functional.softmax(logits_active, dim=-1)
            entropy_current = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            entropy_accumulated[active_mask] += entropy_current
            dist = Categorical(probs)
            next_tokens = dist.sample()
            log_probs_accumulated[active_mask] += dist.log_prob(next_tokens)
            output_ids[active_mask, i] = next_tokens.T

        invalid_move = output_ids[active_mask, i].squeeze(-1) == PAD
        illegal = invalid_move | batch_detect_illegal_moves(output_ids[active_mask, 1:])

        active_indices = torch.nonzero(active_mask).squeeze(-1)
        active_mask = active_mask.clone()
        active_mask[active_indices] &= ~illegal

        if torch.any(active_mask):
            boards = batch_seq_to_board(output_ids[active_mask, 1:])
            winners = batch_check_winner(boards)
            full = batch_board_full(boards)

            active_indices = torch.nonzero(active_mask).squeeze(-1)
            active_mask = active_mask.clone()
            active_mask[active_indices] &= ~full
            active_mask[active_indices] &= winners == 0

    normalized_log_probs = log_probs_accumulated / SEQ_LENGTH

    # Compute rewards for the entire batch
    with torch.no_grad():
        rewards = compute_rewards(output_ids)

    # Compute loss for the entire batch
    neg_advantage = (-normalized_log_probs * rewards.unsqueeze(-1)).mean()
    alpha = 0.0  # hyperparameter to be tuned
    average_entropy = entropy_accumulated.mean()
    loss = neg_advantage - alpha * average_entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if WANDB:
        wandb.log(
            {
                "loss": loss,
                "reward": rewards.mean(),
            }
        )

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}: Loss: {loss.item()} Rewards: {rewards.mean()} NegAdv: {neg_advantage}"
    )

save_checkpoint(model)
