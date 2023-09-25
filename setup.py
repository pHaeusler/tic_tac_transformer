import torch
import os
from tokens import VOCAB_SIZE, SEQ_LENGTH
from model import GPTConfig, GPT


out_dir = "out"
seed = 1337
device = "cuda"

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def init_model(incl_winner: bool = False):
    print("Initializing a new model from scratch")
    config = GPTConfig(
        block_size=SEQ_LENGTH + 1 if incl_winner else SEQ_LENGTH,
        vocab_size=VOCAB_SIZE,
        n_layer=1,
        n_head=1,
        n_embd=14,
        dropout=0.0,
        bias=False,
    )
    print("config:", config)
    return GPT(config)


def load_from_checkpoint():
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = GPT(checkpoint["config"])
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    return model


def save_checkpoint(model):
    os.makedirs(out_dir, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": model.state_dict(),
        "config": model.config,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
