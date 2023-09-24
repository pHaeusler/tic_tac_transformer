import os
import time

import numpy as np
import torch

from tokens import PAD
from setup import init_model, save_checkpoint

save_interval = 1000

wandb_log = True
wandb_project = "ttt"

batch_size = 2048

learning_rate = 6e-3
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

device = "cuda"

data_dir = "data"
train_data = np.load(os.path.join(data_dir, "train.npy")).astype(dtype=np.int64)


def get_batch():
    data = train_data
    ix = torch.randint(data.shape[0], (batch_size,))
    x = torch.from_numpy(data[ix, :])
    y = torch.roll(x, shifts=-1, dims=1)
    y[:, -1] = PAD
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
        device, non_blocking=True
    )
    return x, y


iter_num = 0

model = init_model()
model.to(device)
model.train()

optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device
)


if wandb_log:
    import wandb

    wandb.init(project=wandb_project)

X, Y = get_batch()
t0 = time.time()
while iter_num < max_iters:
    if iter_num > 0 and iter_num % save_interval == 0:
        save_checkpoint(model)

    logits, loss = model(X, Y)

    loss.backward()

    X, Y = get_batch()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    lossf = loss.item()
    print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    if wandb_log:
        wandb.log(
            {
                "iter": iter_num,
                "train/loss": lossf,
                "lr": learning_rate,
            }
        )

    iter_num += 1
