# train.py — the loop that pushes 4.61 down
import torch
from config import config
from src.dataset import Dataset
from src.model import GPT


ds = Dataset(config)
model = GPT(config).to(config.device)
optim = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

@torch.no_grad()
def estimate_loss(model, ds):
    out = {}

    model.eval()
    for split in ("train","val"):
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            xb, yb = ds.get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


# This is the main eval loop
best_val = float("inf") 
for step in range(config.max_iters):
    if step % config.eval_interval == 0:
        losses = estimate_loss(model, ds)
        print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save(model.state_dict(), "checkpoints/model_best.pt")                 

    xb, yb = ds.get_batch("train")
    _, loss = model(xb,yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()



torch.save(model.state_dict(), "model.pt")