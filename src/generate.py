# generate.py — load trained weights, sample from the model
from pathlib import Path
import torch
from config import config
from src.model import GPT
from src.tokenizer import CharTokenizer as tkzr

tok = tkzr.load(Path("data/processed/meta.json"))
config.vocab_size = tok.vocab_size

model = GPT(config)
model.load_state_dict(torch.load(Path("checkpoints/model_best.pt"), map_location=config.device))
model.to(config.device)
model.eval()

# starting context — encode a prompt, or a single zero token for unconditioned sampling
context = torch.tensor([tok.encode("To my friend, Alex, what will BrananoGpy say?")], dtype=torch.long, device=config.device)


out = model.generate(context, max_new_tokens=500, temperature=1.0)
print(f"\n{'='*60}\ntemperature = 1.0 and model = model_best.pt\n{'='*60}")
print(tok.decode(out[0].tolist()))

model.load_state_dict(torch.load(Path("model.pt"), map_location=config.device))
out = model.generate(context, max_new_tokens=500, temperature=1.0)
print(f"\n{'='*60}\ntemperature = 1.0 and model = overfit_model.pt\n{'='*60}")
print(tok.decode(out[0].tolist()))