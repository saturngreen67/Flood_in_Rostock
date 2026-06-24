import numpy as np
import torch
import time

ckpt = torch.load('pimlp_best.pt', map_location='cpu', weights_only=False)

from train_pimlp_gpu2 import PhysicsInformedMLP
hparams     = ckpt['best_hparams']
n_layers    = hparams['n_layers']
width_first = hparams['width_first']
taper       = hparams['taper']
hidden_dims = [max(32, int(width_first * (taper ** i))) for i in range(n_layers)]

model = PhysicsInformedMLP(in_dim=14, hidden=hidden_dims)

# Strip torch.compile() prefix
raw_sd   = ckpt['model_state']
clean_sd = {
    (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
    for k, v in raw_sd.items()
}
model.load_state_dict(clean_sd)
model.eval()

# Time inference over 300 scenarios
N = 64768
x = torch.randn(N, 14)

with torch.no_grad():
    for _ in range(5):       # warm-up
        _ = model(x)

start = time.perf_counter()
with torch.no_grad():
    for _ in range(300):
        _ = model(x)
elapsed = time.perf_counter() - start

ms_per_scenario = (elapsed / 300) * 1000
print(f"Inference time : {ms_per_scenario:.2f} ms/scenario")
print(f"Total 300 test : {elapsed*1000:.1f} ms")
print(f"Speedup vs GAT : {80.04 / ms_per_scenario:.1f}×  (GAT = 80.04 ms)")