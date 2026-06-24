# =============================================================================
#  attention_entropy.py
#
#  Extracts attention weights from ALL four GATConv layers for a small set of
#  test scenarios and produces two publication-quality figures:
#
#  Figure A — 11_attention_entropy_bars.png
#      Bar chart of mean attention entropy per layer, averaged across scenarios.
#      Reference line at log(n_neighbours) = maximum-entropy (uniform) baseline.
#      If all bars are near the reference → uniform attention confirmed.
#      If early bars are lower → earlier layers are more selective.
#
#  Figure B — 12_attention_entropy_maps.png  (one row per scenario)
#      Spatial heatmaps of per-node entropy at each layer side-by-side.
#      Low entropy (dark) = node receives highly concentrated attention.
#      High entropy (bright) = uniform/diffuse attention.
#
#  Requires:
#    gat_results/gat_best.pt
#    scenarios/graph_static.npz
#    scenarios/scenario_?????.npz
#
#  Run from the same directory as scenarios/ and gat_results/:
#    python attention_entropy.py
# =============================================================================

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Config ────────────────────────────────────────────────────────────────────
SCENARIOS_DIR  = "scenarios"
OUTPUT_DIR     = "gat_results"
BEST_PT        = os.path.join(OUTPUT_DIR, "gat_best.pt")
N_SCENARIOS    = 5      # number of test scenarios to analyse (3–5 is enough)
SEED           = 42
TRAIN_FRAC     = 0.70
VAL_FRAC       = 0.15
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── FloodGAT (must be identical to training definition) ──────────────────────
class FloodGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, heads, dropout):
        super().__init__()
        self.num_layers  = num_layers
        self.input_proj  = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.ELU())
        self.gat_layers  = nn.ModuleList([
            GATConv(hidden_channels, hidden_channels, heads=heads,
                    concat=False, dropout=dropout,
                    add_self_loops=True, edge_dim=1)
            for _ in range(num_layers)])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.dropout     = dropout
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2), nn.ELU(),
            nn.Linear(hidden_channels // 2, 1), nn.Softplus())

    def forward(self, x, edge_index, edge_attr):
        x = self.input_proj(x)
        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = norm(x); x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return self.output_head(x).squeeze(-1)

    def forward_with_all_attention(self, x, edge_index, edge_attr):
        """
        Same as forward() but returns a list of attention tensors,
        one per GATConv layer.  Each tensor has shape (E, heads).
        """
        x = self.input_proj(x)
        all_alpha = []
        for j, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
            residual = x
            # return_attention_weights=True returns (x_new, (edge_index, alpha))
            x_new, (ei, alpha) = gat(x, edge_index, edge_attr=edge_attr,
                                     return_attention_weights=True)
            x_new = norm(x_new); x_new = F.elu(x_new)
            x_new = x_new + residual
            x = x_new
            all_alpha.append(alpha)   # (E, heads)
        return self.output_head(x).squeeze(-1), all_alpha, ei

# ── Load checkpoint ───────────────────────────────────────────────────────────
print(f"Loading {BEST_PT} …")
ckpt     = torch.load(BEST_PT, map_location=DEVICE, weights_only=False)
feat_mean = ckpt["feat_mean"]
feat_std  = ckpt["feat_std"]

model = FloodGAT(
    in_channels     = 14,
    hidden_channels = ckpt["hidden_channels"],
    num_layers      = ckpt["num_layers"],
    heads           = ckpt["heads"],
    dropout         = ckpt["dropout"],
).to(DEVICE)

# Strip torch.compile() _orig_mod prefix if present
raw_sd   = ckpt["model_state"]
clean_sd = {(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in raw_sd.items()}
model.load_state_dict(clean_sd)
model.eval()

n_layers = ckpt["num_layers"]
n_heads  = ckpt["heads"]
print(f"  Model: {n_layers} layers, {n_heads} heads, "
      f"hidden={ckpt['hidden_channels']}")

# ── Load static graph ─────────────────────────────────────────────────────────
print("Loading graph …")
g  = np.load(os.path.join(SCENARIOS_DIR, "graph_static.npz"))
H, W = int(g["H"]), int(g["W"])
N    = H * W

static_arr = np.stack([
    g["elevation"].flatten(),
    g["street_mask"].flatten(),
    g["building_mask"].flatten(),
    g["existing_gi"].flatten(),
    g["dist_to_street"].flatten(),
    g["dist_to_outlet"].flatten(),
    g["upstream_area"].flatten(),
    g["water_mask_sink"].flatten(),
], axis=1).astype(np.float32)

edge_index = torch.from_numpy(g["edge_index"].astype(np.int64)).to(DEVICE)
RES = 3.0
elev_flat = static_arr[:, 0]
src_np    = edge_index[0].cpu().numpy()
dst_np    = edge_index[1].cpu().numpy()
dz        = elev_flat[src_np] - elev_flat[dst_np]
dist_m    = (g["edge_distances"].astype(np.float32) * RES
             if "edge_distances" in g
             else np.full(len(src_np), RES, dtype=np.float32))
slope     = dz / dist_m
slope_norm = (slope - slope.mean()) / (slope.std() + 1e-8)
edge_attr  = torch.from_numpy(slope_norm).view(-1, 1).float().to(DEVICE)

# ── Reproduce the same test split as training ─────────────────────────────────
paths  = sorted(glob.glob(os.path.join(SCENARIOS_DIR, "scenario_?????.npz")))
rng    = np.random.default_rng(SEED)
perm   = rng.permutation(len(paths))
n_tr   = int(len(paths) * TRAIN_FRAC)
n_val  = int(len(paths) * VAL_FRAC)
te_paths = [paths[i] for i in perm[n_tr + n_val:]]
print(f"  Test scenarios available : {len(te_paths)}")

# ── Helper: compute per-node entropy from attention weights ───────────────────
def attention_entropy(alpha, dst_nodes, N):
    """
    alpha     : (E,) — mean across heads, already normalised by softmax
    dst_nodes : (E,) — destination node index for each edge
    N         : total number of nodes

    For each destination node i, alpha values from its neighbours form a
    probability distribution.  Entropy = -sum(p * log(p)).
    Returns (N,) array of per-node attention entropy.
    """
    eps   = 1e-10
    # log-entropy contribution per edge
    h_per_edge = -alpha * torch.log(alpha + eps)  # (E,)
    # sum per destination node
    H_node = torch.zeros(N, device=alpha.device)
    H_node = H_node.scatter_add(0, dst_nodes, h_per_edge)
    return H_node  # (N,)

# ── Collect entropy maps for N_SCENARIOS test scenarios ───────────────────────
# Shape: (N_SCENARIOS, n_layers, N)
all_entropy_maps = []
scenario_labels  = []

print(f"\nExtracting attention from {N_SCENARIOS} test scenarios …")
for i, path in enumerate(te_paths[:N_SCENARIOS]):
    d     = np.load(path)
    imp   = d["imperviousness"].flatten().astype(np.float32)
    mann  = d["mannings_n"].flatten().astype(np.float32)
    gi    = d["gi_map"].flatten().astype(np.float32)
    inten = float(d["intensity"])
    dur   = float(d["duration"])
    adopt = float(d["adoption_level"])

    dyn = np.stack([
        imp, mann, gi,
        np.full(N, inten,  dtype=np.float32),
        np.full(N, dur,    dtype=np.float32),
        np.full(N, adopt,  dtype=np.float32),
    ], axis=1)
    x_raw  = np.concatenate([static_arr, dyn], axis=1).astype(np.float32)
    x_norm = (x_raw - feat_mean) / feat_std
    x_t    = torch.from_numpy(x_norm).to(DEVICE)

    with torch.no_grad():
        _, all_alpha, ei = model.forward_with_all_attention(
            x_t, edge_index, edge_attr)

    dst_nodes = ei[1]   # destination node indices, shape (E,)

    layer_entropy_maps = []
    for layer_idx, alpha in enumerate(all_alpha):
        # alpha: (E, heads) — average over heads
        alpha_mean = alpha.mean(dim=1)   # (E,)
        H_node     = attention_entropy(alpha_mean, dst_nodes, N)
        layer_entropy_maps.append(H_node.cpu().numpy().reshape(H, W))

    all_entropy_maps.append(layer_entropy_maps)
    scenario_labels.append(
        f"Sc {i+1}\n$I$={inten:.1f} mm/h\n$D$={dur:.0f} min")
    print(f"  Scenario {i+1}: I={inten:.1f} mm/h  D={dur:.0f} min  "
          f"φ={adopt:.2f}")

all_entropy_maps = np.array(all_entropy_maps)  # (N_SCENARIOS, n_layers, H, W)

# ── Maximum-entropy reference (uniform over 8 neighbours + self = 9) ─────────
# Each node has at most 8 neighbours + itself (self-loops enabled)
# Uniform distribution over 9 entries: H_max = log(9)
H_max = float(np.log(9))

# ── Mean entropy per layer (averaged over all nodes and scenarios) ─────────────
mean_entropy_per_layer = all_entropy_maps.mean(axis=(0, 2, 3))  # (n_layers,)
std_entropy_per_layer  = all_entropy_maps.std(axis=(0, 2, 3))

print(f"\nMean attention entropy per layer (max possible = log(9) = {H_max:.3f}):")
for l, (mu, sd) in enumerate(zip(mean_entropy_per_layer, std_entropy_per_layer)):
    pct = 100 * mu / H_max
    print(f"  Layer {l+1}: {mu:.4f} ± {sd:.4f}  ({pct:.1f}% of maximum)")

# =============================================================================
#  FIGURE A — Bar chart: entropy per layer
# =============================================================================
fig_a, ax = plt.subplots(figsize=(6, 4))

x_pos  = np.arange(1, n_layers + 1)
colors = plt.cm.Blues(np.linspace(0.4, 0.85, n_layers))

bars = ax.bar(x_pos, mean_entropy_per_layer, yerr=std_entropy_per_layer,
              color=colors, width=0.55, capsize=5,
              error_kw={"elinewidth": 1.2, "ecolor": "grey"})

ax.axhline(H_max, color="tomato", lw=1.5, ls="--",
           label=f"Uniform baseline  $H_{{max}} = \\log(9) = {H_max:.2f}$")

# Annotate each bar with % of max
for bar, mu in zip(bars, mean_entropy_per_layer):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{100*mu/H_max:.1f}%",
            ha="center", va="bottom", fontsize=9)

ax.set_xlabel("GAT Layer", fontsize=11)
ax.set_ylabel("Mean Attention Entropy  $H = -\\sum p \\log p$", fontsize=11)
ax.set_title("Per-Layer Attention Entropy\n"
             "(error bars = std across nodes and scenarios)", fontsize=11)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"Layer {l}" for l in x_pos])
ax.set_ylim(0, H_max * 1.15)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

out_a = os.path.join(OUTPUT_DIR, "11_attention_entropy_bars.png")
fig_a.tight_layout()
fig_a.savefig(out_a, dpi=150, bbox_inches="tight")
plt.close(fig_a)
print(f"\nSaved → {out_a}")

# =============================================================================
#  FIGURE B — Spatial entropy maps: rows = scenarios, cols = layers
# =============================================================================
fig_b, axes = plt.subplots(
    N_SCENARIOS, n_layers,
    figsize=(3.5 * n_layers, 3.2 * N_SCENARIOS),
    squeeze=False
)

# Use a shared colour scale across all maps for comparability
vmin = all_entropy_maps.min()
vmax = all_entropy_maps.max()

for sc_idx in range(N_SCENARIOS):
    for l_idx in range(n_layers):
        ax = axes[sc_idx][l_idx]
        im = ax.imshow(all_entropy_maps[sc_idx, l_idx],
                       cmap="viridis", origin="upper",
                       vmin=vmin, vmax=vmax)
        if sc_idx == 0:
            ax.set_title(f"Layer {l_idx + 1}", fontsize=10, fontweight="bold")
        if l_idx == 0:
            ax.set_ylabel(scenario_labels[sc_idx], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

# Shared colorbar
fig_b.subplots_adjust(right=0.88, hspace=0.08, wspace=0.05)
cbar_ax = fig_b.add_axes([0.90, 0.15, 0.02, 0.70])
sm = plt.cm.ScalarMappable(cmap="viridis",
                            norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
fig_b.colorbar(sm, cax=cbar_ax, label="Attention Entropy  $H$")

fig_b.suptitle("Spatial Attention Entropy Maps — All Layers",
               fontsize=12, fontweight="bold", y=1.01)

out_b = os.path.join(OUTPUT_DIR, "12_attention_entropy_maps.png")
fig_b.savefig(out_b, dpi=150, bbox_inches="tight")
plt.close(fig_b)
print(f"Saved → {out_b}")

# =============================================================================
#  FIGURE C — Single summary plot for the paper (compact, publication-ready)
#  Left: entropy bar chart.  Right: spatial map of layer 1 vs layer 4 for
#  the highest-intensity scenario — directly comparable on one figure.
# =============================================================================
high_sc = int(np.argmax([
    float(np.load(te_paths[i])["intensity"])
    for i in range(N_SCENARIOS)
]))

fig_c, axes_c = plt.subplots(1, 4, figsize=(14, 3.8),
                              gridspec_kw={"width_ratios": [1.6, 1, 1, 1]})

# Left: bar chart
ax_bar = axes_c[0]
ax_bar.bar(x_pos, mean_entropy_per_layer, yerr=std_entropy_per_layer,
           color=colors, width=0.55, capsize=4,
           error_kw={"elinewidth": 1.0, "ecolor": "grey"})
ax_bar.axhline(H_max, color="tomato", lw=1.4, ls="--",
               label=f"$H_{{max}}={H_max:.2f}$")
for bar, mu in zip(bars, mean_entropy_per_layer):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{100*mu/H_max:.1f}%",
                ha="center", va="bottom", fontsize=8)
ax_bar.set_xlabel("GAT Layer", fontsize=10)
ax_bar.set_ylabel("Mean Entropy $H$", fontsize=10)
ax_bar.set_title("Entropy per Layer\n(all scenarios)", fontsize=9)
ax_bar.set_xticks(x_pos); ax_bar.set_xticklabels([f"L{l}" for l in x_pos])
ax_bar.set_ylim(0, H_max * 1.15)
ax_bar.legend(fontsize=8)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

# Right: spatial entropy maps for layers 1, 2, 3, 4 of the high-intensity sc
inten_val = float(np.load(te_paths[high_sc])["intensity"])
dur_val   = float(np.load(te_paths[high_sc])["duration"])

for col, l_idx in enumerate(range(n_layers)):
    ax_m = axes_c[col + 1] if col < 3 else None
    if col >= 3:
        break
    ax_m = axes_c[col + 1]
    data = all_entropy_maps[high_sc, l_idx]
    im = ax_m.imshow(data, cmap="viridis", origin="upper",
                     vmin=vmin, vmax=vmax)
    ax_m.set_title(f"Layer {l_idx+1} spatial entropy", fontsize=9)
    ax_m.set_xticks([]); ax_m.set_yticks([])

# Add colorbar on the far right
fig_c.subplots_adjust(right=0.87, wspace=0.12)
cbar_ax_c = fig_c.add_axes([0.89, 0.15, 0.015, 0.70])
fig_c.colorbar(sm, cax=cbar_ax_c, label="$H$")

fig_c.suptitle(
    f"GAT Attention Entropy — "
    f"High-Intensity Scenario ($I={inten_val:.1f}$ mm/h, $D={dur_val:.0f}$ min)",
    fontsize=11, fontweight="bold", y=1.03)

out_c = os.path.join(OUTPUT_DIR, "13_attention_entropy_summary.png")
fig_c.savefig(out_c, dpi=150, bbox_inches="tight")
plt.close(fig_c)
print(f"Saved → {out_c}")

print("\nDone. Outputs:")
print(f"  {out_a}  — bar chart (entropy per layer, all scenarios)")
print(f"  {out_b}  — spatial maps (all scenarios × all layers)")
print(f"  {out_c}  — compact summary figure for the paper")
