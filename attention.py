# =============================================================================
#  run_attention.py
#
#  Standalone script: loads gat_best.pt and generates attention maps.
#  Requires:
#    - scenarios/graph_static.npz
#    - scenarios/scenario_?????.npz  (the 2000 scenario files)
#    - gat_results/gat_best.pt       (from the completed training run)
#
#  Output:
#    - gat_results/10_attention_sc01.png
#    - gat_results/10_attention_sc02.png
#    - gat_results/10_attention_sc03.png
#
#  Run from the same directory as the scenarios/ and gat_results/ folders:
#    python run_attention.py
# =============================================================================

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn   import GATConv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
#  Configuration — must match your training run
# =============================================================================
SCENARIOS_DIR = "scenarios"
OUTPUT_DIR    = "gat_results"
BEST_PT       = os.path.join(OUTPUT_DIR, "gat_best.pt")
SEED          = 42
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
N_ATTENTION   = 3      # number of test scenarios to visualise
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
#  FloodGAT model  (must be identical to training definition)
# =============================================================================
class FloodGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, heads, dropout):
        super().__init__()
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ELU(),
        )
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels  = hidden_channels,
                out_channels = hidden_channels,
                heads        = heads,
                concat       = False,
                dropout      = dropout,
                add_self_loops = True,
                edge_dim     = 1,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        self.dropout    = dropout
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.input_proj(x)
        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual
        return self.output_head(x).squeeze(-1)

# =============================================================================
#  Normaliser
# =============================================================================
class Normalizer:
    def __init__(self):
        self.mean = None
        self.std  = None

# =============================================================================
#  Single-scenario dataset item builder
# =============================================================================
def load_scenario(path, static_arr, edge_index, edge_attr, feat_mean, feat_std):
    """Build a PyG Data object for one scenario."""
    d     = np.load(path)
    N     = static_arr.shape[0]
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

    x_raw = np.concatenate([static_arr, dyn], axis=1).astype(np.float32)
    x_norm = (x_raw - feat_mean) / feat_std

    depth = d["flood_depth"].flatten().astype(np.float32) \
            if "flood_depth" in d else d["depth"].flatten().astype(np.float32)

    data = Data(
        x         = torch.from_numpy(x_norm),
        x_raw     = torch.from_numpy(x_raw),
        edge_index = edge_index,
        edge_attr  = edge_attr,
        y          = torch.from_numpy(depth),
        inten      = torch.full((N,), inten),
        dur        = torch.full((N,), dur),
        imp        = torch.from_numpy(imp),
    )
    return data

# =============================================================================
#  Attention visualisation
# =============================================================================
def visualise_attention(model, te_paths, static_arr, edge_index, edge_attr,
                        feat_mean, feat_std, H, W, out_dir, n_scenarios=3):
    print(f"\n  Computing attention maps for {n_scenarios} test scenarios…")
    model.eval()

    for i, path in enumerate(te_paths[:n_scenarios]):
        data = load_scenario(path, static_arr, edge_index, edge_attr,
                             feat_mean, feat_std).to(DEVICE)

        with torch.no_grad():
            x = model.input_proj(data.x)
            for j, (gat, norm) in enumerate(zip(model.gat_layers, model.norms)):
                is_last = (j == model.num_layers - 1)
                if is_last:
                    x_new, (ei, alpha) = gat(
                        x, data.edge_index,
                        edge_attr=data.edge_attr,
                        return_attention_weights=True)
                else:
                    x_new = gat(x, data.edge_index, edge_attr=data.edge_attr)
                residual = x
                x_new    = norm(x_new)
                x_new    = F.elu(x_new)
                x_new    = x_new + residual
                x        = x_new

        alpha_mean = alpha.mean(dim=1).cpu().numpy()
        dst_nodes  = ei[1].cpu().numpy()
        attn_map   = np.zeros(H * W, dtype=np.float32)
        np.add.at(attn_map, dst_nodes, alpha_mean)
        attn_map   = attn_map.reshape(H, W)

        inten_val = float(data.inten[0].cpu())
        dur_val   = float(data.dur[0].cpu())

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        im0 = axes[0].imshow(data.y.cpu().numpy().reshape(H, W),
                             cmap="Blues", origin="upper")
        fig.colorbar(im0, ax=axes[0], label="Simulated Depth (m)")
        axes[0].set_title(
            f"Simulated Flood Depth\n"
            f"intensity={inten_val:.1f} mm/h  duration={dur_val:.0f} min",
            fontsize=10)

        im1 = axes[1].imshow(attn_map, cmap="hot_r", origin="upper")
        fig.colorbar(im1, ax=axes[1], label="Σ attention received")
        axes[1].set_title(
            "Last-Layer Attention\n"
            "(high = neighbours focus here during message passing)",
            fontsize=10)

        for ax in axes:
            ax.set_xlabel("Column (W→E)")
            ax.set_ylabel("Row (N→S)")
        fig.suptitle(f"GAT Attention Map — Test Scenario {i+1}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"10_attention_sc{i+1:02d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out_path}")

    print("  Done.")

# =============================================================================
#  Main
# =============================================================================
if __name__ == "__main__":
    print(f"Device : {DEVICE}")

    # ── Check checkpoint exists ───────────────────────────────────────────────
    if not os.path.exists(BEST_PT):
        raise FileNotFoundError(
            f"Checkpoint not found: {BEST_PT}\n"
            f"Run the full training script first to produce gat_best.pt."
        )

    # ── Load static graph ─────────────────────────────────────────────────────
    print("Loading graph_static.npz …")
    g  = np.load(os.path.join(SCENARIOS_DIR, "graph_static.npz"))
    H, W = int(g["H"]), int(g["W"])

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

    edge_index = torch.from_numpy(g["edge_index"].astype(np.int64))

    RES       = 3.0
    elev_flat = static_arr[:, 0]
    src_np    = edge_index[0].numpy()
    dst_np    = edge_index[1].numpy()
    dz        = elev_flat[src_np] - elev_flat[dst_np]

    if "edge_distances" in g:
        dist_m = g["edge_distances"].astype(np.float32) * RES
    else:
        dist_m = np.full(len(src_np), RES, dtype=np.float32)
        print("  ⚠️  edge_distances missing — assuming all edges are cardinal.")

    slope      = dz / dist_m
    slope_norm = (slope - slope.mean()) / (slope.std() + 1e-8)
    edge_attr  = torch.from_numpy(slope_norm).view(-1, 1).float()

    print(f"  Grid : {H}×{W} = {H*W:,} nodes   Edges : {edge_index.shape[1]:,}")

    # ── Scenario split (same seed as training) ────────────────────────────────
    paths = sorted(glob.glob(os.path.join(SCENARIOS_DIR, "scenario_?????.npz")))
    rng   = np.random.default_rng(SEED)
    perm  = rng.permutation(len(paths))
    n_tr  = int(len(paths) * TRAIN_FRAC)
    n_val = int(len(paths) * VAL_FRAC)
    te_paths = [paths[i] for i in perm[n_tr + n_val:]]
    print(f"  Test scenarios : {len(te_paths)}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"\nLoading {BEST_PT} …")
    ckpt = torch.load(BEST_PT, map_location=DEVICE, weights_only=False)

    feat_mean = ckpt["feat_mean"]
    feat_std  = ckpt["feat_std"]

    # Print saved metrics
    tm = ckpt.get("test_metrics", {})
    if tm:
        print("\n  Saved test metrics:")
        print(f"    R²              : {tm.get('r2',              'n/a')}")
        print(f"    RMSE            : {tm.get('rmse',            'n/a')} m")
        print(f"    RMSE (>10 cm)   : {tm.get('rmse_10cm',       'n/a')} m")
        print(f"    MAE             : {tm.get('mae',             'n/a')} m")
        print(f"    Bias            : {tm.get('bias',            'n/a')} m")
        print(f"    Speed           : {tm.get('ms_per_scenario', 'n/a')} ms/scenario")

    # ── Rebuild model ─────────────────────────────────────────────────────────
    model = FloodGAT(
        in_channels     = 14,
        hidden_channels = ckpt["hidden_channels"],
        num_layers      = ckpt["num_layers"],
        heads           = ckpt["heads"],
        dropout         = ckpt["dropout"],
    ).to(DEVICE)

    # Strip torch.compile() _orig_mod prefix if present
    raw_sd   = ckpt["model_state"]
    clean_sd = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in raw_sd.items()
    }
    model.load_state_dict(clean_sd)
    print(f"\n  Model loaded  "
          f"(hidden={ckpt['hidden_channels']}, "
          f"layers={ckpt['num_layers']}, "
          f"heads={ckpt['heads']})")

    # ── Generate attention maps ───────────────────────────────────────────────
    visualise_attention(
        model, te_paths, static_arr, edge_index, edge_attr,
        feat_mean, feat_std, H, W, OUTPUT_DIR,
        n_scenarios=N_ATTENTION,
    )
