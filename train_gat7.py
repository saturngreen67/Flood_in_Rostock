# =============================================================================
#  train_gat.py  —  Graph Attention Network (GAT) Surrogate
#                   Rostock Urban Flood  |  Physics-Informed + Optuna HPO
# =============================================================================
#
#  Why GAT instead of GCN?
#  ────────────────────────
#  A plain GCN averages neighbour features with fixed weights.  GAT learns a
#  separate attention score for every edge, so it can decide that the node
#  uphill (high elevation difference) matters more than the node downhill when
#  predicting depth — a physically meaningful bias that GCN cannot express.
#
#  Architecture
#  ────────────
#  Input projection  Linear(14 → hidden)
#       ↓
#  N × GATConv(hidden → hidden, heads, concat=False)
#      + BatchNorm + ELU + Dropout + residual skip
#       ↓
#  Output head       Linear(hidden → 1) + Softplus   (depth ≥ 0)
#
#  Graph topology
#  ──────────────
#  The Rostock raster is a 4-connected grid (H×W nodes, ~240k bidirectional
#  edges) stored once in graph_static.npz.  Every scenario shares the SAME
#  topology; only node features change between scenarios.
#
#  Execution flow
#  ──────────────
#  Phase 1 — HPO (Optuna TPE + MedianPruner)
#    Searches: lr, lambda_phys, weight_decay, hidden_channels,
#              num_layers, heads, dropout, batch_size
#    Saves:    optuna_study_gat.pkl
#              08_hpo_history.png, 09_hpo_parallel.png
#
#  Phase 2 — Full training with best hyperparameters
#    Saves:    gat_best.pt, test_pred.npy, test_true.npy
#    Plots:    01–07 (identical meaning to the MLP plots)
#
#  Node features (14 total — identical to MLP)
#  ────────────────────────────────────────────
#  Static  (8, col 0–7): elevation, street_mask, building_mask, existing_gi,
#                        dist_to_street, dist_to_outlet, upstream_area,
#                        water_mask_sink
#  Dynamic (6, col 8–13): imperviousness, mannings_n, gi_type,
#                         intensity [broadcast], duration [broadcast],
#                         adoption_level
#
#  Physics loss  (same improved version as MLP)
#  ─────────────────────────────────────────────
#  L_total = MSE(pred, depth)  +  λ · L_phys
#  L_phys  = mean[ ReLU(pred − ceiling_up) + 0.5·ReLU(floor − pred)·w_hilltop ]
#  ceiling_up = rain_m × upstr_area × imp / pixel_area
#
#  Install
#  ───────
#  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#  pip install torch_geometric
#  pip install numpy scikit-learn tqdm matplotlib optuna
# =============================================================================

import os
import glob
import pickle
import warnings
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.nn   import GATConv
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import optuna
from optuna.pruners  import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
#  Configuration
# =============================================================================
SCENARIOS_DIR = "scenarios"
OUTPUT_DIR    = "gat_results"

EPOCHS      = 100
HPO_TRIALS  = 40
HPO_EPOCHS  = 15
HPO_WARMUP  = 5

TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15          # test = remaining 0.15

SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
#  Thermal & checkpoint configuration
# =============================================================================
CHECKPOINT_FILE   = os.path.join(OUTPUT_DIR, "gat_train_checkpoint.pt")
CHECKPOINT_EVERY  = 1      # save a checkpoint every N epochs

# ThermalGuard thresholds (°C).  Adjust if your GPU runs hotter/cooler.
TEMP_WARN         = 80     # print a warning, no action yet
TEMP_THROTTLE     = 83     # insert a short sleep between batches
TEMP_CRITICAL     = 87     # pause training for longer to let GPU cool
TEMP_CHECK_EVERY  = 20     # check temperature every N batches

# =============================================================================
#  ThermalGuard
# =============================================================================

class ThermalGuard:
    """
    Monitors GPU temperature via nvidia-smi and throttles the training loop
    before the GPU reaches dangerous temperatures.

    How it works
    ────────────
    Call .check() once every TEMP_CHECK_EVERY batches inside the training
    loop.  It reads the current GPU temperature and takes one of four actions:

      < TEMP_WARN     → nothing
      < TEMP_THROTTLE → print a warning once per degree
      < TEMP_CRITICAL → sleep 2 s between every batch (throttle)
      ≥ TEMP_CRITICAL → pause training for 30 s, then re-check before
                        continuing.  Repeats until temperature drops below
                        TEMP_THROTTLE.

    The guard is completely disabled on CPU (no GPU to monitor).
    If nvidia-smi is not available it silently disables itself.
    """
    def __init__(self):
        self.enabled      = DEVICE.type == "cuda"
        self.last_temp    = 0
        self._warned_at   = set()   # suppress repeated warnings at same temp
        self._throttling  = False
        self._batch_count = 0

        if self.enabled:
            try:
                subprocess.run(
                    ["nvidia-smi", "--query-gpu=temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, check=True, timeout=5)
            except Exception:
                self.enabled = False
                print("  ⚠️  ThermalGuard: nvidia-smi not available — "
                      "thermal monitoring disabled.")

    def _read_temp(self) -> int:
        """Read current GPU 0 temperature in °C.  Returns -1 on failure."""
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip().split("\n")[0])
        except Exception:
            return -1

    def check(self):
        """
        Call this once per batch inside the training loop.
        Handles throttling and pausing transparently — the training loop
        does not need to know anything about temperatures.
        """
        if not self.enabled:
            return

        self._batch_count += 1
        if self._batch_count % TEMP_CHECK_EVERY != 0:
            # If already throttling, still insert the sleep every batch
            if self._throttling:
                time.sleep(2.0)
            return

        temp = self._read_temp()
        if temp < 0:
            return
        self.last_temp = temp

        if temp >= TEMP_CRITICAL:
            # ── Critical: pause until cool ────────────────────────────────────
            print(f"\n  🌡️  GPU CRITICAL: {temp}°C ≥ {TEMP_CRITICAL}°C — "
                  f"pausing training for 30 s to cool down…")
            self._throttling = True
            while True:
                time.sleep(30)
                temp = self._read_temp()
                print(f"  🌡️  GPU temperature now {temp}°C  "
                      f"(target < {TEMP_THROTTLE}°C to resume)")
                if temp < TEMP_THROTTLE:
                    print(f"  ✅  Temperature safe — resuming training.\n")
                    self._throttling = False
                    break

        elif temp >= TEMP_THROTTLE:
            # ── Throttle: sleep 2 s after every batch ─────────────────────────
            if not self._throttling:
                print(f"\n  🌡️  GPU THROTTLE: {temp}°C ≥ {TEMP_THROTTLE}°C — "
                      f"inserting 2 s sleep between batches.")
            self._throttling = True
            time.sleep(2.0)

        else:
            # ── Safe: cancel throttling if it was active ───────────────────────
            if self._throttling:
                print(f"\n  ✅  GPU cooled to {temp}°C — removing throttle.\n")
                self._throttling = False

            if temp >= TEMP_WARN and temp not in self._warned_at:
                print(f"\n  🌡️  GPU WARNING: {temp}°C — approaching throttle "
                      f"threshold ({TEMP_THROTTLE}°C).")
                self._warned_at.add(temp)

    def status_str(self) -> str:
        """Short string for tqdm postfix."""
        if not self.enabled or self.last_temp == 0:
            return ""
        return f"{self.last_temp}°C"


# =============================================================================
#  Dataset
# =============================================================================

class FloodGraphDataset(torch.utils.data.Dataset):
    """
    Each item is a PyG Data object representing one flood scenario.

    Attributes stored on Data
    ─────────────────────────
    x          (N, 14) float32  — normalised node features (filled in loader)
    x_raw      (N, 14) float32  — unnormalised features (for physics loss)
    edge_index (2, E)  int64    — shared grid connectivity
    y          (N,)    float32  — simulated flood depth  [target]
    imp        (N,)    float32  — imperviousness          [physics loss]
    inten      (N,)    float32  — storm intensity mm/h    [physics loss]
    dur        (N,)    float32  — storm duration  min     [physics loss]
    """
    def __init__(self, paths: list, static_arr: np.ndarray,
                 edge_index: torch.Tensor,
                 edge_attr:  torch.Tensor,
                 feat_mean: np.ndarray = None,
                 feat_std:  np.ndarray = None):
        self.paths      = paths
        self.static_arr = static_arr        # (N, 8)
        self.edge_index = edge_index        # (2, E) — same for every scenario
        self.edge_attr  = edge_attr         # (E, 1) normalised dZ — same every scenario
        self.feat_mean  = feat_mean         # (14,) or None
        self.feat_std   = feat_std          # (14,) or None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Data:
        d     = np.load(self.paths[idx])
        N     = self.static_arr.shape[0]

        imp   = d["imperviousness"].flatten().astype(np.float32)
        mann  = d["mannings_n"].flatten().astype(np.float32)
        gi    = d["gi_map"].flatten().astype(np.float32)
        inten = float(d["intensity"])
        dur   = float(d["duration"])
        adopt = float(d["adoption_level"])
        depth = d["flood_depth"].flatten().astype(np.float32)

        dyn = np.stack([
            imp, mann, gi,
            np.full(N, inten,  dtype=np.float32),
            np.full(N, dur,    dtype=np.float32),
            np.full(N, adopt,  dtype=np.float32),
        ], axis=1)                                          # (N, 6)

        x_raw = np.concatenate([self.static_arr, dyn], axis=1)  # (N, 14)

        # Normalise if stats are available
        if self.feat_mean is not None:
            x_norm = (x_raw - self.feat_mean) / self.feat_std
        else:
            x_norm = x_raw.copy()

        return Data(
            x          = torch.from_numpy(x_norm),
            x_raw      = torch.from_numpy(x_raw),
            edge_index = self.edge_index,
            edge_attr  = self.edge_attr,    # (E, 1) slope — same for all scenarios
            y          = torch.from_numpy(depth),
            imp        = torch.from_numpy(imp),
            inten      = torch.full((N,), inten,  dtype=torch.float32),
            dur        = torch.full((N,), dur,    dtype=torch.float32),
        )


def make_geo_dataloaders(tr_paths, val_paths, te_paths,
                         static_arr, edge_index, edge_attr,
                         feat_mean, feat_std, batch_size):
    """Build PyG DataLoaders for train / val / test splits."""
    on_gpu = DEVICE.type == "cuda"
    kw     = dict(
        num_workers      = 4 if on_gpu else 0,
        pin_memory       = on_gpu,
        persistent_workers = on_gpu,
    )
    def _ds(paths):
        return FloodGraphDataset(paths, static_arr, edge_index, edge_attr,
                                 feat_mean, feat_std)

    tr_dl = GeoDataLoader(_ds(tr_paths),  batch_size=batch_size,
                          shuffle=True,  **kw)
    vl_dl = GeoDataLoader(_ds(val_paths), batch_size=batch_size,
                          shuffle=False, **kw)
    te_dl = GeoDataLoader(_ds(te_paths),  batch_size=batch_size,
                          shuffle=False, **kw)
    return tr_dl, vl_dl, te_dl

# =============================================================================
#  Model
# =============================================================================

class FloodGAT(nn.Module):
    """
    Graph Attention Network for node-level flood depth regression.

    Design choices
    ──────────────
    concat=False in GATConv:
        All attention heads vote and their outputs are averaged rather than
        concatenated.  This keeps the hidden dimension constant across layers
        and makes depth (num_layers) the main capacity knob, not heads×width.

    Residual connections:
        Added after every GAT layer.  Without them, stacking ≥3 GAT layers
        causes over-smoothing — all node representations converge to the same
        vector, which erases the spatial variation you need.

    BatchNorm after GATConv:
        Stabilises training and allows higher learning rates.

    Softplus output:
        Guarantees predicted depth ≥ 0 with smooth gradients near zero
        (unlike ReLU which has a dead zone).
    """
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int, heads: int, dropout: float):
        super().__init__()

        # ── Input projection (linear, no graph op) ────────────────────────────
        # Projects raw 14-dim features into the hidden space before the first
        # GAT layer.  This decouples the feature dimension from the hidden
        # dimension and gives the model a chance to mix features before any
        # attention is computed.
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ELU(),
        )

        # ── Stacked GAT layers ────────────────────────────────────────────────
        # edge_dim=1 tells GATConv to incorporate the scalar edge attribute
        # (normalised elevation difference dZ) into the attention coefficient
        # computation.  Internally PyG concatenates the edge feature to the
        # source+destination node pair before the LeakyReLU scoring step, so
        # the attention score for edge (i→j) becomes:
        #   e_ij = LeakyReLU( a^T · [ Wh_i || Wh_j || W_e · dZ_ij ] )
        # This lets the model learn that a steep downhill edge (large +dZ)
        # deserves higher attention weight — directly encoding gravity.
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels  = hidden_channels,
                out_channels = hidden_channels,
                heads        = heads,
                concat       = False,   # average heads → keep dim = hidden_channels
                dropout      = dropout,
                add_self_loops = True,  # every node attends to itself
                edge_dim     = 1,       # scalar edge feature: normalised dZ
            )
            for _ in range(num_layers)
        ])

        # BatchNorm per layer (operates on hidden_channels)
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])

        self.dropout    = dropout
        self.num_layers = num_layers

        # ── Output head ───────────────────────────────────────────────────────
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus(),              # ensures depth ≥ 0
        )

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr:  torch.Tensor) -> torch.Tensor:
        """
        x          : (total_nodes, 14)  — already normalised
        edge_index : (2, total_edges)   — batched connectivity
        edge_attr  : (total_edges, 1)   — normalised elevation diff dZ (src - dst)
        returns    : (total_nodes,)     — predicted flood depths
        """
        # Input projection
        x = self.input_proj(x)          # (N, hidden)

        # Message-passing stack with residual connections
        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            # Pass edge_attr so attention scores incorporate slope (gravity)
            x = gat(x, edge_index, edge_attr=edge_attr)
            x = norm(x)                 # stabilise activations
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + residual            # skip connection — prevents over-smoothing

        return self.output_head(x).squeeze(-1)   # (N,)

# =============================================================================
#  Physics loss  (identical to MLP version — works on any flat node tensor)
# =============================================================================

def physics_loss(pred: torch.Tensor,
                 inten_mm_h: torch.Tensor,
                 dur_min:    torch.Tensor,
                 imp:        torch.Tensor,
                 upstr_area: torch.Tensor,
                 res: float = 3.0) -> torch.Tensor:
    """
    Two-component physics regulariser.

    Component 1 — upstream volume ceiling (one-sided ReLU):
        ceiling_up = rain_m × upstr_area × imp / pixel_area
        Fires only when pred EXCEEDS the ceiling, so correctly-routed deep
        water at valley bottoms is not penalised.

    Component 2 — local rainfall floor on hilltop / isolated impervious cells:
        Gated by hilltop_weight ≈ exp(-upstream_cells / 50) so it is silent
        where routing governs depth and active only near ridgelines / rooftops.
    """
    pixel_area     = res * res
    rain_m         = (inten_mm_h / 1000.0) * (dur_min / 60.0)

    ceiling_up     = rain_m * upstr_area * imp / pixel_area
    loss_up        = torch.relu(pred - ceiling_up)

    upstream_cells = upstr_area / pixel_area
    hilltop_weight = torch.exp(-upstream_cells / 50.0)
    local_floor    = rain_m * imp * hilltop_weight
    loss_floor     = torch.relu(local_floor - pred) * hilltop_weight

    return torch.mean(loss_up + 0.5 * loss_floor)

# =============================================================================
#  Normaliser
# =============================================================================

class Normalizer:
    """Z-score normalisation fitted on a random sample of training scenarios."""
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std:  np.ndarray | None = None

    def fit(self, paths: list, static_arr: np.ndarray,
            n_sample: int = 60, seed: int = 42) -> None:
        rng  = np.random.default_rng(seed)
        idx  = rng.choice(len(paths), size=min(n_sample, len(paths)), replace=False)
        N    = static_arr.shape[0]
        rows = []
        for i in idx:
            d     = np.load(paths[i])
            imp   = d["imperviousness"].flatten().astype(np.float32)
            mann  = d["mannings_n"].flatten().astype(np.float32)
            gi    = d["gi_map"].flatten().astype(np.float32)
            inten = float(d["intensity"])
            dur   = float(d["duration"])
            adopt = float(d["adoption_level"])
            dyn   = np.stack([
                imp, mann, gi,
                np.full(N, inten,  dtype=np.float32),
                np.full(N, dur,    dtype=np.float32),
                np.full(N, adopt,  dtype=np.float32),
            ], axis=1)
            rows.append(np.concatenate([static_arr, dyn], axis=1))
        arr       = np.concatenate(rows, axis=0)
        self.mean = arr.mean(axis=0).astype(np.float32)
        self.std  = (arr.std(axis=0) + 1e-8).astype(np.float32)

# =============================================================================
#  Helpers
# =============================================================================

def safe_r2(true: np.ndarray, pred: np.ndarray) -> float:
    try:
        return float(r2_score(true, pred))
    except Exception:
        return float("nan")


def run_one_epoch_gat(model, loader, mse_fn, lambda_phys,
                      optim=None, train=True):
    """
    One full pass over a DataLoader.
    Returns (avg_total, avg_data, avg_phys, all_pred_np, all_true_np).
    If train=True, performs a backward pass and parameter update.
    """
    model.train(train)
    ep_data, ep_phys = 0.0, 0.0
    all_p, all_t     = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(DEVICE)

            # x_raw col 6 = upstream_area (unnormalised, in m²)
            upstr = batch.x_raw[:, 6]

            pred  = model(batch.x, batch.edge_index, batch.edge_attr)
            ld    = mse_fn(pred, batch.y)
            lp    = physics_loss(pred, batch.inten, batch.dur,
                                 batch.imp, upstr)
            loss  = ld + lambda_phys * lp

            if train:
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            ep_data += ld.item()
            ep_phys += lp.item()
            all_p.append(pred.detach().cpu())
            all_t.append(batch.y.detach().cpu())

    nb   = len(loader)
    tot  = ep_data / nb + lambda_phys * ep_phys / nb
    return (tot,
            ep_data / nb,
            ep_phys / nb,
            torch.cat(all_p).numpy(),
            torch.cat(all_t).numpy())

# =============================================================================
#  Phase 1 — Optuna HPO
# =============================================================================

def objective(trial, tr_paths, val_paths, static_arr, edge_index, edge_attr, normalizer):
    # ── Suggest hyperparameters ───────────────────────────────────────────────
    lr              = trial.suggest_float("lr",              1e-4, 5e-3,  log=True)
    lambda_phys     = trial.suggest_float("lambda_phys",     0.01, 1.0,   log=True)
    weight_decay    = trial.suggest_float("weight_decay",    1e-6, 1e-3,  log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    num_layers      = trial.suggest_int("num_layers",        2, 4)
    heads           = trial.suggest_categorical("heads",     [2, 4])
    dropout         = trial.suggest_float("dropout",         0.0, 0.4)
    batch_size      = trial.suggest_categorical("batch_size", [1, 2])

    # Flush any leftover VRAM from previous trials before allocating the new model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    model  = FloodGAT(in_channels=14,
                      hidden_channels=hidden_channels,
                      num_layers=num_layers,
                      heads=heads,
                      dropout=dropout).to(DEVICE)
    optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_fn = nn.MSELoss()

    tr_dl, vl_dl, _ = make_geo_dataloaders(
        tr_paths, val_paths, [],
        static_arr, edge_index, edge_attr,
        normalizer.mean, normalizer.std,
        batch_size)

    best_val = float("inf")
    try:
        for epoch in range(HPO_EPOCHS):

            run_one_epoch_gat(model, tr_dl, mse_fn, lambda_phys,
                              optim=optim, train=True)
            vtot, *_ = run_one_epoch_gat(model, vl_dl, mse_fn, lambda_phys,
                                         train=False)

            if vtot < best_val:
                best_val = vtot

            trial.report(vtot, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    finally:
        # Always release model + optimiser memory back to the CUDA cache,
        # then flush the cache itself — prevents accumulation across trials.
        del model, optim, tr_dl, vl_dl
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return best_val


def run_hpo(tr_paths, val_paths, static_arr, edge_index, edge_attr, normalizer):
    print(f"\n{'='*65}")
    print(f"  PHASE 1 — Hyperparameter Optimisation  (Optuna TPE)")
    print(f"  Trials : {HPO_TRIALS}   Epochs/trial : {HPO_EPOCHS}   "
          f"Pruner warmup : {HPO_WARMUP}")
    print(f"  Search space : lr, lambda_phys, weight_decay,")
    print(f"                 hidden_channels, num_layers, heads,")
    print(f"                 dropout, batch_size")
    print(f"{'='*65}\n")

    study = optuna.create_study(
        direction = "minimize",
        sampler   = TPESampler(seed=SEED),
        pruner    = MedianPruner(n_startup_trials=5, n_warmup_steps=HPO_WARMUP),
    )

    completed, pruned, failed = [0], [0], [0]

    with tqdm(total=HPO_TRIALS, desc="  HPO trials",
              ncols=105, unit="trial") as pbar:
        def _cb(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                completed[0] += 1
            elif trial.state == optuna.trial.TrialState.PRUNED:
                pruned[0] += 1
            elif trial.state == optuna.trial.TrialState.FAIL:
                failed[0] += 1
            best_val = study.best_value if study.best_trial is not None else float("nan")
            pbar.set_postfix({
                "best":   f"{best_val:.5f}",
                "done":   completed[0],
                "pruned": pruned[0],
                "failed": failed[0],
            })
            pbar.update(1)

        study.optimize(
            lambda t: objective(t, tr_paths, val_paths,
                                static_arr, edge_index, edge_attr, normalizer),
            n_trials       = HPO_TRIALS,
            callbacks      = [_cb],
            gc_after_trial = True,
            # Catch OOM and any other runtime errors — marks trial FAIL
            # instead of crashing the entire study.
            catch          = (RuntimeError,),
        )

    best = study.best_params
    print(f"\n  Best trial  #{study.best_trial.number}  "
          f"(val loss = {study.best_value:.6f})")
    for k, v in best.items():
        print(f"    {k:<22} = {v}")

    study_path = os.path.join(OUTPUT_DIR, "optuna_study_gat.pkl")
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"\n  Study saved → {study_path}")

    plot_hpo(study, OUTPUT_DIR)
    return best, study


def plot_hpo(study, out_dir):
    trials  = [t for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE]
    vals    = [t.value  for t in trials]
    nums    = [t.number for t in trials]
    params  = list(study.best_params.keys())

    # 08 — optimisation history
    best_so_far = np.minimum.accumulate(vals)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(nums, vals, s=22, alpha=0.55, color="steelblue",
               label="Trial val loss")
    ax.plot(nums, best_so_far, color="tomato", lw=2.0, label="Best so far")
    ax.set_xlabel("Trial number"); ax.set_ylabel("Validation loss (total)")
    ax.set_title(f"GAT — Optuna History  ({len(trials)} completed trials)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_hpo_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 09 — per-parameter scatter coloured by val loss
    param_vals = {p: [t.params.get(p) for t in trials] for p in params}
    n_params   = len(params)
    fig, axes  = plt.subplots(1, n_params,
                               figsize=(3.4 * n_params, 5), sharey=False)
    if n_params == 1:
        axes = [axes]

    norm_v = plt.Normalize(vmin=min(vals), vmax=np.percentile(vals, 75))
    cmap   = plt.cm.plasma_r
    rng0   = np.random.default_rng(1)

    for ax, p in zip(axes, params):
        pv     = param_vals[p]
        unique = sorted(set(v for v in pv if v is not None))
        if len(unique) <= 8:
            jitter  = rng0.uniform(-0.15, 0.15, len(pv))
            x_plot  = [unique.index(v) + j if v is not None else 0
                       for v, j in zip(pv, jitter)]
            ax.set_xticks(range(len(unique)))
            ax.set_xticklabels([str(u) for u in unique],
                               fontsize=8, rotation=30)
        else:
            x_plot = [v if v is not None else float("nan") for v in pv]
            if p in ["lr", "lambda_phys", "weight_decay"]:
                ax.set_xscale("log")   # log axis prevents crowding near zero

        sc = ax.scatter(x_plot, vals, c=vals, cmap=cmap, norm=norm_v,
                        s=20, alpha=0.75, rasterized=True)
        best_idx = vals.index(min(vals))
        ax.scatter([x_plot[best_idx]], [vals[best_idx]],
                   marker="*", s=240, color="gold", zorder=5, label="best")
        ax.set_title(p, fontsize=9, pad=4)
        ax.set_ylabel("Val loss" if ax is axes[0] else "")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8)

    fig.colorbar(sc, ax=axes[-1], label="Val loss", shrink=0.75)
    fig.suptitle("GAT HPO — each point is one completed trial",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_hpo_parallel.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  HPO plots saved → {out_dir}/")

# =============================================================================
#  Phase 2 — Full training
# =============================================================================

def train(hparams: dict, tr_paths, val_paths, te_paths,
          static_arr, edge_index, edge_attr, H, W):

    lr              = hparams.get("lr",              1e-3)
    lambda_phys     = hparams.get("lambda_phys",     0.1)
    weight_decay    = hparams.get("weight_decay",    1e-5)
    hidden_channels = hparams.get("hidden_channels", 128)
    num_layers      = hparams.get("num_layers",      3)
    heads           = hparams.get("heads",           4)
    dropout         = hparams.get("dropout",         0.1)
    batch_size      = hparams.get("batch_size",      2)

    print(f"\n{'='*65}")
    print(f"  PHASE 2 — Full GAT Training")
    print(f"  lr={lr:.2e}  lambda_phys={lambda_phys:.4f}  "
          f"weight_decay={weight_decay:.2e}")
    print(f"  hidden={hidden_channels}  layers={num_layers}  "
          f"heads={heads}  dropout={dropout:.2f}  batch={batch_size}")
    print(f"  Epochs : {EPOCHS}")
    print(f"{'='*65}\n")

    # ── Normaliser ────────────────────────────────────────────────────────────
    normalizer = Normalizer()
    print("  Fitting normaliser on 60 training scenarios…")
    normalizer.fit(tr_paths, static_arr, n_sample=60, seed=SEED)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    tr_dl, vl_dl, te_dl = make_geo_dataloaders(
        tr_paths, val_paths, te_paths,
        static_arr, edge_index, edge_attr,
        normalizer.mean, normalizer.std,
        batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model  = FloodGAT(in_channels=14,
                      hidden_channels=hidden_channels,
                      num_layers=num_layers,
                      heads=heads,
                      dropout=dropout).to(DEVICE)
    n_par  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters : {n_par:,}")

    if DEVICE.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("  torch.compile() active\n")
    else:
        print()

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                 optim, patience=8, factor=0.5, min_lr=1e-6)
    mse_fn = nn.MSELoss()

    hist = {k: [] for k in [
        "tr_total","tr_data","tr_phys","tr_r2",
        "val_total","val_data","val_phys","val_r2","lr"]}

    best_val, best_state = float("inf"), None
    start_epoch          = 1

    # ── Resume from checkpoint if one exists ─────────────────────────────────
    if os.path.exists(CHECKPOINT_FILE):
        print(f"  📂 Checkpoint found → {CHECKPOINT_FILE}")
        ckpt_resume = torch.load(CHECKPOINT_FILE, map_location=DEVICE)

        # Verify the checkpoint belongs to this exact run (same hparams)
        if ckpt_resume.get("hparams") == hparams:
            model.load_state_dict(ckpt_resume["model_state"])
            optim.load_state_dict(ckpt_resume["optim_state"])
            sched.load_state_dict(ckpt_resume["sched_state"])
            hist        = ckpt_resume["hist"]
            best_val    = ckpt_resume["best_val"]
            best_state  = ckpt_resume["best_state"]
            start_epoch = ckpt_resume["epoch"] + 1
            print(f"  ✅ Resuming from epoch {start_epoch}  (best val loss so far: {best_val:.6f})\n")
        else:
            print(f"  ⚠️  Checkpoint hparams differ from current run — starting fresh.\n")

    if start_epoch > EPOCHS:
        print("  ✅ Training already complete (all epochs finished). Skipping to test.\n")
    else:
        # ── Thermal guard ─────────────────────────────────────────────────────
        guard = ThermalGuard()

        # ── Epoch loop ────────────────────────────────────────────────────────
        for epoch in range(start_epoch, EPOCHS + 1):

            # Train
            model.train()
            ep_data, ep_phys = 0.0, 0.0
            all_p, all_t     = [], []
            pbar = tqdm(tr_dl, desc=f"Ep {epoch:03d}/{EPOCHS} [Train]",
                        leave=False, ncols=108)
            for batch in pbar:
                guard.check()          # ← thermal check every N batches
                batch = batch.to(DEVICE)
                upstr = batch.x_raw[:, 6]
                pred  = model(batch.x, batch.edge_index, batch.edge_attr)
                ld    = mse_fn(pred, batch.y)
                lp    = physics_loss(pred, batch.inten, batch.dur,
                                     batch.imp, upstr)
                loss  = ld + lambda_phys * lp
                optim.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                ep_data += ld.item(); ep_phys += lp.item()
                all_p.append(pred.detach().cpu()); all_t.append(batch.y.detach().cpu())
                temp_str = guard.status_str()
                pbar.set_postfix({"loss": f"{loss.item():.5f}",
                                  **({"GPU": temp_str} if temp_str else {})})

            nb = len(tr_dl)
            hist["tr_data"].append(ep_data / nb)
            hist["tr_phys"].append(ep_phys / nb)
            hist["tr_total"].append(ep_data / nb + lambda_phys * ep_phys / nb)
            hist["tr_r2"].append(
                safe_r2(torch.cat(all_t).numpy(), torch.cat(all_p).numpy()))

            # Validate
            model.eval()
            vd, vp       = 0.0, 0.0
            vp_l, vt_l   = [], []
            with torch.no_grad():
                for batch in tqdm(vl_dl,
                        desc=f"Ep {epoch:03d}/{EPOCHS} [Val]  ",
                        leave=False, ncols=108):
                    batch = batch.to(DEVICE)
                    upstr = batch.x_raw[:, 6]
                    pred  = model(batch.x, batch.edge_index, batch.edge_attr)
                    vd   += mse_fn(pred, batch.y).item()
                    vp   += physics_loss(pred, batch.inten, batch.dur,
                                         batch.imp, upstr).item()
                    vp_l.append(pred.cpu()); vt_l.append(batch.y.cpu())

            nvb  = len(vl_dl)
            vtot = vd / nvb + lambda_phys * vp / nvb
            hist["val_data"].append(vd / nvb)
            hist["val_phys"].append(vp / nvb)
            hist["val_total"].append(vtot)
            hist["val_r2"].append(
                safe_r2(torch.cat(vt_l).numpy(), torch.cat(vp_l).numpy()))
            hist["lr"].append(optim.param_groups[0]["lr"])

            sched.step(vtot)
            if vtot < best_val:
                best_val   = vtot
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

            print(f"  Ep {epoch:03d} | "
                  f"Tr  total={hist['tr_total'][-1]:.5f} "
                  f"data={hist['tr_data'][-1]:.5f} phys={hist['tr_phys'][-1]:.5f} "
                  f"R²={hist['tr_r2'][-1]:.4f}  ||  "
                  f"Val total={hist['val_total'][-1]:.5f} "
                  f"R²={hist['val_r2'][-1]:.4f}  "
                  f"lr={hist['lr'][-1]:.2e}"
                  + (f"  GPU={guard.status_str()}" if guard.status_str() else ""))

            # ── Save epoch checkpoint ──────────────────────────────────────────
            if epoch % CHECKPOINT_EVERY == 0 or epoch == EPOCHS:
                torch.save({
                    "epoch":       epoch,
                    "hparams":     hparams,
                    "model_state": {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()},
                    "optim_state": optim.state_dict(),
                    "sched_state": sched.state_dict(),
                    "hist":        hist,
                    "best_val":    best_val,
                    "best_state":  best_state,
                    "normalizer_mean": normalizer.mean,
                    "normalizer_std":  normalizer.std,
                }, CHECKPOINT_FILE)

    # ── Test ──────────────────────────────────────────────────────────────────
    # Remove the epoch checkpoint now that training completed successfully.
    # gat_best.pt (saved below) is the permanent artefact; the epoch
    # checkpoint was only needed for crash recovery.
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print(f"  🗑️  Epoch checkpoint removed (training complete).")

    print("\n  Running test set evaluation…")
    model.load_state_dict(best_state)
    model.eval()
    import time
    inference_start = time.time()
    tp_l, tt_l, ti_l, td_l, timp_l = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(te_dl, desc="  Test", ncols=100):
            batch = batch.to(DEVICE)
            pred  = model(batch.x, batch.edge_index, batch.edge_attr)
            tp_l.append(pred.cpu());      tt_l.append(batch.y.cpu())
            ti_l.append(batch.inten.cpu()); td_l.append(batch.dur.cpu())
            timp_l.append(batch.imp.cpu())
    inference_end = time.time()

    tp   = torch.cat(tp_l).numpy();   tt   = torch.cat(tt_l).numpy()
    ti   = torch.cat(ti_l).numpy();   td   = torch.cat(td_l).numpy()
    timp = torch.cat(timp_l).numpy()

    total_test_scenarios = len(te_dl.dataset)
    total_time           = inference_end - inference_start
    ms_per_scenario      = (total_time / total_test_scenarios) * 1000.0

    rmse = float(np.sqrt(np.mean((tp - tt) ** 2)))
    mae  = float(np.mean(np.abs(tp - tt)))
    r2   = safe_r2(tt, tp)
    bias = float(np.mean(tp - tt))
    mask_10cm = tt > 0.10
    rmse_10cm = float(np.sqrt(np.mean((tp[mask_10cm] - tt[mask_10cm]) ** 2)))                 if mask_10cm.sum() > 0 else 0.0

    print(f"\n{'='*65}")
    print(f"  TEST RESULTS (GAT)")
    print(f"    RMSE          : {rmse:.4f} m")
    print(f"    RMSE (>10 cm) : {rmse_10cm:.4f} m")
    print(f"    MAE           : {mae:.4f} m")
    print(f"    R²            : {r2:.4f}")
    print(f"    Bias          : {bias:+.4f} m")
    print(f"    Speed         : {ms_per_scenario:.2f} ms per scenario")
    print(f"{'='*65}\n")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    torch.save({
        "model_state":      best_state,
        "feat_mean":        normalizer.mean,
        "feat_std":         normalizer.std,
        "H": H, "W": W,
        "hidden_channels":  hidden_channels,
        "num_layers":       num_layers,
        "heads":            heads,
        "dropout":          dropout,
        "lambda_phys":      lambda_phys,
        "best_hparams":     hparams,
        "test_metrics":     {"rmse": rmse, "rmse_10cm": rmse_10cm,
                               "mae": mae, "r2": r2, "bias": bias,
                               "ms_per_scenario": ms_per_scenario},
    }, os.path.join(OUTPUT_DIR, "gat_best.pt"))
    np.save(os.path.join(OUTPUT_DIR, "test_pred.npy"), tp)
    np.save(os.path.join(OUTPUT_DIR, "test_true.npy"), tt)
    print(f"  Checkpoint saved → {OUTPUT_DIR}/gat_best.pt")

    make_plots(hist, tp, tt, ti, td, timp, H, W, OUTPUT_DIR)

# =============================================================================
#  Plots 01–07  (identical structure to MLP script)
# =============================================================================

def make_plots(hist, pred, true, inten, dur, imp, H, W, out_dir):
    E   = np.arange(1, len(hist["tr_total"]) + 1)
    res = pred - true
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    rng = np.random.default_rng(0)
    s   = rng.choice(len(pred), size=min(200_000, len(pred)), replace=False)

    # 01 — Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, (tk, vk, title) in zip(axes, [
            ("tr_total","val_total","Total Loss (data + λ·phys)"),
            ("tr_data", "val_data", "Data Loss (MSE)"),
            ("tr_phys", "val_phys", "Physics Loss")]):
        ax.plot(E, hist[tk], color="steelblue", lw=1.8, label="Train")
        ax.plot(E, hist[vk], color="tomato",    lw=1.8, ls="--", label="Val")
        ax.set_yscale("log"); ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(fontsize=10)
    axes[0].set_ylabel("Loss")
    fig.suptitle("GAT — Training Loss Curves",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_loss_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 02 — R² + learning rate
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(E, hist["tr_r2"],  color="steelblue", lw=1.8, label="Train R²")
    axes[0].plot(E, hist["val_r2"], color="tomato",    lw=1.8, ls="--", label="Val R²")
    axes[0].axhline(1.0, color="grey", lw=0.7, ls=":")
    axes[0].set_ylim(-0.1, 1.05); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("R²"); axes[0].set_title("R² over Training")
    axes[0].legend(fontsize=10)
    axes[1].plot(E, hist["lr"], color="mediumpurple", lw=1.8)
    axes[1].set_yscale("log"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate"); axes[1].set_title("LR Schedule")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_r2_and_lr.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 03 — Hex-scatter predicted vs simulated
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae  = float(np.mean(np.abs(pred - true)))
    r2   = safe_r2(true, pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(true[s], pred[s], gridsize=80, cmap="plasma",
                   mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="log₁₀(count)")
    lim = max(true[s].max(), pred[s].max()) * 1.05
    ax.plot([0, lim], [0, lim], "w--", lw=1.3, label="1:1 line")
    ax.set_xlabel("Simulated Depth (m)"); ax.set_ylabel("Predicted Depth (m)")
    ax.set_title(f"GAT Test: Predicted vs Simulated\n"
                 f"RMSE={rmse:.4f} m   MAE={mae:.4f} m   R²={r2:.4f}",
                 fontsize=11)
    ax.legend(fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_scatter_test.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 04 — Residuals
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(res, bins=150, color="steelblue", edgecolor="none", alpha=0.85)
    axes[0].axvline(0,          color="k",     lw=1.2, ls="--", label="zero")
    axes[0].axvline(res.mean(), color="tomato", lw=1.2, ls="--",
                    label=f"mean={res.mean():.4f} m")
    axes[0].set_xlabel("Residual pred−true (m)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Residual Distribution"); axes[0].legend(fontsize=10)
    axes[1].hexbin(true[s], res[s], gridsize=70, cmap="RdBu_r", mincnt=1)
    axes[1].axhline(0, color="k", lw=1.2, ls="--")
    axes[1].set_xlabel("Simulated Depth (m)"); axes[1].set_ylabel("Residual (m)")
    axes[1].set_title("Residuals vs Simulated Depth")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_residuals.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 05 — Physics consistency
    ceiling   = (inten / 1000.0) * (dur / 60.0) * imp
    pct_below = float((pred <= ceiling).mean() * 100)
    fig, ax   = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(ceiling[s], pred[s], c=true[s], cmap="viridis",
                    s=2, alpha=0.4, rasterized=True)
    fig.colorbar(sc, ax=ax, label="Simulated Depth (m)")
    dg = np.linspace(0, ceiling.max(), 200)
    ax.plot(dg, dg, "r--", lw=1.4, label="physics ceiling")
    ax.set_xlabel("Physics Ceiling  I×T×imp (m)")
    ax.set_ylabel("Predicted Depth (m)")
    ax.set_title(f"Physics Consistency\n"
                 f"{pct_below:.1f}% of predictions ≤ rainfall ceiling",
                 fontsize=11)
    ax.legend(markerscale=6, fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_physics_check.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 06 — Spatial MAE map
    N_nodes = H * W
    n_sc    = len(pred) // N_nodes
    if n_sc > 0:
        mae_map = np.mean(
            np.abs(pred[:n_sc * N_nodes].reshape(n_sc, H, W)
                   - true[:n_sc * N_nodes].reshape(n_sc, H, W)), axis=0)
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(mae_map, cmap="hot_r", origin="upper")
        fig.colorbar(im, ax=ax, label="MAE (m)")
        ax.set_title(f"GAT Spatial MAE  ({n_sc} test scenarios)", fontsize=12)
        ax.set_xlabel("Column (W→E)"); ax.set_ylabel("Row (N→S)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "06_spatial_mae.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 07 — MAE by depth percentile bin
    bins   = np.percentile(true, np.linspace(0, 100, 21))
    bin_id = np.clip(np.digitize(true, bins) - 1, 0, len(bins) - 2)
    bin_mae, bin_mid = [], []
    for b in range(len(bins) - 1):
        m = bin_id == b
        if m.sum() > 0:
            bin_mae.append(float(np.mean(np.abs(res[m]))))
            bin_mid.append(0.5 * (bins[b] + bins[b + 1]))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(bin_mae)), bin_mae,
           color=plt.cm.plasma(np.linspace(0.15, 0.85, len(bin_mae))),
           edgecolor="none", alpha=0.9)
    ax.set_xticks(range(len(bin_mae)))
    ax.set_xticklabels([f"{v:.5f}" for v in bin_mid],
                       rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Simulated Depth bin (m)"); ax.set_ylabel("MAE (m)")
    ax.set_title("GAT — MAE by Depth Percentile Bin", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_mae_by_depth.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  All plots (01–07) saved → {out_dir}/")

# =============================================================================
#  Attention weight visualisation  (GAT-specific, no MLP equivalent)
# =============================================================================

def visualise_attention(model, te_paths, static_arr, edge_index, edge_attr,
                        normalizer, H, W, out_dir, n_scenarios=3):
    """
    For each of the first n_scenarios test scenarios, run a forward pass with
    return_attention_weights=True on the LAST GAT layer and plot the mean
    attention weight received by each node as a spatial heatmap.

    High attention weight at a node means its neighbours are relying heavily
    on it during message passing — typically coincides with major drainage
    paths, street network junctions, and GI clusters.
    """
    print(f"\n  Computing attention maps for {n_scenarios} test scenarios…")
    model.eval()

    ds = FloodGraphDataset(te_paths[:n_scenarios], static_arr, edge_index,
                           edge_attr, normalizer.mean, normalizer.std)

    for i, data in enumerate(ds):
        data = data.to(DEVICE)

        # Manual forward pass to intercept attention weights on last GAT layer
        with torch.no_grad():
            x = model.input_proj(data.x)

            for j, (gat, norm) in enumerate(
                    zip(model.gat_layers, model.norms)):

                is_last = (j == model.num_layers - 1)

                if is_last:
                    # Return (edge_index, attention_weights) from last layer
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

        # alpha shape: (n_edges, heads) — average over heads, then
        # scatter-sum to destination nodes to get "attention received"
        alpha_mean = alpha.mean(dim=1).cpu().numpy()   # (n_edges,)
        dst_nodes  = ei[1].cpu().numpy()               # destination of each edge

        attn_map = np.zeros(H * W, dtype=np.float32)
        np.add.at(attn_map, dst_nodes, alpha_mean)
        attn_map = attn_map.reshape(H, W)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        im0 = axes[0].imshow(
            data.y.cpu().numpy().reshape(H, W),
            cmap="Blues", origin="upper")
        fig.colorbar(im0, ax=axes[0], label="Simulated Depth (m)")
        axes[0].set_title(f"Simulated Flood Depth\n"
                          f"intensity={float(data.inten[0]):.1f} mm/h  "
                          f"duration={float(data.dur[0]):.0f} min",
                          fontsize=10)

        im1 = axes[1].imshow(attn_map, cmap="hot_r", origin="upper")
        fig.colorbar(im1, ax=axes[1], label="Σ attention received")
        axes[1].set_title("Last-Layer Attention\n"
                          "(high = neighbours focus here during message passing)",
                          fontsize=10)

        for ax in axes:
            ax.set_xlabel("Column (W→E)"); ax.set_ylabel("Row (N→S)")
        fig.suptitle(f"GAT Attention Map — Test Scenario {i+1}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"10_attention_sc{i+1:02d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Attention maps saved → {out_dir}/10_attention_sc*.png")

# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Startup diagnostics ───────────────────────────────────────────────────
    print(f"{'='*65}")
    print(f"  Graph Attention Network — Rostock Flood Surrogate")
    print(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024**3
        print(f"  GPU    : {props.name}  ({vram:.1f} GB VRAM)")
        print(f"  CUDA   : {torch.version.cuda}")
    else:
        print("  ⚠️  No CUDA GPU detected — running on CPU.")
        print("     Consider using Google Colab (Runtime → T4 GPU).")
    print(f"{'='*65}")

    # ── Load static graph ─────────────────────────────────────────────────────
    g  = np.load(os.path.join(SCENARIOS_DIR, "graph_static.npz"))
    H, W = int(g["H"]), int(g["W"])
    N    = H * W

    static_arr = np.stack([
        g["elevation"].flatten(),       # col 0
        g["street_mask"].flatten(),     # col 1
        g["building_mask"].flatten(),   # col 2
        g["existing_gi"].flatten(),     # col 3
        g["dist_to_street"].flatten(),  # col 4
        g["dist_to_outlet"].flatten(),  # col 5
        g["upstream_area"].flatten(),   # col 6  ← used in physics_loss
        g["water_mask_sink"].flatten(), # col 7
    ], axis=1).astype(np.float32)

    # Edge index: shared by all scenarios, loaded once and kept on CPU.
    # PyG DataLoader moves it to GPU automatically during batching.
    edge_index = torch.from_numpy(
        g["edge_index"].astype(np.int64))                   # (2, E)

    # Edge attributes: slope = dZ / distance  (m/m, dimensionless)
    #
    # With 8-connectivity, edges have two possible lengths:
    #   Cardinal  (N/S/E/W)  : distance = res             = 3.000 m
    #   Diagonal  (NE/NW/SE/SW): distance = res × √2      ≈ 4.243 m
    #
    # Using dZ alone (as before) would give a steeper apparent gradient on
    # diagonal edges for the same real slope, because the nodes are farther
    # apart.  Dividing by distance corrects for this and gives true slope
    # in m/m — the physically meaningful quantity that drives flow.
    #
    # edge_distances is saved in graph_static.npz as a multiplier of res
    # (1.0 for cardinal, √2 for diagonal).  If the file pre-dates 8-connectivity
    # and the key is missing, we fall back to assuming all edges are cardinal.
    RES = 3.0    # grid resolution in metres — must match scenario generation
    elev_flat = static_arr[:, 0]                             # (N,) elevation m
    src_np    = edge_index[0].numpy()
    dst_np    = edge_index[1].numpy()
    dz        = elev_flat[src_np] - elev_flat[dst_np]        # (E,) raw dZ  m

    if "edge_distances" in g:
        # edge_distances stored as multipliers of RES (1.0 or √2)
        dist_m = g["edge_distances"].astype(np.float32) * RES  # (E,) metres
        print("  Edge distances : loaded from graph_static.npz  "
              "(cardinal + diagonal)")
    else:
        # Fallback: old 4-connected file — all edges are cardinal length
        dist_m = np.full(len(src_np), RES, dtype=np.float32)
        print("  ⚠️  edge_distances not found in graph_static.npz — "
              "assuming all edges are cardinal (4-connectivity).")
        print("     Rebuild graph_static.npz with 8-connectivity for "
              "correct diagonal slopes.")

    slope         = dz / dist_m                              # (E,) m/m  true slope
    slope_mean    = slope.mean()
    slope_std     = slope.std() + 1e-8
    slope_norm    = (slope - slope_mean) / slope_std
    edge_attr     = torch.from_numpy(slope_norm).view(-1, 1).float()  # (E, 1)

    n_cardinal = int((dist_m == RES).sum())
    n_diagonal = len(dist_m) - n_cardinal
    print(f"\n  Grid       : {H}×{W} = {N:,} nodes")
    print(f"  Edges      : {edge_index.shape[1]:,}  "
          f"({n_cardinal:,} cardinal + {n_diagonal:,} diagonal)")
    print(f"  Edge attr  : slope (dZ/dist) normalised  "
          f"mean={slope_mean:.4f} m/m  std={slope_std:.4f} m/m")

    # ── Scenario paths + split ────────────────────────────────────────────────
    paths = sorted(glob.glob(os.path.join(SCENARIOS_DIR, "scenario_?????.npz")))
    n     = len(paths)
    print(f"  Scenarios  : {n}")

    rng   = np.random.default_rng(SEED)
    perm  = rng.permutation(n)
    n_tr  = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    tr_paths  = [paths[i] for i in perm[:n_tr]]
    val_paths = [paths[i] for i in perm[n_tr : n_tr + n_val]]
    te_paths  = [paths[i] for i in perm[n_tr + n_val:]]
    print(f"  Split      : {n_tr} train / {n_val} val / {len(te_paths)} test\n")

    # ── Resume detection ─────────────────────────────────────────────────────
    best_pt_path   = os.path.join(OUTPUT_DIR, "gat_best.pt")
    study_pkl_path = os.path.join(OUTPUT_DIR, "optuna_study_gat.pkl")

    if os.path.exists(best_pt_path):
        # ── FULL RESUME: training already complete ────────────────────────────
        print(f"\n  ✅ Found existing checkpoint: {best_pt_path}")
        print(f"  ⏭️  Skipping Phase 1 (HPO) and Phase 2 (training).")
        print(f"  Running attention visualisation only.\n")

        ckpt = torch.load(best_pt_path, map_location=DEVICE, weights_only=False)

        # Restore normaliser from checkpoint (no need to refit)
        norm_vis       = Normalizer()
        norm_vis.mean  = ckpt["feat_mean"]
        norm_vis.std   = ckpt["feat_std"]

        # Rebuild model with saved hparams
        best_model = FloodGAT(
            in_channels     = 14,
            hidden_channels = ckpt["hidden_channels"],
            num_layers      = ckpt["num_layers"],
            heads           = ckpt["heads"],
            dropout         = ckpt["dropout"],
        ).to(DEVICE)
        # torch.compile() prefixes all keys with "_orig_mod." when the
        # model is compiled before saving.  Strip the prefix so the state
        # dict loads cleanly into an uncompiled FloodGAT instance.
        raw_sd     = ckpt["model_state"]
        clean_sd   = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in raw_sd.items()
        }
        best_model.load_state_dict(clean_sd)

        # Print saved test metrics so they are visible in the console
        tm = ckpt.get("test_metrics", {})
        if tm:
            print(f"  Saved test metrics:")
            print(f"    RMSE          : {tm.get('rmse',      'n/a')}")
            print(f"    RMSE (>10 cm) : {tm.get('rmse_10cm', 'n/a')}")
            print(f"    MAE           : {tm.get('mae',       'n/a')}")
            print(f"    R²            : {tm.get('r2',        'n/a')}")
            print(f"    Bias          : {tm.get('bias',      'n/a')}")
            print(f"    Speed         : {tm.get('ms_per_scenario', 'n/a')} ms/scenario")

        visualise_attention(best_model, te_paths, static_arr, edge_index,
                            edge_attr, norm_vis, H, W, OUTPUT_DIR, n_scenarios=3)

    elif os.path.exists(study_pkl_path):
        # ── PARTIAL RESUME: HPO done, training not yet complete ───────────────
        print(f"\n  ✅ Found existing HPO study: {study_pkl_path}")
        print(f"  ⏭️  Skipping Phase 1 — loading saved best hparams.")
        with open(study_pkl_path, "rb") as f:
            study = pickle.load(f)
        best_hparams = study.best_params
        print(f"  Best trial #{study.best_trial.number}  "
              f"(val loss = {study.best_value:.6f})")
        for k, v in best_hparams.items():
            print(f"    {k:<22} = {v}")

        # Shared normaliser needed for Phase 2
        print("\n  Fitting shared normaliser on 60 training scenarios…")
        norm_shared = Normalizer()
        norm_shared.fit(tr_paths, static_arr, n_sample=60, seed=SEED)

        # ── Phase 2 : Full training ───────────────────────────────────────────
        train(best_hparams, tr_paths, val_paths, te_paths,
              static_arr, edge_index, edge_attr, H, W)

        # ── Attention visualisation ───────────────────────────────────────────
        ckpt = torch.load(best_pt_path, map_location=DEVICE, weights_only=False)
        best_model = FloodGAT(
            in_channels     = 14,
            hidden_channels = ckpt["hidden_channels"],
            num_layers      = ckpt["num_layers"],
            heads           = ckpt["heads"],
            dropout         = ckpt["dropout"],
        ).to(DEVICE)
        # torch.compile() prefixes all keys with "_orig_mod." when the
        # model is compiled before saving.  Strip the prefix so the state
        # dict loads cleanly into an uncompiled FloodGAT instance.
        raw_sd     = ckpt["model_state"]
        clean_sd   = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in raw_sd.items()
        }
        best_model.load_state_dict(clean_sd)
        norm_vis       = Normalizer()
        norm_vis.mean  = ckpt["feat_mean"]
        norm_vis.std   = ckpt["feat_std"]
        visualise_attention(best_model, te_paths, static_arr, edge_index,
                            edge_attr, norm_vis, H, W, OUTPUT_DIR, n_scenarios=3)

    else:
        # ── FRESH RUN: nothing saved yet ─────────────────────────────────────
        print("  Fitting shared normaliser on 60 training scenarios…")
        norm_shared = Normalizer()
        norm_shared.fit(tr_paths, static_arr, n_sample=60, seed=SEED)

        # ── Phase 1 : HPO ─────────────────────────────────────────────────────
        best_hparams, study = run_hpo(
            tr_paths, val_paths, static_arr, edge_index, edge_attr, norm_shared)

        # ── Phase 2 : Full training ───────────────────────────────────────────
        train(best_hparams, tr_paths, val_paths, te_paths,
              static_arr, edge_index, edge_attr, H, W)

        # ── Attention visualisation ───────────────────────────────────────────
        ckpt = torch.load(best_pt_path, map_location=DEVICE, weights_only=False)
        best_model = FloodGAT(
            in_channels     = 14,
            hidden_channels = ckpt["hidden_channels"],
            num_layers      = ckpt["num_layers"],
            heads           = ckpt["heads"],
            dropout         = ckpt["dropout"],
        ).to(DEVICE)
        # torch.compile() prefixes all keys with "_orig_mod." when the
        # model is compiled before saving.  Strip the prefix so the state
        # dict loads cleanly into an uncompiled FloodGAT instance.
        raw_sd     = ckpt["model_state"]
        clean_sd   = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in raw_sd.items()
        }
        best_model.load_state_dict(clean_sd)
        norm_vis       = Normalizer()
        norm_vis.mean  = ckpt["feat_mean"]
        norm_vis.std   = ckpt["feat_std"]
        visualise_attention(best_model, te_paths, static_arr, edge_index,
                            edge_attr, norm_vis, H, W, OUTPUT_DIR, n_scenarios=3)
