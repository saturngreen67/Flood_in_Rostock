# =============================================================================
#  train_pimlp.py  —  Physics-Informed MLP + Optuna HPO
#                     Rostock Urban Flood Surrogate
# =============================================================================
#
#  Execution flow
#  ──────────────
#  Phase 1 — Hyperparameter Optimisation (Optuna)
#    • N_TRIALS short runs (HPO_EPOCHS epochs each) on train+val splits
#    • MedianPruner kills bad trials after epoch HPO_WARMUP
#    • Optimises: lr, λ_phys, weight_decay, n_layers, width, batch_size
#    • Saves: optuna_study.pkl, 08_hpo_history.png, 09_hpo_parallel.png
#
#  Phase 2 — Full Training with best hyperparameters
#    • EPOCHS epochs, full progress bars, ReduceLROnPlateau scheduler
#    • Saves: pimlp_best.pt, test_pred.npy, test_true.npy
#    • Plots 01-07 (same as before)
#
#  Node features (14 total)
#  ────────────────────────
#    Static  (8): elevation, street_mask, building_mask, existing_gi,
#                 dist_to_street, dist_to_outlet, upstream_area, water_mask_sink
#    Dynamic (6): imperviousness, mannings_n, gi_type,
#                 intensity [broadcast], duration [broadcast], adoption_level
#
#  Loss
#  ────
#    L_total = L_data  +  λ · L_phys
#    L_data  = MSE(pred, simulated_depth)
#    L_phys  = mean| pred_i − (I[mm/h]/1000 · T[h] · imp_i) |
#
#  Install
#  ───────
#    pip install torch numpy scikit-learn tqdm matplotlib optuna
# =============================================================================

import os
import glob
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import optuna
from optuna.pruners import MedianPruner # stop trials that are not better than the median of completed trials at the same epoch
from optuna.samplers import TPESampler # Bayesian optimisation with Tree-structured Parzen Estimators (TPE)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress per-step noise

# =============================================================================
#  Fixed configuration  (not tuned by Optuna)
# =============================================================================
SCENARIOS_DIR = "scenarios"
OUTPUT_DIR    = "pimlp_results"

EPOCHS        = 100          # final training epochs.an Epoch represents one complete pass of the entire training dataset through the neural network. Since you are generating 2,000 scenarios for your Rostock study, one epoch means your model has "seen" and learned from every single one of those 2,000 flood maps exactly once.
HPO_TRIALS    = 40           # Optuna trials
HPO_EPOCHS    = 15           # epochs per trial (short warm-up)
HPO_WARMUP    = 5            # pruner: min epochs before pruning is allowed

TRAIN_FRAC    = 0.70         # train = 70% of scenarios
VAL_FRAC      = 0.15         # test = remaining 15%

SEED          = 42           # random seed for reproducibility (data splits, normaliser fit, HPO sampling)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
#  Dataset + collate
# =============================================================================

class FloodDataset(Dataset):# PyTorch Dataset class for loading flood scenarios. Each scenario is stored as a .npz file containing both static and dynamic features, as well as the target flood depth. The dataset returns a dictionary with the combined features and target for each scenario.
    def __init__(self, paths: list, static_arr: np.ndarray):
        self.paths      = paths
        self.static_arr = static_arr        # (N_nodes, 8)  float32

    def __len__(self): # Here we define the __len__ method, which returns the number of scenarios in the dataset. This is simply the length of the list of file paths provided to the dataset.
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict: # The __getitem__ method is responsible for loading and processing a single scenario from the dataset. It takes an index as input, loads the corresponding .npz file, extracts the dynamic features and target flood depth, and combines them with the static features to return a dictionary containing the input features (X), target values (y), and some individual dynamic features for convenience.
        d     = np.load(self.paths[idx]) # Load the .npz file corresponding to the given index. This file contains both static and dynamic features, as well as the target flood depth for the scenario.
        N     = self.static_arr.shape[0] # Get the number of nodes (N) from the shape of the static features array. This will be used to ensure that the dynamic features are properly broadcasted to match the number of nodes.
        imp   = d["imperviousness"].flatten().astype(np.float32) # Here d is the loaded .npz file, which is a dictionary-like object. We extract the "imperviousness" feature, flatten it to a 1D array, and convert it to float32. The same process is applied to the "mannings_n" and "gi_map" features. The rainfall intensity, duration, and adoption level are extracted as floats. Finally, we stack all the dynamic features together and concatenate them with the static features to form the final input feature matrix X.
        mann  = d["mannings_n"].flatten().astype(np.float32)
        gi    = d["gi_map"].flatten().astype(np.float32)
        inten = float(d["intensity"]) # Extract the dynamic features from the loaded scenario. The imperviousness, Manning's n, and GI type are flattened and converted to float32 arrays. The rainfall intensity and duration are extracted as floats.
        dur   = float(d["duration"])
        adopt = float(d["adoption_level"])
        dyn   = np.stack([
            imp, mann, gi,
            np.full(N, inten,  dtype=np.float32),
            np.full(N, dur,    dtype=np.float32),
            np.full(N, adopt,  dtype=np.float32),
        ], axis=1)
        return {
            "X":     np.concatenate([self.static_arr, dyn], axis=1), # Combine the static and dynamic features to create the input feature matrix X. The static features are repeated for each node, while the dynamic features are already broadcasted to match the number of nodes. The target variable y is extracted as the flood depth, flattened, and converted to float32. The individual dynamic features (imperviousness, intensity, duration) are also returned separately for convenience in the physics loss calculation.
            "y":     d["flood_depth"].flatten().astype(np.float32),
            "imp":   imp,
            "inten": np.float32(inten),
            "dur":   np.float32(dur),
        } # The reason that the dynamic features are returned separately (imp, inten, dur) is because they are needed in the physics loss calculation. The physics loss function requires the predicted flood depth, rainfall intensity, duration, and imperviousness to compute the physical regularization term. By returning these features separately, we can easily access them during training without having to extract them from the combined feature matrix X.


def collate_fn(batch: list): # The collate_fn function is a custom collate (Jam Avari) function used by the DataLoader to combine multiple samples from the dataset into a single batch. It takes a list of samples (where each sample is a dictionary returned by the __getitem__ method of the FloodDataset) and concatenates the features and targets across the batch dimension. The static features are already included in the "X" key of each sample, while the individual dynamic features (imperviousness, intensity, duration) are extracted and repeated for each node in the batch to match the shape of X.
    N     = batch[0]["X"].shape[0] # Get the number of nodes (N) from the first sample in the batch. This assumes that all samples in the batch have the same number of nodes, which should be the case since they are all from the same dataset. The features and targets from all samples in the batch are concatenated together along the first dimension (the batch dimension). The individual dynamic features (imperviousness, intensity, duration) are repeated for each node in the batch to ensure they have the same shape as X. Finally, all the features and targets are converted to PyTorch tensors and returned as a tuple.
    X     = torch.from_numpy(np.concatenate([b["X"]   for b in batch])) # Concatenate the input features (X) from all samples in the batch and convert to a PyTorch tensor. The same process is applied to the target variable (y) and the individual dynamic features (imp, inten, dur). The intensity and duration features are repeated for each node in the batch to match the shape of X, since they are originally scalars for each scenario.
    y     = torch.from_numpy(np.concatenate([b["y"]   for b in batch])) # Concatenate the target variable (y) from all samples in the batch and convert to a PyTorch tensor. The same process is applied to the input features (X) and the individual dynamic features (imp, inten, dur). The intensity and duration features are repeated for each node in the batch to match the shape of X, since they are originally scalars for each scenario.
    imp   = torch.from_numpy(np.concatenate([b["imp"] for b in batch])) # Concatenate the individual dynamic features (imperviousness) from all samples in the batch and convert to a PyTorch tensor. The same process is applied to the input features (X), target variable (y), and the other dynamic features (inten, dur). The intensity and duration features are repeated for each node in the batch to match the shape of X, since they are originally scalars for each scenario.
    inten = torch.tensor(np.repeat([b["inten"] for b in batch], N), dtype=torch.float32)
    dur   = torch.tensor(np.repeat([b["dur"]   for b in batch], N), dtype=torch.float32)
    return X, y, imp, inten, dur

# =============================================================================
#  Model
# =============================================================================

class PhysicsInformedMLP(nn.Module): # A simple feedforward neural network (MLP) that takes the combined static and dynamic features as input and predicts the flood depth for each node. The architecture consists of an input layer that accepts 14 features (8 static + 6 dynamic), followed by a configurable number of hidden layers with ReLU activations and batch normalization, and an output layer that produces a single predicted depth value for each node. The final activation function is Softplus, which ensures that the predicted depth is non-negative, as negative flood depths would not make physical sense. The number of hidden layers and their widths can be tuned as hyperparameters during the Optuna HPO phase.
    """14 → hidden layers → 1  (Softplus ensures predicted depth >= 0)."""
    def __init__(self, in_dim: int = 14, hidden: list = None):
        super().__init__()
        if hidden is None:
            hidden = [256, 256, 128, 64] # default architecture if not specified
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# =============================================================================
#  Physics loss
# =============================================================================

def physics_loss(pred, inten_mm_h, dur_min, imp, upstr_area, res=3.0):
    """
    Two-component physics regulariser.

    Component 1 — upstream volume ceiling (one-sided, always active):
        A node cannot hold more water than the total effective rainfall
        that can physically arrive from its entire upstream catchment.
        ceiling_up = rain_m × upstr_area × imp / pixel_area
        Uses ReLU so valley nodes with correctly large depths are NOT
        penalised — only genuine over-predictions are penalised.

    Component 2 — local rainfall floor on hilltop/flat impervious cells:
        On nearly-isolated impervious surfaces (rooftops, flat plazas)
        the model should not predict near-zero depth in heavy rain.
        A hilltop_weight (≈1 at ridge cells, ≈0 at valley nodes) gates
        this term so it is silent where routing governs depth.

    Why this fixes the abs() problem:
        The original abs() pulled every prediction toward the LOCAL
        rainfall ceiling, which wrongly punished routing.  Here,
        Component 1 uses the UPSTREAM ceiling (physically correct upper
        bound) and Component 2 only fires where there is no upstream
        area to route from.
    """
    pixel_area     = res * res
    rain_m         = (inten_mm_h / 1000.0) * (dur_min / 60.0)

    # -- Component 1: upstream volume ceiling (one-sided ReLU) ---------------
    ceiling_up     = rain_m * upstr_area * imp / pixel_area
    loss_up        = torch.relu(pred - ceiling_up)

    # -- Component 2: local floor, gated to hilltop/isolated cells only ------
    upstream_cells = upstr_area / pixel_area          # dimensionless cell count
    hilltop_weight = torch.exp(-upstream_cells / 50.0)  # ~1 at hilltops, ~0 downstream
    local_floor    = rain_m * imp * hilltop_weight
    loss_floor     = torch.relu(local_floor - pred) * hilltop_weight

    return torch.mean(loss_up + 0.5 * loss_floor)

# =============================================================================
#  Normaliser
# =============================================================================

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, paths, static_arr, n_sample=60, seed=42):
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

    def to_tensors(self, device):
        return (torch.tensor(self.mean, dtype=torch.float32).to(device),
                torch.tensor(self.std,  dtype=torch.float32).to(device))

# =============================================================================
#  Helpers
# =============================================================================

def safe_r2(true, pred):
    try:
        return float(r2_score(true, pred))
    except Exception:
        return float("nan")


def make_dataloaders(tr_paths, val_paths, te_paths, static_arr, batch_size):
    # num_workers: background processes pre-loading the next batch while the GPU
    #   trains on the current one. 4 is safe; raise to 8 on fast NVMe drives.
    #   Keep at 0 on Windows (multiprocessing spawn limitations).
    # pin_memory: keeps CPU tensors in page-locked RAM so the GPU can fetch them
    #   without an extra copy — measurably faster transfer to CUDA.
    # persistent_workers: keeps worker processes alive between epochs, avoiding
    #   the overhead of spawning/killing them every epoch.
    on_gpu = DEVICE.type == "cuda"
    kw = dict(
        collate_fn=collate_fn,
        num_workers=4 if on_gpu else 0,
        pin_memory=on_gpu,
        persistent_workers=on_gpu,
    )
    tr_dl = DataLoader(FloodDataset(tr_paths,  static_arr),
                       batch_size=batch_size, shuffle=True,  **kw)
    vl_dl = DataLoader(FloodDataset(val_paths, static_arr),
                       batch_size=batch_size, shuffle=False, **kw)
    te_dl = DataLoader(FloodDataset(te_paths,  static_arr),
                       batch_size=batch_size, shuffle=False, **kw)
    return tr_dl, vl_dl, te_dl

# =============================================================================
#  Phase 1 — Optuna objective
# =============================================================================

def objective(trial, tr_paths, val_paths, static_arr, normalizer):
    # ── Suggest hyperparameters ───────────────────────────────────────────────
    lr           = trial.suggest_float("lr",           1e-4, 5e-3, log=True)
    lambda_phys  = trial.suggest_float("lambda_phys",  0.01, 1.0,  log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    n_layers    = trial.suggest_int("n_layers", 2, 6)
    width_first = trial.suggest_categorical("width_first", [128, 256, 512])
    taper       = trial.suggest_float("taper", 0.4, 1.0)
    batch_size  = trial.suggest_categorical("batch_size", [4, 8, 16])

    hidden = [max(32, int(width_first * (taper ** i))) for i in range(n_layers)]

    model      = PhysicsInformedMLP(in_dim=14, hidden=hidden).to(DEVICE)
    t_mean, t_std = normalizer.to_tensors(DEVICE)
    optim_obj  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_fn     = nn.MSELoss()

    on_gpu = DEVICE.type == "cuda"
    kw = dict(
        collate_fn=collate_fn,
        num_workers=4 if on_gpu else 0,
        pin_memory=on_gpu,
        persistent_workers=on_gpu,
    )
    tr_dl = DataLoader(FloodDataset(tr_paths,  static_arr),
                       batch_size=batch_size, shuffle=True,  **kw)
    vl_dl = DataLoader(FloodDataset(val_paths, static_arr),
                       batch_size=batch_size, shuffle=False, **kw)

    best_val = float("inf")
    for epoch in range(HPO_EPOCHS):

        # ── mini train ──
        model.train()
        for X, y, imp, inten, dur in tr_dl:
            X, y            = X.to(DEVICE), y.to(DEVICE)
            imp, inten, dur = imp.to(DEVICE), inten.to(DEVICE), dur.to(DEVICE)
            upstr           = X[:, 6]          # upstream_area (m²), static column 6
            pred = model((X - t_mean) / t_std)
            loss = mse_fn(pred, y) + lambda_phys * physics_loss(
                       pred, inten, dur, imp, upstr)
            optim_obj.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim_obj.step()

        # ── mini val ──
        model.eval()
        vd, vp = 0.0, 0.0
        with torch.no_grad():
            for X, y, imp, inten, dur in vl_dl:
                X, y            = X.to(DEVICE), y.to(DEVICE)
                imp, inten, dur = imp.to(DEVICE), inten.to(DEVICE), dur.to(DEVICE)
                upstr           = X[:, 6]
                pred = model((X - t_mean) / t_std)
                vd  += mse_fn(pred, y).item()
                vp  += physics_loss(pred, inten, dur, imp, upstr).item()

        vtot = vd / len(vl_dl) + lambda_phys * vp / len(vl_dl)
        if vtot < best_val:
            best_val = vtot

        trial.report(vtot, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val


def run_hpo(tr_paths, val_paths, static_arr, normalizer):
    print(f"\n{'='*65}")
    print(f"  PHASE 1 — Hyperparameter Optimisation  (Optuna TPE)")
    print(f"  Trials : {HPO_TRIALS}   Epochs/trial : {HPO_EPOCHS}   "
          f"Pruner warmup : {HPO_WARMUP}")
    print(f"{'='*65}\n")

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=HPO_WARMUP),
    )

    completed, pruned = [0], [0]

    with tqdm(total=HPO_TRIALS, desc="  HPO trials", ncols=105, unit="trial") as pbar:
        def _cb(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                completed[0] += 1
            elif trial.state == optuna.trial.TrialState.PRUNED:
                pruned[0] += 1
            pbar.set_postfix({
                "best":    f"{study.best_value:.5f}",
                "done":    completed[0],
                "pruned":  pruned[0],
            })
            pbar.update(1)

        study.optimize(
            lambda t: objective(t, tr_paths, val_paths, static_arr, normalizer),
            n_trials=HPO_TRIALS,
            callbacks=[_cb],
            gc_after_trial=True,
        )

    best = study.best_params
    print(f"\n  Best trial  #{study.best_trial.number}  "
          f"(val loss = {study.best_value:.6f})")
    for k, v in best.items():
        print(f"    {k:<18} = {v}")

    study_path = os.path.join(OUTPUT_DIR, "optuna_study.pkl")
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"\n  Study saved → {study_path}")

    plot_hpo(study, OUTPUT_DIR)
    return best, study


def plot_hpo(study, out_dir):
    trials  = [t for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE]
    vals    = [t.value for t in trials]
    nums    = [t.number for t in trials]
    params  = list(study.best_params.keys())

    # ── 08. Optimisation history ──────────────────────────────────────────────
    best_so_far = np.minimum.accumulate(vals)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(nums, vals, s=22, alpha=0.55, color="steelblue", label="Trial val loss")
    ax.plot(nums, best_so_far, color="tomato", lw=2.0, label="Best so far")
    ax.set_xlabel("Trial number"); ax.set_ylabel("Validation loss (total)")
    ax.set_title(f"Optuna Optimisation History  ({len(trials)} completed trials)",
                 fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_hpo_history.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 09. Per-parameter scatter (colour = val loss) ─────────────────────────
    param_vals = {p: [t.params.get(p) for t in trials] for p in params}
    n_params   = len(params)
    fig, axes  = plt.subplots(1, n_params, figsize=(3.4 * n_params, 5), sharey=False)
    if n_params == 1:
        axes = [axes]

    norm_v = plt.Normalize(vmin=min(vals), vmax=np.percentile(vals, 75))
    cmap   = plt.cm.plasma_r
    rng0   = np.random.default_rng(1)

    for ax, p in zip(axes, params):
        pv     = param_vals[p]
        unique = sorted(set(v for v in pv if v is not None))
        if len(unique) <= 6:          # treat as categorical
            jitter  = rng0.uniform(-0.15, 0.15, len(pv))
            x_plot  = [unique.index(v) + j if v is not None else 0
                       for v, j in zip(pv, jitter)]
            ax.set_xticks(range(len(unique)))
            ax.set_xticklabels([str(u) for u in unique], fontsize=8, rotation=30)
        else:
            x_plot = [v if v is not None else float("nan") for v in pv]

        sc = ax.scatter(x_plot, vals, c=vals, cmap=cmap, norm=norm_v,
                        s=20, alpha=0.75, rasterized=True)
        best_idx = vals.index(min(vals))
        ax.scatter([x_plot[best_idx]], [vals[best_idx]],
                   marker="*", s=240, color="gold", zorder=5, label="best")
        ax.set_title(p, fontsize=9, pad=4)
        ax.set_ylabel("Val loss" if ax is axes[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, handletextpad=0.2)

    fig.colorbar(sc, ax=axes[-1], label="Val loss", shrink=0.75)
    fig.suptitle("HPO Parameter Sweep — each point is one completed trial",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_hpo_parallel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  HPO plots saved → {out_dir}/08_hpo_history.png, 09_hpo_parallel.png")

# =============================================================================
#  Phase 2 — Full training with best hyperparameters
# =============================================================================

def train(hparams: dict, tr_paths, val_paths, te_paths, static_arr, H, W):
    lr           = hparams.get("lr",           1e-3)
    lambda_phys  = hparams.get("lambda_phys",  0.1)
    weight_decay = hparams.get("weight_decay", 1e-5)
    n_layers    = hparams.get("n_layers", 4)
    width_first = hparams.get("width_first", 256)
    taper       = hparams.get("taper", 0.6)
    batch_size  = hparams.get("batch_size", 8)
    hidden_dims = [max(32, int(width_first * (taper ** i))) for i in range(n_layers)]

    print(f"\n{'='*65}")
    print(f"  PHASE 2 — Full Training")
    print(f"  lr={lr:.2e}  lambda_phys={lambda_phys:.4f}  "
          f"weight_decay={weight_decay:.2e}")
    print(f"  n_layers={n_layers}  width_first={width_first}  taper={taper}  batch_size={batch_size}")
    print(f"  Epochs : {EPOCHS}")
    print(f"{'='*65}\n")

    # Normaliser (refit on full training set for final run)
    normalizer = Normalizer()
    print("  Fitting normaliser on 60 training scenarios…")
    normalizer.fit(tr_paths, static_arr, n_sample=60, seed=SEED)
    t_mean, t_std = normalizer.to_tensors(DEVICE)

    tr_dl, vl_dl, te_dl = make_dataloaders(
        tr_paths, val_paths, te_paths, static_arr, batch_size)

    model  = PhysicsInformedMLP(in_dim=14, hidden=hidden_dims).to(DEVICE)
    n_par  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters : {n_par:,}")

    # torch.compile() (PyTorch >= 2.0) JIT-compiles the forward pass into
    # optimised GPU kernels — typically 10-30% faster at no cost to accuracy.
    # Falls back silently on CPU or older PyTorch versions.
    if DEVICE.type == "cuda" and hasattr(torch, "compile"):
        #model = torch.compile(model)
        print("  torch.compile() active — forward pass will be JIT-compiled\n")
    else:
        print()

    optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                 optim, patience=8, factor=0.5, min_lr=1e-6)
    mse_fn = nn.MSELoss()

    hist = {k: [] for k in [
        "tr_total","tr_data","tr_phys","tr_r2",
        "val_total","val_data","val_phys","val_r2","lr"]}

    best_val, best_state = float("inf"), None

    for epoch in range(1, EPOCHS + 1):

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        ep_data, ep_phys = 0.0, 0.0
        all_p, all_t     = [], []
        pbar = tqdm(tr_dl, desc=f"Ep {epoch:03d}/{EPOCHS} [Train]",
                    leave=False, ncols=108)
        for X, y, imp, inten, dur in pbar:
            X, y            = X.to(DEVICE), y.to(DEVICE)
            imp, inten, dur = imp.to(DEVICE), inten.to(DEVICE), dur.to(DEVICE)
            upstr           = X[:, 6]          # upstream_area (m²), static column 6
            pred = model((X - t_mean) / t_std)
            ld   = mse_fn(pred, y)
            lp   = physics_loss(pred, inten, dur, imp, upstr)
            loss = ld + lambda_phys * lp
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_data += ld.item(); ep_phys += lp.item()
            all_p.append(pred.detach().cpu()); all_t.append(y.detach().cpu())
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        nb = len(tr_dl)
        hist["tr_data"].append(ep_data / nb)
        hist["tr_phys"].append(ep_phys / nb)
        hist["tr_total"].append(ep_data / nb + lambda_phys * ep_phys / nb)
        hist["tr_r2"].append(
            safe_r2(torch.cat(all_t).numpy(), torch.cat(all_p).numpy()))

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        vd, vp       = 0.0, 0.0
        vp_l, vt_l   = [], []
        with torch.no_grad():
            for X, y, imp, inten, dur in tqdm(vl_dl,
                    desc=f"Ep {epoch:03d}/{EPOCHS} [Val]  ",
                    leave=False, ncols=108):
                X, y            = X.to(DEVICE), y.to(DEVICE)
                imp, inten, dur = imp.to(DEVICE), inten.to(DEVICE), dur.to(DEVICE)
                upstr           = X[:, 6]      # upstream_area (m²), static column 6
                pred = model((X - t_mean) / t_std)
                vd  += mse_fn(pred, y).item()
                vp  += physics_loss(pred, inten, dur, imp, upstr).item()
                vp_l.append(pred.cpu()); vt_l.append(y.cpu())

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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Ep {epoch:03d} | "
              f"Tr  total={hist['tr_total'][-1]:.5f} "
              f"data={hist['tr_data'][-1]:.5f} phys={hist['tr_phys'][-1]:.5f} "
              f"R²={hist['tr_r2'][-1]:.4f}  ||  "
              f"Val total={hist['val_total'][-1]:.5f} "
              f"R²={hist['val_r2'][-1]:.4f}  "
              f"lr={hist['lr'][-1]:.2e}")

    # ── Test ──────────────────────────────────────────────────────────────────
    print("\n  Running test set evaluation…")
    model.load_state_dict(best_state)
    model.eval()
    import time
    inference_start = time.time()
    tp_l, tt_l, ti_l, td_l, timp_l = [], [], [], [], []
    with torch.no_grad():
        for X, y, imp, inten, dur in tqdm(te_dl, desc="  Test", ncols=100):
            X, y            = X.to(DEVICE), y.to(DEVICE)
            imp, inten, dur = imp.to(DEVICE), inten.to(DEVICE), dur.to(DEVICE)
            pred = model((X - t_mean) / t_std)
            tp_l.append(pred.cpu()); tt_l.append(y.cpu())
            ti_l.append(inten.cpu()); td_l.append(dur.cpu()); timp_l.append(imp.cpu())
    inference_end = time.time()
    tp   = torch.cat(tp_l).numpy();   tt   = torch.cat(tt_l).numpy()
    ti   = torch.cat(ti_l).numpy();   td   = torch.cat(td_l).numpy()
    timp = torch.cat(timp_l).numpy()

    total_test_scenarios = len(te_dl.dataset)
    total_time = inference_end - inference_start
    ms_per_scenario = (total_time / total_test_scenarios) * 1000.0

    rmse = float(np.sqrt(np.mean((tp - tt) ** 2)))
    mae  = float(np.mean(np.abs(tp - tt)))
    r2   = safe_r2(tt, tp)
    bias = float(np.mean(tp - tt))
    mask_10cm = tt > 0.10
    if mask_10cm.sum() > 0:
        rmse_10cm = float(np.sqrt(np.mean((tp[mask_10cm] - tt[mask_10cm]) ** 2)))
    else:
        rmse_10cm = 0.0

    print(f"\n{'='*65}")
    print(f"  TEST RESULTS")
    print(f"    RMSE : {rmse:.4f} m")
    print(f"    RMSE (>10cm) : {rmse_10cm:.4f} m")
    print(f"    MAE  : {mae:.4f} m")
    print(f"    R²   : {r2:.4f}")
    print(f"    Bias : {bias:+.4f} m")
    print(f"    Speed        : {ms_per_scenario:.2f} ms per scenario")
    print(f"{'='*65}\n")

    torch.save({
        "model_state":  best_state,
        "feat_mean":    normalizer.mean,
        "feat_std":     normalizer.std,
        "H": H, "W": W,
        "hidden_dims":  hidden_dims,
        "lambda_phys":  lambda_phys,
        "best_hparams": hparams,
        "test_metrics": {"rmse": rmse, "rmse_10cm": rmse_10cm, "mae": mae, "r2": r2, "bias": bias, "speed_ms": ms_per_scenario},
    }, os.path.join(OUTPUT_DIR, "pimlp_best.pt"))
    np.save(os.path.join(OUTPUT_DIR, "test_pred.npy"), tp)
    np.save(os.path.join(OUTPUT_DIR, "test_true.npy"), tt)
    print(f"  Checkpoint saved → {OUTPUT_DIR}/pimlp_best.pt")

    make_plots(hist, tp, tt, ti, td, timp, H, W, OUTPUT_DIR)

# =============================================================================
#  Plots 01–07
# =============================================================================

def make_plots(hist, pred, true, inten, dur, imp, H, W, out_dir):
    E = np.arange(1, len(hist["tr_total"]) + 1)
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    rng = np.random.default_rng(0)
    res = pred - true
    s   = rng.choice(len(pred), size=min(200_000, len(pred)), replace=False)

    # 01 — Loss curves (log scale)
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
    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 02 — R² + learning rate
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(E, hist["tr_r2"],  color="steelblue", lw=1.8, label="Train R²")
    axes[0].plot(E, hist["val_r2"], color="tomato",    lw=1.8, ls="--", label="Val R²")
    axes[0].axhline(1.0, color="grey", lw=0.7, ls=":")
    axes[0].set_ylim(-0.1, 1.05); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("R²")
    axes[0].set_title("R² over Training"); axes[0].legend(fontsize=10)
    axes[1].plot(E, hist["lr"], color="mediumpurple", lw=1.8)
    axes[1].set_yscale("log"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate"); axes[1].set_title("Learning Rate Schedule")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_r2_and_lr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 03 — Hex-scatter predicted vs simulated
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae  = float(np.mean(np.abs(pred - true)))
    r2   = safe_r2(true, pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(true[s], pred[s], gridsize=80, cmap="plasma", mincnt=1, bins="log")
    fig.colorbar(hb, ax=ax, label="log₁₀(count)")
    lim = max(true[s].max(), pred[s].max()) * 1.05
    ax.plot([0, lim], [0, lim], "w--", lw=1.3, label="1:1 line")
    ax.set_xlabel("Simulated Depth (m)"); ax.set_ylabel("Predicted Depth (m)")
    ax.set_title(f"Test: Predicted vs Simulated\n"
                 f"RMSE={rmse:.4f} m   MAE={mae:.4f} m   R²={r2:.4f}", fontsize=11)
    ax.legend(fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_scatter_test.png"), dpi=150, bbox_inches="tight")
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
    fig.savefig(os.path.join(out_dir, "04_residuals.png"), dpi=150, bbox_inches="tight")
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
    ax.set_xlabel("Physics Ceiling  I×T×imp (m)"); ax.set_ylabel("Predicted Depth (m)")
    ax.set_title(f"Physics Consistency\n"
                 f"{pct_below:.1f}% of predictions ≤ rainfall ceiling", fontsize=11)
    ax.legend(markerscale=6, fontsize=10); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_physics_check.png"), dpi=150, bbox_inches="tight")
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
        ax.set_title(f"Spatial MAE  ({n_sc} test scenarios)", fontsize=12)
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
    ax.set_xticklabels([f"{v:.3f}" for v in bin_mid],
                       rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Simulated Depth bin (m)"); ax.set_ylabel("MAE (m)")
    ax.set_title("MAE by Depth Percentile Bin", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_mae_by_depth.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  All plots (01–07) saved → {out_dir}/")

# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"{'='*65}")
    print(f"  Physics-Informed MLP — Rostock Flood Surrogate")
    print(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram  = props.total_memory / 1024**3
        print(f"  GPU    : {props.name}  ({vram:.1f} GB VRAM)")
        print(f"  CUDA   : {torch.version.cuda}")
    else:
        print("  ⚠️  No CUDA GPU detected — running on CPU.")
        print("     Training will be significantly slower.")
        print("     See script header for GPU setup instructions.")
    print(f"{'='*65}")

    # ── Load static graph ─────────────────────────────────────────────────────
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

    # ── Scenario paths + split ────────────────────────────────────────────────
    paths = sorted(glob.glob(os.path.join(SCENARIOS_DIR, "scenario_?????.npz")))
    n     = len(paths)
    print(f"\n  Scenarios : {n}  |  Grid : {H}×{W} = {N:,} nodes")

    rng   = np.random.default_rng(SEED)
    perm  = rng.permutation(n)
    n_tr  = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    tr_paths  = [paths[i] for i in perm[:n_tr]]
    val_paths = [paths[i] for i in perm[n_tr : n_tr + n_val]]
    te_paths  = [paths[i] for i in perm[n_tr + n_val:]]
    print(f"  Split     : {n_tr} train / {n_val} val / {len(te_paths)} test\n")

    # ── Shared normaliser (fitted once, used in both phases) ─────────────────
    print("  Fitting shared normaliser on 60 training scenarios…")
    norm_shared = Normalizer()
    norm_shared.fit(tr_paths, static_arr, n_sample=60, seed=SEED)

    # ── Phase 1 : HPO ─────────────────────────────────────────────────────────
    #best_hparams, study = run_hpo(tr_paths, val_paths, static_arr, norm_shared)
    best_hparams = {
        "lr": 0.0009433681230446662,
        "lambda_phys": 0.027832050560952032,
        "weight_decay": 5.268448223015129e-06,
        "n_layers": 6,
        "width_first": 512,
        "taper": 0.7366589196799537,
        "batch_size": 4
    }

    # ── Phase 2 : Full training ───────────────────────────────────────────────
    #train(best_hparams, tr_paths, val_paths, te_paths, static_arr, H, W)
    train(best_hparams, tr_paths, val_paths, te_paths, static_arr, H, W)
