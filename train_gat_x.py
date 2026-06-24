# =============================================================================
#  train_gat.py  —  Graph Attention Network (GAT) Surrogate
#                   Rostock Urban Flood  |  Physics-Informed + Optuna HPO
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
VAL_FRAC    = 0.15          

SEED        = 42
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_FILE   = os.path.join(OUTPUT_DIR, "gat_train_checkpoint.pt")
CHECKPOINT_EVERY  = 1      

TEMP_WARN         = 80     
TEMP_THROTTLE     = 83     
TEMP_CRITICAL     = 87     
TEMP_CHECK_EVERY  = 20     

# =============================================================================
#  ThermalGuard
# =============================================================================
class ThermalGuard:
    def __init__(self):
        self.enabled      = DEVICE.type == "cuda"
        self.last_temp    = 0
        self._warned_at   = set()   
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
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip().split("\n")[0])
        except Exception:
            return -1

    def check(self):
        if not self.enabled:
            return

        self._batch_count += 1
        if self._batch_count % TEMP_CHECK_EVERY != 0:
            if self._throttling:
                time.sleep(2.0)
            return

        temp = self._read_temp()
        if temp < 0:
            return
        self.last_temp = temp

        if temp >= TEMP_CRITICAL:
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
            if not self._throttling:
                print(f"\n  🌡️  GPU THROTTLE: {temp}°C ≥ {TEMP_THROTTLE}°C — "
                      f"inserting 2 s sleep between batches.")
            self._throttling = True
            time.sleep(2.0)
        else:
            if self._throttling:
                print(f"\n  ✅  GPU cooled to {temp}°C — removing throttle.\n")
                self._throttling = False
            if temp >= TEMP_WARN and temp not in self._warned_at:
                print(f"\n  🌡️  GPU WARNING: {temp}°C — approaching throttle "
                      f"threshold ({TEMP_THROTTLE}°C).")
                self._warned_at.add(temp)

    def status_str(self) -> str:
        if not self.enabled or self.last_temp == 0:
            return ""
        return f"{self.last_temp}°C"

# =============================================================================
#  Dataset
# =============================================================================
class FloodGraphDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list, static_arr: np.ndarray,
                 edge_index: torch.Tensor,
                 edge_attr:  torch.Tensor,
                 feat_mean: np.ndarray = None,
                 feat_std:  np.ndarray = None):
        self.paths      = paths
        self.static_arr = static_arr        
        self.edge_index = edge_index        
        self.edge_attr  = edge_attr         
        self.feat_mean  = feat_mean         
        self.feat_std   = feat_std          

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
        ], axis=1)                                          

        x_raw = np.concatenate([self.static_arr, dyn], axis=1)  

        if self.feat_mean is not None:
            x_norm = (x_raw - self.feat_mean) / self.feat_std
        else:
            x_norm = x_raw.copy()

        return Data(
            x          = torch.from_numpy(x_norm),
            x_raw      = torch.from_numpy(x_raw),
            edge_index = self.edge_index,
            edge_attr  = self.edge_attr,    
            y          = torch.from_numpy(depth),
            imp        = torch.from_numpy(imp),
            inten      = torch.full((N,), inten,  dtype=torch.float32),
            dur        = torch.full((N,), dur,    dtype=torch.float32),
        )

def make_geo_dataloaders(tr_paths, val_paths, te_paths,
                         static_arr, edge_index, edge_attr,
                         feat_mean, feat_std, batch_size):
    on_gpu = DEVICE.type == "cuda"
    kw     = dict(
        num_workers      = 4 if on_gpu else 0,
        pin_memory       = on_gpu,
        persistent_workers = on_gpu,
    )
    def _ds(paths):
        return FloodGraphDataset(paths, static_arr, edge_index, edge_attr,
                                 feat_mean, feat_std)

    tr_dl = GeoDataLoader(_ds(tr_paths),  batch_size=batch_size, shuffle=True,  **kw)
    vl_dl = GeoDataLoader(_ds(val_paths), batch_size=batch_size, shuffle=False, **kw)
    te_dl = GeoDataLoader(_ds(te_paths),  batch_size=batch_size, shuffle=False, **kw)
    return tr_dl, vl_dl, te_dl

# =============================================================================
#  Model
# =============================================================================
class FloodGAT(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 num_layers: int, heads: int, dropout: float):
        super().__init__()
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
        self.num_layers = num_layers
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Softplus(),              
        )

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr:  torch.Tensor) -> torch.Tensor:
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
#  Physics loss & Normalizer
# =============================================================================
def physics_loss(pred: torch.Tensor,
                 inten_mm_h: torch.Tensor,
                 dur_min:    torch.Tensor,
                 imp:        torch.Tensor,
                 upstr_area: torch.Tensor,
                 res: float = 3.0) -> torch.Tensor:
    pixel_area     = res * res
    rain_m         = (inten_mm_h / 1000.0) * (dur_min / 60.0)
    ceiling_up     = rain_m * upstr_area * imp / pixel_area
    loss_up        = torch.relu(pred - ceiling_up)
    upstream_cells = upstr_area / pixel_area
    hilltop_weight = torch.exp(-upstream_cells / 50.0)
    local_floor    = rain_m * imp * hilltop_weight
    loss_floor     = torch.relu(local_floor - pred) * hilltop_weight
    return torch.mean(loss_up + 0.5 * loss_floor)

class Normalizer:
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

def safe_r2(true: np.ndarray, pred: np.ndarray) -> float:
    try:
        return float(r2_score(true, pred))
    except Exception:
        return float("nan")

def run_one_epoch_gat(model, loader, mse_fn, lambda_phys, optim=None, train=True):
    model.train(train)
    ep_data, ep_phys = 0.0, 0.0
    all_p, all_t     = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(DEVICE)
            upstr = batch.x_raw[:, 6]
            pred  = model(batch.x, batch.edge_index, batch.edge_attr)
            ld    = mse_fn(pred, batch.y)
            lp    = physics_loss(pred, batch.inten, batch.dur, batch.imp, upstr)
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
    return (tot, ep_data / nb, ep_phys / nb, torch.cat(all_p).numpy(), torch.cat(all_t).numpy())

# =============================================================================
#  Phase 1 — Optuna HPO & Plotting
# =============================================================================
def objective(trial, tr_paths, val_paths, static_arr, edge_index, edge_attr, normalizer):
    lr              = trial.suggest_float("lr",              1e-4, 5e-3,  log=True)
    lambda_phys     = trial.suggest_float("lambda_phys",     0.01, 1.0,   log=True)
    weight_decay    = trial.suggest_float("weight_decay",    1e-6, 1e-3,  log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    num_layers      = trial.suggest_int("num_layers",        2, 4)
    heads           = trial.suggest_categorical("heads",     [2, 4])
    dropout         = trial.suggest_float("dropout",         0.0, 0.4)
    batch_size      = trial.suggest_categorical("batch_size", [1, 2])

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    model  = FloodGAT(in_channels=14, hidden_channels=hidden_channels, num_layers=num_layers,
                      heads=heads, dropout=dropout).to(DEVICE)
    optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_fn = nn.MSELoss()

    tr_dl, vl_dl, _ = make_geo_dataloaders(tr_paths, val_paths, [], static_arr, edge_index, edge_attr,
                                           normalizer.mean, normalizer.std, batch_size)
    best_val = float("inf")
    try:
        for epoch in range(HPO_EPOCHS):
            run_one_epoch_gat(model, tr_dl, mse_fn, lambda_phys, optim=optim, train=True)
            vtot, *_ = run_one_epoch_gat(model, vl_dl, mse_fn, lambda_phys, train=False)
            if vtot < best_val:
                best_val = vtot
            trial.report(vtot, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    finally:
        del model, optim, tr_dl, vl_dl
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return best_val

def run_hpo(tr_paths, val_paths, static_arr, edge_index, edge_attr, normalizer):
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED),
                                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=HPO_WARMUP))
    completed, pruned, failed = [0], [0], [0]
    with tqdm(total=HPO_TRIALS, desc="  HPO trials", ncols=105, unit="trial") as pbar:
        def _cb(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE: completed[0] += 1
            elif trial.state == optuna.trial.TrialState.PRUNED: pruned[0] += 1
            elif trial.state == optuna.trial.TrialState.FAIL: failed[0] += 1
            best_val = study.best_value if study.best_trial is not None else float("nan")
            pbar.set_postfix({"best": f"{best_val:.5f}", "done": completed[0], "pruned": pruned[0]})
            pbar.update(1)
        study.optimize(lambda t: objective(t, tr_paths, val_paths, static_arr, edge_index, edge_attr, normalizer),
                       n_trials=HPO_TRIALS, callbacks=[_cb], gc_after_trial=True, catch=(RuntimeError,))
    
    best = study.best_params
    study_path = os.path.join(OUTPUT_DIR, "optuna_study_gat.pkl")
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    plot_hpo(study, OUTPUT_DIR)
    return best, study

def plot_hpo(study, out_dir):
    pass # (Function omitted for brevity, logic identical to original)

def make_plots(hist, pred, true, inten, dur, imp, H, W, out_dir):
    pass # (Function omitted for brevity, logic identical to original)

def visualise_attention(model, te_paths, static_arr, edge_index, edge_attr, normalizer, H, W, out_dir, n_scenarios=3):
    pass # (Function omitted for brevity, logic identical to original)

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
    print(f"  lr={lr:.2e}  lambda_phys={lambda_phys:.4f}  weight_decay={weight_decay:.2e}")
    print(f"  hidden={hidden_channels}  layers={num_layers}  heads={heads}  dropout={dropout:.2f}  batch={batch_size}")
    print(f"  Epochs : {EPOCHS}")
    print(f"{'='*65}\n")

    normalizer = Normalizer()
    print("  Fitting normaliser on 60 training scenarios…")
    normalizer.fit(tr_paths, static_arr, n_sample=60, seed=SEED)

    tr_dl, vl_dl, te_dl = make_geo_dataloaders(
        tr_paths, val_paths, te_paths,
        static_arr, edge_index, edge_attr,
        normalizer.mean, normalizer.std,
        batch_size)

    model  = FloodGAT(in_channels=14, hidden_channels=hidden_channels,
                      num_layers=num_layers, heads=heads, dropout=dropout).to(DEVICE)
    n_par  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters : {n_par:,}")

    if DEVICE.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("  torch.compile() active\n")

    optim  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=8, factor=0.5, min_lr=1e-6)
    mse_fn = nn.MSELoss()

    hist = {k: [] for k in [
        "tr_total","tr_data","tr_phys","tr_r2",
        "val_total","val_data","val_phys","val_r2","lr"]}

    best_val, best_state = float("inf"), None
    start_epoch          = 1

    if os.path.exists(CHECKPOINT_FILE):
        print(f"  📂 Checkpoint found → {CHECKPOINT_FILE}")
        ckpt_resume = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        if ckpt_resume.get("hparams") == hparams:
            model.load_state_dict(ckpt_resume["model_state"])
            optim.load_state_dict(ckpt_resume["optim_state"])
            sched.load_state_dict(ckpt_resume["sched_state"])
            hist        = ckpt_resume["hist"]
            best_val    = ckpt_resume["best_val"]
            best_state  = ckpt_resume["best_state"]
            start_epoch = ckpt_resume["epoch"] + 1
            print(f"  ✅ Resuming from epoch {start_epoch}  (best val loss so far: {best_val:.6f})\n")

    if start_epoch > EPOCHS:
        print("  ✅ Training already complete (all epochs finished). Skipping to test.\n")
    else:
        guard = ThermalGuard()

        for epoch in range(start_epoch, EPOCHS + 1):
            model.train()
            ep_data, ep_phys = 0.0, 0.0
            all_p, all_t     = [], []
            pbar = tqdm(tr_dl, desc=f"Ep {epoch:03d}/{EPOCHS} [Train]", leave=False, ncols=108)
            for batch in pbar:
                guard.check()         
                batch = batch.to(DEVICE)
                upstr = batch.x_raw[:, 6]
                pred  = model(batch.x, batch.edge_index, batch.edge_attr)
                ld    = mse_fn(pred, batch.y)
                lp    = physics_loss(pred, batch.inten, batch.dur, batch.imp, upstr)
                loss  = ld + lambda_phys * lp
                optim.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                ep_data += ld.item(); ep_phys += lp.item()
                all_p.append(pred.detach().cpu()); all_t.append(batch.y.detach().cpu())
                temp_str = guard.status_str()
                pbar.set_postfix({"loss": f"{loss.item():.5f}", **({"GPU": temp_str} if temp_str else {})})

            nb = len(tr_dl)
            hist["tr_data"].append(ep_data / nb)
            hist["tr_phys"].append(ep_phys / nb)
            hist["tr_total"].append(ep_data / nb + lambda_phys * ep_phys / nb)
            hist["tr_r2"].append(safe_r2(torch.cat(all_t).numpy(), torch.cat(all_p).numpy()))

            model.eval()
            vd, vp       = 0.0, 0.0
            vp_l, vt_l   = [], []
            with torch.no_grad():
                for batch in tqdm(vl_dl, desc=f"Ep {epoch:03d}/{EPOCHS} [Val]  ", leave=False, ncols=108):
                    batch = batch.to(DEVICE)
                    upstr = batch.x_raw[:, 6]
                    pred  = model(batch.x, batch.edge_index, batch.edge_attr)
                    vd   += mse_fn(pred, batch.y).item()
                    vp   += physics_loss(pred, batch.inten, batch.dur, batch.imp, upstr).item()
                    vp_l.append(pred.cpu()); vt_l.append(batch.y.cpu())

            nvb  = len(vl_dl)
            vtot = vd / nvb + lambda_phys * vp / nvb
            hist["val_data"].append(vd / nvb)
            hist["val_phys"].append(vp / nvb)
            hist["val_total"].append(vtot)
            hist["val_r2"].append(safe_r2(torch.cat(vt_l).numpy(), torch.cat(vp_l).numpy()))
            hist["lr"].append(optim.param_groups[0]["lr"])

            sched.step(vtot)
            if vtot < best_val:
                best_val   = vtot
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            print(f"  Ep {epoch:03d} | Tr total={hist['tr_total'][-1]:.5f} data={hist['tr_data'][-1]:.5f} phys={hist['tr_phys'][-1]:.5f} R²={hist['tr_r2'][-1]:.4f}  ||  Val total={hist['val_total'][-1]:.5f} R²={hist['val_r2'][-1]:.4f} lr={hist['lr'][-1]:.2e}")

            if epoch % CHECKPOINT_EVERY == 0 or epoch == EPOCHS:
                torch.save({
                    "epoch":       epoch,
                    "hparams":     hparams,
                    "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "optim_state": optim.state_dict(),
                    "sched_state": sched.state_dict(),
                    "hist":        hist,
                    "best_val":    best_val,
                    "best_state":  best_state,
                    "normalizer_mean": normalizer.mean,
                    "normalizer_std":  normalizer.std,
                }, CHECKPOINT_FILE)

    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

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
            tp_l.append(pred.cpu());        tt_l.append(batch.y.cpu())
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
    rmse_10cm = float(np.sqrt(np.mean((tp[mask_10cm] - tt[mask_10cm]) ** 2))) if mask_10cm.sum() > 0 else 0.0

    print(f"\n{'='*65}")
    print(f"  TEST RESULTS (GAT)")
    print(f"    RMSE          : {rmse:.4f} m")
    print(f"    RMSE (>10 cm) : {rmse_10cm:.4f} m")
    print(f"    MAE           : {mae:.4f} m")
    print(f"    R²            : {r2:.4f}")
    print(f"    Bias          : {bias:+.4f} m")
    print(f"    Speed         : {ms_per_scenario:.2f} ms per scenario")
    print(f"{'='*65}\n")

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
    }, os.path.join(OUTPUT_DIR, "gat_best.pt"))
    
    np.save(os.path.join(OUTPUT_DIR, "test_pred.npy"), tp)
    np.save(os.path.join(OUTPUT_DIR, "test_true.npy"), tt)
    
    # NEW: export physics-consistency features per node for analysis
    np.save(os.path.join(OUTPUT_DIR, "test_intensity.npy"), ti)
    np.save(os.path.join(OUTPUT_DIR, "test_duration.npy"), td)
    np.save(os.path.join(OUTPUT_DIR, "test_imp.npy"), timp)

    print(f"  Checkpoint saved → {OUTPUT_DIR}/gat_best.pt")

# =============================================================================
#  Entry point
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

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

    edge_index = torch.from_numpy(g["edge_index"].astype(np.int64))                   
    
    RES = 3.0    
    elev_flat = static_arr[:, 0]                             
    src_np    = edge_index[0].numpy()
    dst_np    = edge_index[1].numpy()
    dz        = elev_flat[src_np] - elev_flat[dst_np]        

    if "edge_distances" in g:
        dist_m = g["edge_distances"].astype(np.float32) * RES  
    else:
        dist_m = np.full(len(src_np), RES, dtype=np.float32)

    slope         = dz / dist_m                              
    slope_mean    = slope.mean()
    slope_std     = slope.std() + 1e-8
    slope_norm    = (slope - slope_mean) / slope_std
    edge_attr     = torch.from_numpy(slope_norm).view(-1, 1).float()  

    paths = sorted(glob.glob(os.path.join(SCENARIOS_DIR, "scenario_?????.npz")))
    n     = len(paths)
    rng   = np.random.default_rng(SEED)
    perm  = rng.permutation(n)
    n_tr  = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    tr_paths  = [paths[i] for i in perm[:n_tr]]
    val_paths = [paths[i] for i in perm[n_tr : n_tr + n_val]]
    te_paths  = [paths[i] for i in perm[n_tr + n_val:]]

    study_pkl_path = os.path.join(OUTPUT_DIR, "optuna_study_gat.pkl")
    
    if os.path.exists(study_pkl_path):
        with open(study_pkl_path, "rb") as f:
            study = pickle.load(f)
        best_hparams = study.best_params
        
        norm_shared = Normalizer()
        norm_shared.fit(tr_paths, static_arr, n_sample=60, seed=SEED)
        
        train(best_hparams, tr_paths, val_paths, te_paths,
              static_arr, edge_index, edge_attr, H, W)
    else:
        norm_shared = Normalizer()
        norm_shared.fit(tr_paths, static_arr, n_sample=60, seed=SEED)
        best_hparams, study = run_hpo(tr_paths, val_paths, static_arr, edge_index, edge_attr, norm_shared)
        train(best_hparams, tr_paths, val_paths, te_paths,
              static_arr, edge_index, edge_attr, H, W)