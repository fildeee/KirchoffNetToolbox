# train_bare_regression.py  — metrics & NFE summary

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchdiffeq import odeint  # fixed-step RK4 for speed on CPU

from circuit import Circuit  # from bare/circuit.py

# --------------------------
# Fast defaults (tweak later)
# --------------------------
print(">>> starting run...", flush=True)
torch.set_num_threads(1)  # avoid CPU oversubscription on Windows

SEED = 42
BATCH = 128
EPOCHS = 10          # bump after smoke test
LR = 1e-3
STATE_DIM = 6
T0, T1 = 0.0, 1.0
STEP_SIZE = 0.2
N = 1000

torch.manual_seed(SEED)

# --------------------------
# Synthetic regression data
# --------------------------
u = torch.rand(N, 2) * 4 - 2
y = torch.sin(u[:, :1]) + torch.cos(u[:, 1:2])

ds = TensorDataset(u, y)
n_train = int(0.8 * N)
n_val = N - n_train
train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

# --------------------------
# Build Circuit (like example.py)
# --------------------------
device_cfg = {'use_diff': False, 'activation': 'relu'}
net_topo = torch.tensor([
    [0, 1, 2, 3, 4, 5, 6, 5, 3, 2, 1, 0],
    [4, 2, 3, 4, 5, 6, 1, 6, 4, 3, 2, 1]
], dtype=torch.long)
circuit = Circuit(net_topo, device_cfg)

# --------------------------
# NFE wrapper for the ODE func
# --------------------------
class ODEFunc(nn.Module):
    def __init__(self, circuit):
        super().__init__()
        self.circuit = circuit
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        return self.circuit(t, x)

    def pop_nfe(self):
        n = self.nfe
        self.nfe = 0
        return n

# --------------------------
# Model: encoder (2->6) -> ODE -> readout (6->1)
# --------------------------
class BareODERegressor(nn.Module):
    def __init__(self, circuit, in_dim=2, state_dim=STATE_DIM, out_dim=1):
        super().__init__()
        self.func = ODEFunc(circuit)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, state_dim)
        )
        self.readout = nn.Linear(state_dim, out_dim)

    def forward(self, u):
        x0 = self.encoder(u)  # [B, 6]
        t = torch.tensor([T0, T1], dtype=x0.dtype, device=x0.device)
        xt = odeint(self.func, x0, t, method='rk4', options={'step_size': STEP_SIZE})  # [2, B, 6]
        xT = xt[-1]
        yhat = self.readout(xT)
        nfe = self.func.pop_nfe()
        return yhat, nfe

model = BareODERegressor(circuit)
crit = nn.MSELoss()
opt  = optim.AdamW(model.parameters(), lr=LR)

# --------------------------
# Metrics helpers
# --------------------------
def mse(pred, targ):
    return torch.mean((pred - targ) ** 2).item()

def mae(pred, targ):
    return torch.mean(torch.abs(pred - targ)).item()

def r2(pred, targ):
    # 1 - SS_res / SS_tot, add small eps for safety
    eps = 1e-12
    y = targ
    yhat = pred
    ss_res = torch.sum((y - yhat) ** 2)
    ss_tot = torch.sum((y - torch.mean(y)) ** 2) + eps
    return (1.0 - (ss_res / ss_tot)).item()

# --------------------------
# Epoch runner (returns metrics + avg NFE)
# --------------------------
def run_epoch(loader, training=True):
    model.train(training)
    start = time.time()

    total = 0
    sum_mse = 0.0
    sum_mae = 0.0
    sum_nfe = 0

    with torch.set_grad_enabled(training):
        for i, (u_b, y_b) in enumerate(loader):
            if training:
                opt.zero_grad()

            yhat_b, nfe_b = model(u_b)
            loss_b = crit(yhat_b, y_b)

            if training:
                loss_b.backward()
                opt.step()

            # accumulate metrics (on CPU already)
            sum_mse += mse(yhat_b, y_b) * y_b.size(0)
            sum_mae += mae(yhat_b, y_b) * y_b.size(0)
            sum_nfe += nfe_b
            total   += y_b.size(0)

            if i % 5 == 0:
                print(f"   batch {i:03d}/{len(loader)}  loss={loss_b.item():.4f}", flush=True)

    dt = time.time() - start
    avg_mse = sum_mse / max(1, total)
    avg_mae = sum_mae / max(1, total)
    # For R² we compute once on the concatenated tensors for the epoch:
    # Re-run quickly without grads to get epoch-level predictions for R²
    with torch.no_grad():
        yhats, ys = [], []
        for (u_b, y_b) in loader:
            yhat_b, _ = model(u_b)
            yhats.append(yhat_b)
            ys.append(y_b)
        yhats = torch.cat(yhats, dim=0)
        ys    = torch.cat(ys, dim=0)
        avg_r2 = r2(yhats, ys)

    # per-forward NFE (rough: total_nfe / number_of_batches)
    avg_nfe = sum_nfe / max(1, len(loader))
    return {'mse': avg_mse, 'mae': avg_mae, 'r2': avg_r2, 'nfe': avg_nfe, 'time_s': dt}

# --------------------------
# Train loop
# --------------------------
best = float('inf')
hist = {'train': [], 'val': []}
train_wall = time.time()

for ep in range(1, EPOCHS + 1):
    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader,   False)

    hist['train'].append(tr)
    hist['val'].append(va)

    if va['mse'] < best:
        best = va['mse']
        torch.save(model.state_dict(), "bare_best.pth")

    print(f"Epoch {ep:03d} | "
          f"train MSE {tr['mse']:.4f} MAE {tr['mae']:.4f} R2 {tr['r2']:.3f} | "
          f"val MSE {va['mse']:.4f} MAE {va['mae']:.4f} R2 {va['r2']:.3f} | "
          f"avg NFE train {tr['nfe']:.1f} val {va['nfe']:.1f} | "
          f"time train {tr['time_s']:.2f}s val {va['time_s']:.2f}s | "
          f"best val MSE {best:.4f}",
          flush=True)

train_total_s = time.time() - train_wall

# --------------------------
# Final eval (load best) & summary
# --------------------------
model.load_state_dict(torch.load("bare_best.pth", map_location="cpu"))
final = run_epoch(val_loader, False)

print("\n==================== SUMMARY ====================")
print("Best checkpoint on validation set")
print(f"  MSE: {final['mse']:.6f}")
print(f"  MAE: {final['mae']:.6f}")
print(f"  R^2: {final['r2']:.6f}")
print(f"  Avg NFE (per forward): {final['nfe']:.2f}")
print("\nEfficiency")
print(f"  Total training wall time: {train_total_s:.2f}s")
avg_ep_time = sum(x['time_s'] for x in hist['train']) / max(1, len(hist['train']))
print(f"  Avg epoch train time:     {avg_ep_time:.2f}s")
print("=================================================\n")
