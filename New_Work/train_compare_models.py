import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchdiffeq import odeint

# import both circuits
from circuit_original import Circuit as OriginalCircuit
from circuit import Circuit as NewCircuit   # LambertW version

import matplotlib.pyplot as plt
import numpy as np

# --------------------------
# Shared config
# --------------------------
SEED = 42
BATCH = 128
EPOCHS = 250
LR = 1e-3
STATE_DIM = 6
T0, T1 = 0.0, 1.0
STEP_SIZE = 0.2
N = 1000

torch.manual_seed(SEED)

# synthetic data
u = torch.rand(N, 2) * 4 - 2
y = torch.sin(u[:, :1]) + torch.cos(u[:, 1:2])

ds = TensorDataset(u, y)
n_train = int(0.8 * N)
train_ds, val_ds = random_split(ds, [n_train, N-n_train], generator=torch.Generator().manual_seed(SEED))
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

# --------------------------
# ODE wrapper
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
# Model (encoder -> ODE -> readout)
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
        x0 = self.encoder(u)
        t = torch.tensor([T0, T1], dtype=x0.dtype, device=x0.device)
        xt = odeint(self.func, x0, t, method='rk4', options={'step_size': STEP_SIZE})
        xT = xt[-1]
        yhat = self.readout(xT)
        nfe = self.func.pop_nfe()
        return yhat, nfe

# --------------------------
# Metrics
# --------------------------
def mse(pred, targ): return torch.mean((pred - targ)**2).item()
def mae(pred, targ): return torch.mean(torch.abs(pred - targ)).item()
def r2(pred, targ):
    eps = 1e-12
    ss_res = torch.sum((targ - pred)**2)
    ss_tot = torch.sum((targ - torch.mean(targ))**2) + eps
    return (1 - ss_res/ss_tot).item()

# --------------------------
# Epoch runner
# --------------------------
def run_epoch(model, loader, opt=None):
    training = opt is not None
    model.train(training)
    total, sum_mse, sum_mae, sum_nfe = 0,0,0,0
    yhats, ys = [], []
    for (u_b, y_b) in loader:
        if training: opt.zero_grad()
        yhat_b, nfe_b = model(u_b)
        loss = nn.MSELoss()(yhat_b, y_b)
        if training:
            loss.backward()
            opt.step()
        sum_mse += mse(yhat_b, y_b) * y_b.size(0)
        sum_mae += mae(yhat_b, y_b) * y_b.size(0)
        sum_nfe += nfe_b
        total   += y_b.size(0)
        yhats.append(yhat_b); ys.append(y_b)
    yhats = torch.cat(yhats); ys = torch.cat(ys)
    return {
        'mse': sum_mse/total,
        'mae': sum_mae/total,
        'r2': r2(yhats, ys),
        'nfe': sum_nfe/len(loader)
    }

# --------------------------
# Experiment runner
# --------------------------
def experiment(train_circuit_cls, test_circuit_cls, label):
    print(f"\n=== {label} ===")
    net_topo = torch.tensor([
        [0,1,2,3,4,5,6,5,3,2,1,0],
        [4,2,3,4,5,6,1,6,4,3,2,1]
    ], dtype=torch.long)

    # build train model
    train_circuit = train_circuit_cls(net_topo, {'use_diff': False, 'activation': 'relu'})
    model = BareODERegressor(train_circuit)
    opt = optim.AdamW(model.parameters(), lr=LR)

    start_time = time.time()
    # train
    for ep in range(1, EPOCHS+1):
        tr = run_epoch(model, train_loader, opt)
        va = run_epoch(model, val_loader, None)
        if ep % 25 == 0:
            print(f"Epoch {ep:03d}: train MSE={tr['mse']:.4f} val MSE={va['mse']:.4f} R2={va['r2']:.3f}")
    train_time = time.time() - start_time

    # if test model is different, swap circuit
    if test_circuit_cls is not train_circuit_cls:
        test_circuit = test_circuit_cls(net_topo, {'use_diff': False, 'activation': 'relu'})
        model.func = ODEFunc(test_circuit)

    final = run_epoch(model, val_loader, None)
    print(f"Final results ({label}): MSE={final['mse']:.4f} MAE={final['mae']:.4f} R2={final['r2']:.3f}")
    final['time_s'] = train_time
    return final

# --------------------------
# Run experiments and collect results
# --------------------------
results = {}
results["Original->Original"] = experiment(OriginalCircuit, OriginalCircuit, "Original->Original")
results["New->New"]           = experiment(NewCircuit, NewCircuit, "New->New")
results["Original->New"]      = experiment(OriginalCircuit, NewCircuit, "Original->New")

# --------------------------
# Plot MSE & MAE together
# --------------------------
metrics = ["MSE", "MAE"]
mses   = [results[k]['mse'] for k in results.keys()]
maes   = [results[k]['mae'] for k in results.keys()]
values_err = np.array([mses, maes])  # shape: (2 metrics, n_experiments)

x = np.arange(len(metrics))  # [0,1]
width = 0.25
colors = ['skyblue', 'lightgreen', 'salmon']
labels_exp = list(results.keys())

fig, ax = plt.subplots(figsize=(10,6))
for i, label in enumerate(labels_exp):
    ax.bar(x + i*width, values_err[:, i], width, label=label, color=colors[i])

ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylabel("Error")
ax.set_title("Comparison of Experiments: MSE & MAE")
ax.legend()

plt.tight_layout()
plt.show()

# --------------------------
# Plot R² separately
# --------------------------
r2s = [results[k]['r2'] for k in results.keys()]

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(labels_exp, r2s, color=colors)
ax.set_ylabel("R²")
ax.set_ylim(0, 1.05)  # keep scale clear
ax.set_title("Comparison of Experiments: R²")

plt.tight_layout()
plt.show()

# --------------------------
# Plot training times separately
# --------------------------
times = [results[k]['time_s'] for k in results.keys()]

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(labels_exp, times, color=colors)
ax.set_ylabel("Time (s)")
ax.set_title("Training Time per Experiment")

plt.tight_layout()
plt.show()
