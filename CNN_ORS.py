
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader


dataset_name = "rabi"   # change to: "kharif", "rabi", "hyderabad"

if dataset_name == "kharif":
    df = pd.read_csv("Miryalaguda_ET_Kharif_training.csv")
elif dataset_name == "rabi":
    df = pd.read_csv("Miryalaguda_ET_Rabi_training.csv")
else:
    df = pd.read_csv("Hyderabad_ET_training new.csv")

# Create save directory
save_dir = f"plots/{dataset_name}"
os.makedirs(save_dir, exist_ok=True)

#  CLEAN DATA

for col in ['system:index', '.geo']:
    if col in df.columns:
        df = df.drop(columns=[col])

print("Columns:", df.columns.tolist())

#  TARGET + FEATURES

target_col = 'MODIS_ET' if 'MODIS_ET' in df.columns else df.columns[-1]

X = df.drop(columns=[target_col]).values
y = df[target_col].values

n_features = X.shape[1]
print(f"Dataset: {dataset_name} | Samples: {len(y)} | Features: {n_features}")

#  SPLIT  (70 / 15 / 15)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

#  SCALING  (fit on train only – avoids data leakage)

feature_scaler = StandardScaler()
X_train = feature_scaler.fit_transform(X_train)
X_val   = feature_scaler.transform(X_val)
X_test  = feature_scaler.transform(X_test)

target_scaler = StandardScaler()
y_train = target_scaler.fit_transform(y_train.reshape(-1, 1))
y_val   = target_scaler.transform(y_val.reshape(-1, 1))
y_test  = target_scaler.transform(y_test.reshape(-1, 1))

#  RESHAPE  →  (Batch, 1, n_features) for Conv1D

X_train = X_train.reshape(-1, 1, n_features)
X_val   = X_val.reshape(-1, 1, n_features)
X_test  = X_test.reshape(-1, 1, n_features)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=128)

#  MODEL
#    Architecture (per report & Che et al. 2022):
#      Conv1D(1→16, k=3) + BN + ReLU
#      Conv1D(16→32, k=2) + BN + ReLU
#      Flatten
#      FC(32 * conv_out → 64) + ReLU
#      Dropout(0.3)          ← corrected from 0.25
#      FC(64 → 1)
#
#    Batch Normalisation: reduces internal covariate shift,
#    acts as implicit regulariser (Che et al. 2022, Fig. 2b)
class ET_CNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        # Conv block 1
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.bn1   = nn.BatchNorm1d(16)            # FIX: added BN

        # Conv block 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=2)
        self.bn2   = nn.BatchNorm1d(32)            # FIX: added BN

        # Output length after two Conv1d layers (no padding, no pooling):
        #   after conv1: n_features - (3-1) = n_features - 2
        #   after conv2: (n_features - 2) - (2-1) = n_features - 3
        conv_output_size = n_features - 3

        # Fully connected head
        self.fc1     = nn.Linear(32 * conv_output_size, 64)
        self.dropout = nn.Dropout(0.3)             # FIX: 0.25 → 0.3 (matches report)
        self.fc2     = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))        # FIX: BN before ReLU
        x = F.relu(self.bn2(self.conv2(x)))        # FIX: BN before ReLU
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = ET_CNN(n_features)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

#  TRAINING
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs        = 200
patience      = 15
best_val_loss = float('inf')
counter       = 0
best_model    = None

train_losses = []
val_losses   = []

for epoch in range(epochs):

    # --- Train ---
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # --- Validate ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred     = model(xb)
            loss     = criterion(pred, yb)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch:>3d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter       = 0
        best_model    = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

model.load_state_dict(best_model)

#  EVALUATION
#    Metrics: RMSE, R², MAE, Bias (Mean Error)
#    All in original units (mm/8-day)
model.eval()
with torch.no_grad():
    pred_scaled = model(X_test).numpy()
    true_scaled = y_test.numpy()

# Inverse-transform back to mm/8-day
pred = target_scaler.inverse_transform(pred_scaled)
true = target_scaler.inverse_transform(true_scaled)

rmse = np.sqrt(mean_squared_error(true, pred))
r2   = r2_score(true, pred)
mae  = mean_absolute_error(true, pred)          # FIX: added MAE
bias = np.mean(pred - true)                     # FIX: added Bias (Mean Error)

print("\n" + "="*45)
print(f"FINAL RESULTS  —  {dataset_name.upper()}")
print("="*45)
print(f"  RMSE  : {rmse:.4f} mm/8-day")
print(f"  R²    : {r2:.4f}")
print(f"  MAE   : {mae:.4f} mm/8-day")        # FIX
print(f"  Bias  : {bias:.4f} mm/8-day")        # FIX
print("="*45)

#  PLOTS  (SAVE + SHOW)
#     All plots have proper axis labels and units

dataset_label = {
    "kharif":    "Kharif — Miryalaguda",
    "rabi":      "Rabi — Miryalaguda",
    "hyderabad": "Hyderabad"
}.get(dataset_name, dataset_name)

# ── 10.1  Loss Curve ──────────────────────────────────────────
plt.figure(figsize=(7, 4))
plt.plot(train_losses, label="Train Loss",      color="#1C7293")
plt.plot(val_losses,   label="Validation Loss", color="#F4A261")
plt.xlabel("Epoch",    fontsize=12)                    # FIX: label
plt.ylabel("MSE Loss (scaled)", fontsize=12)           # FIX: label
plt.title(f"Training Loss Curve — {dataset_label}", fontsize=13)
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f"{save_dir}/loss_curve.png", dpi=300)
plt.show()

# ── 10.2  Scatter Plot ────────────────────────────────────────
plt.figure(figsize=(6, 6))
plt.scatter(true, pred, alpha=0.4, s=15, color="#1C7293", label="Predictions")

min_val = min(true.min(), pred.min()) - 0.5
max_val = max(true.max(), pred.max()) + 0.5
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label="1:1 line")

# Annotate metrics on plot
plt.text(0.05, 0.93, f"R² = {r2:.3f}\nRMSE = {rmse:.3f} mm/8-day\nMAE = {mae:.3f} mm/8-day",  # FIX
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

plt.xlabel("Actual ET (mm/8-day)",    fontsize=12)     # FIX: units
plt.ylabel("Predicted ET (mm/8-day)", fontsize=12)     # FIX: units
plt.title(f"Actual vs Predicted ET — {dataset_label}", fontsize=13)
plt.legend(fontsize=10)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f"{save_dir}/scatter_plot.png", dpi=300)
plt.show()

# ── 10.3  Residual Plot ───────────────────────────────────────
residuals = true.flatten() - pred.flatten()

plt.figure(figsize=(7, 4))
plt.scatter(pred.flatten(), residuals, alpha=0.3, s=12, color="#1C7293")
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
plt.xlabel("Predicted ET (mm/8-day)", fontsize=12)     # FIX: label
plt.ylabel("Residual (mm/8-day)",     fontsize=12)     # FIX: label
plt.title(f"Residual Plot — {dataset_label}",          fontsize=13)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f"{save_dir}/residual_plot.png", dpi=300)
plt.show()

# ── 10.4  Error Histogram ─────────────────────────────────────
plt.figure(figsize=(7, 4))
plt.hist(residuals, bins=30, color="#1C7293", edgecolor='white', alpha=0.85)
plt.axvline(x=0,    color='red',    linestyle='--', linewidth=1.5, label="Zero error")
plt.axvline(x=bias, color='orange', linestyle='-',  linewidth=1.5, label=f"Bias = {bias:.3f}")
plt.xlabel("Residual (mm/8-day)", fontsize=12)         # FIX: label + units
plt.ylabel("Frequency",           fontsize=12)         # FIX: label
plt.title(f"Error Distribution — {dataset_label}",    fontsize=13)
plt.legend(fontsize=10)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f"{save_dir}/error_hist.png", dpi=300)
plt.show()

# ── 10.5  Correlation Heatmap ─────────────────────────────────
plt.figure(figsize=(10, 8))
corr_matrix = df.corr(numeric_only=True)               # FIX: numeric_only=True
im = plt.imshow(corr_matrix, cmap='viridis', vmin=-1, vmax=1)
plt.colorbar(im, label="Pearson r")                    # FIX: label
cols = corr_matrix.columns.tolist()
plt.xticks(range(len(cols)), cols, rotation=45, ha='right', fontsize=9)
plt.yticks(range(len(cols)), cols, fontsize=9)
plt.title(f"Feature Correlation Heatmap — {dataset_label}", fontsize=13)
plt.tight_layout()
plt.savefig(f"{save_dir}/correlation_heatmap.png", dpi=300)
plt.show()

# ── 10.6  Prediction vs Index (first 200 samples) ────────────
plt.figure(figsize=(10, 4))
plt.plot(true[:200].flatten(), label="Actual ET",    color="#1C7293", linewidth=1.2)
plt.plot(pred[:200].flatten(), label="Predicted ET", color="#F4A261", linewidth=1.2, alpha=0.85)
plt.xlabel("Sample Index",        fontsize=12)         # FIX: label
plt.ylabel("ET (mm/8-day)",       fontsize=12)         # FIX: label + units
plt.title(f"Predicted vs Actual ET (first 200 test samples) — {dataset_label}", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(f"{save_dir}/prediction_vs_index.png", dpi=300)
plt.show()

print(f"\nAll plots saved to: {save_dir}/")