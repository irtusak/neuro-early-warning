"""
PyTorch autoencoder for learning normal neurological report distributions.

Fixes applied from Health Canada expert review:
- Data leakage fixed: local features computed against BASELINE tree only (ML Engineer #1)
- Validation split for threshold calibration (ML Engineer #6)
- Early stopping on validation loss (ML Engineer #4)
- Reproducible seeds (ML Engineer #5)
- Symptom column alignment with novel symptom warnings (ML Engineer #2)
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree

from config import SERIOUS_SYMPTOMS

logger = logging.getLogger("neurowatch")


class ReportAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def _compute_local_features(
    df: pd.DataFrame,
    baseline_tree: BallTree,
    baseline_df: pd.DataFrame,
    radius_km: float = 50.0,
) -> pd.DataFrame:
    """
    Compute per-report local context features using ONLY baseline data.

    FIX (ML Engineer #1): The BallTree is built from baseline data only.
    When scoring new reports, we query their neighborhoods against the
    baseline tree — so local features reflect the HISTORICAL context,
    not the current (potentially anomalous) data.
    """
    coords_rad = np.radians(df[["latitude", "longitude"]].values)
    radius_rad = radius_km / 6371.0
    neighbors = baseline_tree.query_radius(coords_rad, r=radius_rad, return_distance=False)

    # Pre-compute baseline arrays for vectorized access
    bl_severity = baseline_df["severity"].values.astype(float)
    bl_serious = baseline_df["symptom"].isin(SERIOUS_SYMPTOMS).values.astype(float)
    bl_age = baseline_df["age"].values.astype(float)

    local_count = np.array([len(n) for n in neighbors], dtype=np.float64)
    local_severity_mean = np.array([bl_severity[n].mean() if len(n) > 0 else 0 for n in neighbors])
    local_severity_std = np.array([bl_severity[n].std() if len(n) > 1 else 0 for n in neighbors])
    local_serious_frac = np.array([bl_serious[n].mean() if len(n) > 0 else 0 for n in neighbors])
    local_age_mean = np.array([bl_age[n].mean() if len(n) > 0 else 0 for n in neighbors])

    return pd.DataFrame({
        "local_count": local_count,
        "local_severity_mean": local_severity_mean,
        "local_severity_std": local_severity_std,
        "local_serious_frac": local_serious_frac,
        "local_age_mean": local_age_mean,
    }, index=df.index)


class AutoencoderDetector:
    """Train an autoencoder on baseline data; flag high-reconstruction-error reports."""

    def __init__(self, latent_dim: int = 8, epochs: int = 80, lr: float = 1e-3,
                 threshold_percentile: float = 95.0, patience: int = 10, seed: int = 42):
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.threshold_percentile = threshold_percentile
        self.patience = patience
        self.seed = seed
        self._model = None
        self._scaler = None
        self._threshold = None
        self._input_dim = None
        self._symptom_columns = None
        self._baseline_tree = None
        self._baseline_df = None

    def _prepare(self, df: pd.DataFrame, use_baseline_tree: bool = True) -> np.ndarray:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["day_ordinal"] = (df["date"] - df["date"].min()).dt.days
        sex_code = (df["sex"] == "M").astype(float)

        # One-hot symptoms with alignment
        symptom_dummies = pd.get_dummies(df["symptom"], prefix="sym").astype(float)
        if self._symptom_columns is None:
            self._symptom_columns = symptom_dummies.columns.tolist()
        else:
            novel = set(symptom_dummies.columns) - set(self._symptom_columns)
            if novel:
                logger.warning(f"Novel symptoms not seen in training: {novel}")
            for col in self._symptom_columns:
                if col not in symptom_dummies.columns:
                    symptom_dummies[col] = 0.0
            symptom_dummies = symptom_dummies[self._symptom_columns]

        # Local features from BASELINE tree only (FIX: no data leakage)
        if use_baseline_tree and self._baseline_tree is not None:
            local = _compute_local_features(df, self._baseline_tree, self._baseline_df)
        else:
            # During fit, tree is built from this data (which IS the baseline)
            coords_rad = np.radians(df[["latitude", "longitude"]].values)
            tree = BallTree(coords_rad, metric="haversine")
            local = _compute_local_features(df, tree, df)

        features = pd.concat([
            df[["latitude", "longitude", "day_ordinal", "severity", "age"]].reset_index(drop=True),
            pd.DataFrame({"sex": sex_code.values}),
            symptom_dummies.reset_index(drop=True),
            local.reset_index(drop=True),
        ], axis=1)

        return features.values.astype(np.float32)

    def fit(self, df_baseline: pd.DataFrame):
        # Set seeds for reproducibility (ML Engineer fix #5)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Store baseline for leakage-free scoring
        self._baseline_df = df_baseline.copy()
        self._baseline_df["date"] = pd.to_datetime(self._baseline_df["date"])
        coords_rad = np.radians(self._baseline_df[["latitude", "longitude"]].values)
        self._baseline_tree = BallTree(coords_rad, metric="haversine")

        X = self._prepare(df_baseline, use_baseline_tree=False)
        self._scaler = StandardScaler().fit(X)
        X_scaled = self._scaler.transform(X)
        self._input_dim = X_scaled.shape[1]

        # Split into train/calibration for threshold (ML Engineer fix #6)
        n = len(X_scaled)
        n_train = int(n * 0.8)
        indices = np.random.permutation(n)
        X_train = X_scaled[indices[:n_train]]
        X_cal = X_scaled[indices[n_train:]]

        self._model = ReportAutoencoder(self._input_dim, self.latent_dim)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataset = TensorDataset(torch.from_numpy(X_train))
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, generator=g)

        cal_tensor = torch.from_numpy(X_cal)

        # Training with early stopping (ML Engineer fix #4)
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self._model.train()
            for (batch,) in train_loader:
                recon = self._model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation loss on calibration set
            self._model.eval()
            with torch.no_grad():
                val_recon = self._model(cal_tensor)
                val_loss = criterion(val_recon, cal_tensor).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}")
                    break

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)

        # Compute threshold on CALIBRATION set (not training set - ML Engineer fix #6)
        self._model.eval()
        with torch.no_grad():
            cal_recon = self._model(cal_tensor)
            cal_errors = ((cal_recon - cal_tensor) ** 2).mean(dim=1).numpy()
        self._threshold = np.percentile(cal_errors, self.threshold_percentile)
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return dataframe with reconstruction_error and ae_anomaly columns."""
        X = self._prepare(df, use_baseline_tree=True)  # uses baseline tree, no leakage
        X_scaled = self._scaler.transform(X)

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_scaled)
            recon = self._model(X_tensor)
            errors = ((recon - X_tensor) ** 2).mean(dim=1).numpy()

        result = df.copy()
        result["reconstruction_error"] = errors
        result["ae_anomaly"] = errors > self._threshold
        return result
