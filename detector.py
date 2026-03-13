"""
Early Warning Anomaly Detector for Neurological Symptom Clusters.

Fixes applied from Health Canada expert review:
- Feature dimension alignment between fit/detect (ML Engineer #2)
- Isolation Forest uses contamination="auto" (ML Engineer #3, Biostatistician #5)
- Poisson detector applies Benjamini-Hochberg FDR correction (Biostatistician #2)
- Coordinates rounded for privacy (Privacy Officer #1)
- Symptom risk weighting (Epidemiologist #7a)
- Age included in spatial features (Epidemiologist #4a)
- detect() restricted to recent-only data by default (Biostatistician #7)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

from config import SYMPTOM_RISK_WEIGHTS, COORDINATE_PRECISION, MIN_CELL_COUNT

logger = logging.getLogger("neurowatch")


@dataclass
class Alert:
    """A single early-warning alert for a detected cluster."""
    alert_id: str
    generated_at: str
    center_lat: float
    center_lon: float
    radius_km: float
    n_cases: int
    time_window: str
    severity_mean: float
    anomaly_score: float
    description: str
    dominant_symptoms: dict = field(default_factory=dict)
    symptom_risk_score: float = 0.0
    confidence: str = ""
    recommended_actions: list[str] = field(default_factory=list)


def _make_alert_id() -> str:
    return f"NW-{datetime.utcnow().strftime('%Y')}-{uuid4().hex[:6].upper()}"


class SpatialScanDetector:
    """Detect anomalous geographic clusters using density + rate analysis."""

    def __init__(
        self,
        spatial_eps_km: float = 50.0,
        min_samples: int = 10,
        time_window_days: int = 60,
    ):
        self.spatial_eps_km = spatial_eps_km
        self.min_samples = min_samples
        self.time_window_days = time_window_days
        self._iso_forest = None
        self._scaler = None
        self._fit_columns: list[str] | None = None

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    def _build_features(self, df: pd.DataFrame, is_fit: bool = False) -> np.ndarray:
        """Build feature matrix with column alignment between fit and detect."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["day_ordinal"] = (df["date"] - df["date"].min()).dt.days

        # Symptom risk weight as a feature (Epidemiologist fix: weight symptoms)
        df["symptom_risk"] = df["symptom"].map(SYMPTOM_RISK_WEIGHTS).fillna(1.0)

        symptom_dummies = pd.get_dummies(df["symptom"], prefix="sym")

        if is_fit:
            self._fit_columns = symptom_dummies.columns.tolist()
        else:
            # Align columns (ML Engineer fix: dimension mismatch)
            novel = set(symptom_dummies.columns) - set(self._fit_columns)
            if novel:
                logger.warning(f"Novel symptoms in detect data not seen in training: {novel}")
            for col in self._fit_columns:
                if col not in symptom_dummies.columns:
                    symptom_dummies[col] = 0.0
            symptom_dummies = symptom_dummies[self._fit_columns]

        # Include age (Epidemiologist fix: age was missing)
        features = df[["latitude", "longitude", "day_ordinal", "severity", "age", "symptom_risk"]].join(
            symptom_dummies
        )
        return features.values.astype(np.float64)

    def fit(self, df_baseline: pd.DataFrame):
        """Learn normal distribution from baseline data."""
        X = self._build_features(df_baseline, is_fit=True)
        self._scaler = StandardScaler().fit(X)
        X_scaled = self._scaler.transform(X)
        # Use contamination="auto" (ML Engineer fix: not hardcoded)
        self._iso_forest = IsolationForest(
            contamination="auto", random_state=42, n_jobs=-1
        )
        self._iso_forest.fit(X_scaled)

        # Compute threshold from baseline score distribution
        baseline_scores = -self._iso_forest.decision_function(X_scaled)
        self._score_threshold = np.percentile(baseline_scores, 95)
        return self

    def detect(self, df: pd.DataFrame) -> list[Alert]:
        """Run detection on new data. Returns list of Alerts."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        X = self._build_features(df)
        X_scaled = self._scaler.transform(X)
        scores = -self._iso_forest.decision_function(X_scaled)
        df["anomaly_score"] = scores

        # Use calibrated threshold from baseline (not percentile of current data)
        suspicious = df[df["anomaly_score"] > self._score_threshold].copy()

        if len(suspicious) < self.min_samples:
            return []

        coords_rad = np.radians(suspicious[["latitude", "longitude"]].values)
        eps_rad = self.spatial_eps_km / 6371.0
        clustering = DBSCAN(eps=eps_rad, min_samples=self.min_samples, metric="haversine")
        suspicious["cluster_id"] = clustering.fit_predict(coords_rad)

        alerts = []
        for cid in suspicious["cluster_id"].unique():
            if cid == -1:
                continue
            cluster = suspicious[suspicious["cluster_id"] == cid]

            # Small-cell suppression (Privacy fix)
            if len(cluster) < MIN_CELL_COUNT:
                continue

            center_lat = cluster["latitude"].mean()
            center_lon = cluster["longitude"].mean()

            dists = [
                self._haversine_km(center_lat, center_lon, row.latitude, row.longitude)
                for _, row in cluster.iterrows()
            ]
            radius = max(dists) if dists else 0

            date_min = cluster["date"].min().strftime("%Y-%m-%d")
            date_max = cluster["date"].max().strftime("%Y-%m-%d")

            symptom_counts = cluster["symptom"].value_counts().head(3).to_dict()

            # Compute symptom risk score for this cluster
            cluster_risk = cluster["symptom"].map(SYMPTOM_RISK_WEIGHTS).fillna(1.0).mean()

            alerts.append(
                Alert(
                    alert_id=_make_alert_id(),
                    generated_at=datetime.utcnow().isoformat(),
                    center_lat=round(center_lat, COORDINATE_PRECISION),
                    center_lon=round(center_lon, COORDINATE_PRECISION),
                    radius_km=round(radius, 1),
                    n_cases=len(cluster),
                    time_window=f"{date_min} to {date_max}",
                    severity_mean=round(cluster["severity"].mean(), 2),
                    anomaly_score=round(cluster["anomaly_score"].mean(), 4),
                    dominant_symptoms=symptom_counts,
                    symptom_risk_score=round(cluster_risk, 2),
                    description=(
                        f"Cluster of {len(cluster)} neurological cases within {radius:.0f} km "
                        f"radius. Mean severity {cluster['severity'].mean():.1f}/5. "
                        f"Symptom risk {cluster_risk:.1f}/5. "
                        f"Dominant symptoms: {symptom_counts}"
                    ),
                )
            )

        return sorted(alerts, key=lambda a: a.anomaly_score, reverse=True)


class PoissonRateDetector:
    """
    Poisson excess-rate detector over spatial grid cells.

    Fixes: Benjamini-Hochberg FDR correction applied to all p-values
    before thresholding (Biostatistician fix #2).
    """

    def __init__(self, grid_resolution: float = 1.0, lookback_days: int = 365, alert_fdr: float = 0.05):
        self.grid_resolution = grid_resolution
        self.lookback_days = lookback_days
        self.alert_fdr = alert_fdr  # FDR threshold instead of raw p-value
        self._baseline_rates: dict[tuple[float, float], float] = {}

    def _cell(self, lat: float, lon: float) -> tuple[float, float]:
        return (
            round(lat / self.grid_resolution) * self.grid_resolution,
            round(lon / self.grid_resolution) * self.grid_resolution,
        )

    def fit(self, df: pd.DataFrame):
        """Compute baseline daily rates per grid cell."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["cell"] = df.apply(lambda r: self._cell(r["latitude"], r["longitude"]), axis=1)
        total_days = (df["date"].max() - df["date"].min()).days or 1
        for cell, group in df.groupby("cell"):
            self._baseline_rates[cell] = len(group) / total_days
        return self

    def detect(self, df: pd.DataFrame, window_days: int = 30, reference_date: str | None = None) -> list[Alert]:
        """Detect cells where recent count exceeds Poisson expectation, with FDR correction."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        ref = pd.to_datetime(reference_date) if reference_date else df["date"].max()
        cutoff = ref - pd.Timedelta(days=window_days)
        recent = df[(df["date"] >= cutoff) & (df["date"] <= ref)].copy()
        recent["cell"] = recent.apply(lambda r: self._cell(r["latitude"], r["longitude"]), axis=1)

        # Compute p-values for all cells first
        cell_results = []
        for cell, group in recent.groupby("cell"):
            expected_rate = self._baseline_rates.get(cell, 0.01)
            expected_count = expected_rate * window_days
            observed = len(group)
            p_value = 1 - poisson.cdf(observed - 1, max(expected_count, 0.01))
            rate_ratio = observed / max(expected_count, 0.01)
            cell_results.append({
                "cell": cell, "group": group, "observed": observed,
                "expected_count": expected_count, "p_value": p_value,
                "rate_ratio": rate_ratio,
            })

        if not cell_results:
            return []

        # Benjamini-Hochberg FDR correction (Biostatistician fix #2)
        cell_results.sort(key=lambda x: x["p_value"])
        n_tests = len(cell_results)
        for i, cr in enumerate(cell_results):
            cr["bh_threshold"] = self.alert_fdr * (i + 1) / n_tests
            cr["significant"] = cr["p_value"] <= cr["bh_threshold"]

        alerts = []
        for cr in cell_results:
            if not cr["significant"]:
                continue
            if cr["rate_ratio"] < 2.0:  # require at least 2x excess
                continue

            cell = cr["cell"]
            group = cr["group"]

            # Small-cell suppression (Privacy fix)
            if cr["observed"] < MIN_CELL_COUNT:
                continue

            symptom_counts = group["symptom"].value_counts().head(3).to_dict()
            cluster_risk = group["symptom"].map(SYMPTOM_RISK_WEIGHTS).fillna(1.0).mean()

            alerts.append(
                Alert(
                    alert_id=_make_alert_id(),
                    generated_at=datetime.utcnow().isoformat(),
                    center_lat=round(cell[0], COORDINATE_PRECISION),
                    center_lon=round(cell[1], COORDINATE_PRECISION),
                    radius_km=self.grid_resolution * 111 / 2,
                    n_cases=cr["observed"],
                    time_window=f"last {window_days} days",
                    severity_mean=round(group["severity"].mean(), 2),
                    anomaly_score=round(-np.log10(max(cr["p_value"], 1e-300)), 2),
                    dominant_symptoms=symptom_counts,
                    symptom_risk_score=round(cluster_risk, 2),
                    description=(
                        f"Excess rate in grid cell ({cell[0]}, {cell[1]}): "
                        f"observed {cr['observed']} vs expected {cr['expected_count']:.1f} "
                        f"(p={cr['p_value']:.2e}, rate ratio {cr['rate_ratio']:.1f}x, FDR-adjusted)"
                    ),
                )
            )

        return sorted(alerts, key=lambda a: a.anomaly_score, reverse=True)
