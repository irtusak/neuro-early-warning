#!/usr/bin/env python3
"""
Early Warning Pipeline — end-to-end detection of neurological symptom clusters.

Fixes applied from Health Canada expert review:
- Audit logging (Privacy Officer #3)
- detect() on recent data only, not full dataset (Biostatistician #7)
- Input validation (Privacy Officer #5)
- Reproducible seeds (ML Engineer #5)
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN

from config import COORDINATE_PRECISION, MIN_CELL_COUNT
from synthetic_data import build_dataset
from detector import SpatialScanDetector, PoissonRateDetector, Alert
from autoencoder import AutoencoderDetector
from ensemble import rank_alerts, format_alert_report

# --- Audit Logging (Privacy Officer fix #3) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("neurowatch_audit.log", mode="a"),
    ],
)
logger = logging.getLogger("neurowatch")


def _validate_input(path: str) -> str:
    """Validate input CSV path (Privacy Officer fix #5)."""
    real_path = os.path.realpath(path)
    if not os.path.isfile(real_path):
        raise FileNotFoundError(f"Input file not found: {real_path}")
    if not real_path.endswith(".csv"):
        raise ValueError(f"Input must be a .csv file, got: {real_path}")
    return real_path


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean input data (Epidemiologist fix #6a: missing data)."""
    required = {"date", "latitude", "longitude", "symptom", "severity", "age", "sex"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Input CSV missing required columns: {missing_cols}")

    n_before = len(df)
    df = df.dropna(subset=["latitude", "longitude", "date", "symptom"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"Dropped {n_dropped} rows with missing required fields")

    df["severity"] = df["severity"].fillna(df["severity"].median())
    df["age"] = df["age"].fillna(df["age"].median()).astype(int)
    df["sex"] = df["sex"].fillna("U")

    return df


def _ae_to_cluster_alerts(scored: pd.DataFrame) -> list[Alert]:
    """Convert per-report autoencoder flags into cluster-level alerts via DBSCAN."""
    from detector import _make_alert_id
    from config import SYMPTOM_RISK_WEIGHTS

    flagged = scored[scored["ae_anomaly"]].copy()
    if len(flagged) < MIN_CELL_COUNT:
        return []

    coords_rad = np.radians(flagged[["latitude", "longitude"]].values)
    eps_rad = 50.0 / 6371.0
    clustering = DBSCAN(eps=eps_rad, min_samples=5, metric="haversine")
    flagged["cluster_id"] = clustering.fit_predict(coords_rad)

    alerts = []
    for cid in flagged["cluster_id"].unique():
        if cid == -1:
            continue
        cluster = flagged[flagged["cluster_id"] == cid]

        if len(cluster) < MIN_CELL_COUNT:
            continue

        center_lat = cluster["latitude"].mean()
        center_lon = cluster["longitude"].mean()

        dists = [
            SpatialScanDetector._haversine_km(center_lat, center_lon, r.latitude, r.longitude)
            for _, r in cluster.iterrows()
        ]

        date_min = pd.to_datetime(cluster["date"]).min().strftime("%Y-%m-%d")
        date_max = pd.to_datetime(cluster["date"]).max().strftime("%Y-%m-%d")

        symptom_counts = cluster["symptom"].value_counts().head(3).to_dict()
        cluster_risk = cluster["symptom"].map(SYMPTOM_RISK_WEIGHTS).fillna(1.0).mean()

        alerts.append(Alert(
            alert_id=_make_alert_id(),
            generated_at=datetime.utcnow().isoformat(),
            center_lat=round(center_lat, COORDINATE_PRECISION),
            center_lon=round(center_lon, COORDINATE_PRECISION),
            radius_km=round(max(dists) if dists else 0, 1),
            n_cases=len(cluster),
            time_window=f"{date_min} to {date_max}",
            severity_mean=round(cluster["severity"].mean(), 2),
            anomaly_score=round(cluster["reconstruction_error"].mean(), 4),
            dominant_symptoms=symptom_counts,
            symptom_risk_score=round(cluster_risk, 2),
            description=(
                f"Autoencoder cluster: {len(cluster)} high-error cases near "
                f"({center_lat:.1f}, {center_lon:.1f})"
            ),
        ))

    return alerts


def main():
    parser = argparse.ArgumentParser(description="Neurological Cluster Early Warning System")
    parser.add_argument("--csv", type=str, help="Path to input CSV (uses synthetic data if omitted)")
    parser.add_argument("--skip-autoencoder", action="store_true", help="Skip PyTorch autoencoder")
    args = parser.parse_args()

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== Pipeline run {run_id} started by user={os.getenv('USER', 'unknown')} ===")

    # --- Load data ---
    if args.csv:
        validated_path = _validate_input(args.csv)
        df = pd.read_csv(validated_path)
        df = _validate_dataframe(df)
        logger.info(f"Loaded {len(df)} reports from {validated_path}")
    else:
        df = build_dataset()
        logger.info(f"Generated {len(df)} synthetic reports ({df['is_anomaly'].sum()} injected anomalies)")

    # --- Split baseline / recent ---
    # FIX (Biostatistician #7): Use a fixed baseline period, detect on RECENT only
    df["date"] = pd.to_datetime(df["date"])
    cutoff = df["date"].quantile(0.7)
    baseline = df[df["date"] <= cutoff]
    recent = df[df["date"] > cutoff]
    logger.info(f"Baseline: {len(baseline)} reports up to {cutoff.date()}")
    logger.info(f"Recent:   {len(recent)} reports after {cutoff.date()}")

    # --- Detector 1: Spatial Scan (Isolation Forest + DBSCAN) ---
    logger.info("[1] Running Spatial Scan Detector...")
    spatial = SpatialScanDetector(spatial_eps_km=50, min_samples=8)
    spatial.fit(baseline)
    alerts_spatial = spatial.detect(recent)  # FIX: recent only, not df
    logger.info(f"    Found {len(alerts_spatial)} spatial alerts")
    for i, a in enumerate(alerts_spatial):
        logger.info(f"    #{i+1}: {a.description}")

    # --- Detector 2: Poisson Rate ---
    logger.info("[2] Running Poisson Rate Detector (with FDR correction)...")
    poisson_det = PoissonRateDetector(grid_resolution=0.5, alert_fdr=0.05)
    poisson_det.fit(baseline)
    alerts_poisson = poisson_det.detect(df, window_days=90, reference_date="2025-12-31")
    logger.info(f"    Found {len(alerts_poisson)} rate alerts")
    for i, a in enumerate(alerts_poisson):
        logger.info(f"    #{i+1}: {a.description}")

    # --- Detector 3: Autoencoder ---
    alerts_ae = []
    if not args.skip_autoencoder:
        logger.info("[3] Training Autoencoder Detector (with early stopping)...")
        ae = AutoencoderDetector(latent_dim=8, epochs=80, threshold_percentile=95, seed=42)
        ae.fit(baseline)
        scored = ae.score(recent)  # FIX: score recent only, not df
        n_flagged = scored["ae_anomaly"].sum()
        logger.info(f"    Flagged {n_flagged} / {len(recent)} recent reports")

        if "is_anomaly" in recent.columns:
            true_pos = scored[scored["is_anomaly"] & scored["ae_anomaly"]]
            total_true = recent["is_anomaly"].sum()
            if total_true > 0:
                logger.info(f"    Recall on injected anomalies: {len(true_pos)}/{total_true} "
                            f"({100*len(true_pos)/total_true:.1f}%)")

        alerts_ae = _ae_to_cluster_alerts(scored)
        logger.info(f"    Formed {len(alerts_ae)} autoencoder cluster alerts")

    # --- Ensemble ---
    logger.info("[4] Running Ensemble Scorer (normalized scores, symptom-weighted)...")
    ranked = rank_alerts(alerts_spatial, alerts_poisson, alerts_ae)
    report = format_alert_report(ranked)
    print(report)

    # Log alert summary for audit
    for r in ranked:
        logger.info(f"ALERT {r.alert_id}: {r.confidence} | ({r.center_lat},{r.center_lon}) | "
                     f"{r.n_cases} cases | detectors={r.detectors_fired}")

    # --- Environmental Risk Analysis ---
    logger.info("[5] Analyzing environmental risk factors...")
    from environmental_data import enrich_alerts_with_environment
    env_data = enrich_alerts_with_environment(ranked)
    for r in ranked:
        env = env_data.get(r.alert_id, {})
        socio = env.get('socioeconomic', {})
        socio_name = socio.get('name', 'N/A') if socio else 'N/A'
        logger.info(f"  {r.alert_id} env_risk={env.get('env_risk_score', 0)}/5 "
                     f"socio_risk={env.get('socio_risk_score', 0)}/5 "
                     f"combined={env.get('combined_risk_score', 0)}/5 "
                     f"region={socio_name}: {env.get('risk_summary', 'N/A')}")

    # --- Visualize ---
    logger.info("[6] Generating visualizations...")
    from visualize import plot_alert_map, plot_timeline
    viz_alerts = [
        Alert(
            alert_id=r.alert_id, generated_at=r.generated_at,
            center_lat=r.center_lat, center_lon=r.center_lon,
            radius_km=r.radius_km, n_cases=r.n_cases,
            time_window=r.time_window, severity_mean=r.severity_mean,
            anomaly_score=r.ensemble_score,
            confidence=r.confidence,
            description=f"[{r.confidence}] {r.description}",
        )
        for r in ranked
    ]

    # Use full df for visualization (shows both baseline and recent)
    plot_alert_map(df, viz_alerts, env_data=env_data)
    plot_timeline(df, viz_alerts)

    logger.info(f"=== Pipeline run {run_id} completed. {len(ranked)} alerts generated. ===")
    print(f"\nDone. {len(ranked)} alerts. Review alert_map.png and alert_timeline.png")


if __name__ == "__main__":
    main()
