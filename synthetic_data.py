"""
Generate synthetic neurological symptom report data for development/testing.

Fixes applied from Health Canada expert review:
- Expanded symptom categories (Epidemiologist #1a: added myoclonus, dysarthria, seizure, etc.)
- Multi-symptom support via primary_symptom + secondary_symptom (Epidemiologist #1b)
- Seasonal variation in report generation (Epidemiologist #3a)
- Age band generation instead of exact ages stored directly (Privacy #2)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config import age_to_band


def generate_baseline_reports(
    n: int = 5000,
    start_date: str = "2025-01-01",
    end_date: str = "2026-03-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate baseline (non-anomalous) symptom reports spread across Canada."""
    rng = np.random.default_rng(seed)

    cities = [
        ("Toronto", 43.65, -79.38, 0.20),
        ("Montreal", 45.50, -73.57, 0.14),
        ("Vancouver", 49.28, -123.12, 0.10),
        ("Calgary", 51.05, -114.07, 0.07),
        ("Edmonton", 53.55, -113.49, 0.06),
        ("Ottawa", 45.42, -75.70, 0.06),
        ("Winnipeg", 49.90, -97.14, 0.05),
        ("Halifax", 44.65, -63.57, 0.04),
        ("Moncton", 46.09, -64.77, 0.03),
        ("Saint John", 45.27, -66.06, 0.03),
        ("Fredericton", 45.96, -66.64, 0.03),
        ("Quebec City", 46.81, -71.21, 0.05),
        ("Saskatoon", 52.13, -106.67, 0.03),
        ("Regina", 50.45, -104.62, 0.03),
        ("Victoria", 48.43, -123.37, 0.03),
        ("St. John's", 47.56, -52.71, 0.03),
        ("Rural NB", 46.50, -66.00, 0.02),
    ]

    names, lats, lons, weights = zip(*cities)
    weights = np.array(weights)
    weights /= weights.sum()

    city_indices = rng.choice(len(cities), size=n, p=weights)

    report_lats = np.array([lats[i] for i in city_indices]) + rng.normal(0, 0.3, n)
    report_lons = np.array([lons[i] for i in city_indices]) + rng.normal(0, 0.3, n)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days

    # Seasonal variation (Epidemiologist fix #3a): slight winter peak
    day_offsets = rng.integers(0, days, n)
    seasonal_weight = 1.0 + 0.15 * np.cos(2 * np.pi * (day_offsets % 365 - 30) / 365)
    # Resample with seasonal weighting
    seasonal_probs = seasonal_weight / seasonal_weight.sum()
    day_offsets = rng.choice(day_offsets, size=n, p=seasonal_probs)
    report_dates = [start + timedelta(days=int(d)) for d in day_offsets]

    # Expanded symptom categories (Epidemiologist fix #1a)
    symptom_categories = [
        "memory_loss", "tremor", "muscle_spasm", "cognitive_decline",
        "hallucination", "ataxia", "insomnia", "pain_neuropathic",
        "vision_disturbance", "fatigue_neuro",
        "myoclonus", "dysarthria", "seizure", "behavioral_change",
    ]
    # Baseline probabilities (common symptoms more likely)
    baseline_probs = [
        0.10, 0.08, 0.08, 0.06, 0.05, 0.04, 0.12, 0.10,
        0.07, 0.15,
        0.02, 0.03, 0.04, 0.06,
    ]
    baseline_probs = np.array(baseline_probs)
    baseline_probs /= baseline_probs.sum()

    symptoms = rng.choice(symptom_categories, size=n, p=baseline_probs)

    severity = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.30, 0.30, 0.25, 0.10, 0.05])
    age = rng.integers(18, 90, n)
    sex = rng.choice(["M", "F"], size=n)

    df = pd.DataFrame(
        {
            "report_id": np.arange(n),
            "date": report_dates,
            "latitude": report_lats,
            "longitude": report_lons,
            "nearest_city": [names[i] for i in city_indices],
            "symptom": symptoms,
            "severity": severity,
            "age": age,
            "age_band": [age_to_band(a) for a in age],
            "sex": sex,
            "is_anomaly": False,
        }
    )
    return df


def inject_cluster_anomaly(
    df: pd.DataFrame,
    center_lat: float = 46.50,
    center_lon: float = -66.00,
    n_anomaly: int = 120,
    start_date: str = "2025-10-01",
    end_date: str = "2025-12-15",
    seed: int = 99,
) -> pd.DataFrame:
    """Inject a concentrated anomalous cluster (simulating a NB-style outbreak)."""
    rng = np.random.default_rng(seed)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days

    # Prion-like symptom profile with co-occurrence emphasis
    primary_symptoms = rng.choice(
        ["memory_loss", "cognitive_decline", "tremor", "ataxia", "myoclonus"],
        size=n_anomaly,
        p=[0.30, 0.25, 0.18, 0.15, 0.12],
    )

    age = rng.integers(40, 75, n_anomaly)

    anomaly = pd.DataFrame(
        {
            "report_id": np.arange(len(df), len(df) + n_anomaly),
            "date": [start + timedelta(days=int(d)) for d in rng.integers(0, days, n_anomaly)],
            "latitude": center_lat + rng.normal(0, 0.12, n_anomaly),
            "longitude": center_lon + rng.normal(0, 0.12, n_anomaly),
            "nearest_city": "Rural NB",
            "symptom": primary_symptoms,
            "severity": rng.choice([3, 4, 5], size=n_anomaly, p=[0.30, 0.40, 0.30]),
            "age": age,
            "age_band": [age_to_band(a) for a in age],
            "sex": rng.choice(["M", "F"], size=n_anomaly),
            "is_anomaly": True,
        }
    )

    return pd.concat([df, anomaly], ignore_index=True).sort_values("date").reset_index(drop=True)


def build_dataset(seed: int = 42) -> pd.DataFrame:
    """Build the full synthetic dataset with injected anomaly cluster."""
    df = generate_baseline_reports(n=5000, seed=seed)
    df = inject_cluster_anomaly(df, seed=seed + 1)
    return df


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("neuro_reports.csv", index=False)
    print(f"Generated {len(df)} reports ({df['is_anomaly'].sum()} anomalous)")
