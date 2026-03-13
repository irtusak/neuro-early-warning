"""
Centralized configuration for the Early Warning System.

Symptom risk weights, privacy settings, and detection parameters
are defined here so all detectors use consistent clinical judgments.
"""

# Symptom clinical risk weights (higher = more concerning for novel neurological syndrome)
# Calibrated against prion-like / NB-cluster symptom profiles
SYMPTOM_RISK_WEIGHTS: dict[str, float] = {
    "memory_loss": 3.0,
    "cognitive_decline": 4.0,
    "ataxia": 4.0,
    "tremor": 2.0,
    "myoclonus": 5.0,
    "dysarthria": 4.0,
    "seizure": 3.0,
    "behavioral_change": 3.0,
    "hallucination": 2.5,
    "vision_disturbance": 2.0,
    "muscle_spasm": 1.0,
    "pain_neuropathic": 1.0,
    "insomnia": 1.0,
    "fatigue_neuro": 1.0,
}

# Symptoms considered "serious" for local context features
SERIOUS_SYMPTOMS: set[str] = {
    k for k, v in SYMPTOM_RISK_WEIGHTS.items() if v >= 3.0
}

# Privacy: coordinate rounding precision (1 decimal ≈ 11 km resolution)
COORDINATE_PRECISION: int = 1

# Privacy: age band boundaries
AGE_BANDS: list[tuple[int, int]] = [
    (0, 17), (18, 29), (30, 39), (40, 49),
    (50, 59), (60, 69), (70, 79), (80, 120),
]

# Privacy: minimum case count for geographic output (small-cell suppression)
MIN_CELL_COUNT: int = 5


def age_to_band(age: int) -> str:
    """Convert exact age to a privacy-safe age band string."""
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "unknown"


def round_coordinates(lat: float, lon: float) -> tuple[float, float]:
    """Round coordinates to configured privacy-safe precision."""
    return (round(lat, COORDINATE_PRECISION), round(lon, COORDINATE_PRECISION))


# Environmental risk analysis
ENV_SEARCH_RADIUS_KM: float = 100.0
ENV_CONCERN_THRESHOLD: float = 0.7  # water contaminant level above this is flagged
