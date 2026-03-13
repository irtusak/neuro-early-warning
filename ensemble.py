"""
Ensemble scorer that combines signals from all three detectors into
a single ranked alert list with confidence levels.

Fixes applied from Health Canada expert review:
- Normalize scores to common scale before combining (Biostatistician #6)
- Symptom risk weighting in confidence (Epidemiologist #7b)
- Actionable response guidance per confidence level (Operations #2)
- Alert IDs and metadata (Operations #5)
- Privacy-safe coordinate rounding (Privacy #1)
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

from detector import Alert, SpatialScanDetector
from config import COORDINATE_PRECISION


@dataclass
class RankedAlert:
    """An alert enriched with ensemble scoring and confidence level."""
    alert_id: str
    generated_at: str
    rank: int
    center_lat: float
    center_lon: float
    radius_km: float
    n_cases: int
    time_window: str
    severity_mean: float
    ensemble_score: float
    confidence: str  # CRITICAL, HIGH, MEDIUM, LOW
    detectors_fired: list[str]
    description: str
    symptom_risk_score: float = 0.0
    recommended_actions: list[str] = field(default_factory=list)


# Response protocols per confidence level (Operations fix #2)
RESPONSE_PROTOCOLS = {
    "CRITICAL": [
        "Notify provincial Chief Medical Officer within 1 hour",
        "Initiate case investigation protocol",
        "Request environmental sampling within 24 hours",
        "Prepare public communications briefing",
    ],
    "HIGH": [
        "Queue for next-day epidemiological review",
        "Request detailed case reports from regional health authority",
        "Cross-reference with environmental monitoring data",
    ],
    "MEDIUM": [
        "Include in weekly surveillance summary",
        "Monitor for trend escalation in next pipeline run",
    ],
    "LOW": [
        "Log for trend monitoring, no immediate action required",
    ],
}


def _alerts_overlap(a1: Alert, a2: Alert, threshold_km: float = 100.0) -> bool:
    """Check if two alerts refer to roughly the same geographic area."""
    dist = SpatialScanDetector._haversine_km(
        a1.center_lat, a1.center_lon, a2.center_lat, a2.center_lon
    )
    return dist < threshold_km


def _normalize_scores(alerts: list[tuple[Alert, str]]) -> dict[int, float]:
    """
    Normalize anomaly scores to [0, 1] per detector type.

    FIX (Biostatistician #6): Poisson scores (~300), Isolation Forest scores (~0.02),
    and autoencoder scores (~0.1) are on completely different scales. Normalizing
    to rank percentiles within each detector makes them comparable.
    """
    scores_by_source: dict[str, list[tuple[int, float]]] = {}
    for i, (a, source) in enumerate(alerts):
        scores_by_source.setdefault(source, []).append((i, a.anomaly_score))

    normalized = {}
    for source, indexed_scores in scores_by_source.items():
        sorted_scores = sorted(indexed_scores, key=lambda x: x[1])
        n = len(sorted_scores)
        for rank, (idx, _) in enumerate(sorted_scores):
            normalized[idx] = (rank + 1) / n if n > 1 else 0.5

    return normalized


def rank_alerts(
    spatial_alerts: list[Alert],
    poisson_alerts: list[Alert],
    ae_cluster_alerts: list[Alert] | None = None,
) -> list[RankedAlert]:
    """Merge alerts from multiple detectors with normalized scoring."""
    tagged: list[tuple[Alert, str]] = []
    for a in spatial_alerts:
        tagged.append((a, "spatial_scan"))
    for a in poisson_alerts:
        tagged.append((a, "poisson_rate"))
    for a in (ae_cluster_alerts or []):
        tagged.append((a, "autoencoder"))

    if not tagged:
        return []

    # Normalize scores to [0,1] per detector (Biostatistician fix #6)
    norm_scores = _normalize_scores(tagged)

    # Greedy merge overlapping alerts
    groups: list[list[tuple[int, Alert, str]]] = []
    used = set()
    for i, (a1, s1) in enumerate(tagged):
        if i in used:
            continue
        group = [(i, a1, s1)]
        used.add(i)
        for j, (a2, s2) in enumerate(tagged):
            if j in used:
                continue
            if _alerts_overlap(a1, a2):
                group.append((j, a2, s2))
                used.add(j)
        groups.append(group)

    ranked = []
    for group in groups:
        alerts_in_group = [a for _, a, _ in group]
        sources = list({s for _, _, s in group})
        indices = [i for i, _, _ in group]

        total_cases = sum(a.n_cases for a in alerts_in_group)
        center_lat = sum(a.center_lat * a.n_cases for a in alerts_in_group) / total_cases
        center_lon = sum(a.center_lon * a.n_cases for a in alerts_in_group) / total_cases
        radius = max(a.radius_km for a in alerts_in_group)
        severity = np.mean([a.severity_mean for a in alerts_in_group])

        # Use normalized scores (Biostatistician fix: comparable scales)
        max_norm_score = max(norm_scores.get(i, 0) for i in indices)
        agreement_boost = 1.0 + 0.5 * (len(sources) - 1)
        severity_boost = severity / 3.0

        # Include symptom risk in scoring (Epidemiologist fix #7b)
        symptom_risk = np.mean([a.symptom_risk_score for a in alerts_in_group if a.symptom_risk_score > 0]) if any(a.symptom_risk_score > 0 for a in alerts_in_group) else 1.0
        symptom_boost = symptom_risk / 3.0

        ensemble_score = max_norm_score * agreement_boost * severity_boost * symptom_boost

        # Confidence level (includes symptom risk — Epidemiologist fix)
        n_detectors = len(sources)
        if n_detectors >= 3 or (n_detectors >= 2 and severity >= 4.0) or (n_detectors >= 2 and symptom_risk >= 3.5):
            confidence = "CRITICAL"
        elif n_detectors >= 2 or severity >= 4.0 or symptom_risk >= 3.5:
            confidence = "HIGH"
        elif total_cases >= 20:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        time_windows = [a.time_window for a in alerts_in_group]
        time_str = " | ".join(sorted(set(time_windows)))

        ranked.append(RankedAlert(
            alert_id=alerts_in_group[0].alert_id,
            generated_at=datetime.utcnow().isoformat(),
            rank=0,
            center_lat=round(center_lat, COORDINATE_PRECISION),
            center_lon=round(center_lon, COORDINATE_PRECISION),
            radius_km=round(radius, 1),
            n_cases=total_cases,
            time_window=time_str,
            severity_mean=round(severity, 2),
            ensemble_score=round(ensemble_score, 4),
            confidence=confidence,
            detectors_fired=sources,
            symptom_risk_score=round(symptom_risk, 2),
            recommended_actions=RESPONSE_PROTOCOLS.get(confidence, []),
            description=(
                f"{total_cases} cases near ({center_lat:.1f}, {center_lon:.1f}), "
                f"severity {severity:.1f}/5, symptom risk {symptom_risk:.1f}/5. "
                f"Detected by: {', '.join(sources)}"
            ),
        ))

    ranked.sort(key=lambda r: r.ensemble_score, reverse=True)
    for i, r in enumerate(ranked):
        r.rank = i + 1

    return ranked


def format_alert_report(ranked_alerts: list[RankedAlert]) -> str:
    """Format alerts into a human-readable report."""
    lines = [
        "=" * 70,
        "  NEUROLOGICAL CLUSTER EARLY WARNING REPORT",
        f"  Generated: {datetime.utcnow().isoformat()}",
        "=" * 70,
        "",
    ]

    if not ranked_alerts:
        lines.append("  No anomalous clusters detected.")
        return "\n".join(lines)

    for alert in ranked_alerts:
        marker = {"CRITICAL": "!!!", "HIGH": "!!", "MEDIUM": "!", "LOW": "."}
        lines.extend([
            f"  [{alert.confidence}] {marker.get(alert.confidence, '')} ALERT {alert.alert_id} (#{alert.rank})",
            f"  Location:     ({alert.center_lat}, {alert.center_lon}), radius {alert.radius_km} km",
            f"  Cases:        {alert.n_cases}",
            f"  Severity:     {alert.severity_mean}/5",
            f"  Symptom Risk: {alert.symptom_risk_score}/5",
            f"  Time:         {alert.time_window}",
            f"  Score:        {alert.ensemble_score}",
            f"  Detectors:    {', '.join(alert.detectors_fired)}",
        ])
        if alert.recommended_actions:
            lines.append(f"  Actions:")
            for action in alert.recommended_actions:
                lines.append(f"    -> {action}")
        lines.append("-" * 70)

    lines.append("")
    critical = sum(1 for a in ranked_alerts if a.confidence == "CRITICAL")
    high = sum(1 for a in ranked_alerts if a.confidence == "HIGH")
    lines.append(f"  Summary: {len(ranked_alerts)} alerts — {critical} CRITICAL, {high} HIGH")
    lines.append("=" * 70)

    return "\n".join(lines)
