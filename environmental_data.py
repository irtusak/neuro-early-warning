"""
Synthetic environmental hazard data and proximity analysis.

Places industrial sites, water quality stations, air quality monitors,
and major highway corridors at realistic Canadian locations. For each
alert, computes proximity-based environmental risk.

NOTE: All data is synthetic. In production this would pull from:
- National Pollutant Release Inventory (NPRI)
- Environment Canada air quality monitoring
- Provincial water quality databases
- StatCan road network data
"""

from math import radians, sin, cos, asin, sqrt
from config import ENV_SEARCH_RADIUS_KM, ENV_CONCERN_THRESHOLD


def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two points."""
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


# ============================================================
# Synthetic Environmental Data
# ============================================================

INDUSTRIAL_SITES = [
    # New Brunswick (clustered near the anomaly zone at ~46.5, -66.0)
    {"name": "Irving Oil Refinery", "lat": 46.38, "lon": -66.12, "type": "Petroleum Refinery",
     "emissions": "Benzene, toluene, xylene, PM2.5, SO2", "annual_tonnes": 1850},
    {"name": "AV Nackawic Pulp Mill", "lat": 46.20, "lon": -66.60, "type": "Pulp & Paper Mill",
     "emissions": "Mercury, organochlorines, dioxins", "annual_tonnes": 420},
    {"name": "Belledune Lead Smelter", "lat": 47.91, "lon": -65.83, "type": "Lead/Zinc Smelter",
     "emissions": "Lead, cadmium, arsenic, SO2", "annual_tonnes": 2100},
    {"name": "Sussex Potash Mine", "lat": 46.10, "lon": -65.80, "type": "Mining Operation",
     "emissions": "Heavy metals, particulate matter", "annual_tonnes": 310},
    {"name": "Moncton Industrial Park", "lat": 46.13, "lon": -65.40, "type": "Mixed Industrial",
     "emissions": "VOCs, PM2.5, NOx", "annual_tonnes": 180},
    # Alberta
    {"name": "Suncor Oil Sands", "lat": 57.00, "lon": -111.60, "type": "Oil Sands Extraction",
     "emissions": "PAHs, naphthenic acids, SO2, NOx", "annual_tonnes": 12500},
    {"name": "Shell Scotford Refinery", "lat": 53.62, "lon": -113.00, "type": "Petroleum Refinery",
     "emissions": "Benzene, SO2, NOx, PM2.5", "annual_tonnes": 3200},
    {"name": "Lafarge Exshaw Cement", "lat": 51.05, "lon": -115.22, "type": "Cement Plant",
     "emissions": "PM10, NOx, mercury, dioxins", "annual_tonnes": 890},
    # British Columbia
    {"name": "Teck Trail Smelter", "lat": 49.10, "lon": -117.70, "type": "Lead/Zinc Smelter",
     "emissions": "Lead, arsenic, cadmium, SO2", "annual_tonnes": 4100},
    {"name": "Burnaby Chevron Refinery", "lat": 49.28, "lon": -122.95, "type": "Petroleum Refinery",
     "emissions": "Benzene, toluene, SO2", "annual_tonnes": 1100},
    # Ontario
    {"name": "Sarnia Chemical Valley", "lat": 42.97, "lon": -82.40, "type": "Petrochemical Complex",
     "emissions": "Benzene, vinyl chloride, ethylene oxide", "annual_tonnes": 8900},
    {"name": "Sudbury Vale Nickel", "lat": 46.49, "lon": -81.00, "type": "Nickel Smelter",
     "emissions": "Nickel, SO2, arsenic, cobalt", "annual_tonnes": 5600},
    # Saskatchewan
    {"name": "Mosaic Belle Plaine Potash", "lat": 50.42, "lon": -105.20, "type": "Potash Mine",
     "emissions": "Particulate matter, salt dust, ammonia", "annual_tonnes": 250},
    # Quebec
    {"name": "Valero Levis Refinery", "lat": 46.80, "lon": -71.15, "type": "Petroleum Refinery",
     "emissions": "SO2, NOx, VOCs, benzene", "annual_tonnes": 2800},
    # Manitoba
    {"name": "HudBay Flin Flon Smelter", "lat": 54.77, "lon": -101.88, "type": "Copper/Zinc Smelter",
     "emissions": "Arsenic, cadmium, mercury, SO2", "annual_tonnes": 1900},
    # Newfoundland
    {"name": "Come By Chance Refinery", "lat": 47.82, "lon": -53.95, "type": "Petroleum Refinery",
     "emissions": "SO2, NOx, benzene, PM2.5", "annual_tonnes": 1600},
]

# CCME Water Quality Index (WQI) — Canadian Council of Ministers of the Environment
# Scale: Excellent (95-100), Good (80-94), Fair (65-79), Marginal (45-64), Poor (0-44)
WATER_QUALITY_STATIONS = [
    # NB — poor/marginal quality (correlated with anomaly zone ~46.5, -66.0)
    {"name": "Kennebecasis River", "lat": 46.30, "lon": -65.95, "type": "River",
     "wqi": 38, "wqi_rating": "Poor",
     "contaminants": {"manganese": 0.82, "lead": 0.45, "BMAA": 0.38}},
    {"name": "Petitcodiac River", "lat": 46.25, "lon": -65.50, "type": "River",
     "wqi": 42, "wqi_rating": "Poor",
     "contaminants": {"glyphosate": 0.71, "manganese": 0.55, "aluminum": 0.42}},
    {"name": "Saint John River (Fredericton)", "lat": 46.55, "lon": -66.50, "type": "River",
     "wqi": 58, "wqi_rating": "Marginal",
     "contaminants": {"manganese": 0.68, "mercury": 0.31, "BMAA": 0.45}},
    {"name": "Miramichi River", "lat": 47.00, "lon": -65.50, "type": "River",
     "wqi": 62, "wqi_rating": "Marginal",
     "contaminants": {"glyphosate": 0.62, "lead": 0.28, "manganese": 0.51}},
    # Other provinces — good/excellent quality
    {"name": "St. Lawrence (Montreal)", "lat": 45.50, "lon": -73.55, "type": "River",
     "wqi": 82, "wqi_rating": "Good",
     "contaminants": {"lead": 0.15, "manganese": 0.12, "mercury": 0.08}},
    {"name": "Bow River (Calgary)", "lat": 51.05, "lon": -114.07, "type": "River",
     "wqi": 88, "wqi_rating": "Good",
     "contaminants": {"lead": 0.11, "manganese": 0.18, "glyphosate": 0.22}},
    {"name": "North Saskatchewan River", "lat": 53.55, "lon": -113.49, "type": "River",
     "wqi": 85, "wqi_rating": "Good",
     "contaminants": {"lead": 0.14, "manganese": 0.21, "mercury": 0.05}},
    {"name": "Fraser River (Vancouver)", "lat": 49.20, "lon": -123.10, "type": "River",
     "wqi": 83, "wqi_rating": "Good",
     "contaminants": {"lead": 0.19, "manganese": 0.16, "mercury": 0.07}},
    {"name": "Red River (Winnipeg)", "lat": 49.88, "lon": -97.13, "type": "River",
     "wqi": 72, "wqi_rating": "Fair",
     "contaminants": {"lead": 0.22, "manganese": 0.25, "glyphosate": 0.31}},
    {"name": "Lake Ontario (Toronto)", "lat": 43.63, "lon": -79.38, "type": "Lake",
     "wqi": 91, "wqi_rating": "Good",
     "contaminants": {"lead": 0.13, "manganese": 0.09, "BMAA": 0.11}},
    {"name": "Exploits River (NL)", "lat": 48.95, "lon": -55.67, "type": "River",
     "wqi": 86, "wqi_rating": "Good",
     "contaminants": {"manganese": 0.20, "mercury": 0.12, "lead": 0.10}},
]

# Air Quality Health Index (AQHI) — Government of Canada / Ontario
# Scale: 1-3 Low Risk, 4-6 Moderate Risk, 7-10 High Risk, 10+ Very High Risk
AIR_QUALITY_MONITORS = [
    # NB — elevated readings (near anomaly zone ~46.5, -66.0)
    {"name": "Moncton AQ Station", "lat": 46.15, "lon": -65.45,
     "pm25": 18.5, "no2": 22, "so2": 8, "aqhi": 5, "risk": "Moderate Risk"},
    {"name": "Saint John AQ Station", "lat": 46.35, "lon": -66.10,
     "pm25": 24.1, "no2": 28, "so2": 15, "aqhi": 7, "risk": "High Risk"},
    {"name": "Fredericton AQ Station", "lat": 46.55, "lon": -66.55,
     "pm25": 12.3, "no2": 14, "so2": 5, "aqhi": 3, "risk": "Low Risk"},
    # Other cities — low risk
    {"name": "Toronto AQ Station", "lat": 43.66, "lon": -79.39,
     "pm25": 11.2, "no2": 18, "so2": 4, "aqhi": 3, "risk": "Low Risk"},
    {"name": "Montreal AQ Station", "lat": 45.51, "lon": -73.57,
     "pm25": 13.1, "no2": 20, "so2": 6, "aqhi": 3, "risk": "Low Risk"},
    {"name": "Vancouver AQ Station", "lat": 49.28, "lon": -123.12,
     "pm25": 9.8, "no2": 15, "so2": 3, "aqhi": 2, "risk": "Low Risk"},
    {"name": "Calgary AQ Station", "lat": 51.05, "lon": -114.07,
     "pm25": 10.5, "no2": 16, "so2": 5, "aqhi": 2, "risk": "Low Risk"},
    {"name": "Edmonton AQ Station", "lat": 53.55, "lon": -113.49,
     "pm25": 14.2, "no2": 19, "so2": 7, "aqhi": 3, "risk": "Low Risk"},
    {"name": "Winnipeg AQ Station", "lat": 49.90, "lon": -97.14,
     "pm25": 8.9, "no2": 12, "so2": 3, "aqhi": 2, "risk": "Low Risk"},
    {"name": "Halifax AQ Station", "lat": 44.65, "lon": -63.57,
     "pm25": 7.8, "no2": 10, "so2": 2, "aqhi": 1, "risk": "Low Risk"},
]

MAJOR_HIGHWAYS = [
    {"name": "Trans-Canada Hwy (NB section)", "lat": 46.45, "lon": -66.30,
     "description": "High truck traffic through rural NB, diesel exhaust corridor"},
    {"name": "Trans-Canada Hwy (Moncton)", "lat": 46.15, "lon": -65.45,
     "description": "Major interchange, heavy freight traffic"},
    {"name": "Hwy 401 (Toronto)", "lat": 43.70, "lon": -79.42,
     "description": "Busiest highway in North America, 400k+ vehicles/day"},
    {"name": "Trans-Canada Hwy (Calgary)", "lat": 51.05, "lon": -114.10,
     "description": "Major east-west corridor through Alberta"},
    {"name": "Hwy 1 (Vancouver)", "lat": 49.25, "lon": -123.00,
     "description": "Port truck traffic, heavy diesel emissions"},
    {"name": "Trans-Canada Hwy (Winnipeg)", "lat": 49.88, "lon": -97.15,
     "description": "Major prairies freight corridor"},
]


SOCIOECONOMIC_REGIONS = [
    # New Brunswick — lower income, higher unemployment, limited healthcare access
    {"name": "Rural New Brunswick", "lat": 46.5, "lon": -66.0, "radius_km": 80,
     "median_income": 38200, "unemployment_pct": 11.8, "postsecondary_pct": 42,
     "physician_per_100k": 98, "nearest_hospital_km": 45, "indigenous_pct": 4.1,
     "food_insecurity_pct": 18.5, "pop_density_per_km2": 8.2,
     "vulnerability": "HIGH",
     "notes": "Rural, aging population with limited specialist access. Nearest neurologist ~120 km away."},
    {"name": "Moncton Region", "lat": 46.1, "lon": -64.8, "radius_km": 40,
     "median_income": 44500, "unemployment_pct": 8.2, "postsecondary_pct": 51,
     "physician_per_100k": 145, "nearest_hospital_km": 5, "indigenous_pct": 2.8,
     "food_insecurity_pct": 13.1, "pop_density_per_km2": 85,
     "vulnerability": "MEDIUM",
     "notes": "Regional centre with hospital, but serves large rural catchment area."},
    # Alberta
    {"name": "Calgary Metro", "lat": 51.0, "lon": -114.1, "radius_km": 50,
     "median_income": 72400, "unemployment_pct": 6.1, "postsecondary_pct": 67,
     "physician_per_100k": 215, "nearest_hospital_km": 8, "indigenous_pct": 3.0,
     "food_insecurity_pct": 8.9, "pop_density_per_km2": 1500,
     "vulnerability": "LOW",
     "notes": "Well-resourced urban centre with multiple hospitals and specialists."},
    {"name": "Edmonton Metro", "lat": 53.5, "lon": -113.5, "radius_km": 50,
     "median_income": 68100, "unemployment_pct": 6.8, "postsecondary_pct": 64,
     "physician_per_100k": 205, "nearest_hospital_km": 7, "indigenous_pct": 5.4,
     "food_insecurity_pct": 9.5, "pop_density_per_km2": 1200,
     "vulnerability": "LOW",
     "notes": "Provincial capital with university hospital and research facilities."},
    # British Columbia
    {"name": "Vancouver Metro", "lat": 49.3, "lon": -123.1, "radius_km": 50,
     "median_income": 65300, "unemployment_pct": 5.2, "postsecondary_pct": 69,
     "physician_per_100k": 240, "nearest_hospital_km": 5, "indigenous_pct": 2.5,
     "food_insecurity_pct": 10.2, "pop_density_per_km2": 5400,
     "vulnerability": "LOW",
     "notes": "Major urban centre, excellent healthcare infrastructure."},
    # Quebec
    {"name": "Quebec City Region", "lat": 46.8, "lon": -71.2, "radius_km": 40,
     "median_income": 52800, "unemployment_pct": 5.5, "postsecondary_pct": 61,
     "physician_per_100k": 230, "nearest_hospital_km": 6, "indigenous_pct": 1.2,
     "food_insecurity_pct": 7.8, "pop_density_per_km2": 950,
     "vulnerability": "LOW",
     "notes": "Provincial capital with CHUQ university hospital network."},
    # Saskatchewan
    {"name": "Regina Region", "lat": 50.5, "lon": -104.6, "radius_km": 50,
     "median_income": 58900, "unemployment_pct": 6.5, "postsecondary_pct": 56,
     "physician_per_100k": 155, "nearest_hospital_km": 10, "indigenous_pct": 9.3,
     "food_insecurity_pct": 12.4, "pop_density_per_km2": 45,
     "vulnerability": "MEDIUM",
     "notes": "Provincial capital but serves large rural/Indigenous communities."},
    {"name": "Saskatoon Region", "lat": 52.1, "lon": -106.7, "radius_km": 50,
     "median_income": 56200, "unemployment_pct": 7.0, "postsecondary_pct": 58,
     "physician_per_100k": 165, "nearest_hospital_km": 8, "indigenous_pct": 10.5,
     "food_insecurity_pct": 13.8, "pop_density_per_km2": 55,
     "vulnerability": "MEDIUM",
     "notes": "University city, but surrounding reserves have very limited access."},
    # Manitoba
    {"name": "Winnipeg Region", "lat": 49.9, "lon": -97.1, "radius_km": 50,
     "median_income": 55400, "unemployment_pct": 6.3, "postsecondary_pct": 59,
     "physician_per_100k": 185, "nearest_hospital_km": 7, "indigenous_pct": 12.2,
     "food_insecurity_pct": 14.1, "pop_density_per_km2": 1400,
     "vulnerability": "MEDIUM",
     "notes": "Significant urban Indigenous population, health disparities documented."},
    # Newfoundland
    {"name": "St. John's Region", "lat": 47.6, "lon": -52.7, "radius_km": 50,
     "median_income": 51200, "unemployment_pct": 9.5, "postsecondary_pct": 55,
     "physician_per_100k": 175, "nearest_hospital_km": 10, "indigenous_pct": 3.8,
     "food_insecurity_pct": 15.2, "pop_density_per_km2": 230,
     "vulnerability": "MEDIUM",
     "notes": "Provincial capital but aging population, outmigration of young workers."},
]

# Canadian national averages for comparison
NATIONAL_AVERAGES = {
    "median_income": 61400,
    "unemployment_pct": 5.4,
    "postsecondary_pct": 62,
    "physician_per_100k": 210,
    "food_insecurity_pct": 10.5,
}


# ============================================================
# Environmental Risk Analysis
# ============================================================

def analyze_environmental_risk(
    alert_lat: float,
    alert_lon: float,
    alert_radius_km: float = 50.0,
    search_radius_km: float = ENV_SEARCH_RADIUS_KM,
) -> dict:
    """Compute environmental risk factors for a single alert location."""

    nearby_industrial = []
    for site in INDUSTRIAL_SITES:
        dist = _haversine_km(alert_lat, alert_lon, site["lat"], site["lon"])
        if dist <= search_radius_km:
            nearby_industrial.append({
                "name": site["name"], "distance_km": round(dist, 1),
                "type": site["type"], "emissions": site["emissions"],
                "annual_tonnes": site["annual_tonnes"],
            })
    nearby_industrial.sort(key=lambda x: x["distance_km"])

    nearby_water = []
    elevated_water = 0
    for station in WATER_QUALITY_STATIONS:
        dist = _haversine_km(alert_lat, alert_lon, station["lat"], station["lon"])
        if dist <= search_radius_km:
            elevated = {k: v for k, v in station["contaminants"].items() if v >= ENV_CONCERN_THRESHOLD}
            if elevated:
                elevated_water += 1
            nearby_water.append({
                "name": station["name"], "distance_km": round(dist, 1),
                "type": station["type"], "contaminants": station["contaminants"],
                "elevated": elevated,
            })
    nearby_water.sort(key=lambda x: x["distance_km"])

    nearby_air = []
    poor_air = 0
    for monitor in AIR_QUALITY_MONITORS:
        dist = _haversine_km(alert_lat, alert_lon, monitor["lat"], monitor["lon"])
        if dist <= search_radius_km:
            if monitor["aqhi"] >= 4:  # Moderate Risk or higher
                poor_air += 1
            nearby_air.append({
                "name": monitor["name"], "distance_km": round(dist, 1),
                "pm25": monitor["pm25"], "aqhi": monitor["aqhi"],
                "risk": monitor["risk"],
            })
    nearby_air.sort(key=lambda x: x["distance_km"])

    nearby_highways = []
    for hwy in MAJOR_HIGHWAYS:
        dist = _haversine_km(alert_lat, alert_lon, hwy["lat"], hwy["lon"])
        if dist <= search_radius_km:
            nearby_highways.append({
                "name": hwy["name"], "distance_km": round(dist, 1),
                "description": hwy["description"],
            })
    nearby_highways.sort(key=lambda x: x["distance_km"])

    # Socioeconomic data — find nearest matching region
    nearest_socio = None
    best_socio_dist = float('inf')
    for region in SOCIOECONOMIC_REGIONS:
        dist = _haversine_km(alert_lat, alert_lon, region["lat"], region["lon"])
        if dist <= region["radius_km"] and dist < best_socio_dist:
            best_socio_dist = dist
            nearest_socio = region
    # Fall back to nearest region if none within radius
    if nearest_socio is None:
        for region in SOCIOECONOMIC_REGIONS:
            dist = _haversine_km(alert_lat, alert_lon, region["lat"], region["lon"])
            if dist < best_socio_dist:
                best_socio_dist = dist
                nearest_socio = region

    # Compute socioeconomic vulnerability score (0-5)
    socio_score = 0.0
    if nearest_socio:
        nat = NATIONAL_AVERAGES
        # Below-average income
        if nearest_socio["median_income"] < nat["median_income"] * 0.75:
            socio_score += 1.5
        elif nearest_socio["median_income"] < nat["median_income"] * 0.9:
            socio_score += 0.8
        # Above-average unemployment
        if nearest_socio["unemployment_pct"] > nat["unemployment_pct"] * 1.5:
            socio_score += 1.0
        elif nearest_socio["unemployment_pct"] > nat["unemployment_pct"] * 1.2:
            socio_score += 0.5
        # Low physician access
        if nearest_socio["physician_per_100k"] < nat["physician_per_100k"] * 0.6:
            socio_score += 1.5
        elif nearest_socio["physician_per_100k"] < nat["physician_per_100k"] * 0.8:
            socio_score += 0.8
        # Food insecurity
        if nearest_socio["food_insecurity_pct"] > nat["food_insecurity_pct"] * 1.5:
            socio_score += 0.5
        # Remote hospital access
        if nearest_socio["nearest_hospital_km"] > 30:
            socio_score += 0.5
    socio_score = round(min(5.0, socio_score), 1)

    # Compute env risk score (0-5)
    industrial_within_50 = sum(1 for s in nearby_industrial if s["distance_km"] <= 50)
    env_risk_score = min(5.0, industrial_within_50 * 0.8 + elevated_water * 1.0 + poor_air * 0.5)
    env_risk_score = round(env_risk_score, 1)

    # Combined risk score (env + socio, weighted)
    combined_risk = round(min(5.0, env_risk_score * 0.6 + socio_score * 0.4), 1)

    # Build summary
    parts = []
    if nearby_industrial:
        types = ", ".join(set(s["type"] for s in nearby_industrial[:3]))
        parts.append(f"{len(nearby_industrial)} industrial site(s) within {search_radius_km:.0f} km ({types})")
    if elevated_water:
        contams = set()
        for w in nearby_water:
            contams.update(w["elevated"].keys())
        parts.append(f"Elevated {', '.join(contams)} in {elevated_water} water station(s)")
    if poor_air:
        parts.append(f"{poor_air} air monitor(s) showing moderate+ readings")
    if nearby_highways:
        parts.append(f"{len(nearby_highways)} major highway(s) nearby")
    if nearest_socio and socio_score >= 2.0:
        parts.append(f"Socioeconomic vulnerability {nearest_socio['vulnerability']} ({nearest_socio['name']})")

    risk_summary = ". ".join(parts) + "." if parts else "No significant environmental factors identified."

    return {
        "env_risk_score": env_risk_score,
        "socio_risk_score": socio_score,
        "combined_risk_score": combined_risk,
        "risk_summary": risk_summary,
        "nearby_industrial": nearby_industrial,
        "nearby_water": nearby_water,
        "nearby_air": nearby_air,
        "nearby_highways": nearby_highways,
        "elevated_water_count": elevated_water,
        "poor_air_count": poor_air,
        "socioeconomic": nearest_socio,
    }


def enrich_alerts_with_environment(ranked_alerts) -> dict:
    """Compute environmental risk for each ranked alert. Returns dict keyed by alert_id."""
    results = {}
    for alert in ranked_alerts:
        env = analyze_environmental_risk(
            alert.center_lat, alert.center_lon,
            getattr(alert, "radius_km", 50.0),
        )
        results[alert.alert_id] = env
    return results


def get_all_env_sites() -> dict:
    """Return all environmental site data for map visualization."""
    return {
        "industrial": INDUSTRIAL_SITES,
        "water": WATER_QUALITY_STATIONS,
        "air": AIR_QUALITY_MONITORS,
        "highways": MAJOR_HIGHWAYS,
        "socioeconomic": SOCIOECONOMIC_REGIONS,
    }
