"""Visualization utilities for early warning alerts."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def plot_alert_map(df: pd.DataFrame, alerts: list, title: str = "Early Warning: Neurological Cluster Detection", env_data: dict | None = None):
    """Generate an interactive Folium map of Canada with report heat, alert zones, and environmental layers."""
    import folium
    from math import radians, sin, cos, asin, sqrt

    # Centre on Canada — dark theme with clear borders
    m = folium.Map(location=[56.0, -96.0], zoom_start=4,
                   tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
                   attr="&copy; OpenStreetMap &copy; CARTO",
                   control_scale=False,
                   scrollWheelZoom=False)

    # Confidence colour mapping
    confidence_colors = {
        "CRITICAL": "#8b1a2b",
        "HIGH": "#ff6b6b",
        "MEDIUM": "#ffa94d",
        "LOW": "#2ecc71",
    }

    # --- Assign each report to the nearest alert (by proximity to center) ---
    def _haversine(lat1, lon1, lat2, lon2):
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return 6371 * 2 * asin(sqrt(a))

    def _find_nearest_alert(row_lat, row_lon, alerts_list):
        """Find the nearest alert. Returns (confidence, distance_km) or (None, inf)."""
        best_conf = None
        best_dist = float('inf')
        for alert in alerts_list:
            conf = getattr(alert, "confidence", "")
            dist = _haversine(row_lat, row_lon, alert.center_lat, alert.center_lon)
            # Use a generous radius: the alert's own radius or 80km, whichever is larger
            max_radius = max(alert.radius_km, 80)
            if dist <= max_radius and dist < best_dist:
                best_dist = dist
                best_conf = conf
        return best_conf

    # Build lookup of report confidence assignments
    report_confs = {}
    coords = df[["latitude", "longitude"]].dropna()
    for idx, row in coords.iterrows():
        report_confs[idx] = _find_nearest_alert(row["latitude"], row["longitude"], alerts)

    # --- Background reports (not near any alert) — dim grey ---
    bg_group = folium.FeatureGroup(name="Background reports")
    for idx, row in coords.iterrows():
        if report_confs.get(idx) is None:
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=1.5, color="#555", fillColor="#666",
                fill=True, fill_opacity=0.15, weight=0,
            ).add_to(bg_group)
    bg_group.add_to(m)

    # --- Reports near alert zones — coloured by confidence level ---
    case_group = folium.FeatureGroup(name="Cases (colour-coded by confidence)")
    for idx, row in coords.iterrows():
        conf = report_confs.get(idx)
        if conf is not None:
            color = confidence_colors.get(conf, "#888")
            symptom = df.at[idx, "symptom"] if "symptom" in df.columns else ""
            severity = df.at[idx, "severity"] if "severity" in df.columns else ""
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3.5, color=color, fillColor=color,
                fill=True, fill_opacity=0.7, weight=1,
                popup=f"<b>{conf}</b><br>{symptom} | severity {severity}",
            ).add_to(case_group)
    case_group.add_to(m)

    # --- Alert zone circles (outline only, no fill — so env layers show through) ---
    alert_group = folium.FeatureGroup(name="Alert zones")

    # Collect alert data for the click-to-zoom script
    alert_zoom_data = []

    for i, alert in enumerate(alerts):
        confidence = getattr(alert, "confidence", "")
        color = confidence_colors.get(confidence, "#8b1a2b")

        zoom_level = 10 if alert.radius_km < 30 else 9 if alert.radius_km < 60 else 8
        alert_zoom_data.append({
            "lat": alert.center_lat, "lon": alert.center_lon,
            "radius": alert.radius_km, "zoom": zoom_level,
        })

        # Alert radius circle — outline only, no fill
        folium.Circle(
            location=[alert.center_lat, alert.center_lon],
            radius=alert.radius_km * 1000,
            color=color, weight=3, fill=False,
            popup=folium.Popup(
                f"<div style='min-width:220px'>"
                f"<b style='font-size:14px'>ALERT #{i+1}</b>"
                f"{'  <span style=\"color:' + color + ';font-weight:bold\">' + confidence + '</span>' if confidence else ''}"
                f"<hr style='margin:4px 0'>"
                f"<b>Cases:</b> {alert.n_cases}<br>"
                f"<b>Severity:</b> {alert.severity_mean:.1f}/5<br>"
                f"<b>Radius:</b> {alert.radius_km:.0f} km<br>"
                f"<b>Window:</b> {alert.time_window}<br>"
                f"<b>Score:</b> {alert.anomaly_score:.2f}<br>"
                f"{'<b>Symptoms:</b> ' + str(getattr(alert, 'dominant_symptoms', '')) + '<br>' if getattr(alert, 'dominant_symptoms', None) else ''}"
                f"<hr style='margin:4px 0'>"
                f"<i>{alert.description}</i>"
                f"</div>",
                max_width=350,
            ),
        ).add_to(alert_group)

        # Centre marker with label
        folium.Marker(
            location=[alert.center_lat, alert.center_lon],
            icon=folium.DivIcon(
                html=f'<div style="cursor:pointer;font-size:11px;font-weight:bold;color:{color};'
                     f'text-shadow:1px 1px 2px black,-1px -1px 2px black;white-space:nowrap">'
                     f'#{i+1} — {alert.n_cases} cases</div>',
                icon_size=(150, 20), icon_anchor=(75, 10),
            ),
        ).add_to(alert_group)

    alert_group.add_to(m)

    # --- Environmental layers ---
    from environmental_data import get_all_env_sites
    env_sites = get_all_env_sites()

    # Industrial sites (pink)
    ind_group = folium.FeatureGroup(name="Industrial Sites", show=True)
    for site in env_sites["industrial"]:
        folium.CircleMarker(
            location=[site["lat"], site["lon"]],
            radius=7, color="#e91e90", fillColor="#e91e90",
            fill=True, fill_opacity=1.0, weight=1.5,
            popup=folium.Popup(
                f"<div style='min-width:200px'>"
                f"<b style='color:#e91e90'>{site['name']}</b><br>"
                f"<b>Type:</b> {site['type']}<br>"
                f"<b>Emissions:</b> {site['emissions']}<br>"
                f"<b>Annual:</b> {site['annual_tonnes']} tonnes"
                f"</div>", max_width=300,
            ),
        ).add_to(ind_group)
    ind_group.add_to(m)

    # Water quality — CCME Water Quality Index
    # Poor (0-44) / Marginal (45-64) / Fair (65-79) / Good (80-94) / Excellent (95-100)
    water_group = folium.FeatureGroup(name="Water Quality (CCME WQI)", show=True)
    for station in env_sites["water"]:
        wqi = station.get("wqi", 80)
        wqi_rating = station.get("wqi_rating", "Good")
        if wqi < 45:
            wcolor = "#00e5ff"  # bright cyan — Poor
        elif wqi < 65:
            wcolor = "#4169e1"  # royal blue — Marginal
        elif wqi < 80:
            wcolor = "#1e3a8a"  # dark blue — Fair
        else:
            wcolor = "#172554"  # very dark navy — Good/Excellent
        contam_lines = "<br>".join(
            f"{'<b style=\"color:#00e5ff\">' if v >= 0.7 else '<b>'}{k}:</b> {v:.2f}"
            for k, v in station["contaminants"].items()
        )
        folium.CircleMarker(
            location=[station["lat"], station["lon"]],
            radius=6, color=wcolor, fillColor=wcolor,
            fill=True, fill_opacity=1.0, weight=1.5,
            popup=folium.Popup(
                f"<div style='min-width:200px'>"
                f"<b style='color:{wcolor}'>{station['name']}</b><br>"
                f"<b>Type:</b> {station['type']}<br>"
                f"<b>CCME WQI:</b> {wqi}/100 — <b>{wqi_rating}</b><br>"
                f"<hr style='margin:4px 0'>"
                f"{contam_lines}"
                f"</div>", max_width=300,
            ),
        ).add_to(water_group)
    water_group.add_to(m)

    # Air quality — AQHI (Air Quality Health Index)
    # Low Risk (1-3) / Moderate Risk (4-6) / High Risk (7-10) / Very High Risk (10+)
    air_group = folium.FeatureGroup(name="Air Quality (AQHI)", show=True)
    for monitor in env_sites["air"]:
        aqhi = monitor.get("aqhi", 2)
        risk = monitor.get("risk", "Low Risk")
        if aqhi >= 7:
            acolor = "#d4a0ff"  # bright lavender — High Risk
        elif aqhi >= 4:
            acolor = "#9c5ec7"  # medium purple — Moderate Risk
        else:
            acolor = "#4a2d73"  # dark purple — Low Risk
        folium.CircleMarker(
            location=[monitor["lat"], monitor["lon"]],
            radius=6, color=acolor, fillColor=acolor,
            fill=True, fill_opacity=1.0, weight=1.5,
            popup=folium.Popup(
                f"<div style='min-width:180px'>"
                f"<b style='color:{acolor}'>{monitor['name']}</b><br>"
                f"<b>AQHI:</b> {aqhi}/10 — <b>{risk}</b><br>"
                f"<hr style='margin:4px 0'>"
                f"<b>PM2.5:</b> {monitor['pm25']} µg/m³<br>"
                f"<b>NO2:</b> {monitor['no2']} ppb<br>"
                f"<b>SO2:</b> {monitor['so2']} ppb"
                f"</div>", max_width=250,
            ),
        ).add_to(air_group)
    air_group.add_to(m)

    # Major highways
    hwy_group = folium.FeatureGroup(name="Major Highways", show=True)
    for hwy in env_sites["highways"]:
        folium.CircleMarker(
            location=[hwy["lat"], hwy["lon"]],
            radius=5, color="#ffffff", fillColor="#ffffff",
            fill=True, fill_opacity=1.0, weight=1,
            popup=f"<b>{hwy['name']}</b><br>{hwy['description']}",
        ).add_to(hwy_group)
    hwy_group.add_to(m)

    # Socioeconomic vulnerability zones
    socio_group = folium.FeatureGroup(name="Socioeconomic Data", show=True)
    socio_colors = {"HIGH": "#ff6b6b", "MEDIUM": "#ffa94d", "LOW": "#69db7c"}
    for region in env_sites.get("socioeconomic", []):
        scolor = socio_colors.get(region["vulnerability"], "#69db7c")
        folium.Circle(
            location=[region["lat"], region["lon"]],
            radius=region["radius_km"] * 1000,
            color=scolor, weight=2, fill=False,
            dash_array="8 4",
            popup=folium.Popup(
                f"<div style='min-width:220px'>"
                f"<b style='color:{scolor};font-size:13px'>{region['name']}</b>"
                f"<br><b>Vulnerability:</b> {region['vulnerability']}"
                f"<hr style='margin:4px 0'>"
                f"<b>Median Income:</b> ${region['median_income']:,}<br>"
                f"<b>Unemployment:</b> {region['unemployment_pct']}%<br>"
                f"<b>Post-Secondary:</b> {region['postsecondary_pct']}%<br>"
                f"<b>Physicians:</b> {region['physician_per_100k']} per 100k<br>"
                f"<b>Nearest Hospital:</b> {region['nearest_hospital_km']} km<br>"
                f"<b>Food Insecurity:</b> {region['food_insecurity_pct']}%<br>"
                f"<b>Indigenous Pop:</b> {region['indigenous_pct']}%<br>"
                f"<b>Pop Density:</b> {region['pop_density_per_km2']}/km²"
                f"<hr style='margin:4px 0'>"
                f"<i>{region['notes']}</i>"
                f"</div>", max_width=300,
            ),
        ).add_to(socio_group)
    socio_group.add_to(m)

    # --- Legend ---
    legend_html = """
    <div style="position:fixed;bottom:12px;left:12px;z-index:1000;
                background:rgba(20,22,30,0.92);padding:8px 12px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.4);font-size:10px;
                font-family:'Segoe UI',system-ui,sans-serif;
                color:#ccc;line-height:1.7;border:1px solid rgba(255,255,255,0.06)">
        <div style="display:flex;gap:6px;align-items:center;margin-bottom:2px">
            <span style="font-weight:600;color:#999;font-size:9px;text-transform:uppercase;letter-spacing:.5px">Alerts</span>
            <span style="color:#8b1a2b">&#9679;</span><span>Critical</span>
            <span style="color:#ff6b6b">&#9679;</span><span>High</span>
            <span style="color:#ffa94d">&#9679;</span><span>Medium</span>
            <span style="color:#2ecc71">&#9679;</span><span>Low</span>
            <span style="color:#555">&#9679;</span><span style="color:#777">Background</span>
        </div>
        <div style="display:flex;gap:6px;align-items:center;margin-bottom:2px">
            <span style="font-weight:600;color:#999;font-size:9px;text-transform:uppercase;letter-spacing:.5px">Water Quality <span style="font-size:7px">(CCME WQI)</span></span>
            <span style="color:#00e5ff">&#9679;</span><span>Poor</span>
            <span style="color:#4169e1">&#9679;</span><span>Marginal</span>
            <span style="color:#1e3a8a">&#9679;</span><span>Fair</span>
            <span style="color:#172554">&#9679;</span><span>Good</span>
        </div>
        <div style="display:flex;gap:6px;align-items:center;margin-bottom:2px">
            <span style="font-weight:600;color:#999;font-size:9px;text-transform:uppercase;letter-spacing:.5px">Air Quality <span style="font-size:7px">(AQHI)</span></span>
            <span style="color:#d4a0ff">&#9679;</span><span>High Risk</span>
            <span style="color:#9c5ec7">&#9679;</span><span>Moderate Risk</span>
            <span style="color:#4a2d73">&#9679;</span><span>Low Risk</span>
        </div>
        <div style="display:flex;gap:6px;align-items:center;margin-bottom:2px">
            <span style="font-weight:600;color:#999;font-size:9px;text-transform:uppercase;letter-spacing:.5px">Socioeconomic Status</span>
            <span style="border-top:2px dashed #ff6b6b;width:16px;display:inline-block"></span><span>High Vuln.</span>
            <span style="border-top:2px dashed #ffa94d;width:16px;display:inline-block"></span><span>Med</span>
            <span style="border-top:2px dashed #69db7c;width:16px;display:inline-block"></span><span>Low</span>
        </div>
        <div style="display:flex;gap:6px;align-items:center">
            <span style="color:#e91e90">&#9679;</span><span>Industrial Site</span>
            <span style="border-left:1px solid rgba(255,255,255,0.1);height:10px"></span>
            <span style="color:#ffffff">&#9679;</span><span>Highways</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- Title banner ---
    title_html = f"""
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:1000;
                background:rgba(255,255,255,0.92);padding:8px 24px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2);font-family:sans-serif">
        <b style="font-size:16px">{title}</b>
        <span style="font-size:12px;color:#666;margin-left:12px">{len(alerts)} alerts | {len(df)} reports</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # Layer control toggle (collapsed to save space, hover to expand)
    folium.LayerControl(collapsed=True).add_to(m)

    # Compact the layer control styling
    layer_style = """
    <style>
    .leaflet-control-layers { font-size:11px !important; }
    .leaflet-control-layers-list { padding:4px 8px !important; }
    .leaflet-control-layers label { margin-bottom:1px !important; }
    </style>
    """
    m.get_root().html.add_child(folium.Element(layer_style))

    # --- Inject click-to-zoom on alert circles + postMessage listener ---
    import json
    alert_zoom_js_data = json.dumps(alert_zoom_data)
    interaction_js = f"""
    <script>
    (function() {{
        // Find the Leaflet map instance
        function getMap() {{
            for (var key in window) {{
                try {{ if (window[key] instanceof L.Map) return window[key]; }} catch(e) {{}}
            }}
            return null;
        }}

        // Alert zone data for click-to-zoom
        var alertZones = {alert_zoom_js_data};

        // Wait for map to be ready, then attach click-to-zoom on all circle layers
        function setupClickZoom() {{
            var map = getMap();
            if (!map) {{ setTimeout(setupClickZoom, 200); return; }}

            // Listen for clicks on the map and check proximity to alert centers
            map.on('click', function(e) {{
                var lat = e.latlng.lat, lng = e.latlng.lng;
                for (var i = 0; i < alertZones.length; i++) {{
                    var a = alertZones[i];
                    var dist = map.distance(e.latlng, L.latLng(a.lat, a.lon)) / 1000;
                    if (dist <= a.radius * 1.2) {{
                        map.flyTo([a.lat, a.lon], a.zoom, {{duration: 1}});
                        return;
                    }}
                }}
            }});
        }}
        setupClickZoom();

        // PostMessage listener for cross-iframe zoom (from dashboard table)
        window.addEventListener('message', function(e) {{
            if (e.data && e.data.action === 'flyTo') {{
                var map = getMap();
                if (map) map.flyTo([e.data.lat, e.data.lon], e.data.zoom || 8, {{duration: 1.5}});
            }}
        }});
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(interaction_js))

    m.save("alert_map.html")
    print("Saved alert_map.html (interactive map)")

    # Also save a static PNG for logs/reports
    _plot_alert_map_static(df, alerts, title)


def _plot_alert_map_static(df: pd.DataFrame, alerts: list, title: str):
    """Fallback static matplotlib map (saved as PNG)."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    normal = df[~df.get("is_anomaly", False)]
    anomalous = df[df.get("is_anomaly", False)]

    ax.scatter(normal["longitude"], normal["latitude"], c="steelblue", alpha=0.15, s=8, label="Normal reports")
    if len(anomalous):
        ax.scatter(anomalous["longitude"], anomalous["latitude"],
                   c="crimson", alpha=0.5, s=20, label="Anomalous reports")

    for i, alert in enumerate(alerts):
        radius_deg = alert.radius_km / 111.0
        circle = plt.Circle(
            (alert.center_lon, alert.center_lat), radius_deg,
            fill=False, edgecolor="red", linewidth=2, linestyle="--",
        )
        ax.add_patch(circle)
        ax.annotate(
            f"ALERT #{i+1}\n{alert.n_cases} cases\nseverity {alert.severity_mean:.1f}",
            (alert.center_lon, alert.center_lat),
            fontsize=8, fontweight="bold", color="darkred",
            ha="center", va="bottom",
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("alert_map.png", dpi=150)
    plt.close()


def plot_timeline(df: pd.DataFrame, alerts: list):
    """Plot daily case count timeline with alert windows highlighted."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    daily = df.groupby("date").size()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily.index, daily.values, color="steelblue", linewidth=0.8)
    ax.fill_between(daily.index, daily.values, alpha=0.2, color="steelblue")

    for alert in alerts:
        # Handle merged time windows like "2025-10-01 to 2025-12-01 | last 90 days"
        for segment in alert.time_window.split(" | "):
            parts = segment.strip().split(" to ")
            if len(parts) == 2:
                try:
                    start, end = pd.to_datetime(parts[0]), pd.to_datetime(parts[1])
                    ax.axvspan(start, end, alpha=0.2, color="red", label="Alert window")
                except Exception:
                    pass

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Reports")
    ax.set_title("Neurological Report Timeline")
    plt.tight_layout()
    plt.savefig("alert_timeline.png", dpi=150)
    plt.close()
    print("Saved alert_timeline.png")
