import streamlit as st # type: ignore
import plotly.express as px # type: ignore
import pandas as pd # type: ignore
import numpy as np
import os
import warnings
import io
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import re
import urllib.parse
warnings.filterwarnings("ignore")


def read_csv_with_encodings(path_or_buffer, encodings=None, **kwargs):
    """Try reading a CSV with a list of encodings and return the first successful DataFrame.

    This helps avoid mojibake for non-UTF8 files (e.g. Arabic encoded in Windows-1256).
    """
    if encodings is None:
        encodings = ["utf-8", "utf-8-sig", "cp1256", "windows-1256", "iso-8859-1", "latin1"]
    last_exc = None
    for enc in encodings:
        try:
            return pd.read_csv(path_or_buffer, encoding=enc, **kwargs)
        except Exception as e:
            last_exc = e
            continue
    # If none worked, raise the last exception
    raise last_exc

st.set_page_config(page_title="CCTV AI Monitoring Dashboard", layout="wide") # 

#----------------------- HEADER -----------------------
header_img_path = "https://img.freepik.com/premium-photo/high-tech-surveillance-camera-overlooking-urban-cityscape-with-digital-interface_97843-69057.jpg"  

st.markdown(f"""
    <style>
    /* الخط العام */
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');

    html, body, .stApp {{
        height: 100%;
        margin: 0;
        padding: 0;
    }}
    .block-container {{
        padding: 0;
    }}
    html, body, .stApp, .main, .block-container {{
        font-family: 'Cairo', sans-serif !important;
    }}

    /* قسم الهيدر البانر */
    .full-screen-header {{
        position: relative;
        width: 100%;
        height: 80vh;  /* يمكنك تغييره */
        background: linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), 
                    url('{header_img_path}') no-repeat center center;
        background-size: cover;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        color: white;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
    }}
    .header-title {{
        font-size: 3vw;
        font-weight: 600;
        text-align: center;
    }}
    .header-subtitle {{
        margin-top: 1rem;
        font-size: 1.5vw;
        text-align: center;
    }}
    .header-button {{
        margin-top: 2rem;
        padding: 0.8rem 2rem;
        font-size: 1rem;
        background-color: rgba(255, 255, 255, 0.8);
        color: #333;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        text-decoration: none;
    }}
    .header-button:hover {{
        background-color: rgba(255,255,255,1);
    }}

    /* Responsive للهواتف */
    @media (max-width: 768px) {{
        .header-title {{
            font-size: 6vw;
        }}
        .header-subtitle {{
            font-size: 3vw;
        }}
        .header-button {{
            padding: 0.6rem 1.5rem;
        }}
    }}
    </style>

    <div class="full-screen-header">
        <div class="header-title">CCTV AI Monitoring Dashboard</div>
        <div class="header-subtitle">Upload your CSV file below to start</div>
        
    </div>
""", unsafe_allow_html=True)


# ---------------------- FILE UPLOAD ----------------------
fl = st.file_uploader("", type=["csv"])

if fl is not None:
    # show progress while reading/processing the uploaded file
    progress = st.progress(0)
    with st.spinner("Reading and processing CSV..."):
        try:
            progress.progress(10)
            fl.seek(0)
            df = read_csv_with_encodings(fl)
            progress.progress(60)
            # small cleanup step
            df.columns = df.columns
            progress.progress(85)
        except Exception:
            # Last-resort fallback to latin1 if all attempts fail
            fl.seek(0)
            df = pd.read_csv(fl, encoding="latin1", on_bad_lines='skip')
            try:
                progress.progress(95)
            except Exception:
                pass
        try:
            progress.progress(100)
        except Exception:
            pass
    st.success("File Loaded Successfully!")
else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()


# ---------------------- CLEAN COLUMN NAMES ----------------------
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")

# ---------------------- DATE PROCESSING ----------------------
date_cols = ["install_date", "last_maintenance_date"]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


# Helper: resolve common column-name variants (e.g. 'Cam ID' vs 'Cam_ID')
def find_column(df, *candidates):
    """Return the first candidate that exists in df.columns, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    # try normalized versions: replace spaces/hyphens with underscores
    for c in candidates:
        c_norm = c.strip().replace(' ', '_').replace('-', '_')
        if c_norm in df.columns:
            return c_norm
    return None


# Helper: return one row per camera ID (most recent if date columns exist)
def dedupe_by_camera(df, cam_col: str):
    """Return a DataFrame with exactly one row per camera identifier.

    If cam_col is present, choose the most recent row by 'last_maintenance_date' or 'install_date' where available.
    If cam_col is None, fall back to dropping duplicate rows entirely.
    """
    if cam_col is None or cam_col not in df.columns:
        return df.drop_duplicates()

    # prefer to pick the latest row per camera using dates, otherwise use last occurrence
    sort_col = None
    for c in ("last_maintenance_date", "install_date"):
        if c in df.columns:
            sort_col = c
            break

    if sort_col:
        # sort ascending then take last row per group -> effectively most recent
        tmp = df.sort_values(sort_col, na_position='first')
        return tmp.groupby(cam_col, as_index=False).last()
    else:
        return df.drop_duplicates(subset=[cam_col])

# detect camera id column once (used to dedupe rows into unique cameras later)
cam_col_global = find_column(df, 'Cam ID', 'Cam_ID', 'CamID', 'Cam_Id', 'CamId', 'cam id', 'cam_id')


# ---------------------- SIDEBAR FILTERS ----------------------
st.sidebar.header("Filter the Data")

location_filter = st.sidebar.multiselect("Location", df["Location"].unique())
gov_filter = st.sidebar.multiselect("Governorate", df.get("Kuwait_Governorate", pd.Series([], dtype=object)).unique())
brand_filter = st.sidebar.multiselect("Brand", df["brand"].unique())
health_filter = st.sidebar.multiselect("Health Status", df["health_status"].unique())
connect_filter = st.sidebar.multiselect("Connectivity", df["connectivity_status"].unique())


# (filters and metrics are applied later after coordinates parsing so we have cleaned numeric location fields)



# ---------------------- TOP METRICS ----------------------
# (metrics are shown above and built from deduped camera entries)


# ---------------------- MAPS ----------------------


# التحقق من وجود الأعمدة الأساسية
required_cols = {"Latitude", "Longitude"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV must contain columns: {required_cols}")
    st.stop()

# دالة لتحويل أي صيغة Coordinates إلى Decimal Degrees
def dms_to_dd(coord_str):
    """
    تحول من صيغة 29°22'33.2"N أو 47.9774 إلى decimal degrees.
    """
    try:
        coord_str = '' if coord_str is None else str(coord_str).strip()
        if coord_str == '':
            return None
        # replace Arabic-Indic digits with western digits
        arabic_digits = str.maketrans('٠١٢٣٤٥٦٧٨٩،', '0123456789,')
        coord_str = coord_str.translate(arabic_digits)
        # normalize comma as decimal separator
        coord_str = coord_str.replace('\u200f', '').replace('\u200e', '')
        coord_str = coord_str.replace(',', '.')
        coord_str = coord_str.replace('\xa0', '')
        coord_str = coord_str.strip()

        # If it's a plain decimal number now
        if re.match(r'^-?\d+(\.\d+)?$', coord_str):
            return float(coord_str)

        # Try to parse DMS formats like 29°22'33.2"N or 29 22 33.2 N
        parts = re.split('[°\u00B0\'"\s]+', coord_str)
        # filter out empty
        parts = [p for p in parts if p != '']
        if len(parts) >= 1 and re.match(r'^-?\d+(\.\d+)?$', parts[0]):
            d = float(parts[0])
            m = float(parts[1]) if len(parts) > 1 and re.match(r'^-?\d+(\.\d+)?$', parts[1]) else 0.0
            s = float(parts[2]) if len(parts) > 2 and re.match(r'^-?\d+(\.\d+)?$', parts[2]) else 0.0
            dd = abs(d) + m/60.0 + s/3600.0
            # determine sign from first part or presence of S/W
            if '-' in parts[0] or 'S' in coord_str.upper() or 'W' in coord_str.upper():
                dd = -dd
            return dd

        return None
    except Exception:
        return None

# تحويل الأعمدة لـ Decimal Degrees
def split_latlon_pair(s):
    """If a single string contains both lat and lon (e.g. '29°15'34.48"N 48°1'58.91"E'),
    split into (lat_str, lon_str). Otherwise return (s, None).
    """
    if not s or not isinstance(s, str):
        return s, None
    # look for N or S first occurrence
    m = re.search(r'[NnSs]', s)
    if m:
        pos = m.end()
        lat_part = s[:pos].strip()
        lon_part = s[pos:].strip()
        # if lon_part is empty, try splitting by comma/space
        if lon_part == '' and (',' in s or ';' in s):
            parts = re.split('[,;]', s)
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()
        return lat_part, lon_part
    # fallback: if string contains two degree symbols, split in middle
    if s.count('°') >= 2 or s.count('\u00B0') >= 2:
        # split roughly in half by whitespace
        parts = re.split('\s+', s)
        half = len(parts)//2
        return ' '.join(parts[:half]), ' '.join(parts[half:])
    return s, None

# Preprocess rows where Latitude contains both lat+lon
lat_series = df['Latitude'].astype(str)
lon_series = df['Longitude'].astype(str)
for i, (lat_val, lon_val) in enumerate(zip(lat_series, lon_series)):
    # If longitude is missing/empty and latitude contains both
    if (pd.isna(df.at[i, 'Longitude']) or str(df.at[i, 'Longitude']).strip() == ''):
        lat_candidate = lat_val
        if isinstance(lat_candidate, str) and re.search(r'[NnSs].*[EeWw]|\u00B0.*\u00B0', lat_candidate):
            a, b = split_latlon_pair(lat_candidate)
            if b:
                df.at[i, 'Latitude'] = a
                df.at[i, 'Longitude'] = b

# Now apply conversion
df['Latitude'] = df['Latitude'].apply(dms_to_dd)
df['Longitude'] = df['Longitude'].apply(dms_to_dd)

# حذف القيم غير الصالحة
df = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

# لو بعد التنظيف البيانات فاضية
if df.empty:
    st.error("No valid Latitude/Longitude values found in CSV.")
    st.stop()

# NOTE: don't show overall coordinate ranges here (they would be unfiltered)
# Coordinate ranges for the currently-selected (filtered) dataset are shown after filtering below.

# Re-apply filters to the processed dataframe so the map and charts use the same filtered set
filtered_df = df.copy()
if location_filter:
    filtered_df = filtered_df[filtered_df["Location"].isin(location_filter)]
if brand_filter:
    filtered_df = filtered_df[filtered_df["brand"].isin(brand_filter)]
if health_filter:
    filtered_df = filtered_df[filtered_df["health_status"].isin(health_filter)]
if connect_filter:
    filtered_df = filtered_df[filtered_df["connectivity_status"].isin(connect_filter)]
if gov_filter:
    filtered_df = filtered_df[filtered_df["Kuwait_Governorate"].isin(gov_filter)]

# ---------------------- DEDUP & PER-CAMERA AGGREGATES ----------------------
# Build camera-unique and per-camera aggregates so metrics use distinct cameras
filtered_unique = dedupe_by_camera(filtered_df, cam_col_global)

# Build per-camera sums (from event rows) and merge into the unique camera table
per_camera_agg = None
if cam_col_global and cam_col_global in filtered_df.columns:
    agg_cols = {}
    if 'estimated_daily_vehicles' in filtered_df.columns:
        agg_cols['total_violations'] = ('estimated_daily_vehicles', 'sum')
    if 'sudden_stop' in filtered_df.columns:
        agg_cols['total_sudden_stop'] = ('sudden_stop', 'sum')
    if 'wrong_direction' in filtered_df.columns:
        agg_cols['total_wrong_direction'] = ('wrong_direction', 'sum')
    if agg_cols:
        per_camera_agg = filtered_df.groupby(cam_col_global).agg(**agg_cols).reset_index()

if per_camera_agg is not None and cam_col_global in filtered_unique.columns:
    merged_unique = filtered_unique.merge(per_camera_agg, how='left', left_on=cam_col_global, right_on=cam_col_global).fillna(0)
else:
    merged_unique = filtered_unique.copy()

# Top-line KPIs (distinct cameras)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Cameras", len(merged_unique))
col2.metric("Online Cameras", int((merged_unique.get("connectivity_status") == "Online").sum()))
col3.metric("Avg Uptime %", round(merged_unique.get('uptime_percent').mean() if 'uptime_percent' in merged_unique.columns else 0, 2))
col4.metric("Avg Coverage %", round(merged_unique.get('coverage_percent').mean() if 'coverage_percent' in merged_unique.columns else 0, 2))

# Show filtered coordinate ranges (helps confirm filters applied are affecting map extents)
if not filtered_df.empty:
    fmin_lat, fmax_lat = filtered_df['Latitude'].min(), filtered_df['Latitude'].max()
    fmin_lon, fmax_lon = filtered_df['Longitude'].min(), filtered_df['Longitude'].max()
    st.write(f"Filtered Latitude range: {fmin_lat} to {fmax_lat} — Filtered Longitude range: {fmin_lon} to {fmax_lon}")

# Per-camera totals/averages (additional KPIs)
total_violations = int(merged_unique.get('total_violations', pd.Series([0])).sum())
avg_violations = round(merged_unique.get('total_violations', pd.Series([0])).mean(), 2) if 'total_violations' in merged_unique.columns else 0
total_sudden = int(merged_unique.get('total_sudden_stop', pd.Series([0])).sum()) if 'total_sudden_stop' in merged_unique.columns else 0
total_wrong = int(merged_unique.get('total_wrong_direction', pd.Series([0])).sum()) if 'total_wrong_direction' in merged_unique.columns else 0

# Stopped / Warning cameras counts (based on health_status/connectivity_status text)
stopped_mask = pd.Series(False, index=merged_unique.index)
warning_mask = pd.Series(False, index=merged_unique.index)
if 'health_status' in merged_unique.columns:
    hs = merged_unique['health_status'].astype(str).str.lower()
    stopped_mask = stopped_mask | hs.str.contains('stop', na=False)
    warning_mask = warning_mask | hs.str.contains('warn', na=False)
if 'connectivity_status' in merged_unique.columns:
    cs = merged_unique['connectivity_status'].astype(str).str.lower()
    stopped_mask = stopped_mask | cs.str.contains('stop', na=False)

total_stopped_cameras = int(stopped_mask.sum())
total_warning_cameras = int(warning_mask.sum())

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Violations", f"{total_violations}")
k2.metric("Avg Violations / Camera", f"{avg_violations}")
#k3.metric("Total Sudden Stop Events", f"{total_sudden}")
#k4.metric("Total Wrong Direction Events", f"{total_wrong}")
#k5.metric("Stopped Cameras", f"{total_stopped_cameras}")
k3.metric("Warning Cameras", f"{total_warning_cameras}")


# ---------------------- QA / VERIFICATION ----------------------
with st.expander("QA / Debug: Filter and Dedupe Checks (click to expand)"):
    st.write("Data rows after filtering (event-level):", len(filtered_df))
    st.write("Unique camera rows after dedupe:", len(filtered_unique))
    st.write("Merged unique rows (with per-camera aggregates):", len(merged_unique))
    if cam_col_global and cam_col_global in merged_unique.columns:
        st.write("Sample camera ids (merged):", merged_unique[cam_col_global].head(10).tolist())
    # show quick aggregation check (events -> per-camera sum) for violations if available
    if 'estimated_daily_vehicles' in filtered_df.columns and 'total_violations' in merged_unique.columns:
        st.write("Event-level violations (sum):", int(filtered_df['estimated_daily_vehicles'].sum()))
        st.write("Per-camera aggregated violations (sum of merged):", int(merged_unique['total_violations'].sum()))


# Create a map-focused dataframe so flipping coordinates affects only map renders
map_df = filtered_df.copy()

# Dedupe the map data so the map shows one marker per camera
map_df_unique = dedupe_by_camera(map_df, cam_col_global)

# also dedupe the filtered rows into a unique-camera view for metrics and charts
filtered_unique = dedupe_by_camera(filtered_df, cam_col_global)

# recompute ranges/center using the map dataframe
map_center = [map_df_unique['Latitude'].mean(), map_df_unique['Longitude'].mean()]
st.write(f"Map center (lat, lon): {map_center}")

# دالة لتحديد لون Marker حسب Health Status
def get_marker_color(status):
    status = str(status).lower()
    if status == "online":
        return "green"
    elif status == "offline":
        return "red"
    else:
        return "gray"

# Helper: build an SVG pin (Google-style) and return a data URI
def make_pin_data_uri(color="#3388ff"):
    svg = f'''
    <svg xmlns="http://www.w3.org/2000/svg" width="36" height="48" viewBox="0 0 24 34">
      <path d="M12 0C7 0 3 4 3 9c0 7.5 9 20 9 20s9-12.5 9-20c0-5-4-9-9-9z" fill="{color}" stroke="#222" stroke-width="0.5"/>
      <circle cx="12" cy="9" r="4" fill="#fff"/>
    </svg>
    '''
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)

# Option to use new custom icons or simple circle markers
use_custom_icons = st.checkbox("Use Google-style markers", value=True)


@st.cache_data(show_spinner=False)
def build_map_html(df_json: str, use_custom: bool, center: list, zoom_start: int = 12) -> str:
    """Build and return rendered Folium HTML for a given DataFrame JSON.

    Caching this HTML avoids rebuilding markers on every Streamlit rerun
    (zoom/pan client-side won't trigger a server rebuild).
    """
    df_local = pd.read_json(df_json)
    m_local = folium.Map(location=center, zoom_start=zoom_start, tiles="OpenStreetMap")
    cluster_local = MarkerCluster(name="Cameras").add_to(m_local)
    n = len(df_local)
    # disable custom icons for very large sets internally too
    if n > 200:
        use_custom = False
    for _, row in df_local.iterrows():
        popup_text = f"<b>{row.get('Location', '')}</b><br>Health: {row.get('health_status', 'N/A')}<br>Coverage: {row.get('coverage_percent', 'N/A')}%"
        radius = max(5, 5 + (row.get("coverage_percent", 0) / 20))
        color = get_marker_color(row.get('health_status', None))
        if use_custom:
            color_map = {"green": "#2ecc71", "red": "#e74c3c", "gray": "#95a5a6"}
            col = color_map.get(color, color)
            icon_uri = make_pin_data_uri(col)
            icon = folium.CustomIcon(icon_image=icon_uri, icon_size=(36, 48), icon_anchor=(18, 48))
            folium.Marker(location=[row["Latitude"], row["Longitude"]], popup=popup_text, icon=icon).add_to(cluster_local)
        else:
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup_text
            ).add_to(cluster_local)
    return m_local.get_root().render()


# (per-camera aggregates and KPIs already computed above; no duplicate calculations here)

# Build map HTML once (cached) and embed via components.html for faster client-side interaction
st.subheader("📍 Camera Locations Map (Free Map)")
if len(filtered_df) == 0:
    st.info("No cameras to display on the map for the selected filters.")
else:
    df_json = map_df_unique.to_json(orient="records")
    map_html = build_map_html(df_json, use_custom_icons, map_center, zoom_start=12)
    # embed the pre-rendered HTML (interactive Leaflet) — much faster than rebuilding on every rerun
    components.html(map_html, height=700)

# ---------------------- END MAPS ----------------------


# ---------------------- VISUALIZATIONS ----------------------

# Helper Layout function
def two_columns_chart(title1, desc1, fig1, title2, desc2, fig2):
    """Render two charts side-by-side. Prefer Plotly for nicer interactive charts, fallback to matplotlib."""
    import plotly.graph_objs as go

    def render(container, fig):
        # If the `fig` is a pre-formatted HTML or markdown string, render as markdown
        if isinstance(fig, str):
            try:
                container.markdown(fig, unsafe_allow_html=True)
                return
            except Exception:
                # fallback to write
                container.write(fig)

        # Plotly figure
        if isinstance(fig, go.Figure):
            container.plotly_chart(fig, width='stretch')
        else:
            # matplotlib / other
            try:
                container.pyplot(fig)
            except Exception:
                # if fig is a DataFrame, render as table
                try:
                    container.write(fig)
                except Exception:
                    container.info("Unable to render chart")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(title1)
        st.write(desc1)
        render(col1, fig1)

    with col2:
        st.subheader(title2)
        st.write(desc2)
        render(col2, fig2)


# ========================================================
# 1 & 2 — Status Pie + Working/Not Working By Gov
# ========================================================
# Status distribution (Plotly) — use distinct cameras
if 'health_status' in filtered_unique.columns and len(filtered_unique) > 0:
    fig1 = px.pie(filtered_unique, names='health_status', title='Camera Status Distribution', hole=0.0)
else:
    fig1 = px.pie(names=['No data'], values=[1], title='No health_status data')

# Working vs Not working by governorate (stacked bar)
cam_col = cam_col_global
if 'Kuwait_Governorate' in filtered_unique.columns and 'health_status' in filtered_unique.columns:
    group = filtered_unique.groupby(['Kuwait_Governorate', 'health_status']).size().reset_index(name='count')
    fig2 = px.bar(group, x='Kuwait_Governorate', y='count', color='health_status', title='Working vs Not Working Cameras by Governorate')
else:
    fig2 = px.bar(x=[], y=[], title='No governorate/health_status data')

two_columns_chart(
    "Camera Status Distribution (Pie Chart)",
    "Shows the distribution of camera health statuses across the entire network.",
    fig1,
    "Working vs Not Working Cameras by Governorate",
    "Helps identify which governorates have higher offline or unhealthy cameras.",
    fig2
)


# ========================================================
# 3 & 4 — Reliable Manufacturers + Avg Signal Strength
# ========================================================
if 'brand' in filtered_unique.columns and len(filtered_unique) > 0:
    top_m = filtered_unique['brand'].value_counts().head(10).reset_index()
    top_m.columns = ['brand', 'count']
    fig3 = px.bar(top_m, x='count', y='brand', orientation='h', title='Top 10 Most Reliable Manufacturers')
else:
    fig3 = px.bar(x=[], y=[], title='No brand data')

if 'Kuwait_Governorate' in filtered_unique.columns and 'bandwidth_mbps' in filtered_unique.columns:
    sig = filtered_unique.groupby('Kuwait_Governorate')['bandwidth_mbps'].mean().reset_index()
    fig4 = px.bar(sig, x='Kuwait_Governorate', y='bandwidth_mbps', title='Average Signal Strength by Governorate', labels={'bandwidth_mbps':'Avg Bandwidth (Mbps)'})
else:
    fig4 = px.bar(x=[], y=[], title='No bandwidth data')

two_columns_chart(
    "Top 10 Most Reliable Manufacturers",
    "Counts how many active cameras each brand has. Indicates reliability and deployment preference.",
    fig3,
    "Average Signal Strength by Governorate",
    "Shows signal quality across each governorate to detect weak connectivity zones.",
    fig4
)


# ========================================================
# 5 & 6 — Avg Uptime + Days Since Maintenance Histogram
# ========================================================
if 'camera_type' in filtered_unique.columns and 'uptime_percent' in filtered_unique.columns:
    up = filtered_unique.groupby('camera_type')['uptime_percent'].mean().reset_index()
    fig5 = px.bar(up, x='camera_type', y='uptime_percent', title='Average Uptime by Camera Type', labels={'uptime_percent':'Uptime %'})
else:
    fig5 = px.bar(x=[], y=[], title='No uptime data')

if 'days_since_last_failure' in filtered_unique.columns:
    fig6 = px.histogram(filtered_unique, x='days_since_last_failure', nbins=20, title='Days Since Last Maintenance (Histogram)')
else:
    fig6 = px.histogram(x=[], title='No maintenance-days data')

two_columns_chart(
    "Average Uptime by Camera Type",
    "Shows which camera types deliver best operational stability.",
    fig5,
    "Days Since Last Maintenance (Histogram)",
    "Distribution of time since last maintenance to identify overdue maintenance.",
    fig6
)


# ========================================================
# 7 & 8 — Maintenance Interval + AI Recommendations Summary (textual)
# ========================================================
if 'days_since_last_failure' in filtered_unique.columns and 'brand' in filtered_unique.columns:
    filtered_unique['maint_interval'] = filtered_unique['days_since_last_failure']
    maint = filtered_unique.groupby('brand')['maint_interval'].mean().reset_index()
    fig7 = px.bar(maint, x='brand', y='maint_interval', title='Avg Maintenance Interval by Manufacturer', labels={'maint_interval':'Average Interval (Days)'})
else:
    fig7 = px.bar(x=[], y=[], title='No maintenance interval data')


def generate_ai_recommendations_from_notes(cam_df: pd.DataFrame, cam_id_col: str = None, max_items: int = 8, include_events: bool = False, event_df: pd.DataFrame = None) -> str:
    """Generate plain-text AI recommendations prioritizing technical_notes.

    - If `technical_notes` rows exist in cam_df, return per-camera actionable items (Arabic + English).
    - If not present and include_events=True, fall back to event-based checks (sudden_stop/wrong_direction) using event_df.
    - If neither available, return a helpful instruction message.
    """
    parts = []

    # Helper keyword -> (english, arabic) map for technical notes
    keyword_map = {
        'lens': ('Clean lens / inspect lens alignment', 'تنظيف العدسة / فحص محاذاة العدسة'),
        'dirty': ('Clean camera housing / glass', 'تنظيف غطاء الكاميرا/الزجاج'),
        'power': ('Check power supply / cabling', 'التحقق من مزود الطاقة / الأسلاك'),
        'cable': ('Check cable connections or replace cable', 'التحقق من توصيلات الكابل أو استبدال الكابل'),
        'firmware': ('Schedule firmware update / reboot', 'جدولة تحديث البرنامج الثابت / إعادة التشغيل'),
        'network': ('Investigate connectivity / switch port', 'التحقق من الاتصال / منفذ التبديل'),
        'mount': ('Check mount / tighten screws', 'التحقق من حامل الكاميرا / شد المسامير'),
        'obstruction': ('Clear obstructions / reposition camera', 'إزالة العوائق / إعادة توجيه الكاميرا')
    }

    if 'technical_notes' in cam_df.columns and cam_df['technical_notes'].dropna().astype(str).str.strip().any():
        notes_df = cam_df[cam_df['technical_notes'].astype(str).str.strip() != '']
        # Build a compact HTML table with small font for recommendations
        rows = []
        for _, row in notes_df.head(max_items).iterrows():
            cid = row.get(cam_id_col, '<no-id>') if cam_id_col else '<no-id>'
            loc = row.get('Location', 'Unknown')
            note = str(row.get('technical_notes', '')).strip()
            # generate recommendation text via keyword matching
            matched_recs = []
            matched_recs_ar = []
            for kw, (en, ar) in keyword_map.items():
                if kw in note.lower():
                    matched_recs.append(en)
                    matched_recs_ar.append(ar)
            if not matched_recs:
                matched_recs = ['Inspect and log technician findings for this camera.']
                matched_recs_ar = ['تفقد الكاميرا وتسجيل ملاحظات الفني.']
            rows.append({
                'camera_id': cid,
                'location': loc,
                'note': note,
                'rec_en': ' | '.join(matched_recs),
                'rec_ar': ' | '.join(matched_recs_ar)
            })

        # render HTML table (small font) — we return HTML string for the helper to render
        html = ['<div style="font-family: Cairo, Segoe UI, Arial, sans-serif; font-size:13px;">']
        html.append('<table style="border-collapse:collapse; width:100%; font-size:13px;">')
        html.append('<thead><tr style="background:#f3f4f6;"><th style="padding:6px;border:1px solid #ddd;text-align:left">Camera ID</th><th style="padding:6px;border:1px solid #ddd;text-align:left">Location</th><th style="padding:6px;border:1px solid #ddd;text-align:left">Note</th><th style="padding:6px;border:1px solid #ddd;text-align:left">Recommendation (EN)</th><th style="padding:6px;border:1px solid #ddd;text-align:left">Recommendation (AR)</th></tr></thead>')
        html.append('<tbody>')
        for r in rows:
            html.append(f"<tr><td style=\"padding:6px;border:1px solid #eee;\">{r['camera_id']}</td><td style=\"padding:6px;border:1px solid #eee;\">{r['location']}</td><td style=\"padding:6px;border:1px solid #eee;\">{r['note']}</td><td style=\"padding:6px;border:1px solid #eee;\">{r['rec_en']}</td><td style=\"padding:6px;border:1px solid #eee;\">{r['rec_ar']}</td></tr>")
        html.append('</tbody></table></div>')
        parts.append('\n'.join(html))

        # If user doesn't want event-based recs, stop here
        if not include_events:
            return '\n'.join(parts)

    # If technical_notes absent or include_events True, optionally include behavior-based recs
    if include_events and event_df is not None:
        # check for sudden_stop / wrong_direction columns safely
        s_series = event_df['sudden_stop'] if 'sudden_stop' in event_df.columns else pd.Series(0, index=event_df.index)
        w_series = event_df['wrong_direction'] if 'wrong_direction' in event_df.columns else pd.Series(0, index=event_df.index)
        evs = event_df[((s_series > 0) | (w_series > 0))].copy()
        if evs.empty:
            parts.append('No behavior-based events found (sudden_stop/wrong_direction).')
            return '\n'.join(parts)

        # Prefer per-camera aggregates if cam_df contains totals
        if cam_id_col and cam_id_col in cam_df.columns and ('total_sudden_stop' in cam_df.columns or 'total_wrong_direction' in cam_df.columns):
            cam_df = cam_df.copy()
            cam_df['combined_events'] = cam_df.get('total_sudden_stop', 0) + cam_df.get('total_wrong_direction', 0)
            flagged = cam_df[cam_df['combined_events'] > 0].sort_values('combined_events', ascending=False)
            if not flagged.empty:
                parts.append('AI Recommendations — cameras with violation events (sudden stop / wrong direction):')
                for _, r in flagged.head(max_items).iterrows():
                    cid = r.get(cam_id_col, '<no-id>')
                    loc = r.get('Location', 'Unknown')
                    s_cnt = int(r.get('total_sudden_stop', 0)) if 'total_sudden_stop' in r else 0
                    w_cnt = int(r.get('total_wrong_direction', 0)) if 'total_wrong_direction' in r else 0
                    parts.append(f"• Camera {cid} — {loc} — sudden_stop: {s_cnt}, wrong_direction: {w_cnt}")
                    parts.append('  Recommendation (AR): في التوقيتات أعلاه، يُنصح بإرسال سيارة دورية لمتابعة المركبات المخالفة.')
                    parts.append('  Recommendation (EN): Dispatch a patrol to monitor and follow up on violating vehicles.')
                    parts.append('---')
                # format flagged as a table similar to notes table
                rows = []
                for _, r in flagged.head(max_items).iterrows():
                    cid = r.get(cam_id_col, '<no-id>')
                    loc = r.get('Location', 'Unknown')
                    s_cnt = int(r.get('total_sudden_stop', 0)) if 'total_sudden_stop' in r else 0
                    w_cnt = int(r.get('total_wrong_direction', 0)) if 'total_wrong_direction' in r else 0
                    rows.append({'camera_id': cid, 'location': loc, 'sudden': s_cnt, 'wrong': w_cnt})
                html = ['<div style="font-family: Cairo, Segoe UI, Arial, sans-serif; font-size:13px;">']
                html.append('<table style="border-collapse:collapse; width:100%; font-size:13px;">')
                html.append('<thead><tr style="background:#f3f4f6;"><th style="padding:6px;border:1px solid #ddd;text-align:left">Camera ID</th><th style="padding:6px;border:1px solid #ddd;text-align:left">Location</th><th style="padding:6px;border:1px solid #ddd;text-align:left">Sudden Stop</th><th style="padding:6px;border:1px solid #ddd;text-align:left">Wrong Direction</th><th style="padding:6px;border:1px solid #ddd;text-align:left">Recommendation</th></tr></thead>')
                html.append('<tbody>')
                for r in rows:
                    html.append(f"<tr><td style=\"padding:6px;border:1px solid #eee;\">{r['camera_id']}</td><td style=\"padding:6px;border:1px solid #eee;\">{r['location']}</td><td style=\"padding:6px;border:1px solid #eee;\">{r['sudden']}</td><td style=\"padding:6px;border:1px solid #eee;\">{r['wrong']}</td><td style=\"padding:6px;border:1px solid #eee;\">Dispatch a patrol / إرسال دورية</td></tr>")
                html.append('</tbody></table></div>')
                parts.append('\n'.join(html))
                return '\n'.join(parts)

        # fallback to listing example events
        parts.append('AI Recommendations — Violating event examples:')
        for _, r in evs.head(max_items).iterrows():
            tscol = find_column(event_df, 'timestamp', 'time', 'event_time', 'date')
            ts = r.get(tscol, '<no-timestamp>') if tscol else '<no-timestamp>'
            loc = r.get('Location', 'Unknown')
            s_cnt = int(r.get('sudden_stop', 0)) if 'sudden_stop' in r else 0
            w_cnt = int(r.get('wrong_direction', 0)) if 'wrong_direction' in r else 0
            parts.append(f"• {ts} — {loc} — sudden_stop: {s_cnt}, wrong_direction: {w_cnt}")
            parts.append('  Recommendation (AR): إرسال دورية لمتابعة الحادث.')
            parts.append('  Recommendation (EN): Send a patrol to investigate the incident.')

        return '\n'.join(parts)

    # final fallback
    if parts:
        return '\n'.join(parts)
    return 'No technical notes or relevant event columns found to generate recommendations.'


# UI toggle: if user wants to include behavior-based recommendations (sudden_stop/wrong_direction)
include_event_recs = st.checkbox('Also include behavior-based recommendations (sudden_stop / wrong_direction)', value=False)

recommendations_text = generate_ai_recommendations_from_notes(merged_unique, cam_col_global, max_items=8, include_events=include_event_recs, event_df=filtered_df)
fig8 = recommendations_text

two_columns_chart(
    "Avg Maintenance Interval by Manufacturer",
    "Shows how frequently each brand requires maintenance based on failure history.",
    fig7,
    "AI Recommendation Summary",
    "Textual recommendations derived from technical notes (toggle in the panel to include behavior events).",
    fig8
)

# ========================================================
# 9 & 10 — High Risk by Technician + Temperature Heatmap
# ========================================================
if 'technical_notes' in filtered_unique.columns and len(filtered_unique[filtered_unique['health_status'] != 'Healthy']) > 0:
    if cam_col:
        risk_series = filtered_unique[filtered_unique['health_status'] != 'Healthy'].groupby('technical_notes')[cam_col].nunique()
    else:
        risk_series = filtered_unique[filtered_unique['health_status'] != 'Healthy'].groupby('technical_notes').size()
    risk = risk_series.reset_index()
    risk.columns = ['technical_notes', 'count']
    fig9 = px.bar(risk, x='technical_notes', y='count', title='High-Risk Camera Count by Technician')
else:
    fig9 = px.bar(x=[], y=[], title='No high-risk data')

if 'ambient_temp_c' in filtered_unique.columns and 'Kuwait_Governorate' in filtered_unique.columns:
    pivot_temp = filtered_unique.pivot_table(values='ambient_temp_c', index='Kuwait_Governorate', aggfunc='mean').reset_index()
    fig10 = px.bar(pivot_temp, x='Kuwait_Governorate', y='ambient_temp_c', title='Avg Temperature by Governorate', labels={'ambient_temp_c':'Avg Temp (°C)'})
else:
    fig10 = px.bar(x=[], y=[], title='No temperature data')

two_columns_chart(
    "High-Risk Cameras by Technicians",
    "Shows which technicians are assigned more high-risk cameras.",
    fig9,
    "Average Temperature by Governorate (Heatmap)",
    "Highlights environmental heat exposure possibly affecting camera health.",
    fig10
)


# ========================================================
# 11 & 12 — Humidity vs Signal + Power vs Uptime Bubble
# ========================================================
if 'humidity_percent' in filtered_unique.columns and 'bandwidth_mbps' in filtered_unique.columns:
    fig11 = px.scatter(filtered_unique, x='humidity_percent', y='bandwidth_mbps', title='Humidity vs Signal Strength', labels={'humidity_percent':'Humidity %', 'bandwidth_mbps':'Signal Strength (Mbps)'})
else:
    fig11 = px.scatter(x=[], y=[], title='No humidity/signal data')

if 'uptime_percent' in filtered_unique.columns:
    size_col = filtered_unique['power_consumption'] if 'power_consumption' in filtered_unique.columns else filtered_unique.get('bandwidth_mbps', pd.Series([1]*len(filtered_unique)))
    fig12 = px.scatter(filtered_unique, x=size_col, y='uptime_percent', size=filtered_unique['uptime_percent'] * 2, title='Power Usage vs Uptime (Bubble Chart)', labels={'x':'Power Usage (Simulated)', 'uptime_percent':'Uptime %'})
else:
    fig12 = px.scatter(x=[], y=[], title='No uptime data')

two_columns_chart(
    "Humidity vs Signal Strength (Scatter)",
    "Shows the environmental impact of humidity on signal quality.",
    fig11,
    "Power Usage vs Uptime (Bubble Chart)",
    "Bubble size indicates uptime. Helps evaluate energy efficiency.",
    fig12
)


# ========================================================
# 13 & 14 — Sudden Stop / Wrong Direction + Violations by Gov
# ========================================================
if len(filtered_unique) > 0:
    tmp = filtered_unique.copy()
    tmp['sudden_stop'] = np.random.randint(0,5,len(tmp))
    tmp['wrong_direction'] = np.random.randint(0,5,len(tmp))
    fig13 = px.line(tmp.reset_index(), y=['sudden_stop','wrong_direction'], title='Sudden Stop & Wrong Direction (Line Chart)')
else:
    fig13 = px.line(title='No data')

if 'estimated_daily_vehicles' in filtered_unique.columns and 'Kuwait_Governorate' in filtered_unique.columns:
    viol_df = filtered_unique.groupby('Kuwait_Governorate')['estimated_daily_vehicles'].sum().reset_index()
    fig14 = px.bar(viol_df, x='Kuwait_Governorate', y='estimated_daily_vehicles', title='Violations Detected by Governorate', labels={'estimated_daily_vehicles':'Violations Count (Simulated)'})
else:
    fig14 = px.bar(x=[], y=[], title='No violations data')

two_columns_chart(
    "Sudden Stop & Wrong Direction (Line Chart)",
    "Behavior-based detection trends over time.",
    fig13,
    "Violations Detected by Governorate",
    "Shows areas with highest traffic violations or incidents.",
    fig14
)


# ========================================================
# 15 & 16 — Vehicles vs Risk + Motion Alerts (Area chart)
# ========================================================
if 'estimated_daily_vehicles' in filtered_unique.columns and 'health_status' in filtered_unique.columns:
    risk_map = filtered_unique['health_status'].apply(lambda x: 3 if x != 'Healthy' else 1)
    fig15 = px.scatter(filtered_unique, x='estimated_daily_vehicles', y=risk_map, title='Vehicle Count vs Risk Level', labels={'x':'Vehicle Count', 'y':'Risk Level'})
else:
    fig15 = px.scatter(x=[], y=[], title='No vehicle/risk data')

if len(filtered_unique) > 0:
    alerts = np.random.randint(10,200,len(filtered_unique))
    fig16 = px.area(y=alerts, title='Motion Alerts Per Month (Area Chart)')
else:
    fig16 = px.area(title='No alerts data')

two_columns_chart(
    "Vehicle Count vs Risk Level (Scatter)",
    "Correlation between traffic load and camera risk/health status.",
    fig15,
    "Motion Alerts Per Month (Area Chart)",
    "Shows how alert volume changes monthly.",
    fig16
)


# ========================================================
# 17 & 18 — Maintenance Cluster Map + Predict Next Failure
# ========================================================
st.subheader("Maintenance Cluster Map")
st.write("Clusters cameras by location to identify maintenance hotspots.")
# dedupe for the maintenance map as well
if {'Latitude','Longitude'}.issubset(map_df_unique.columns) and len(map_df_unique) > 0:
    st.map(map_df_unique.rename(columns={'Latitude':'lat','Longitude':'lon'})[['lat','lon']])
else:
    st.info('No geographic points to display on maintenance cluster map')

st.subheader("AI Predict Next Failure")
st.write("Predicts next failure using a simple uptime + temperature + humidity model.")
if {'uptime_percent','ambient_temp_c','humidity_percent'}.issubset(filtered_unique.columns):
    filtered_unique['next_failure_prediction'] = (
        (100 - filtered_unique['uptime_percent']) +
        (filtered_unique['ambient_temp_c'] / 10) +
        (filtered_unique['humidity_percent'] / 20)
    )
    st.line_chart(filtered_unique['next_failure_prediction'])
else:
    st.info('Not enough columns to predict next failure')


# ========================================================
# 19 & 20 — Risk Level AI + Correlation Matrix
# ========================================================
st.subheader("AI Risk Level Recommendation")
st.write("AI assigns Risk Level based on health, temperature, humidity, and uptime.")
if {'uptime_percent','ambient_temp_c','humidity_percent'}.issubset(merged_unique.columns):
    # Compute AI risk at camera-level (merged_unique) so we respect filters and de-duplication
    merged_unique['AI_risk_score'] = (
        (100 - merged_unique['uptime_percent']) +
        merged_unique['ambient_temp_c'] +
        merged_unique['humidity_percent']
    )
    fig_risk = px.bar(merged_unique.reset_index(drop=True).reset_index(), x='index', y='AI_risk_score', title='AI Risk Score per Camera')
    st.plotly_chart(fig_risk, width='stretch')
else:
    st.info('Insufficient columns for AI risk score')

st.subheader("Correlation Matrix Feature Selection")
num = merged_unique.select_dtypes(include=['float64','int64'])
if not num.empty:
    corr = num.corr()
    fig_corr = px.imshow(corr, color_continuous_scale='RdBu', title='Correlation Matrix')
    st.plotly_chart(fig_corr, width='stretch')
else:
    st.info('No numeric columns to compute correlation matrix')


#--------------------------- TABLE VIEW ---------------------------
with st.expander("View Filtered Data Table"):
    st.write(filtered_df)
    st.download_button("Download Filtered CSV",
                       filtered_df.to_csv(index=False).encode("utf-8"),
                       "SurveillanceCameras_latest-update2.csv")
