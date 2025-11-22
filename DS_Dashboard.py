import streamlit as st # type: ignore
import plotly.express as px # type: ignore
import pandas as pd # type: ignore
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

st.set_page_config(page_title="CCTV Monitoring Dashboard", layout="wide", page_icon="📹")
st.title("📹 CCTV AI Monitoring Dashboard")
#st.markdown("<style>div.block-container{padding-top:2rem;}</style>", unsafe_allow_html=True)
st.markdown("""
    <style>
    * { font-family: 'Segoe UI', Arial, sans-serif; }
    </style>
""", unsafe_allow_html=True)

# ---------------------- FILE UPLOAD ----------------------
fl = st.file_uploader("Upload CCTV CSV file", type=["csv"])

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


# ---------------------- SIDEBAR FILTERS ----------------------
st.sidebar.header("Filter the Data")

location_filter = st.sidebar.multiselect("Location", df["Location"].unique())
brand_filter = st.sidebar.multiselect("Brand", df["brand"].unique())
health_filter = st.sidebar.multiselect("Health Status", df["health_status"].unique())
connect_filter = st.sidebar.multiselect("Connectivity", df["connectivity_status"].unique())


filtered_df = df.copy()

if location_filter:
    filtered_df = filtered_df[filtered_df["Location"].isin(location_filter)]
if brand_filter:
    filtered_df = filtered_df[filtered_df["brand"].isin(brand_filter)]
if health_filter:
    filtered_df = filtered_df[filtered_df["health_status"].isin(health_filter)]
if connect_filter:
    filtered_df = filtered_df[filtered_df["connectivity_status"].isin(connect_filter)]


# ---------------------- TOP METRICS ----------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Cameras", len(filtered_df))
col2.metric("Online Cameras", (filtered_df["connectivity_status"] == "Online").sum())
col3.metric("Avg Uptime %", round(filtered_df["uptime_percent"].mean(), 2))
col4.metric("Avg Coverage %", round(filtered_df["coverage_percent"].mean(), 2))


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

# Diagnostic: show coordinate ranges
min_lat, max_lat = df["Latitude"].min(), df["Latitude"].max()
min_lon, max_lon = df["Longitude"].min(), df["Longitude"].max()
st.write(f"Latitude range: {min_lat} to {max_lat} — Longitude range: {min_lon} to {max_lon}")

# Permanently swap Latitude/Longitude (user requested to remove the swap checkbox)
df[["Latitude", "Longitude"]] = df[["Longitude", "Latitude"]]
st.info("Latitude and Longitude columns swapped for mapping (permanent).")

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

# recompute ranges/center using filtered data
map_center = [filtered_df["Latitude"].mean(), filtered_df["Longitude"].mean()]
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


# Build map HTML once (cached) and embed via components.html for faster client-side interaction
st.subheader("📍 Camera Locations Map (Free Map)")
if len(filtered_df) == 0:
    st.info("No cameras to display on the map for the selected filters.")
else:
    df_json = filtered_df.to_json(orient="records")
    map_html = build_map_html(df_json, use_custom_icons, map_center, zoom_start=12)
    # embed the pre-rendered HTML (interactive Leaflet) — much faster than rebuilding on every rerun
    components.html(map_html, height=700)

# ---------------------- END MAPS ----------------------


# ---------------------- VISUALIZATIONS ----------------------

st.subheader("Cameras by Location")
fig1 = px.bar(filtered_df, x="Location", title="Cameras by Location", color="Location")
st.plotly_chart(fig1, use_container_width=True)


st.subheader("Cameras by Health Status")
fig2 = px.pie(filtered_df, names="health_status", title="Health Distribution", hole=0.5)
st.plotly_chart(fig2, use_container_width=True)


st.subheader("Connectivity Status")
fig3 = px.bar(filtered_df, x="connectivity_status", color="connectivity_status",
              title="Online vs Offline Cameras")
st.plotly_chart(fig3, use_container_width=True)


st.subheader("Camera Brands Distribution")
fig4 = px.histogram(filtered_df, x="brand", color="brand", title="Brand Count")
st.plotly_chart(fig4, use_container_width=True)


st.subheader("Resolution Distribution")
fig5 = px.box(filtered_df, x="brand", y="resolution_mp", color="brand",
              title="Resolution per Brand")
st.plotly_chart(fig5, use_container_width=True)


st.subheader("Motion Events Per Day")
fig6 = px.scatter(filtered_df, x="motion_events_per_day", y="uptime_percent",
                  size="resolution_mp", color="brand",
                  title="Motion Events vs Uptime")
st.plotly_chart(fig6, use_container_width=True)


st.subheader("Temperature vs Humidity")
fig7 = px.scatter(filtered_df, x="ambient_temp_c", y="humidity_percent",
                  color="health_status", size="coverage_percent",
                  title="Environmental Conditions")
st.plotly_chart(fig7, use_container_width=True)

# ============================
#   COVERAGE & MOTION ANALYTICS
# ============================


st.header("📡 Coverage & Motion Activity Analytics")
st.markdown("تحليل أداء الكاميرات بناءً على نسبة التغطية وعدد الأحداث اليومية (Motion Events).")


# =======================================================
# 1) Coverage Distribution Histogram
# =======================================================
st.subheader("📊 Coverage % Distribution")
st.caption("هذا المخطط يوضح توزيع نسبة التغطية لجميع الكاميرات لمعرفة مستوى التغطية العام.")

fig_cov_hist = px.histogram(
    filtered_df,
    x="coverage_percent",
    nbins=40,
    title="Coverage Percentage Distribution",
    template="plotly_white"
)
st.plotly_chart(fig_cov_hist, use_container_width=True)


# =======================================================
# 2) Coverage Box Plot (Detect Low-Coverage Cameras)
# =======================================================
st.subheader("📉 Coverage Outliers (Box Plot)")
st.caption("هذا الرسم يوضح القيم الشاذة للكاميرات ذات التغطية المنخفضة للمساعدة في اكتشاف المشاكل بسرعة.")

fig_cov_box = px.box(
    filtered_df,
    y="coverage_percent",
    points="all",
    title="Coverage Outliers",
    template="seaborn"
)
st.plotly_chart(fig_cov_box, use_container_width=True)


# =======================================================
# 3) Scatter: Coverage vs Motion Events
# =======================================================
st.subheader("📈 Relationship: Coverage vs Motion Events")
st.caption("هنا نرى العلاقة بين التغطية وعدد الأحداث المسجلة — مفيد لمعرفة الكاميرات النشطة والمواقع ذات الحركة العالية.")

fig_cov_motion = px.scatter(
    filtered_df,
    x="coverage_percent",
    y="motion_events_per_day",
    color="brand",
    size="resolution_mp",
    hover_name="Location",
    title="Coverage vs Motion Events Per Day",
    template="plotly_white"
)
st.plotly_chart(fig_cov_motion, use_container_width=True)


# =======================================================
# 4) Motion Events Distribution
# =======================================================
st.subheader("🎯 Motion Events Distribution")
st.caption("يوضح توزيع عدد الأحداث اليومية لكل الكاميرات لمعرفة الكاميرات ذات النشاط العالي أو المنخفض.")

fig_motion_hist = px.histogram(
    filtered_df,
    x="motion_events_per_day",
    nbins=50,
    title="Motion Events Per Day Distribution",
    template="simple_white"
)
st.plotly_chart(fig_motion_hist, use_container_width=True)


# =======================================================
# 5) Line Trend: Motion vs Coverage
# =======================================================
st.subheader("📉 Motion Trend by Coverage Level")
st.caption("منحنى يوضح تأثير التغطية على عدد الأحداث التي تسجلها الكاميرا.")

fig_motion_line = px.line(
    filtered_df.sort_values("coverage_percent"),
    x="coverage_percent",
    y="motion_events_per_day",
    color="brand",
    title="Motion Events Trend by Coverage",
    markers=True,
    template="plotly_white"
)
st.plotly_chart(fig_motion_line, use_container_width=True)


# =======================================================
# 6) Heatmap – Motion × Coverage × Health Status
# =======================================================
st.subheader("🔥 Heatmap: Coverage vs Motion by Health Status")
st.caption("مخطط حراري يوضح العلاقة بين التغطية والنشاط مع مقارنة الحالة الصحية لكل كاميرا.")

fig_heat = px.density_heatmap(
    filtered_df,
    x="coverage_percent",
    y="motion_events_per_day",
    facet_col="health_status",
    color_continuous_scale="Viridis",
    title="Heatmap – Coverage vs Motion by Health Status"
)
st.plotly_chart(fig_heat, use_container_width=True)

# ================= END =================


#--------------------------- TABLE VIEW ---------------------------
with st.expander("View Filtered Data Table"):
    st.write(filtered_df)
    st.download_button("Download Filtered CSV",
                       filtered_df.to_csv(index=False).encode("utf-8"),
                       "SurveillanceCameras_updated_1.csv")