import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import pgeocode
import numpy as np
import os
import re
from urllib.parse import quote
from branca.element import Element

# --- PDF GENERATION LIBRARY ---
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# Updated page title to match your new branding
st.set_page_config(page_title="Food Surplus Scout", layout="wide")

# ------------------------------------------------------------------
# TITLE & INTRODUCTORY TEXT
# ------------------------------------------------------------------
st.title("SURPLUS FOOD SCOUT (Beta v1.0)")

# CONSTRAINT: [3, 1] ratio keeps text readable (75% width) 
c_content, c_spacer = st.columns([3, 1])

with c_content:
    st.markdown("""
    ### A tool for food rescuers to scout surpluses and make hit lists.
    
    This app uses the [EPA Excess Food Opportunities Data (v3.1)](https://www.epa.gov/sustainable-management-food/excess-food-opportunities-map) to surface potential sources of food surplus near you. The data is far from perfect, but hopefully helps guide you toward leads you wouldn’t have found or thought of otherwise.      
    #### ⚠️ Keep in Mind:
    * **Estimates ≠ Reality:** The waste estimates below are a ["best guess"](https://www.epa.gov/system/files/documents/2025-02/efom_v3_1_technical_methodology.pdf) based on other, potentially correlated data–not actual measurements. In some cases, they are obviously, very, comically wrong. Use your judgement!
    * **Ignores Existing Rescue:** The model does not know if a generator already has a rescue partner.
    * **Data Quirks:** Data may be outdated, inaccurate, or just plain weird. Many entries have '0' estimated waste, which is likely wrong, and due to missing data elsewhere. Worth scoping these! 
    * **Beta Limitations:** Results are currently limited to a 50 mile radius of Dayton, Ohio. 
    """)

st.markdown("---")

# --- CONFIG ---
TARGET_FILENAME = 'dayton_combined_50mi_radius.csv'

# --- HELPERS ---
def make_label_readable(text):
    text = text.replace(".xlsx", "").replace(".csv", "")
    text = re.sub(r"(\w)([A-Z])", r"\1 \2", text)
    return text

def clean_zip(value):
    """Ensures Zip is a clean string without decimals"""
    try:
        if pd.isna(value):
            return ""
        return str(int(float(value)))
    except:
        return str(value)

def generate_search_url(row):
    """Creates a Google Search URL: Name + Address"""
    # Safe access to fields in case City/State are missing
    addr = str(row.get('Address', ''))
    city = str(row.get('City', ''))
    state = str(row.get('State', ''))
    
    # Construct a robust search query
    query = f"{row['Name']} {addr} {city} {state}"
    return f"https://www.google.com/search?q={quote(query)}"

def create_pdf(df):
    """Generates a PDF report matching the table view"""
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Food Rescue Hit List Report", ln=True, align='C')
    pdf.ln(10)
    
    # Disclaimer in PDF
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, "Note: Waste estimates are derived from EPA models and may include inedible materials. Values are estimates, not actual measurements.")
    pdf.ln(5)

    # Table Header
    pdf.set_font("Arial", 'B', 9) 
    pdf.set_fill_color(200, 220, 255)
    
    # Define columns and widths
    headers = ['Name', 'Sector', 'Waste (T)', 'Address', 'City', 'St', 'Zip', 'Mi']
    widths =  [50,     45,       25,          60,        30,     15,   25,    20] 
    
    for col, w in zip(headers, widths):
        pdf.cell(w, 10, col, border=1, fill=True)
    pdf.ln()
    
    # Table Rows
    pdf.set_font("Arial", size=8)
    for _, row in df.iterrows():
        name = str(row.get('Name', ''))[:30]
        sector = str(row.get('Sector', ''))[:25]
        waste = f"{row.get('Est. Waste (Tons/Yr)', 0):,.1f}"
        addr = str(row.get('Address', ''))[:35]
        city = str(row.get('City', ''))[:15]
        state = str(row.get('State', ''))[:5]
        zip_code = str(row.get('Zip', ''))
        dist = str(row.get('Distance_Miles', ''))
        
        data = [name, sector, waste, addr, city, state, zip_code, dist]
        
        for item, w in zip(data, widths):
            pdf.cell(w, 10, str(item), border=1)
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1', 'ignore')

# --- 1. Loader ---
@st.cache_data
def load_data(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(script_dir, 'data', filename),
        os.path.join('data', filename),
        filename
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, dtype={'Zip': object})
            except:
                continue
    return None

# --- 2. Math Engine ---
@st.cache_data
def get_nearby_data(df, center_lat, center_lon, radius_miles, min_waste):
    deg_buffer = radius_miles / 60 
    df_fast = df[
        (df['Latitude'].between(center_lat - deg_buffer, center_lat + deg_buffer)) &
        (df['Longitude'].between(center_lon - deg_buffer, center_lon + deg_buffer))
    ].copy()
    
    if df_fast.empty: return df_fast

    R = 3958.8 
    lat1, lon1 = np.radians(center_lat), np.radians(center_lon)
    lat2, lon2 = np.radians(df_fast['Latitude']), np.radians(df_fast['Longitude'])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    df_fast['Distance_Miles'] = R * c
    
    return df_fast[
        (df_fast['Distance_Miles'] <= radius_miles) & 
        (df_fast['Waste_Tons'] >= min_waste)
    ]

# --- 3. Main Execution ---
df = load_data(TARGET_FILENAME)

if df is None:
    st.error(f"❌ Cleaned data file `{TARGET_FILENAME}` not found. Run `clean_data.py` first!")
    st.stop()

# ------------------------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------------------------
st.sidebar.header("📍 Scout Settings")

# 1. Zip Code Input
zip_code = st.sidebar.text_input("Enter Zip Code", value="45402", max_chars=5)

# 2. IMMEDIATE GEOCODING
nomi = pgeocode.Nominatim('us')
location = nomi.query_postal_code(zip_code)

# 3. FEEDBACK
if pd.isna(location.latitude):
    st.sidebar.error("Invalid Zip Code")
    st.stop()
else:
    hq_lat, hq_lon = location.latitude, location.longitude
    st.sidebar.success(f"📍 {location.place_name}, {location.state_name}")

# 4. Sliders
search_radius = st.sidebar.slider("Search Radius (Miles)", 1, 50, 10)
min_waste = st.sidebar.slider("Min. Waste (Tons/Year)", 0, 100, 0)

st.sidebar.header("📂 Filter Categories")

selected_sources = []
if 'Source_File' in df.columns:
    available_sources = sorted(df['Source_File'].dropna().unique())
    st.sidebar.caption("Select sectors to include:")
    for source in available_sources:
        readable_label = make_label_readable(str(source))
        if st.sidebar.checkbox(readable_label, value=True, key=source):
            selected_sources.append(source)
    
    if not selected_sources:
        st.warning("Please select at least one category.")
        st.stop()
        
    filtered_df = df[df['Source_File'].isin(selected_sources)]
else:
    filtered_df = df
# 5. Footer
st.sidebar.markdown("---")
st.sidebar.info("👋 I'd love your ideas for making this more useful. 📫 Get in touch:  \n"
                "[ericmrothman@gmail.com](mailto:ericmrothman@gmail.com)")

# Math Engine
final_df = get_nearby_data(filtered_df, hq_lat, hq_lon, search_radius, min_waste)

# --- MAP CONSTRUCTION ---
m = folium.Map(location=[hq_lat, hq_lon], zoom_start=12, tiles="CartoDB positron")

map_custom_css = """
<style>
.leaflet-popup-content-wrapper {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
    padding: 0 !important;
}
.leaflet-popup-tip-container {
    display: none !important;
}
.leaflet-popup-content {
    margin: 0 !important;
    width: auto !important;
}
.leaflet-container a.leaflet-popup-close-button {
    top: 5px !important;
    right: 5px !important;
    color: #444 !important;
    text-shadow: 0 0 2px white;
}
.leaflet-tooltip {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.leaflet-tooltip-left:before, .leaflet-tooltip-right:before, 
.leaflet-tooltip-bottom:before, .leaflet-tooltip-top:before {
    display: none !important;
}
</style>
"""
m.get_root().html.add_child(Element(map_custom_css))

# --- CHECK DATA ---
if final_df.empty:
    st.warning("No targets found matching criteria.")
else:
    # HARDCODED SORT
    final_df = final_df.sort_values("Waste_Tons", ascending=False)

    # Add Markers
    for _, row in final_df.head(200).iterrows():
        search_url = generate_search_url(row)
        
        tooltip_html = f"""
        <div style="
            font-family: sans-serif; 
            background-color: white;
            padding: 6px 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.3);
            white-space: nowrap;
        ">
            <strong style="color: #2E7D32;">{row['Name']}</strong><br>
            <span style="font-size: 11px;">🥕 {row['Waste_Tons']:,.0f} Tons</span>
        </div>
        """

        popup_html = f"""
        <div style="
            font-family: sans-serif; 
            width: 240px;
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 12px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        ">
            <h4 style="margin: 0 0 8px 0; color: #2E7D32; font-size: 16px; padding-right: 10px;">{row['Name']}</h4>
            
            <div style="margin-bottom: 8px;">
                <span style="background-color: #E8F5E9; color: #2E7D32; padding: 3px 8px; border-radius: 4px; font-size: 10px; font-weight: bold; border: 1px solid #C8E6C9;">
                    {row['Sector']}
                </span>
            </div>
            
            <div style="font-size: 13px; color: #333; line-height: 1.5;">
                <b>🥕 Waste:</b> {row['Waste_Tons']:,.1f} Tons/Yr<br>
                <b>📏 Dist:</b> {row['Distance_Miles']:.1f} miles
            </div>
            
            <div style="margin-top: 10px;">
                <a href="{search_url}" target="_blank" style="
                    display: inline-block;
                    text-decoration: none;
                    color: #1565C0;
                    font-size: 12px;
                    font-weight: bold;
                    border: 1px solid #90CAF9;
                    background-color: #E3F2FD;
                    padding: 4px 8px;
                    border-radius: 4px;
                ">
                    🔎 Google This Source
                </a>
            </div>
            
            <hr style="margin: 10px 0; border: 0; border-top: 1px solid #eee;">
            
            <div style="font-size: 11px; color: #666; font-style: italic;">
                📍 {row['Address']}
            </div>
        </div>
        """

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8, 
            color="#228B22", 
            fill=True, 
            fill_color="#32CD32", 
            fill_opacity=0.7,
            tooltip=folium.Tooltip(tooltip_html, sticky=True),
            popup=folium.Popup(popup_html, max_width=280)
        ).add_to(m)

# ------------------------------------------------------------------
# RENDER MAP
# ------------------------------------------------------------------
st.markdown("### 🗺️ Interactive Map")
st.caption("Hover for summary. Click for details.")

c1, c2 = st.columns([19, 1]) 

with c1:
    st_folium(m, width=None, height=500, returned_objects=[])

st.markdown("---")

# --- DATA TABLE & EXPORTS ---
if not final_df.empty:
    st.markdown("### 📋 Hit List")
    
    # ----------------------------------------------------------------
    # PREPARE DATA
    # ----------------------------------------------------------------
    base_df = final_df.copy()
    
    # 1. Rename & Clean
    base_df = base_df.rename(columns={'Waste_Tons': 'Est. Waste (Tons/Yr)'})
    
    if 'Zip' in base_df.columns:
        base_df['Zip'] = base_df['Zip'].apply(clean_zip)
        
    base_df['Distance_Miles'] = base_df['Distance_Miles'].round(1)
    base_df['Est. Waste (Tons/Yr)'] = base_df['Est. Waste (Tons/Yr)'].round(1)

    # 2. DEFINE COLUMNS
    desired_cols = [
        'Name', 'Sector', 'Est. Waste (Tons/Yr)', 
        'Address', 'City', 'State', 'Zip', 'Distance_Miles'
    ]
    
    # Filter to only columns that actually exist
    final_cols = [c for c in desired_cols if c in base_df.columns]
    
    # 3. DOWNLOAD DF
    download_df = base_df[final_cols].copy()
    
    # 4. DISPLAY DF (With Link as FIRST column)
    display_df = base_df[final_cols].copy()
    display_df['Research Link'] = base_df.apply(generate_search_url, axis=1)

    # REORDER: Put 'Research Link' first
    display_cols_ordered = ['Research Link'] + final_cols
    display_df = display_df[display_cols_ordered]

# ----------------------------------------------------------------
    # EXPORT BUTTONS (Responsive Side-by-Side -> Stack)
    # ----------------------------------------------------------------
    st.markdown("""
        <style>
        /* 1. Force the columns container to allow wrapping */
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 10px !important; /* Adds space when they wrap */
        }
        
        /* 2. Set a minimum width for the columns themselves */
        div[data-testid="column"] {
            min-width: 350px !important; /* Forces wrap if screen is too narrow */
            flex: 1 1 auto !important;   /* Allows them to fill space evenly */
        }

        /* 3. Make buttons fill their respective columns */
        div.stDownloadButton button {
            width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)

    csv_data = download_df.to_csv(index=False).encode('utf-8')
    
    # Use standard columns, but the CSS above will make them 'smart'
    c1, c2, _ = st.columns([1, 1, 2])
    
    with c1:
        st.download_button(
            label="📥 Download CSV",
            data=csv_data, 
            file_name="scout_hit_list.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with c2:
        if HAS_FPDF:
            pdf_bytes = create_pdf(download_df)
            st.download_button(
                label="📄 Download PDF",
                data=pdf_bytes,
                file_name="scout_hit_list.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("Install `fpdf` for PDF export")
    # ----------------------------------------------------------------
    # TABLE DISPLAY
    # ----------------------------------------------------------------
    t1, t2 = st.columns([19, 1])
    
    with t1:
        st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Research Link": st.column_config.LinkColumn(
                    "Research",           
                    help="Click to Google this source",
                    validate="^https://", 
                    display_text="Search Web" 
                ),
                "Distance_Miles": st.column_config.NumberColumn(
                    "Distance (Mi)",
                    format="%.1f"
                ),
                "Est. Waste (Tons/Yr)": st.column_config.NumberColumn(
                    "Est. Waste (Tons/Yr)",
                    format="%.1f"
                )
            }
        )