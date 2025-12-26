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

st.set_page_config(page_title="Food Surplus Scout", layout="wide")

# --- SESSION STATE SETUP ---
if 'shortlist' not in st.session_state:
    st.session_state.shortlist = pd.DataFrame()

# ------------------------------------------------------------------
# TITLE & INTRODUCTORY TEXT
# ------------------------------------------------------------------
st.markdown("""
    <style>
    .mobile-cta { display: none; }
    @media only screen and (max-width: 768px) {
        .mobile-cta {
            display: block;
            background-color: #e8f5e9; color: #1b5e20;
            padding: 12px; border-radius: 5px; 
            margin-bottom: 25px; border-left: 6px solid #2E7D32;
            font-size: 16px;
        }
    }
    </style>
    <div class="mobile-cta">
        <strong>☝️ Scroll down to see the map and filters.</strong>
    </div>
""", unsafe_allow_html=True)

st.title("SURPLUS FOOD SCOUT (Beta v1.1)")

c_content, c_spacer = st.columns([3, 1])
with c_content:
    st.markdown("""
### 🥣 A tool to help food rescue operations discover untapped, nearby surplus
This tool uses [EPA data](https://www.epa.gov/sustainable-management-food/excess-food-opportunities-map) to surface hidden, high-volume generators—like wholesalers, manufacturers, and institutions–guiding you toward potential surplus sources you might not have considered.    """)

# 3. Disclaimer 
st.info("""
**⚠️ Keep in Mind:**
* **Estimates ≠ Reality:** The waste estimates are a ["best guess"](https://www.epa.gov/system/files/documents/2025-02/efom_v3_1_technical_methodology.pdf) based on potentially correlated data–not actual measurements. Estimates also include non-edible, organic waste. 
* **Ignores Existing Rescue:** The model does not know if a generator already has a rescue partner.
* **Data Quirks:** Data may be outdated or inaccurate. Some entries have '0' estimated waste, likely due to missing data. Still worth investigating!
* **Beta Limitations:** Results are currently limited to a 50 mile radius of Dayton, Ohio. 
""")



st.markdown("---")

# --- CONFIG & HELPERS ---
TARGET_FILENAME = 'dayton_combined_50mi_radius.csv'

def make_label_readable(text):
    text = text.replace(".xlsx", "").replace(".csv", "")
    text = re.sub(r"(\w)([A-Z])", r"\1 \2", text)
    return text

def clean_zip(value):
    try:
        if pd.isna(value): return ""
        return str(int(float(value)))
    except:
        return str(value)

def generate_search_url(row):
    addr = str(row.get('Address', ''))
    city = str(row.get('City', ''))
    state = str(row.get('State', ''))
    query = f"{row['Name']} {addr} {city} {state}"
    return f"https://www.google.com/search?q={quote(query)}"

def create_pdf(df):
    pdf = FPDF(orientation='L', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Surplus Food Contact List", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, "Note: Waste estimates are derived from EPA models and may include inedible materials.")
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 9) 
    pdf.set_fill_color(200, 220, 255)
    
    headers = ['Name', 'Sector', 'Waste (T)', 'Address', 'City', 'St', 'Zip', 'Mi']
    widths =  [50,     45,       25,          60,        30,     15,   25,    20] 
    
    for col, w in zip(headers, widths):
        pdf.cell(w, 10, col, border=1, fill=True)
    pdf.ln()
    
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

@st.cache_data
def load_data(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [os.path.join(script_dir, 'data', filename), os.path.join('data', filename), filename]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, dtype={'Zip': object})
            except:
                continue
    return None

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
    a = np.sin((lat2-lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    df_fast['Distance_Miles'] = R * c
    
    return df_fast[
        (df_fast['Distance_Miles'] <= radius_miles) & 
        (df_fast['Waste_Tons'] >= min_waste)
    ]

# --- LOAD DATA ---
df = load_data(TARGET_FILENAME)
if df is None:
    st.error(f"❌ Data file `{TARGET_FILENAME}` not found.")
    st.stop()

# ------------------------------------------------------------------
# 1. PRIMARY CONTROLS (Zip & Radius)
# ------------------------------------------------------------------
c_zip, c_radius, c_buffer = st.columns([2, 4, 4])

with c_zip:
    zip_code = st.text_input("📍 Start Zip Code", value="45402", max_chars=5)

with c_radius:
    search_radius = st.slider("📏 Search Radius (Miles)", 1, 50, 10)

# ------------------------------------------------------------------
# 2. COMPACT ADVANCED FILTERS (Expander)
# ------------------------------------------------------------------
with st.expander("🛠️ Filter by Sector & Waste Volume", expanded=False):
    f_col1, f_col2 = st.columns(2)
    
    with f_col1:
        min_waste = st.slider("Min. Waste (Tons/Year)", 0, 100, 0)
    
    with f_col2:
        st.markdown("**Filter by Sector**")
        selected_sources = []
        
        if 'Source_File' in df.columns:
            raw_sources = sorted(df['Source_File'].dropna().unique())
            
            # Loop through sources and create a checkbox for each
            for source in raw_sources:
                readable_label = make_label_readable(str(source))
                
                # 'value=True' makes them checked by default
                if st.checkbox(readable_label, value=True, key=f"check_{source}"):
                    selected_sources.append(source)
        else:
            selected_sources = []
# --- APPLY FILTERS ---
if 'Source_File' in df.columns:
    if not selected_sources:
        st.warning("Please select at least one sector.")
        filtered_df = df[df['Source_File'].isin([])]
    else:
        filtered_df = df[df['Source_File'].isin(selected_sources)]
else:
    filtered_df = df

# --- GEOCODING ---
nomi = pgeocode.Nominatim('us')
location = nomi.query_postal_code(zip_code)

if pd.isna(location.latitude):
    st.error(f"Could not find coordinates for Zip Code: {zip_code}")
    st.stop()
else:
    hq_lat, hq_lon = location.latitude, location.longitude
    with c_buffer:
        st.markdown(f"<div style='margin-top: 30px; color: #666;'>Searching near <b>{location.place_name}, {location.state_name}</b></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# MAP COMPUTATION
# ------------------------------------------------------------------
final_df = get_nearby_data(filtered_df, hq_lat, hq_lon, search_radius, min_waste)

m = folium.Map(location=[hq_lat, hq_lon], tiles="CartoDB positron")
lat_offset = search_radius / 69.0
lon_offset = search_radius / (69.0 * np.cos(np.radians(hq_lat)))
m.fit_bounds([[hq_lat - lat_offset, hq_lon - lon_offset], [hq_lat + lat_offset, hq_lon + lon_offset]])

# Custom CSS for clean tooltips
map_custom_css = """
<style>
.leaflet-popup-content-wrapper { background: transparent !important; box-shadow: none !important; border: none !important; padding: 0 !important; }
.leaflet-popup-tip-container { display: none !important; }
.leaflet-popup-content { margin: 0 !important; width: auto !important; }
.leaflet-container a.leaflet-popup-close-button { top: 5px !important; right: 5px !important; color: #444 !important; }
.leaflet-tooltip { background-color: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
</style>
"""
m.get_root().html.add_child(Element(map_custom_css))

if final_df.empty:
    st.warning("No targets found matching criteria.")
else:
    final_df = final_df.sort_values("Waste_Tons", ascending=False)
    for _, row in final_df.head(200).iterrows():
        tooltip_html = f"""
        <div style="font-family: sans-serif; background-color: white; padding: 6px 10px; border: 1px solid #ccc; border-radius: 6px; box-shadow: 0 3px 8px rgba(0,0,0,0.3); white-space: nowrap;">
            <strong style="color: #2E7D32;">{row['Name']}</strong><br>
            <span style="font-size: 11px;">🥕 {row['Waste_Tons']:,.0f} Tons</span>
        </div>
        """
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=8, color="#228B22", fill=True, fill_color="#32CD32", fill_opacity=0.7,
            tooltip=folium.Tooltip(tooltip_html, sticky=True)
        ).add_to(m)

# ------------------------------------------------------------------
# RENDER LAYOUT
# ------------------------------------------------------------------
count = len(final_df)

# 1. THE BANNER
st.markdown(f"""
    <div style="text-align: center; background-color: rgba(46, 125, 50, 0.15); padding: 15px; border-radius: 10px; border: 1px solid #2e7d32; margin-bottom: 20px;">
        <span style="font-size: 28px; margin-right: 5px;">🥕</span> 
        <span style="font-size: 35px; font-weight: bold; color: #66bb6a;">{count}</span>
        <span style="font-size: 20px; margin: 0 8px; color: #EEE;">surpluses near</span>
        <span style="font-size: 28px; font-weight: bold; color: #ffa726; font-family: monospace;">{zip_code}</span>
    </div>
""", unsafe_allow_html=True)

# 2. MAIN COLUMNS (Map + Details)
col_map, col_details = st.columns([7, 3])

with col_map:
    # Map Output
    map_output = st_folium(m, width=None, height=500, returned_objects=["last_object_clicked"])

with col_details:
    st.markdown("### 📍 Location Details")
    clicked_row = None
    
    # Check for clicks
    if map_output['last_object_clicked']:
        click_lat = map_output['last_object_clicked']['lat']
        click_lng = map_output['last_object_clicked']['lng']
        match = final_df[np.isclose(final_df['Latitude'], click_lat, atol=1e-5) & np.isclose(final_df['Longitude'], click_lng, atol=1e-5)]
        if not match.empty:
            clicked_row = match.iloc[0]

    # Render Card
    if clicked_row is not None:
        d_name = clicked_row['Name']
        d_addr = clicked_row['Address']
        d_city = f"{clicked_row['City']}, {clicked_row['State']} {clean_zip(clicked_row['Zip'])}"
        d_waste = clicked_row['Waste_Tons']
        d_sector = clicked_row['Sector']
        d_url = generate_search_url(clicked_row)
        
        # FIXED: Removed indentation inside the string to prevent code-block rendering
        html_card = f"""
<div style="background-color: #f0f8f0; padding: 20px; border-radius: 10px; border: 1px solid #c8e6c9;">
<h4 style="margin-top:0; color:#2E7D32; font-weight: 700;">{d_name}</h4>
<p style="font-size:13px; color:#1b1b1b; font-style: italic; margin-bottom: 15px;">{d_sector}</p>
<p style="color: #333333; margin: 5px 0;">
<strong style="color: #2E7D32;">🥕 Est. Waste:</strong> {d_waste:,.1f} Tons/Yr
</p>
<p style="color: #333333; margin-top: 15px; margin-bottom: 5px;">
<strong style="color: #555;">🏢 Address:</strong>
</p>
<p style="color: #1b1b1b; margin-top: 0; line-height: 1.4;">
{d_addr}<br>{d_city}
</p>
<div style="margin-top: 20px;">
<a href="{d_url}" target="_blank" style="display: block; text-align:center; background-color: #E3F2FD; color: #1565C0; padding: 10px; border-radius: 6px; text-decoration:none; border: 1px solid #90CAF9; font-weight:bold; font-size: 14px;">
🔎 Google This Source
</a>
</div>
</div>
"""
        st.markdown(html_card, unsafe_allow_html=True)
        st.write("")
        
        # Add Button
        btn_key = f"add_{d_name}_{clicked_row['Latitude']}"
        if st.button("➕ Add to Shortlist", key=btn_key, type="primary", use_container_width=True):
            row_to_add = pd.DataFrame([clicked_row])
            row_to_add = row_to_add.rename(columns={'Waste_Tons': 'Est. Waste (Tons/Yr)'})
            if 'Zip' in row_to_add.columns:
                row_to_add['Zip'] = row_to_add['Zip'].apply(clean_zip)
            st.session_state.shortlist = pd.concat([st.session_state.shortlist, row_to_add])
            st.session_state.shortlist = st.session_state.shortlist.drop_duplicates(subset=['Name', 'Address'])
            st.success(f"Added {d_name}!")
    else:
        st.info("👈 Click a green dot on the map to see details and add it to your list.")

st.markdown("---")

# ----------------------------------------------------------------
# DATA TABLE
# ----------------------------------------------------------------
if not final_df.empty:
    # 1. Prepare Base Data & Download Data First
    base_df = final_df.copy()
    base_df = base_df.rename(columns={'Waste_Tons': 'Est. Waste (Tons/Yr)'})
    if 'Zip' in base_df.columns: base_df['Zip'] = base_df['Zip'].apply(clean_zip)
    base_df['Distance_Miles'] = base_df['Distance_Miles'].round(1)
    base_df['Est. Waste (Tons/Yr)'] = base_df['Est. Waste (Tons/Yr)'].round(1)

    desired_cols = ['Name', 'Sector', 'Est. Waste (Tons/Yr)', 'Address', 'City', 'State', 'Zip', 'Distance_Miles']
    final_cols = [c for c in desired_cols if c in base_df.columns]
    download_df = base_df[final_cols]

    # 2. TOP ROW: Header + Download Buttons
    # We use vertical_alignment="bottom" (Streamlit 1.31+) so buttons align with text
    c_head, c_spacer, c_csv, c_pdf = st.columns([3, 1, 1.5, 1.5], vertical_alignment="bottom")

    with c_head:
        st.markdown("### 📋 All Results")
    
    with c_csv:
        st.download_button(
            label="📥 CSV", 
            data=download_df.to_csv(index=False).encode('utf-8'), 
            file_name="scout_full_list.csv", 
            mime="text/csv", 
            use_container_width=True
        )
        
    with c_pdf:
        if HAS_FPDF:
            st.download_button(
                label="📄 PDF", 
                data=create_pdf(download_df), 
                file_name="scout_full_list.pdf", 
                mime="application/pdf", 
                use_container_width=True
            )

    # 3. Prepare Table Data
    display_df = base_df[final_cols].copy()
    display_df['Research Link'] = base_df.apply(generate_search_url, axis=1)
    display_df = display_df[['Research Link'] + final_cols]
    display_df.insert(0, "Select", False)

    # 4. Render Table
    edited_df = st.data_editor(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Select": st.column_config.CheckboxColumn("Add?", default=False, width="small"),
            "Research Link": st.column_config.LinkColumn("Research", display_text="Search Web"),
            "Distance_Miles": st.column_config.NumberColumn("Distance (Mi)", format="%.1f"),
            "Est. Waste (Tons/Yr)": st.column_config.NumberColumn("Est. Waste", format="%.1f")
        }
    )

    # 5. BOTTOM ACTION: Add Selected (Must be below table to capture input)
    if st.button("➕ Add Selected to My Shortlist", use_container_width=True):
        selected_rows = edited_df[edited_df.Select]
        if not selected_rows.empty:
            to_save = selected_rows.drop(columns=['Select'])
            
            if 'shortlist' not in st.session_state:
                st.session_state.shortlist = pd.DataFrame()
                
            st.session_state.shortlist = pd.concat([st.session_state.shortlist, to_save])
            st.session_state.shortlist = st.session_state.shortlist.drop_duplicates(subset=['Name', 'Address'])
            st.success(f"Added {len(to_save)} items!")
            st.rerun()
        else:
            st.warning("Please check at least one box.")# ------------------------------------------------------------------
# SIDEBAR: SHORTLIST (MOVED TO BOTTOM)
# ------------------------------------------------------------------
# Placed here so it reflects updates from buttons above immediately
st.sidebar.header(f"🛒 Your Shortlist ({len(st.session_state.shortlist)})")

if not st.session_state.shortlist.empty:
    with st.sidebar.expander("View Selected Items", expanded=True):
        st.dataframe(st.session_state.shortlist[['Name', 'Est. Waste (Tons/Yr)']], hide_index=True)
        if st.button("Clear List", key="clear_shortlist"):
            st.session_state.shortlist = pd.DataFrame()
            st.rerun()

    shortlist_csv = st.session_state.shortlist.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("📥 Download Shortlist CSV", shortlist_csv, "my_custom_hit_list.csv", "text/csv")

    if HAS_FPDF:
        shortlist_pdf = create_pdf(st.session_state.shortlist)
        st.sidebar.download_button("📄 Download Shortlist PDF", shortlist_pdf, "my_custom_hit_list.pdf", "application/pdf")
else:
    st.sidebar.info("Click 'Add' on the map details or table to build your export list.")

st.sidebar.markdown("---")
st.sidebar.info("👋 I'd love your ideas for making this more useful. 📫 Get in touch: [ericmrothman@gmail.com](mailto:ericmrothman@gmail.com)")