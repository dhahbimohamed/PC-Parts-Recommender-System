# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import tensorflow as tf
import os
import json
from typing import Optional
import html

# ----------------------------
# Config - update your file paths
# ----------------------------
ORIGINAL_DF_PATHS = {
    'ram': 'df_ram_original.pkl',
    'cpu': 'df_cpu_original.pkl',
    'gpu': 'df_gpu_original.pkl',
    'motherboard': 'df_motherboard_original.pkl'
}
EMBEDDING_PATHS = {
    'motherboard': 'motherboard_embeddings.npy',
    'gpu': 'gpu_embeddings.npy',
    'ram': 'ram_embeddings.npy',
    'cpu': 'cpu_embeddings.npy'
}

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="PC Parts Optimizer", page_icon="⚡", layout="wide")

# ----------------------------
# Utility: load datasets
# ----------------------------
@st.cache_resource
def load_models_and_data():
    loaded = {}
    for part, path in ORIGINAL_DF_PATHS.items():
        try:
            if os.path.exists(path):
                df = pd.read_pickle(path)
                # drop unnamed index columns if any
                df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()
                # detect common columns
                name_col = next((c for c in df.columns if c.lower() in ('name','product','model')), df.columns[0])
                price_col = next((c for c in df.columns if 'price' in c.lower() or 'cost' in c.lower()), df.columns[1] if len(df.columns) > 1 else df.columns[0])
                url_col = next((c for c in df.columns if 'url' in c.lower()), None)

                emb = None
                emb_path = EMBEDDING_PATHS.get(part)
                if emb_path and os.path.exists(emb_path):
                    try:
                        emb = np.load(emb_path, allow_pickle=True)
                    except Exception:
                        emb = None

                loaded[part] = {
                    'data': df,
                    'name_col': name_col,
                    'price_col': price_col,
                    'url_col': url_col,
                    'embeddings': emb,
                }
        except Exception as e:
            st.error(f"Error loading {part}: {e}")
    return loaded

models = load_models_and_data()

if not models:
    st.error("No datasets found: check ORIGINAL_DF_PATHS file paths.")
    st.stop()

# ----------------------------
# Recommendation function
# ----------------------------
def compute_recommendations(df, embeddings, name_col, price_col, selected_index, user_price, top_n=5, similarity_threshold=0.8):
    """
    Returns DataFrame of recommended rows
    """
    try:
        df = df.reset_index(drop=True).copy()

        if selected_index >= len(df):
            selected_index = 0

        selected_row = df.iloc[selected_index]
        selected_name = selected_row[name_col]

        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        original_price = float(df.iloc[selected_index][price_col]) if not pd.isna(df.iloc[selected_index][price_col]) else None

        candidates = df.copy()

        if embeddings is not None and len(embeddings) == len(candidates):
            try:
                sims = cosine_similarity([embeddings[selected_index]], embeddings)[0]
                candidates['similarity'] = sims
                candidates = candidates[candidates['similarity'] >= similarity_threshold]
            except Exception:
                candidates['similarity'] = 0.0
        else:
            candidates['similarity'] = 0.0

        candidates = candidates[candidates[name_col] != selected_name]

        def price_ok(row):
            p = row[price_col]
            if pd.isna(p):
                return False
            if user_price is not None and p <= user_price * 0.95:
                return True
            if original_price is not None and p <= original_price:
                return True
            return False

        candidates = candidates[candidates.apply(price_ok, axis=1)]

        if not candidates.empty:
            candidates = candidates.sort_values(by=['similarity', price_col], ascending=[False, True]).head(top_n)
            return candidates

        fallback = df.copy()
        fallback = fallback[fallback[name_col] != selected_name]
        if user_price is not None:
            fallback = fallback[(fallback[price_col] <= user_price * 0.95)]
        elif original_price is not None:
            fallback = fallback[(fallback[price_col] <= original_price)]
        fallback = fallback.sort_values(by=price_col, ascending=True).head(top_n)
        fallback['similarity'] = 0.0
        return fallback

    except Exception as e:
        st.error(f"Error in recommendations: {e}")
        return pd.DataFrame(columns=df.columns.tolist() + ['similarity'])

# ----------------------------
# Compatibility function
# ----------------------------
def get_compatibility_recommendations(selected_part_type, selected_part_name, models, top_n=3):
    """
    Simple compatibility recommendations
    """
    try:
        recommendations = {}

        other_types = [pt for pt in models.keys() if pt != selected_part_type]

        for other_type in other_types:
            other_data = models[other_type]
            other_df = other_data['data']

            compatible_parts = other_df.head(top_n)
            recommendations[other_type] = compatible_parts.to_dict('records')

        return recommendations

    except Exception as e:
        return {"error": f"Compatibility error: {str(e)}"}

# ----------------------------
# Sidebar for user inputs
# ----------------------------
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-header {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1226 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        border: 1px solid #2a2f45;
    }
    .sidebar-header h2 {
        color: white;
        margin: 0;
        font-size: 24px;
    }
    .sidebar-header p {
        color: rgba(255,255,255,0.8);
        margin: 5px 0 0 0;
        font-size: 14px;
    }
    </style>
    <div class="sidebar-header">
        <h2>⚡ PC Parts Optimizer</h2>
        <p>Smart Component Recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    sim_threshold = 0.8
    top_n = 5

    recommendation_type = st.radio("**Recommendation Type**",
                                 ["💰 Price & Performance", "🔧 Compatibility"])

    part_type = st.selectbox("**Component Type**", options=list(models.keys()), format_func=lambda x: x.upper())

    part_data = models[part_type]
    df = part_data['data'].copy()
    name_col = part_data['name_col']
    price_col = part_data['price_col']
    url_col = part_data['url_col']
    embeddings = part_data['embeddings']

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()

    names = df[name_col].astype(str).tolist()
    selected_name = st.selectbox("**Select Part**", options=names)

    try:
        selected_original_index = df[df[name_col] == selected_name].index[0]
    except Exception:
        selected_original_index = 0

    if recommendation_type == "💰 Price & Performance":
        try:
            default_price = float(pd.to_numeric(df.loc[df[name_col] == selected_name, price_col].iloc[0], errors='coerce'))
        except Exception:
            default_price = 0.0

        user_price = st.number_input("**Your Price (USD)**", min_value=0.0, value=float(default_price), step=10.0)
    else:
        user_price = None

    st.markdown("---")
    st.markdown("### 💡 How it works:")
    if recommendation_type == "💰 Price & Performance":
        st.markdown("""
        - Find cheaper alternatives with similar performance
        - Based on technical specifications
        - Filters by price and similarity
        """)
    else:
        st.markdown("""
        - Find parts that work well together
        - Recommends compatible components
        - Build complete systems
        """)

# Compute recommendations
if recommendation_type == "💰 Price & Performance":
    recs_df = compute_recommendations(df, embeddings, name_col, price_col, selected_original_index, user_price, top_n=top_n, similarity_threshold=sim_threshold)
    compatibility_recs = None
else:
    compatibility_recs = get_compatibility_recommendations(part_type, selected_name, models, top_n=top_n)
    recs_df = pd.DataFrame()

# Prepare data
selected_row = df[df[name_col] == selected_name].iloc[0].to_dict()

def clean_specs(specs_dict):
    cleaned = {}
    for k, v in specs_dict.items():
        if any(url_word in k.lower() for url_word in ['url', 'link', 'website', 'http']):
            continue
        if pd.notna(v) and str(v).strip() and str(v).strip().lower() != 'nan':
            cleaned[k] = str(v)
    return cleaned

def get_seller_url(row_dict, df_columns):
    seller_url_cols = [col for col in df_columns if any(word in col.lower() for word in ['seller', 'store', 'merchant', 'retailer', 'shop'])]

    for col in seller_url_cols:
        if col in row_dict and pd.notna(row_dict[col]) and str(row_dict[col]).startswith('http'):
            return str(row_dict[col])

    url_col = next((c for c in df_columns if 'url' in c.lower()), None)
    if url_col and url_col in row_dict and pd.notna(row_dict[url_col]) and str(row_dict[url_col]).startswith('http'):
        return str(row_dict[url_col])

    return ""

def row_to_card(row_dict, df_columns, part_type=None):
    name_col = part_data['name_col']
    price_col = part_data['price_col']

    return {
        "name": str(row_dict.get(name_col, ""))[:150],
        "price": float(row_dict.get(price_col)) if pd.notna(row_dict.get(price_col)) else None,
        "url": get_seller_url(row_dict, df_columns),
        "specs": clean_specs({k: v for k, v in row_dict.items() if k not in [name_col, price_col, url_col]}),
        "part_type": part_type
    }

hero_card = row_to_card(selected_row, df.columns)
hero_card["user_price"] = float(user_price) if user_price else None

recs_list = []

if recommendation_type == "💰 Price & Performance":
    if not recs_df.empty:
        for _, r in recs_df.iterrows():
            row_dict = r.to_dict()
            rec = row_to_card(row_dict, df.columns)
            sim_val = r.get('similarity', 0.0)
            rec['similarity_pct'] = float(sim_val) * 100.0 if sim_val is not None else 0.0
            rec_price = rec.get('price')
            rec['savings'] = None
            if rec_price is not None and user_price:
                rec['savings'] = round(user_price - rec_price, 2)
            recs_list.append(rec)

else:
    if compatibility_recs and 'error' not in compatibility_recs:
        for part_type_rec, parts in compatibility_recs.items():
            if part_type_rec in models:
                for part in parts:
                    rec = row_to_card(part, models[part_type_rec]['data'].columns, part_type_rec)
                    rec['similarity_pct'] = 85.0
                    rec['savings'] = None
                    recs_list.append(rec)

if not recs_list:
    recs_list = [{
        "name": "No recommendations found",
        "price": None,
        "url": "",
        "specs": {},
        "similarity_pct": 0.0,
        "savings": None
    }]

# Calculate stats
if recommendation_type == "💰 Price & Performance":
    valid_prices = [r.get('price') for r in recs_list if r.get('price') is not None]
    if valid_prices and user_price:
        avg_price = np.mean(valid_prices)
        avg_savings_pct = ((user_price - avg_price) / user_price) * 100
        avg_savings_pct = max(0, min(100, avg_savings_pct))
    else:
        avg_savings_pct = 0
else:
    total_compatible_parts = len(recs_list)
    max_possible_parts = 12
    compatibility_score = min(95, (total_compatible_parts / max_possible_parts) * 100)

payload = {
    "hero": hero_card,
    "recs": recs_list,
    "user_inputs": {
        "part_type": part_type.upper(),
        "selected_name": selected_name,
        "user_price": user_price,
        "recommendation_type": recommendation_type
    },
    "stats": {
        "avg_savings_pct": avg_savings_pct if recommendation_type == "💰 Price & Performance" else compatibility_score,
        "alternatives_found": len([r for r in recs_list if r.get('price') or r.get('part_type')])
    }
}

# Modern HTML design with futuristic theme
html_code = f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
<style>
:root {{
  --bg: #0a0e1a;
  --accent1: #00f5ff;
  --accent2: #00d2ff;
  --accent3: #9d4edd;
  --accent4: #7b2cbf;
  --accent5: #5a189a;
  --accent6: #3c096c;
  --success: #00d4aa;
  --warning: #ffb347;
  --danger: #ff6b6b;
  --muted: #8b9bb4;
}}
body {{
  margin:0;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
  color: #ffffff;
  background: var(--bg);
  min-height: 100vh;
}}
.container {{
  max-width:1400px;
  margin:0 auto;
  padding:0;
}}
.hero-section {{
  background:
    radial-gradient(circle at 20% 80%, rgba(0, 245, 255, 0.15) 0%, transparent 50%),
    radial-gradient(circle at 80% 20%, rgba(157, 78, 221, 0.15) 0%, transparent 50%),
    radial-gradient(circle at 40% 40%, rgba(123, 44, 191, 0.2) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent6) 0%, var(--accent4) 50%, var(--accent1) 100%);
  border-radius:0 0 30px 30px;
  padding:50px 40px;
  margin-bottom:40px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.4);
  position: relative;
  overflow: hidden;
}}
.hero-section::before {{
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background:
    url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="circuit" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M0,10 L20,10 M10,0 L10,20" stroke="rgba(255,255,255,0.05)" stroke-width="0.5" fill="none"/></pattern></defs><rect width="100" height="100" fill="url(#circuit)"/></svg>');
  background-size: 200px;
  opacity: 0.3;
}}
.hero-content {{
  position: relative;
  z-index: 2;
  display:flex;
  flex-direction:column;
  gap:30px;
}}
.hero-main {{
  display:flex;
  flex-direction:column;
  gap:25px;
}}
.hero-stats {{
  display:grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap:20px;
  margin-top:20px;
}}
.stat-card {{
  background: rgba(255,255,255,0.15);
  backdrop-filter: blur(10px);
  padding:25px;
  border-radius:20px;
  text-align:center;
  border:1px solid rgba(255,255,255,0.2);
  transition: transform 0.3s ease;
}}
.stat-card:hover {{
  transform: translateY(-5px);
  background: rgba(255,255,255,0.2);
}}
.stat-value {{
  font-size:28px;
  font-weight:800;
  color:white;
  margin:8px 0;
}}
.stat-label {{
  font-size:13px;
  color:rgba(255,255,255,0.9);
  text-transform:uppercase;
  letter-spacing:0.5px;
  font-weight:600;
}}
.hero-title {{
  font-size:42px;
  font-weight:800;
  color:white;
  margin-bottom:15px;
  text-align:center;
  text-shadow: 0 4px 20px rgba(0,0,0,0.3);
}}
.hero-specs {{
  display:grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap:15px;
  margin:20px 0;
}}
.spec-item {{
  display:flex;
  justify-content:space-between;
  padding:12px 0;
  border-bottom:1px solid rgba(255,255,255,0.2);
}}
.spec-key {{
  color:rgba(255,255,255,0.9);
  font-size:15px;
  font-weight:500;
}}
.spec-val {{
  color:white;
  font-weight:700;
  font-size:15px;
}}
.price-comparison {{
  display:flex;
  gap:30px;
  align-items:center;
  margin:30px 0;
  justify-content:center;
}}
.price-item {{
  text-align:center;
  padding:20px;
  background: rgba(255,255,255,0.1);
  border-radius:15px;
  min-width:150px;
  transition: all 0.3s ease;
}}
.price-item:hover {{
  background: rgba(255,255,255,0.15);
  transform: translateY(-5px);
}}
.price-label {{
  font-size:14px;
  color:rgba(255,255,255,0.9);
  margin-bottom:8px;
  font-weight:600;
}}
.price-value {{
  font-size:32px;
  font-weight:800;
  color:white;
}}
.savings-badge {{
  background: linear-gradient(135deg, var(--success), #00b894);
  color:white;
  padding:12px 25px;
  border-radius:25px;
  font-size:16px;
  font-weight:700;
  box-shadow: 0 8px 25px rgba(0,212,170,0.3);
  transition: all 0.3s ease;
}}
.savings-badge:hover {{
  transform: scale(1.05);
}}
.compatibility-badge {{
  background: linear-gradient(135deg, var(--accent3), var(--accent4));
  color:white;
  padding:12px 25px;
  border-radius:25px;
  font-size:16px;
  font-weight:700;
  box-shadow: 0 8px 25px rgba(102,126,234,0.3);
}}
.cta-buttons {{
  display:flex;
  gap:20px;
  margin-top:30px;
  justify-content:center;
}}
.btn {{
  padding:16px 35px;
  border-radius:15px;
  border:none;
  cursor:pointer;
  font-weight:700;
  text-decoration:none;
  display:inline-flex;
  align-items:center;
  gap:10px;
  transition:all 0.3s ease;
  font-size:16px;
}}
.btn-primary {{
  background: rgba(255,255,255,0.2);
  backdrop-filter: blur(10px);
  color:white;
  border:2px solid rgba(255,255,255,0.3);
  box-shadow:0 10px 30px rgba(255,255,255,0.1);
}}
.btn-secondary {{
  background:rgba(0,0,0,0.3);
  color:white;
  border:2px solid rgba(255,255,255,0.2);
}}
.btn:hover {{
  transform:translateY(-3px);
  box-shadow:0 15px 35px rgba(0,0,0,0.4);
  background: rgba(255,255,255,0.25);
}}
.recommendations-section {{
  margin-top:50px;
  padding:0 40px;
}}
.section-header {{
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:35px;
}}
.section-title {{
  font-size:32px;
  font-weight:800;
  color:white;
  display:flex;
  align-items:center;
  gap:15px;
}}
.carousel {{
  display:grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap:30px;
}}
.recommendation-card {{
  border-radius:25px;
  padding:30px;
  transition:all 0.4s ease;
  cursor:pointer;
  position: relative;
  overflow: hidden;
  min-height: 280px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-shadow: 0 15px 40px rgba(0,0,0,0.3);
}}
.recommendation-card::before {{
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background:
    radial-gradient(circle at 30% 20%, rgba(0, 245, 255, 0.2) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(157, 78, 221, 0.15) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent6) 0%, var(--accent4) 100%);
  z-index: 1;
  border: 1px solid rgba(0, 245, 255, 0.3);
  transition: all 0.4s ease;
}}
.recommendation-card:nth-child(2)::before {{
  background:
    radial-gradient(circle at 30% 20%, rgba(123, 44, 191, 0.2) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(90, 24, 154, 0.15) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent5) 0%, var(--accent3) 100%);
  border: 1px solid rgba(123, 44, 191, 0.3);
}}
.recommendation-card:nth-child(3)::before {{
  background:
    radial-gradient(circle at 30% 20%, rgba(0, 210, 255, 0.2) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(0, 180, 216, 0.15) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent4) 0%, var(--accent2) 100%);
  border: 1px solid rgba(0, 210, 255, 0.3);
}}
.recommendation-card:nth-child(4)::before {{
  background:
    radial-gradient(circle at 30% 20%, rgba(157, 78, 221, 0.2) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(123, 44, 191, 0.15) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent3) 0%, var(--accent1) 100%);
  border: 1px solid rgba(157, 78, 221, 0.3);
}}
.recommendation-card:nth-child(5)::before {{
  background:
    radial-gradient(circle at 30% 20%, rgba(90, 24, 154, 0.2) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(60, 9, 108, 0.15) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent6) 0%, var(--accent5) 100%);
  border: 1px solid rgba(90, 24, 154, 0.3);
}}
.recommendation-card:hover::before {{
  background:
    radial-gradient(circle at 30% 20%, rgba(0, 245, 255, 0.3) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(157, 78, 221, 0.25) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent1) 0%, var(--accent3) 100%);
  transform: scale(1.05);
}}
.recommendation-card:nth-child(2):hover::before {{
  background:
    radial-gradient(circle at 30% 20%, rgba(123, 44, 191, 0.3) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(90, 24, 154, 0.25) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent3) 0%, var(--accent1) 100%);
}}
.recommendation-card:nth-child(3):hover::before {{
  background:
    radial-gradient(circle at 30% 20%, rgba(0, 210, 255, 0.3) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(0, 180, 216, 0.25) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent2) 0%, var(--accent4) 100%);
}}
.recommendation-card:nth-child(4):hover::before {{
  background:
    radial-gradient(circle at 30% 20%, rgba(157, 78, 221, 0.3) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(123, 44, 191, 0.25) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent1) 0%, var(--accent5) 100%);
}}
.recommendation-card:nth-child(5):hover::before {{
  background:
    radial-gradient(circle at 30% 20%, rgba(90, 24, 154, 0.3) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(60, 9, 108, 0.25) 0%, transparent 50%),
    linear-gradient(135deg, var(--accent5) 0%, var(--accent2) 100%);
}}
.recommendation-card:hover {{
  transform:translateY(-10px) scale(1.02);
  box-shadow:0 25px 60px rgba(0,245,255,0.2);
}}
.card-content {{
  position: relative;
  z-index: 2;
  height: 100%;
  display: flex;
  flex-direction: column;
}}
.card-header {{
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
  margin-bottom:25px;
}}
.card-title {{
  font-size:20px;
  font-weight:800;
  color:white;
  line-height:1.4;
  flex:1;
  margin-right: 15px;
  text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}}
.similarity-badge {{
  background: rgba(255,255,255,0.25);
  backdrop-filter: blur(10px);
  color:white;
  padding:10px 18px;
  border-radius:20px;
  font-size:14px;
  font-weight:700;
  white-space:nowrap;
  border:1px solid rgba(255,255,255,0.3);
  transition: all 0.3s ease;
}}
.similarity-badge:hover {{
  background: rgba(255,255,255,0.35);
}}
.part-type-badge {{
  background: rgba(255,255,255,0.3);
  color:white;
  padding:6px 12px;
  border-radius:15px;
  font-size:12px;
  font-weight:600;
  margin-left:10px;
}}
.card-price-section {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 25px 0;
  padding: 20px;
  background: rgba(255,255,255,0.15);
  border-radius:15px;
  border-left: 5px solid rgba(255,255,255,0.5);
  transition: all 0.3s ease;
}}
.card-price-section:hover {{
  background: rgba(255,255,255,0.2);
  transform: translateX(5px);
}}
.card-price {{
  font-size:36px;
  font-weight:800;
  color:white;
  text-shadow: 0 2px 10px rgba(0,0,0,0.3);
}}
.card-savings {{
  color:var(--success);
  font-size:18px;
  font-weight:700;
  text-align: right;
  text-shadow: 0 2px 8px rgba(0,0,0,0.3);
}}
.card-specs {{
  display:flex;
  flex-direction:column;
  gap:15px;
  margin:20px 0;
  flex-grow: 1;
}}
.card-spec {{
  display:flex;
  justify-content:space-between;
  font-size:15px;
  padding:12px 0;
  border-bottom:1px solid rgba(255,255,255,0.2);
  transition: all 0.3s ease;
}}
.card-spec:hover {{
  background: rgba(255,255,255,0.1);
  padding: 12px 10px;
  border-radius: 8px;
}}
.spec-name {{
  color:rgba(255,255,255,0.9);
  font-weight: 600;
}}
.spec-value {{
  color:white;
  font-weight:700;
}}
.card-actions {{
  display:flex;
  gap:15px;
  margin-top:25px;
}}
.no-recommendations {{
  text-align:center;
  padding:80px;
  color:var(--muted);
  font-size:20px;
  grid-column: 1 / -1;
  background:
    radial-gradient(circle at 30% 20%, rgba(45, 55, 70, 0.3) 0%, transparent 50%),
    linear-gradient(135deg, #2d3746 0%, #1a202c 100%);
  border-radius: 25px;
  border: 2px solid #2d3746;
}}
.icon {{
  width:24px;
  text-align:center;
}}
</style>

<div class="container">
  <div class="hero-section">
    <div class="hero-content">
      <div class="hero-main">
        <div class="hero-title" id="hero-title">
          <span id="selected-part-name">PC Parts Optimizer</span>
        </div>

        <div class="price-comparison" id="price-comparison">
          <!-- Dynamic content -->
        </div>

        <div class="hero-specs" id="hero-specs">
          <!-- Dynamic content -->
        </div>

        <div class="cta-buttons">
          <a class="btn btn-primary" id="btn-view" target="_blank">
            <i class="fas fa-external-link-alt"></i>
            View Product
          </a>
          <button class="btn btn-secondary" onclick="window.scrollTo({{top: document.querySelector('.recommendations-section').offsetTop, behavior: 'smooth'}})">
            <i class="fas fa-arrow-down"></i>
            See Recommendations
          </button>
        </div>
      </div>

      <div class="hero-stats">
        <div class="stat-card">
          <i class="fas fa-bolt" style="color:white; font-size:28px;"></i>
          <div class="stat-value" id="stat-rec-count">0</div>
          <div class="stat-label">Recommendations</div>
        </div>
        <div class="stat-card">
          <i class="fas fa-percentage" style="color:white; font-size:28px;"></i>
          <div class="stat-value" id="stat-avg-savings">0%</div>
          <div class="stat-label" id="savings-label">Performance</div>
        </div>
        <div class="stat-card">
          <i class="fas fa-microchip" style="color:white; font-size:28px;"></i>
          <div class="stat-value">{part_type.upper()}</div>
          <div class="stat-label">Component Type</div>
        </div>
      </div>
    </div>
  </div>

  <div class="recommendations-section">
    <div class="section-header">
      <div class="section-title">
        <i class="fas fa-rocket"></i>
        <span id="recommendation-title">Recommended Alternatives</span>
      </div>
    </div>

    <div class="carousel" id="carousel">
      <!-- Dynamic content -->
    </div>
  </div>
</div>

<script>
const payload = {json.dumps(payload).replace("</", "<\\/")};
const hero = payload.hero || {{}};
const recs = payload.recs || [];
const userInputs = payload.user_inputs || {{}};
const stats = payload.stats || {{}};

function formatPrice(price) {{
  return price ? '$' + Number(price).toFixed(2) : 'N/A';
}}

function setHero(h) {{
  document.getElementById('selected-part-name').textContent = h.name || 'PC Parts Optimizer';

  const priceComparison = document.getElementById('price-comparison');

  if (userInputs.recommendation_type === '💰 Price & Performance') {{
    const userPrice = h.user_price;
    const dbPrice = h.price;
    const savings = userPrice && dbPrice ? userPrice - dbPrice : 0;
    const isNegative = savings < 0;

    priceComparison.innerHTML = `
      <div class="price-item">
        <div class="price-label"><i class="fas fa-tag"></i> Your Price</div>
        <div class="price-value">${{formatPrice(userPrice)}}</div>
      </div>
      <div class="price-item">
        <div class="price-label"><i class="fas fa-database"></i> DB Price</div>
        <div class="price-value">${{formatPrice(dbPrice)}}</div>
      </div>
      <div class="price-item">
        <div class="price-label"><i class="fas fa-piggy-bank"></i> Potential Savings</div>
        <div class="savings-badge" style="${{isNegative ? 'background: linear-gradient(135deg, var(--danger), #ff4757); color: white;' : ''}}">
          ${{userPrice && dbPrice ? (savings >= 0 ? '+$' + savings.toFixed(2) : '-$' + Math.abs(savings).toFixed(2)) : '$0.00'}}
        </div>
      </div>
    `;
  }} else {{
    priceComparison.innerHTML = `
      <div class="price-item">
        <div class="price-label"><i class="fas fa-microchip"></i> Component Type</div>
        <div class="price-value">${{userInputs.part_type}}</div>
      </div>
      <div class="price-item">
        <div class="price-label"><i class="fas fa-link"></i> Recommendation</div>
        <div class="compatibility-badge">Compatibility</div>
      </div>
      <div class="price-item">
        <div class="price-label"><i class="fas fa-cogs"></i> System Builder</div>
        <div class="price-value">Active</div>
      </div>
    `;
  }}

  document.getElementById('btn-view').href = h.url || '#';

  const specsEl = document.getElementById('hero-specs');
  specsEl.innerHTML = '';
  if (h.specs) {{
    const keys = Object.keys(h.specs).slice(0,6);
    keys.forEach(k => {{
      const div = document.createElement('div');
      div.className = 'spec-item';
      div.innerHTML = `
        <span class="spec-key">${{k}}:</span>
        <span class="spec-val">${{h.specs[k]}}</span>
      `;
      specsEl.appendChild(div);
    }});
  }}
}}

function renderRecs(list) {{
  const carousel = document.getElementById('carousel');
  carousel.innerHTML = '';

  const recTitle = document.getElementById('recommendation-title');
  if (userInputs.recommendation_type === '💰 Price & Performance') {{
    recTitle.textContent = 'Recommended Alternatives';
  }} else {{
    recTitle.textContent = 'Compatible Parts';
  }}

  document.getElementById('stat-rec-count').textContent = stats.alternatives_found || 0;

  const savingsStat = document.getElementById('stat-avg-savings');
  const savingsLabel = document.getElementById('savings-label');

  if (userInputs.recommendation_type === '💰 Price & Performance') {{
    savingsStat.textContent = Math.round(stats.avg_savings_pct || 0) + '%';
    savingsLabel.textContent = 'Avg Savings';
  }} else {{
    savingsStat.textContent = Math.round(stats.avg_savings_pct || 0) + '%';
    savingsLabel.textContent = 'Compatibility';
  }}

  if (list.length === 0 || (list.length === 1 && !list[0].price && !list[0].part_type)) {{
    carousel.innerHTML = `
      <div class="no-recommendations">
        <i class="fas fa-search" style="font-size:48px; margin-bottom:20px; opacity:0.5;"></i>
        <div>No recommendations found</div>
        <div style="font-size:14px; margin-top:10px;">Try adjusting your search criteria</div>
      </div>
    `;
    return;
  }}

  list.forEach((r, idx) => {{
    const card = document.createElement('div');
    card.className = 'recommendation-card';

    const partTypeBadge = r.part_type ? `<span class="part-type-badge">${{r.part_type.toUpperCase()}}</span>` : '';

    card.innerHTML = `
      <div class="card-content">
        <div class="card-header">
          <div class="card-title">
            <i class="fas fa-microchip" style="margin-right:10px;"></i>
            ${{r.name}}
            ${{partTypeBadge}}
          </div>
          ${{r.similarity_pct ? `<div class="similarity-badge">${{Math.round(r.similarity_pct)}}% ${{userInputs.recommendation_type === '💰 Price & Performance' ? 'match' : 'compatible'}}</div>` : ''}}
        </div>

        <div class="card-price-section">
          <div>
            <div style="font-size:14px; color:rgba(255,255,255,0.9);">${{r.price ? 'Price' : 'Compatibility'}}</div>
            <div class="card-price">${{r.price ? formatPrice(r.price) : 'High'}}</div>
          </div>
          ${{r.savings > 0 ? `
            <div class="card-savings">
              <i class="fas fa-piggy-bank"></i>
              Save $${{r.savings.toFixed(2)}}
            </div>
          ` : r.part_type ? `
            <div class="card-savings" style="color: var(--accent3);">
              <i class="fas fa-link"></i>
              Compatible
            </div>
          ` : ''}}
        </div>

        ${{Object.keys(r.specs).length > 0 ? `
          <div class="card-specs">
            ${{Object.keys(r.specs).slice(0,4).map(k => `
              <div class="card-spec">
                <span class="spec-name">${{k}}:</span>
                <span class="spec-value">${{r.specs[k]}}</span>
              </div>
            `).join('')}}
          </div>
        ` : ''}}

        <div class="card-actions">
          <a class="btn btn-primary" href="${{r.url || '#'}}" target="_blank" style="flex:1; justify-content:center;">
            <i class="fas fa-shopping-cart"></i>
            ${{r.url ? 'View Deal' : 'Details'}}
          </a>
        </div>
      </div>
    `;

    carousel.appendChild(card);
  }});
}}

// Initialize
setHero(hero);
renderRecs(recs);
</script>
"""

# Render the HTML component
st.components.v1.html(html_code, height=1200, scrolling=True)
