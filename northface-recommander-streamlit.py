# ====================================
# North Face Streamlit Recommender
# Created by Christophe Noret - 2025
# ====================================
# Run this app with:
#   streamlit run northface-recommander-streamlit.py
# ====================================

# Requirements:
#   - Python 3.x
#   - pandas
#   - streamlit

# Required File:
#   - ./data/clustered_data.csv (exported from the notebook)
#     It must contain at least:
#       • an index representing product IDs
#       • a 'cluster' column indicating the cluster ID
# ====================================

import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    """Load the clustered data from CSV."""
    df = pd.read_csv("./data/clustered_data.csv", index_col=0)
    return df


data = load_data()


def find_similar_items(item_id, n=5):
    "" "Return a list of n similar item ids from the same cluster." ""
    if item_id not in data.index:
        return []
    cluster_id = data.loc[item_id, "cluster"]
    if cluster_id == -1:
        return []
    cluster_items = data[(data["cluster"] == cluster_id) & (data.index != item_id)]
    return cluster_items.index[:n].tolist()


st.set_page_config(page_title="North Face Recommender", layout="centered")

st.markdown(
    """
<style>
    .title {font-size: 2.3em; font-weight: bold; text-align: center; color: #1f4e79;}
    .subtitle {font-size: 1.2em; text-align: center; color: #444;}
    .desc-block {background-color: #f0f2f6; padding: 1em; border-radius: 8px; margin-bottom: 1em;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="title">🧢 The North Face Product Recommender</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Find similar products using machine learning clustering</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# Product selection
item_id = st.number_input(
    "Enter a product ID",
    min_value=int(data.index.min()),
    max_value=int(data.index.max()),
    step=1,
)

# Recommender
if st.button("🔍 Recommend Similar Products"):
    similar = find_similar_items(item_id)
    if similar:
        title = data.loc[item_id, "title"]
        st.success(f"Found {len(similar)} products similar to **{title}**:")
        for sid in similar:
            st.markdown(
                f'<div class="desc-block"><strong>🧢 {data.loc[sid, "title"]} (ID: {sid})</strong><br>{data.loc[sid, "description"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning(
            "⚠️ No similar items found. The product might be an outlier or the ID is invalid."
        )
        valid_ids = data[data["cluster"] != -1].index.tolist()
        st.caption(f"Valid product IDs : {valid_ids[:50]}...")

st.markdown("---")
st.caption("Made with ❤️ by Christophe NORET")
