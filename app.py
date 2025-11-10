import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Soccer Player Clustering App",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Player Clustering & Role Profiling App")
st.markdown("""
This app analyzes a fixed soccer player dataset using hierarchical clustering (Ward’s method).  
Players are grouped into **10 clusters**, visualized on a **2D PCA plot** and compared using radar charts and statistical summaries.
""")

# ============================================================
# LOAD DATA (fixed file)
# ============================================================
DATA_PATH = "Cleaned_Football_Player_Data.csv"

try:
    df = pd.read_csv(DATA_PATH)
    st.success(f"✅ Loaded dataset: {DATA_PATH} ({df.shape[0]} players, {df.shape[1]} columns)")
except FileNotFoundError:
    st.error("❌ Dataset not found. Make sure 'Cleaned_Football_Player_Data.csv' is in the same folder as this script.")
    st.stop()

# ============================================================
# FEATURE SELECTION
# ============================================================
exclude_cols = [
    'Player ID', 'Player Name', 'Player Team', 'Mins Played', 
    'Player Height', 'Player Weight', 'rating'
]
feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in exclude_cols]

X = df[feature_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# PCA REDUCTION
# ============================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca.index = X.index

# ============================================================
# CLUSTERING (fixed 10)
# ============================================================
n_clusters = 10
cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters = cluster_model.fit_predict(X_scaled)

df_clusters = df.loc[X.index].copy()
df_clusters['Cluster'] = clusters
df_clusters['PC1'] = df_pca['PC1']
df_clusters['PC2'] = df_pca['PC2']

# ============================================================
# CLUSTER SUMMARIES
# ============================================================
cluster_means = df_clusters.groupby("Cluster")[feature_cols].mean()
overall_means = df_clusters[feature_cols].mean()
cluster_diff = cluster_means - overall_means

# Identify top 5 defining features for each cluster
top_features = {}
for cluster_id in cluster_diff.index:
    top = cluster_diff.loc[cluster_id].abs().sort_values(ascending=False).head(5)
    top_features[cluster_id] = top.index.tolist()

# ============================================================
# 2D PCA SCATTERPLOT
# ============================================================
st.header("📊 Player Clusters (2D PCA Projection)")
fig_scatter = px.scatter(
    df_clusters,
    x="PC1",
    y="PC2",
    color=df_clusters["Cluster"].astype(str),
    hover_data=["Player Name", "Player Team"] if "Player Name" in df_clusters.columns else None,
    title="Players grouped into 10 clusters based on statistical similarity",
    width=900,
    height=600
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# CLUSTER DETAIL VIEW
# ============================================================
st.header("🔍 Cluster Exploration")

selected_cluster = st.selectbox("Select a cluster to view details:", df_clusters["Cluster"].unique())

col1, col2 = st.columns(2)

# ------------------ TOP 5 FACTORS ------------------
with col1:
    st.subheader(f"Top 5 Defining Factors — Cluster {selected_cluster}")
    top_df = pd.DataFrame({
        "Feature": top_features[selected_cluster],
        "Cluster Mean": cluster_means.loc[selected_cluster, top_features[selected_cluster]].values,
        "Overall Mean": overall_means[top_features[selected_cluster]].values
    })
    top_df["Difference"] = top_df["Cluster Mean"] - top_df["Overall Mean"]
    st.dataframe(top_df.style.format({"Cluster Mean": "{:.3f}", "Overall Mean": "{:.3f}", "Difference": "{:+.3f}"}))

# ------------------ RADAR CHART ------------------
with col2:
    st.subheader(f"Radar Chart — Cluster {selected_cluster} vs Dataset Average")

    cluster_avg = cluster_means.loc[selected_cluster]
    overall_avg = overall_means[cluster_avg.index]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=cluster_avg.values,
        theta=cluster_avg.index,
        fill='toself',
        name=f'Cluster {selected_cluster}',
        line=dict(color='royalblue')
    ))
    fig.add_trace(go.Scatterpolar(
        r=overall_avg.values,
        theta=overall_avg.index,
        fill='toself',
        name='Dataset Average',
        line=dict(color='orange', dash='dash')
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showline=True, linewidth=0.5)),
        showlegend=True,
        title=f"Cluster {selected_cluster} Profile vs Dataset Average",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DOWNLOAD SECTION
# ============================================================
st.markdown("---")
st.subheader("📁 Export Clustered Dataset")

output_csv = df_clusters.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Clustered Data (CSV)",
    data=output_csv,
    file_name="clustered_players_fixed.csv",
    mime="text/csv"
