import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Soccer Player Clustering Dashboard",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Soccer Player Clustering Dashboard")
st.markdown("""
This dashboard analyzes player performance data from a fixed dataset and groups players into **10 distinct clusters** 
using Ward's hierarchical clustering.  
Clusters represent player archetypes derived from statistical performance, visualized through PCA and radar charts.
""")

# ============================================================
# LOAD FIXED DATASET
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Football_Player_Data.csv")
    return df

df = load_data()
st.success(f"✅ Dataset loaded: {df.shape[0]} players, {df.shape[1]} columns")

# ============================================================
# FEATURE SELECTION (EXCLUDE NON-PERFORMANCE COLUMNS)
# ============================================================
exclude_cols = [
    'rating', 'Player Name', 'Player Team', 'Mins Played', 
    'Height', 'Weight', 'Player ID', 'Player Age'
]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

# ============================================================
# STANDARDIZE + PCA
# ============================================================
X = df[feature_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ============================================================
# CLUSTERING (FIXED TO 10)
# ============================================================
n_clusters = 10
model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters = model.fit_predict(X_scaled)

df_clusters = df.loc[X.index].copy()
df_clusters['Cluster'] = clusters
df_clusters['PC1'] = X_pca[:, 0]
df_clusters['PC2'] = X_pca[:, 1]

# ============================================================
# CLUSTER SUMMARIES
# ============================================================
cluster_means = df_clusters.groupby("Cluster")[feature_cols].mean()
overall_means = df_clusters[feature_cols].mean()
cluster_diff = cluster_means - overall_means

top_features = {}
for cluster_id in cluster_diff.index:
    top = cluster_diff.loc[cluster_id].abs().sort_values(ascending=False).head(5)
    top_features[cluster_id] = top.index.tolist()

# ============================================================
# PCA SCATTER PLOT
# ============================================================
st.header("📊 PCA 2D Cluster Visualization")

fig_scatter = px.scatter(
    df_clusters,
    x="PC1",
    y="PC2",
    color=df_clusters["Cluster"].astype(str),
    hover_data=["Player Name"],
    title="PCA Projection of Player Clusters (10 Total)",
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig_scatter.update_layout(
    legend_title_text="Cluster ID",
    height=700,
    xaxis_title="Principal Component 1 (Attacking tendency)",
    yaxis_title="Principal Component 2 (Defensive tendency)"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# CLUSTER DETAILS
# ============================================================
st.header("🔍 Cluster Insights")

selected_cluster = st.selectbox("Select a Cluster to Explore:", sorted(df_clusters["Cluster"].unique()))
st.subheader(f"Cluster {selected_cluster} Summary")

# Show top defining factors
st.write("**Top 5 Defining Features (most different from overall average):**")
top_df = pd.DataFrame({
    "Feature": top_features[selected_cluster],
    "Cluster Mean": cluster_means.loc[selected_cluster, top_features[selected_cluster]].values,
    "Dataset Mean": overall_means[top_features[selected_cluster]].values
})
top_df["Difference"] = top_df["Cluster Mean"] - top_df["Dataset Mean"]
st.dataframe(top_df.style.format({"Cluster Mean": "{:.3f}", "Dataset Mean": "{:.3f}", "Difference": "{:+.3f}"}))

# ============================================================
# RADAR CHART (WITH CLEAR LEGEND)
# ============================================================
st.subheader(f"📈 Radar Chart Comparison – Cluster {selected_cluster}")

cluster_avg = cluster_means.loc[selected_cluster]
overall_avg = overall_means[cluster_avg.index]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=cluster_avg.values,
    theta=cluster_avg.index,
    fill='toself',
    name=f'Cluster {selected_cluster} Average',
    line=dict(color='royalblue', width=3)
))

fig.add_trace(go.Scatterpolar(
    r=overall_avg.values,
    theta=overall_avg.index,
    fill='toself',
    name='Overall Dataset Average',
    line=dict(color='orange', width=2, dash='dash')
))

fig.update_layout(
    title=f"Cluster {selected_cluster} vs. Dataset Average",
    polar=dict(
        radialaxis=dict(visible=True, showline=True, linewidth=1, gridcolor='lightgrey')
    ),
    showlegend=True,
    legend=dict(
        title="Legend",
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="center",
        x=0.5,
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor="black",
        borderwidth=1
    ),
    height=750
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# DOWNLOAD OUTPUT
# ============================================================
st.markdown("---")
st.subheader("📁 Export Clustered Dataset")

output_csv = df_clusters.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Clustered Data (CSV)",
    data=output_csv,
    file_name="clustered_players_fixed.csv",
    mime="text/csv"
)
