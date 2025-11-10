import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

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
This dashboard groups players into **10 archetypes** based on performance metrics.  
Each archetype represents a different style of player contribution across the pitch.
""")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_Football_Player_Data.csv")
    return df

df = load_data()
st.success(f"✅ Dataset loaded successfully: {df.shape[0]} players, {df.shape[1]} columns")

# ============================================================
# FEATURE SELECTION
# ============================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_keywords = ['rating', 'name', 'team', 'min', 'height', 'weight', 'id', 'age']
feature_cols = [
    c for c in numeric_cols
    if not any(keyword.lower() in c.lower() for keyword in exclude_keywords)
]

# Try to detect position columns for export reordering later
pos_cols = [c for c in df.columns if "position" in c.lower()]

# ============================================================
# STANDARDIZE + PCA
# ============================================================
X = df[feature_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ============================================================
# CLUSTERING (10 FIXED)
# ============================================================
n_clusters = 10
model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters = model.fit_predict(X_scaled)

df_clusters = df.loc[X.index].copy()
df_clusters['Cluster'] = clusters
df_clusters['PC1'] = X_pca[:, 0]
df_clusters['PC2'] = X_pca[:, 1]

# ============================================================
# CLUSTER NAMES (YOUR UPDATED LIST)
# ============================================================
cluster_names = {
    0: "All-Around Goal Generator",
    1: "Roaming Defender",
    2: "Primary Goal Scorer",
    3: "Wide Playmaker",
    4: "Utility Man",
    5: "Ball-Winning Distributer",
    6: "Dribbling Winger",
    7: "Goal Keeper",
    8: "Secondary Goal Scorer",
    9: "Holding Defender"
}

df_clusters['Cluster Name'] = df_clusters['Cluster'].map(cluster_names)

# ============================================================
# CLUSTER SUMMARY STATS
# ============================================================
cluster_means = df_clusters.groupby("Cluster")[feature_cols].mean()
overall_means = df_clusters[feature_cols].mean()
feature_std = df_clusters[feature_cols].std()
z_diff = (cluster_means - overall_means) / feature_std

# ============================================================
# PCA CLUSTER VISUALIZATION
# ============================================================
st.header("📊 PCA 2D Cluster Visualization")

cluster_select = st.multiselect(
    "Select clusters to display:",
    options=["All"] + [f"{i} – {cluster_names[i]}" for i in sorted(df_clusters['Cluster'].unique())],
    default=["All"]
)

# Determine which clusters to display
if "All" in cluster_select or len(cluster_select) == 0:
    filtered_df = df_clusters.copy()
else:
    selected_numbers = [int(c.split("–")[0].strip()) for c in cluster_select]
    filtered_df = df_clusters[df_clusters['Cluster'].isin(selected_numbers)]

# Create combined cluster label for legend
df_clusters['Cluster Label'] = df_clusters['Cluster'].astype(str) + " – " + df_clusters['Cluster Name']

fig_scatter = px.scatter(
    filtered_df,
    x="PC1",
    y="PC2",
    color=df_clusters.loc[filtered_df.index, "Cluster Label"],
    hover_data=["Player Name"] if "Player Name" in df_clusters.columns else None,
    title="PCA Projection of Player Archetypes (10 Clusters)",
    color_discrete_sequence=px.colors.qualitative.Set3
)
fig_scatter.update_layout(
    legend_title_text="Cluster (Number – Archetype)",
    height=700,
    xaxis_title="Principal Component 1 (Attacking Tendency)",
    yaxis_title="Principal Component 2 (Defensive Tendency)"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ============================================================
# FEATURE DIFFERENCE BAR CHART
# ============================================================
st.markdown("---")
st.header("📊 Cluster-Specific Feature Differences")

selected_cluster = st.selectbox(
    "Select a Cluster to Explore:",
    sorted(df_clusters["Cluster"].unique()),
    format_func=lambda x: f"{x} – {cluster_names[x]}"
)

selected_z = z_diff.loc[selected_cluster].sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(12, 6))
colors = ['green' if v > 0 else 'red' for v in selected_z]
ax2.barh(selected_z.index, selected_z.values, color=colors)
ax2.axvline(0, color='black', linewidth=1)
ax2.set_xlabel("Z-score Difference from Dataset Mean")
ax2.set_title(f"{selected_cluster} – {cluster_names[selected_cluster]}: Feature Deviations")
plt.gca().invert_yaxis()
st.pyplot(fig2)

st.markdown("""
**Interpretation Tip:**  
Green = above-average values for this archetype; Red = below-average values.  
The longer the bar, the more defining that feature is for the cluster.
""")

# ============================================================
# EXPORT CLUSTERED DATASET (FINALIZED)
# ============================================================
st.markdown("---")
st.subheader("📁 Export Clustered Dataset")

# Build export dataframe (remove unwanted columns)
export_df = df_clusters.drop(columns=["PC1", "PC2", "Cluster", "Cluster Label"], errors="ignore")

# Reorder columns to place Cluster Name right after position columns
cols = list(export_df.columns)
if pos_cols:
    insert_at = max(cols.index(pos_cols[-1]) + 1, 1)
    reordered_cols = cols[:insert_at] + ['Cluster Name'] + [c for c in cols if c not in pos_cols + ['Cluster Name']]
else:
    reordered_cols = ['Cluster Name'] + [c for c in cols if c != 'Cluster Name']

export_df = export_df[reordered_cols]

# Download button
output_csv = export_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Clustered Data (CSV)",
    data=output_csv,
    file_name="clustered_players_named.csv",
    mime="text/csv"
)
