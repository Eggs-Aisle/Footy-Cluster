import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import plotly.graph_objects as go

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Soccer Player Clustering App",
    page_icon="⚽",
    layout="wide"
)

st.title("Player Clustering & Role Profiling App")
st.markdown("""
This app groups players into 10 performance-based clusters using hierarchical clustering (Ward's method).
Each cluster represents a distinct player archetype based on statistical profiles.  
Use the panels below to explore cluster-defining features and radar charts comparing each cluster’s average
values to the dataset mean.
""")

# ============================================================
# DATA UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Upload your cleaned player dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success(f"✅ Data loaded successfully! ({df.shape[0]} players, {df.shape[1]} columns)")

    # ============================================================
    # SELECT FEATURES
    # ============================================================
    # Automatically detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude ID/demographic columns if present
    exclude_cols = ['Player ID', 'Player Name', 'Player Team', 'Position 1', 'Position 2']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    st.sidebar.header("Feature Selection")
    selected_features = st.sidebar.multiselect(
        "Select features for clustering:",
        feature_cols,
        default=feature_cols
    )

    # ============================================================
    # STANDARDIZE + PCA
    # ============================================================
    X = df[selected_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # ============================================================
    # CLUSTERING
    # ============================================================
    n_clusters = 10
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    clusters = cluster_model.fit_predict(X_scaled)

    df_clusters = df.loc[X.index].copy()
    df_clusters['Cluster'] = clusters

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Total Clusters:** {n_clusters}")
    st.sidebar.write(f"**Explained Variance (2 PCs):** {pca.explained_variance_ratio_.sum():.2%}")

    # ============================================================
    # CLUSTER SUMMARIES
    # ============================================================
    cluster_means = df_clusters.groupby("Cluster")[selected_features].mean()
    overall_means = df_clusters[selected_features].mean()
    cluster_diff = cluster_means - overall_means

    # Top 5 factors per cluster
    top_features = {}
    for cluster_id in cluster_diff.index:
        top = cluster_diff.loc[cluster_id].abs().sort_values(ascending=False).head(5)
        top_features[cluster_id] = top.index.tolist()

    # ============================================================
    # DISPLAY SECTION
    # ============================================================
    st.header("🔍 Explore Cluster Profiles")

    tab1, tab2 = st.tabs(["Cluster Summary", "Radar Chart"])

    with tab1:
        selected_cluster = st.selectbox("Select a cluster to view", cluster_means.index)
        st.subheader(f"Cluster {selected_cluster} Summary")

        st.write(f"**Top 5 defining factors:**")
        top_df = pd.DataFrame({
            "Feature": top_features[selected_cluster],
            "Cluster Mean": cluster_means.loc[selected_cluster, top_features[selected_cluster]].values,
            "Overall Mean": overall_means[top_features[selected_cluster]].values
        })
        top_df["Difference"] = top_df["Cluster Mean"] - top_df["Overall Mean"]
        st.dataframe(top_df.style.format({"Cluster Mean": "{:.3f}", "Overall Mean": "{:.3f}", "Difference": "{:+.3f}"}))

    with tab2:
        st.subheader(f"Radar Chart: Cluster {selected_cluster} vs Dataset Average")

        # Radar data
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
            polar=dict(
                radialaxis=dict(visible=True, showline=True, linewidth=0.5)
            ),
            showlegend=True,
            height=700,
            title=f"Cluster {selected_cluster} vs Dataset Average"
        )

        st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # DOWNLOAD SECTION
    # ============================================================
    st.markdown("---")
    st.subheader("Export Clustered Dataset")

    output_csv = df_clusters.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Clustered Data (CSV)",
        data=output_csv,
        file_name="clustered_players.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a dataset to get started.")
