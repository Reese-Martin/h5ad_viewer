import scanpy as sc
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np


def stratified_downsample(adata, groupby_col, target_total_cells=5000, seed=42):
    # Downsample cells while preserving group proportions from `groupby_col`.
    np.random.seed(seed)
    counts = adata.obs[groupby_col].value_counts()
    proportions = counts / counts.sum()
    sample_sizes = (proportions * target_total_cells).astype(int)

    sampled_indices = []
    for group, n in sample_sizes.items():
        group_indices = adata.obs[adata.obs[groupby_col] == group].index
        if len(group_indices) <= n:
            sampled_indices.extend(group_indices)
        else:
            sampled_indices.extend(np.random.choice(group_indices, size=n, replace=False))

    return adata[sampled_indices, :].copy()


st.title("🧬 h5ad Viewer")

uploaded_file = st.file_uploader("Upload an .h5ad file", type=["h5ad"])

if uploaded_file:
    adata = sc.read_h5ad(uploaded_file)
    st.sidebar.markdown("### 📉 Downsampling (optional)")

    enable_downsampling = st.sidebar.checkbox("Enable smart downsampling", value=True)

    if enable_downsampling and adata.n_obs > 5000:
        groupby_col = st.sidebar.selectbox("Preserve cell proportions by:",
                                           options=[col for col in adata.obs.columns if 1 < adata.obs[col].nunique() < 100],
                                           help="Used to proportionally sample cells from each group.")
        target_n = st.sidebar.slider("Target number of cells", min_value=1000, max_value=20000, value=5000, step=500)

        with st.spinner(f"Downsampling to {target_n} cells..."):
            adata = stratified_downsample(adata, groupby_col, target_total_cells=target_n)
        st.sidebar.success(f"Downsampled to {adata.n_obs} cells.")


    st.sidebar.markdown("### 🔎 Variable Filtering")
    var_columns = adata.var.columns.tolist()
    bool_cols = [col for col in var_columns if pd.api.types.is_bool_dtype(adata.var[col])]

    if bool_cols:
        filter_col = st.sidebar.selectbox("Filter genes by boolean column:", bool_cols)
        adata = adata[:, adata.var[filter_col].values]  # Subset columns (genes)
        st.sidebar.success(f"Subsetted to {adata.shape[1]} genes using `{filter_col}`")
    else:
        st.sidebar.info("No boolean columns found in `adata.var` for filtering.")

    st.sidebar.markdown("### 🧫 Cell Filtering")
    obs_columns = adata.obs.columns.tolist()
    cat_cols = [col for col in obs_columns if isinstance(adata.obs[col].dtype, pd.CategoricalDtype) or adata.obs[col].nunique() < 50]

    if cat_cols:
        cell_filter_col = st.sidebar.selectbox("Filter cells by category:", cat_cols)
        categories = adata.obs[cell_filter_col].unique().tolist()
        selected_labels = st.sidebar.multiselect(f"Select {cell_filter_col} values to include:", categories,
                                                 default=categories)
        if selected_labels:
            mask = adata.obs[cell_filter_col].isin(selected_labels)
            adata = adata[mask, :]
            st.sidebar.success(f"Subsetted to {adata.shape[0]} cells.")
        else:
            st.sidebar.warning("No labels selected. Using full dataset.")
    else:
        st.sidebar.info("No low-cardinality categorical columns found in `adata.obs`.")
    st.sidebar.header("Navigation")

    view = st.sidebar.radio("View", ["Summary", "UMAP", "PCA", "Gene Expression", "obs/var"])

    max_cells = 5000
    if adata.n_obs > max_cells:
        idx = np.random.choice(adata.n_obs, size=max_cells, replace=False)
        adata_plot = adata[idx, :].copy()
    else:
        adata_plot = adata

    if view == "Summary":
        st.write("Shape:", adata.shape)
        st.write("obs columns:", list(adata.obs.columns))
        st.write("var columns:", list(adata.var.columns))
        st.write("uns keys:", list(adata.uns.keys()))

    elif view == "UMAP":
        st.subheader("🧭 UMAP View")

        # Default to 'X_umap' but allow user to enter other .obsm key
        default_umap_key = "X_umap"
        umap_key = st.text_input("Enter UMAP key from `.obsm`", value=default_umap_key)

        if umap_key in adata_plot.obsm:
            umap = adata_plot.obsm[umap_key]
            umap_df = pd.DataFrame(umap[:, :2], columns=["UMAP1", "UMAP2"], index=adata_plot.obs_names)
            umap_df = umap_df.join(adata_plot.obs)
            color_by = st.selectbox("Color by:", adata_plot.obs.columns)
            # fig, ax = plt.subplots()
            # ax.scatter(adata_plot.obsm["X_umap"][:, 0], adata_plot.obsm["X_umap"][:, 1], s=1)
            # st.pyplot(fig)
            fig = px.scatter(umap_df, x="UMAP1", y="UMAP2", color=color_by)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Could not find `{umap_key}` in `.obsm`. Available keys: {list(adata.obsm.keys())}")

    elif view == "PCA":
        # Default to 'X_umap' but allow user to enter other .obsm key
        default_pca_key = "X_pca"
        pca_key = st.text_input("Enter PCA key from `.obsm`", value=default_pca_key)

        if pca_key in adata.obsm:
            n_pcs = adata.obsm["X_pca"].shape[1]
            pcs = [f"PC_{i + 1}" for i in range(n_pcs)]
            pca_df = pd.DataFrame(adata.obsm["X_pca"], columns=pcs, index=adata.obs_names)
            pca_df = pca_df.join(adata.obs)
            col1, col2 = st.columns(2)
            with col1:
                pc_x = st.selectbox("X-axis PC", pcs, index=0)
            with col2:
                pc_y = st.selectbox("Y-axis PC", pcs, index=1)

            color_by = st.selectbox("Color by:", adata.obs.columns)
            fig = px.scatter(
                pca_df,
                x=pc_x,
                y=pc_y,
                color=color_by,
                labels={pc_x: pc_x, pc_y: pc_y},
                title=f"PCA: {pc_x} vs {pc_y}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Could not find `{pca_key}` in `.obsm`. Available keys: {list(adata.obsm.keys())}")

    elif view == "Gene Expression":
        gene = st.selectbox("Select a gene", adata.var_names)
        expr = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, 'toarray') else adata[:, gene].X.flatten()
        # umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
        # umap_df[gene] = expr
        # fig = px.scatter(umap_df, x="UMAP1", y="UMAP2", color=gene, color_continuous_scale="Viridis")
        st.dataframe(pd.DataFrame({gene: expr}, index=adata.obs_names))
        # tab = st.selectbox("Table", adata)

    elif view == "obs/var":
        tab = st.selectbox("Table", ["obs", "var"])
        df = adata.obs if tab == "obs" else adata.var
        st.dataframe(df)