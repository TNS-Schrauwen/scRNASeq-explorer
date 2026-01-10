import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils.preprocessing import Preprocessing
from utils.plotting import Plotting
from utils.workspace_manager import WorkspaceManager


# Common Plotly template and style settings for publication-quality white background
COMMON_LAYOUT = dict(
    template="simple_white",
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Arial", size=12, color="black"),
    title_font=dict(family="Arial", size=16, color="black"),
    legend_title_font=dict(color="black"),
    legend_font=dict(color="black"),
)

# Predefined readable color names (colorblind-friendly + distinct)
COLOR_OPTIONS = [
    "blue", "red", "green", "orange", "purple", "brown", "pink", "gray",
    "olive", "cyan", "magenta", "yellow", "black", "darkblue", "darkred",
    "darkgreen", "gold", "violet", "teal", "coral"
]


def apply_publication_style(fig):
    """Apply consistent publication-quality styling to a Plotly figure."""
    fig.update_layout(**COMMON_LAYOUT)
    fig.update_xaxes(title_font=dict(weight="bold"), tickfont=dict(color="black"), linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(title_font=dict(weight="bold"), tickfont=dict(color="black"), linewidth=1, linecolor="black", mirror=True)
    fig.update_traces(marker=dict(line=dict(width=0)))  # Remove point borders
    return fig


def create_custom_color_map(group_labels, selected_colors):
    """Create a dictionary mapping each unique group to a selected color."""
    unique_groups = np.unique(group_labels)
    color_cycle = selected_colors * (len(unique_groups) // len(selected_colors) + 1)
    return dict(zip(unique_groups, color_cycle[:len(unique_groups)]))


def main():
    st.title("📊 Dimensionality Reduction")
    st.markdown("Reduce data dimensionality using PCA, UMAP, and t-SNE for visualization and analysis.")
    
    # Check if preprocessing is done
    if st.session_state.get('adata') is None:
        st.error("Please complete preprocessing first!")
        st.stop()
    
    adata = st.session_state.adata
    workspace_manager = st.session_state.workspace_manager
    
    st.info(f"Current dataset: **{adata.n_obs} cells** × **{adata.n_vars} genes**")
    
    # Sidebar for parameters
    st.sidebar.header("Dimensionality Reduction Parameters")
    
    # PCA parameters
    st.sidebar.subheader("PCA")
    n_pcs = st.sidebar.slider("Number of PCs", min_value=10, max_value=100, value=50)
    use_hvg = st.sidebar.checkbox("Use highly variable genes only", value=True)
    
    # UMAP parameters
    st.sidebar.subheader("UMAP")
    umap_min_dist = st.sidebar.slider("UMAP min_dist", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    umap_spread = st.sidebar.slider("UMAP spread", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    compute_umap_3d = st.sidebar.checkbox("Compute 3D UMAP", value=True)
    
    # t-SNE parameters
    st.sidebar.subheader("t-SNE")
    compute_tsne = st.sidebar.checkbox("Compute t-SNE", value=True)
    tsne_perplexity = st.sidebar.slider("t-SNE perplexity", min_value=5, max_value=100, value=30)
    tsne_learning_rate = st.sidebar.slider("t-SNE learning rate", min_value=10, max_value=1000, value=200)
    
    # Neighbors parameters
    st.sidebar.subheader("Neighborhood Graph")
    n_neighbors = st.sidebar.slider("Number of neighbors", min_value=5, max_value=100, value=15)
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🧮 PCA", "🗺️ UMAP", "🌀 t-SNE", "📈 Run All"])
    
    with tab1:
        st.header("Principal Component Analysis (PCA)")
        
        if st.button("Compute PCA", key="pca_button"):
            with st.spinner("Computing PCA..."):
                try:
                    # Compute PCA
                    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_hvg)
                    
                    # Plot variance explained with elbow point
                    fig_variance = Plotting.plot_pca_variance(adata, n_pcs)
                    fig_variance = apply_publication_style(fig_variance)
                    fig_variance.update_layout(title="PCA Variance Explained")
                    st.plotly_chart(fig_variance, use_container_width=True)
                    st.caption("**Summary:** This scree plot shows the proportion of variance explained by each principal component. The elbow point suggests the optimal number of PCs to retain.")
                    
                    # Show PCA loadings
                    st.subheader("PCA Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        variance_ratio = adata.uns['pca']['variance_ratio']
                        st.metric("Total Variance Explained", f"{np.sum(variance_ratio) * 100:.1f}%")
                    
                    with col2:
                        st.metric("PCs Computed", n_pcs)
                    
                    with col3:
                        st.metric("Using HVG", "Yes" if use_hvg else "No")
                    
                    with col4:
                        elbow_point = Plotting._find_elbow_point(variance_ratio)
                        st.metric("Suggested PCs", f"{elbow_point + 1}")
                    
                    # PCA scatter plot options
                    st.subheader("PCA Visualization")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        color_options = ['None'] + list(adata.obs.columns)
                        pca_color_by = st.selectbox("Color by (metadata)", color_options, key="pca_color_by")
                    
                    with col2:
                        pc1 = st.number_input("PC for X-axis", min_value=1, max_value=n_pcs, value=1)
                    
                    with col3:
                        pc2 = st.number_input("PC for Y-axis", min_value=1, max_value=n_pcs, value=2)
                    
                    # Plot PCA scatter
                    if pca_color_by == 'None':
                        fig_pca = Plotting.plot_pca_scatter(adata, color=None, pc1=pc1, pc2=pc2)
                    else:
                        # Custom color selection
                        unique_groups = adata.obs[pca_color_by].unique()
                        st.write(f"**Select colors for each group in '{pca_color_by}'**")
                        group_color_map = {}
                        for group in unique_groups:
                            default_idx = list(unique_groups).index(group) % len(COLOR_OPTIONS)
                            color = st.selectbox(f"Color for {group}", COLOR_OPTIONS, index=default_idx, key=f"pca_color_{group}")
                            group_color_map[group] = color
                        
                        fig_pca = Plotting.plot_pca_scatter(
                            adata, color=pca_color_by, pc1=pc1, pc2=pc2,
                            custom_color_map=create_custom_color_map(adata.obs[pca_color_by], list(group_color_map.values()))
                        )
                    
                    fig_pca = apply_publication_style(fig_pca)
                    fig_pca.update_layout(title=f"PCA: PC{pc1} vs PC{pc2}")
                    st.plotly_chart(fig_pca, use_container_width=True)
                    st.caption("**Summary:** PCA reduces high-dimensional gene expression data into principal components. This 2D scatter plot helps identify major sources of variation and potential cell clusters.")
                    
                    # Show top genes for selected PCs
                    if 'pca' in adata.uns and 'PCs' in adata.varm:
                        with st.expander(f"Top Genes for PC{pc1} and PC{pc2}"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                loadings_pc1 = adata.varm['PCs'][:, pc1-1]
                                top_genes_pc1_idx = np.argsort(np.abs(loadings_pc1))[-10:][::-1]
                                top_genes_pc1 = pd.DataFrame({
                                    'Gene': adata.var_names[top_genes_pc1_idx],
                                    'Loading': loadings_pc1[top_genes_pc1_idx]
                                })
                                st.write(f"**Top genes for PC{pc1}:**")
                                st.dataframe(top_genes_pc1, use_container_width=True)
                            
                            with col2:
                                loadings_pc2 = adata.varm['PCs'][:, pc2-1]
                                top_genes_pc2_idx = np.argsort(np.abs(loadings_pc2))[-10:][::-1]
                                top_genes_pc2 = pd.DataFrame({
                                    'Gene': adata.var_names[top_genes_pc2_idx],
                                    'Loading': loadings_pc2[top_genes_pc2_idx]
                                })
                                st.write(f"**Top genes for PC{pc2}:**")
                                st.dataframe(top_genes_pc2, use_container_width=True)
                    
                    st.success("PCA computed successfully!")
                    
                except Exception as e:
                    st.error(f"Error computing PCA: {str(e)}")
        else:
            st.info("Click 'Compute PCA' to run principal component analysis")
    
    with tab2:
        st.header("UMAP Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Compute UMAP", key="umap_2d"):
                with st.spinner("Computing UMAP..."):
                    try:
                        if 'neighbors' not in adata.uns:
                            st.info("Computing neighborhood graph first...")
                            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.n_vars))
                        
                        sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread)
                        
                        st.success("2D UMAP computed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error computing UMAP: {str(e)}")
        
        with col2:
            if compute_umap_3d and st.button("Compute 3D UMAP", key="umap_3d"):
                with st.spinner("Computing 3D UMAP..."):
                    try:
                        if 'neighbors' not in adata.uns:
                            sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.n_vars))
                        
                        sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread, n_components=3)
                        
                        st.success("3D UMAP computed successfully!")
                        
                    except Exception as e:
                        st.error(f"Error computing 3D UMAP: {str(e)}")
        
        # Display UMAP plots
        if 'X_umap' in adata.obsm:
            st.subheader("UMAP Visualization")
            
            color_options = ['None'] + list(adata.obs.columns)
            umap_color_by = st.selectbox("Color by (metadata)", color_options, key="umap_color_by")
            
            # Custom color selection if coloring by a column
            custom_color_map = None
            if umap_color_by != 'None':
                unique_groups = adata.obs[umap_color_by].unique()
                st.write(f"**Select colors for each group in '{umap_color_by}'**")
                group_color_map = {}
                for group in unique_groups:
                    default_idx = list(unique_groups).index(group) % len(COLOR_OPTIONS)
                    color = st.selectbox(f"Color for {group}", COLOR_OPTIONS, index=default_idx, key=f"umap_color_{group}")
                    group_color_map[group] = color
                custom_color_map = create_custom_color_map(adata.obs[umap_color_by], list(group_color_map.values()))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 2D UMAP
                fig_umap_2d = Plotting.plot_embedding(
                    adata, basis='umap', color=umap_color_by if umap_color_by != 'None' else None,
                    custom_color_map=custom_color_map
                )
                fig_umap_2d = apply_publication_style(fig_umap_2d)
                fig_umap_2d.update_layout(title="2D UMAP Projection")
                st.plotly_chart(fig_umap_2d, use_container_width=True)
                st.caption("**Summary:** UMAP preserves local structure better than t-SNE and is widely used for visualizing cell clusters in single-cell data.")
            
            with col2:
                # 3D UMAP if available
                if adata.obsm['X_umap'].shape[1] >= 3:
                    fig_umap_3d = Plotting.plot_embedding(
                        adata, basis='umap', color=umap_color_by if umap_color_by != 'None' else None,
                        use_3d=True, custom_color_map=custom_color_map,
                        marker_opacity=0.55, marker_size=4  # Translucent, no border (handled in apply_style)
                    )
                    fig_umap_3d = apply_publication_style(fig_umap_3d)
                    fig_umap_3d.update_layout(
                        title="3D UMAP Projection",
                        scene=dict(
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                            aspectmode='cube'
                        )
                    )
                    st.plotly_chart(fig_umap_3d, use_container_width=True)
                    st.caption("**Summary:** 3D UMAP provides a more comprehensive view of data topology. Translucent points help reveal overlaps and density.")
                else:
                    st.info("3D UMAP not computed. Enable 'Compute 3D UMAP' and run again.")
    
    with tab3:
        st.header("t-SNE Visualization")
        
        if compute_tsne and st.button("Compute t-SNE"):
            with st.spinner("Computing t-SNE (this may take a while for large datasets)..."):
                try:
                    sc.tl.tsne(adata, n_pcs=min(n_pcs, adata.n_vars), 
                              perplexity=tsne_perplexity, learning_rate=tsne_learning_rate)
                    
                    st.success("t-SNE computed successfully!")
                    
                except Exception as e:
                    st.error(f"Error computing t-SNE: {str(e)}")
        
        # Display t-SNE plot
        if 'X_tsne' in adata.obsm:
            st.subheader("t-SNE Visualization")
            
            color_options = ['None'] + list(adata.obs.columns)
            tsne_color_by = st.selectbox("Color by (metadata)", color_options, key="tsne_color_by")
            
            custom_color_map = None
            if tsne_color_by != 'None':
                unique_groups = adata.obs[tsne_color_by].unique()
                st.write(f"**Select colors for each group in '{tsne_color_by}'**")
                group_color_map = {}
                for group in unique_groups:
                    default_idx = list(unique_groups).index(group) % len(COLOR_OPTIONS)
                    color = st.selectbox(f"Color for {group}", COLOR_OPTIONS, index=default_idx, key=f"tsne_color_{group}")
                    group_color_map[group] = color
                custom_color_map = create_custom_color_map(adata.obs[tsne_color_by], list(group_color_map.values()))
            
            fig_tsne = Plotting.plot_embedding(
                adata, basis='tsne', color=tsne_color_by if tsne_color_by != 'None' else None,
                custom_color_map=custom_color_map
            )
            fig_tsne = apply_publication_style(fig_tsne)
            fig_tsne.update_layout(title="t-SNE Projection")
            st.plotly_chart(fig_tsne, use_container_width=True)
            st.caption("**Summary:** t-SNE excels at revealing local structure and distinct clusters but may distort global distances.")
            
            # t-SNE parameters
            st.subheader("t-SNE Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Perplexity", tsne_perplexity)
            with col2:
                st.metric("Learning Rate", tsne_learning_rate)
            with col3:
                st.metric("PCs Used", min(n_pcs, adata.n_vars))
        else:
            st.info("Click 'Compute t-SNE' to run t-SNE analysis")
    
    with tab4:
        st.header("Run Complete Dimensionality Reduction")
        
        st.subheader("Processing Steps")
        
        steps = [
            f"🔲 Compute PCA ({n_pcs} components)",
            f"🔲 Compute neighborhood graph ({n_neighbors} neighbors)",
            f"🔲 Compute 2D UMAP (min_dist={umap_min_dist})",
            f"🔲 {'Compute 3D UMAP' if compute_umap_3d else 'Skip 3D UMAP'}",
            f"🔲 {'Compute t-SNE' if compute_tsne else 'Skip t-SNE'}"
        ]
        
        for step in steps:
            st.write(step)
        
        if st.button("Run Complete Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running complete dimensionality reduction pipeline..."):
                try:
                    progress_bar = st.progress(0)
                    
                    # Step 1: PCA
                    st.write("**1. Computing PCA...**")
                    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_hvg)
                    progress_bar.progress(20)
                    
                    # Step 2: Neighbors
                    st.write("**2. Computing neighborhood graph...**")
                    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, adata.n_vars))
                    progress_bar.progress(40)
                    
                    # Step 3: UMAP 2D
                    st.write("**3. Computing 2D UMAP...**")
                    sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread)
                    progress_bar.progress(60)
                    
                    # Step 4: UMAP 3D
                    if compute_umap_3d:
                        st.write("**4. Computing 3D UMAP...**")
                        sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread, n_components=3)
                    progress_bar.progress(80)
                    
                    # Step 5: t-SNE
                    if compute_tsne:
                        st.write("**5. Computing t-SNE...**")
                        sc.tl.tsne(adata, n_pcs=min(n_pcs, adata.n_vars), 
                                  perplexity=tsne_perplexity, learning_rate=tsne_learning_rate)
                    progress_bar.progress(100)
                    
                    # Update session state
                    st.session_state.adata = adata
                    
                    # Update workspace
                    workspace_manager.update_workspace_metadata(
                        st.session_state.workspace,
                        {
                            "steps_completed": ["data_upload", "quality_control", "preprocessing", "dim_reduction"],
                            "dim_reduction_parameters": {
                                "n_pcs": n_pcs,
                                "use_hvg": use_hvg,
                                "n_neighbors": n_neighbors,
                                "umap_min_dist": umap_min_dist,
                                "umap_spread": umap_spread,
                                "compute_umap_3d": compute_umap_3d,
                                "compute_tsne": compute_tsne,
                                "tsne_perplexity": tsne_perplexity,
                                "tsne_learning_rate": tsne_learning_rate
                            }
                        }
                    )
                    
                    st.success("Dimensionality reduction completed successfully!")
                    
                    # Show summary
                    st.subheader("Results Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("PCA Components", n_pcs)
                    with col2:
                        st.metric("Neighbors", n_neighbors)
                    with col3:
                        st.metric("2D UMAP", "Computed")
                    with col4:
                        st.metric("3D UMAP", "Computed" if compute_umap_3d else "Skipped")
                    
                    # Next steps
                    st.divider()
                    st.success("Dimensionality reduction completed! Proceed to **Clustering**.")
                    
                    if st.button("Go to Clustering →", use_container_width=True):
                        st.switch_page("pages/7_Clustering.py")
                
                except Exception as e:
                    st.error(f"Error during dimensionality reduction: {str(e)}")


if __name__ == "__main__":
    main()