import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocessing import Preprocessing
from utils.plotting import Plotting
from utils.workspace_manager import WorkspaceManager


# Predefined readable color names (colorblind-friendly + distinct)
COLOR_OPTIONS = [
    "blue", "red", "green", "orange", "purple", "brown", "pink", "gray",
    "olive", "cyan", "magenta", "yellow", "black", "darkblue", "darkred",
    "darkgreen", "gold", "violet", "teal", "coral"
]


def apply_plotly_publication_style(fig):
    """Apply consistent publication-quality styling to a Plotly figure."""
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12, color="black"),
        title_font=dict(family="Arial", size=16, color="black"),
    )
    fig.update_xaxes(title_font=dict(weight="bold"), tickfont=dict(color="black"), linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(title_font=dict(weight="bold"), tickfont=dict(color="black"), linewidth=1, linecolor="black", mirror=True)
    return fig

def set_publication_style():
    """Set matplotlib style for publication-quality plots"""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'axes.labelcolor': 'black',
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'font.weight': 'bold',
        'font.family': 'sans-serif',
        'font.size': 12,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.facecolor': 'white',
        'grid.color': 'lightgray',
        'grid.linestyle': '--',
        'grid.alpha': 0.3
    })


def create_custom_color_map(group_labels, selected_colors):
    """Create a dictionary mapping each unique group to a selected color."""
    unique_groups = np.unique(group_labels)
    color_cycle = selected_colors * (len(unique_groups) // len(selected_colors) + 1)
    return dict(zip(unique_groups, color_cycle[:len(unique_groups)]))


def main():
    st.title("Dimensionality Reduction")
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
    tab1, tab2, tab3, tab4 = st.tabs(["PCA", "UMAP", "t-SNE", "Run All"])
    
    with tab1:
        st.header("Principal Component Analysis (PCA)")
        
        if st.button("Compute PCA", key="pca_button"):
            with st.spinner("Computing PCA..."):
                try:
                    # Compute PCA
                    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_hvg)
                    
                    # Plot variance explained with elbow point
                    set_publication_style()
                    fig_variance, ax = plt.subplots(figsize=(10, 6))
                    
                    variance_ratio = adata.uns['pca']['variance_ratio'][:n_pcs]
                    cumulative_variance = np.cumsum(variance_ratio)
                    elbow_point = Plotting._find_elbow_point(variance_ratio)
                    
                    # Plot individual variance
                    ax.bar(range(1, len(variance_ratio)+1), variance_ratio, 
                          alpha=0.6, color='lightgray', edgecolor='black', label='Individual Variance')
                    
                    # Plot cumulative variance
                    ax2 = ax.twinx()
                    ax2.plot(range(1, len(variance_ratio)+1), cumulative_variance, 
                            'o-', color='black', linewidth=2, markersize=6, label='Cumulative Variance')
                    
                    # Elbow line
                    ax.axvline(elbow_point + 1, color='#d62728', linestyle='--', linewidth=2, 
                              label=f'Elbow: PC{elbow_point + 1}')
                    
                    ax.set_xlabel('Principal Component', fontweight='bold')
                    ax.set_ylabel('Individual Variance Ratio', fontweight='bold')
                    ax2.set_ylabel('Cumulative Variance Ratio', fontweight='bold')
                    ax.set_title('PCA Variance Explained', fontweight='bold')
                    
                    # Combine legends
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines + lines2, labels + labels2, loc='center right')
                    
                    st.pyplot(fig_variance)
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
                    
                    # Prepare for static plot
                    set_publication_style()
                    fig_pca, ax = plt.subplots(figsize=(8, 8))
                    
                    x = adata.obsm['X_pca'][:, pc1-1]
                    y = adata.obsm['X_pca'][:, pc2-1]
                    
                    if pca_color_by == 'None':
                        ax.scatter(x, y, c='gray', s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
                    else:
                        # Check if categorical or continuous
                        if pd.api.types.is_numeric_dtype(adata.obs[pca_color_by]) and len(adata.obs[pca_color_by].unique()) > 50:
                            # Continuous
                            sc_plot = ax.scatter(x, y, c=adata.obs[pca_color_by], cmap='viridis', 
                                               s=40, alpha=0.8, edgecolors='white', linewidth=0.2)
                            plt.colorbar(sc_plot, ax=ax, label=pca_color_by)
                        else:
                            # Categorical - Allow color selection
                            unique_groups = adata.obs[pca_color_by].unique()
                            st.write(f"**Select colors for each group in '{pca_color_by}'**")
                            group_color_map = {}
                            cols = st.columns(min(len(unique_groups), 4))
                            for i, group in enumerate(unique_groups):
                                with cols[i % 4]:
                                    default_idx = list(unique_groups).index(group) % len(COLOR_OPTIONS)
                                    color = st.selectbox(f"{group}", COLOR_OPTIONS, index=default_idx, key=f"pca_color_{group}")
                                    group_color_map[group] = color
                            
                            sns.scatterplot(x=x, y=y, hue=adata.obs[pca_color_by], 
                                          palette=group_color_map, ax=ax, s=50, 
                                          edgecolor='white', linewidth=0.5, alpha=0.9)
                            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title=pca_color_by)
                    
                    ax.set_xlabel(f'PC{pc1}', fontweight='bold')
                    ax.set_ylabel(f'PC{pc2}', fontweight='bold')
                    ax.set_title(f'PCA: PC{pc1} vs PC{pc2}', fontweight='bold')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    st.pyplot(fig_pca)
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
                                st.dataframe(top_genes_pc1, width='stretch')
                            
                            with col2:
                                loadings_pc2 = adata.varm['PCs'][:, pc2-1]
                                top_genes_pc2_idx = np.argsort(np.abs(loadings_pc2))[-10:][::-1]
                                top_genes_pc2 = pd.DataFrame({
                                    'Gene': adata.var_names[top_genes_pc2_idx],
                                    'Loading': loadings_pc2[top_genes_pc2_idx]
                                })
                                st.write(f"**Top genes for PC{pc2}:**")
                                st.dataframe(top_genes_pc2, width='stretch')
                    
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
            
            # Custom color selection map for both 2D (static) and 3D (interactive)
            group_color_map = {}
            custom_color_map = None
            
            if umap_color_by != 'None':
                if not (pd.api.types.is_numeric_dtype(adata.obs[umap_color_by]) and len(adata.obs[umap_color_by].unique()) > 50):
                    unique_groups = adata.obs[umap_color_by].unique()
                    st.write(f"**Select colors for each group in '{umap_color_by}'**")
                    cols = st.columns(min(len(unique_groups), 4))
                    for i, group in enumerate(unique_groups):
                        with cols[i % 4]:
                            default_idx = list(unique_groups).index(group) % len(COLOR_OPTIONS)
                            color = st.selectbox(f"{group}", COLOR_OPTIONS, index=default_idx, key=f"umap_color_{group}")
                            group_color_map[group] = color
                    
                    # For 3D plot
                    custom_color_map = create_custom_color_map(adata.obs[umap_color_by], list(group_color_map.values()))
            
            # 2D UMAP - Static Publication Quality
            set_publication_style()
            # Landscape, bigger size, high DPI (600+)
            fig_umap_2d, ax = plt.subplots(figsize=(16, 10), dpi=600)
            
            x = adata.obsm['X_umap'][:, 0]
            y = adata.obsm['X_umap'][:, 1]
            
            # Reduced point size and opacity for better visibility of overlapping groups
            point_size = 5
            alpha_val = 0.4

            if umap_color_by == 'None':
                ax.scatter(x, y, c='gray', s=point_size, alpha=alpha_val, edgecolors='none')
            else:
                if pd.api.types.is_numeric_dtype(adata.obs[umap_color_by]) and len(adata.obs[umap_color_by].unique()) > 50:
                    sc_plot = ax.scatter(x, y, c=adata.obs[umap_color_by], cmap='viridis', 
                                       s=point_size, alpha=alpha_val, edgecolors='none')
                    plt.colorbar(sc_plot, ax=ax, label=umap_color_by)
                else:
                    sns.scatterplot(x=x, y=y, hue=adata.obs[umap_color_by], 
                                  palette=group_color_map, ax=ax, s=point_size, 
                                  edgecolor='none', alpha=alpha_val)
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title=umap_color_by)
            
            ax.set_xlabel('UMAP1', fontweight='bold')
            ax.set_ylabel('UMAP2', fontweight='bold')
            ax.set_title('2D UMAP Projection', fontweight='bold')
            ax.grid(False) # Clean look for UMAP
            
            st.pyplot(fig_umap_2d, use_container_width=True)
            st.caption("**Summary:** UMAP preserves local structure better than t-SNE and is widely used for visualizing cell clusters in single-cell data.")
            
            st.subheader("Interactive 2D UMAP")
            fig_umap_2d_int = Plotting.plot_embedding(
                adata, basis='umap', color=umap_color_by if umap_color_by != 'None' else None,
                use_3d=False, custom_color_map=custom_color_map,
                marker_opacity=0.4, marker_size=3
            )
            fig_umap_2d_int = apply_plotly_publication_style(fig_umap_2d_int)
            fig_umap_2d_int.update_layout(height=700)
            st.plotly_chart(fig_umap_2d_int, use_container_width=True)
            
            # 3D UMAP if available
            if adata.obsm['X_umap'].shape[1] >= 3:
                st.subheader("Interactive 3D UMAP")
                fig_umap_3d = Plotting.plot_embedding(
                    adata, basis='umap', color=umap_color_by if umap_color_by != 'None' else None,
                    use_3d=True, custom_color_map=custom_color_map,
                    marker_opacity=0.4, marker_size=2
                )
                fig_umap_3d = apply_plotly_publication_style(fig_umap_3d)
                fig_umap_3d.update_layout(
                    title="3D UMAP Projection",
                    height=800,
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
            
            group_color_map = {}
            custom_color_map = None
            if tsne_color_by != 'None':
                if not (pd.api.types.is_numeric_dtype(adata.obs[tsne_color_by]) and len(adata.obs[tsne_color_by].unique()) > 50):
                    unique_groups = adata.obs[tsne_color_by].unique()
                    st.write(f"**Select colors for each group in '{tsne_color_by}'**")
                    cols = st.columns(min(len(unique_groups), 4))
                    for i, group in enumerate(unique_groups):
                        with cols[i % 4]:
                            default_idx = list(unique_groups).index(group) % len(COLOR_OPTIONS)
                            color = st.selectbox(f"{group}", COLOR_OPTIONS, index=default_idx, key=f"tsne_color_{group}")
                            group_color_map[group] = color
                    custom_color_map = create_custom_color_map(adata.obs[tsne_color_by], list(group_color_map.values()))
            
            # Static t-SNE Plot
            set_publication_style()
            # Increased figure size
            fig_tsne, ax = plt.subplots(figsize=(12, 12))
            
            x = adata.obsm['X_tsne'][:, 0]
            y = adata.obsm['X_tsne'][:, 1]
            
            # Reduced point size and opacity
            point_size = 5
            alpha_val = 0.4
            
            if tsne_color_by == 'None':
                ax.scatter(x, y, c='gray', s=point_size, alpha=alpha_val, edgecolors='none')
            else:
                if pd.api.types.is_numeric_dtype(adata.obs[tsne_color_by]) and len(adata.obs[tsne_color_by].unique()) > 50:
                    sc_plot = ax.scatter(x, y, c=adata.obs[tsne_color_by], cmap='viridis', 
                                       s=point_size, alpha=alpha_val, edgecolors='none')
                    plt.colorbar(sc_plot, ax=ax, label=tsne_color_by)
                else:
                    sns.scatterplot(x=x, y=y, hue=adata.obs[tsne_color_by], 
                                  palette=group_color_map, ax=ax, s=point_size, 
                                  edgecolor='none', alpha=alpha_val)
                    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title=tsne_color_by)
            
            ax.set_xlabel('t-SNE1', fontweight='bold')
            ax.set_ylabel('t-SNE2', fontweight='bold')
            ax.set_title('t-SNE Projection', fontweight='bold')
            ax.grid(False)
            
            st.pyplot(fig_tsne)
            st.caption("**Summary:** t-SNE excels at revealing local structure and distinct clusters but may distort global distances.")
            
            st.subheader("Interactive t-SNE")
            fig_tsne_int = Plotting.plot_embedding(
                adata, basis='tsne', color=tsne_color_by if tsne_color_by != 'None' else None,
                use_3d=False, custom_color_map=custom_color_map,
                marker_opacity=0.4, marker_size=3
            )
            fig_tsne_int = apply_plotly_publication_style(fig_tsne_int)
            st.plotly_chart(fig_tsne_int, use_container_width=True)
            
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