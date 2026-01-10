import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
from utils.clustering import Clustering
from utils.plotting import Plotting
from utils.workspace_manager import WorkspaceManager

def main():
    st.title("Clustering")
    st.markdown("Identify cell clusters using Leiden or Louvain algorithms.")
    
    # Check if dimensionality reduction is done
    if st.session_state.get('adata') is None:
        st.error("Please complete dimensionality reduction first!")
        st.stop()
    
    adata = st.session_state.adata
    workspace_manager = st.session_state.workspace_manager
    
    st.info(f"Current dataset: **{adata.n_obs} cells** × **{adata.n_vars} genes**")
    
    # Sidebar for clustering parameters
    st.sidebar.header("Clustering Parameters")
    
    # Clustering method
    clustering_method = st.sidebar.selectbox(
        "Clustering Algorithm",
        options=['leiden', 'louvain'],
        index=0
    )
    
    # Resolution parameter
    resolution = st.sidebar.slider(
        "Resolution",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher resolution finds more clusters"
    )
    
    # Key added
    cluster_key = st.sidebar.text_input(
        "Cluster key",
        value=f"{clustering_method}_{resolution}",
        help="Key to store clustering results in adata.obs"
    )
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["Clustering", "Results", "Comparison"])
    
    with tab1:
        st.header("Cell Clustering")
        
        # Check if neighbors are computed
        if 'neighbors' not in adata.uns:
            st.warning("Neighborhood graph not found. Computing neighbors first...")
            sc.pp.neighbors(adata)
            st.session_state.adata = adata
        
        # Run clustering
        if st.button("Run Clustering", type="primary"):
            with st.spinner("Running clustering algorithm..."):
                try:
                    if clustering_method == 'leiden':
                        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
                    else:  # louvain
                        sc.tl.louvain(adata, resolution=resolution, key_added=cluster_key)
                    
                    st.success(f"Clustering completed! Found {adata.obs[cluster_key].nunique()} clusters.")
                    
                    # Update session state
                    st.session_state.adata = adata
                    
                    # Update workspace
                    workspace_manager.update_workspace_metadata(
                        st.session_state.workspace,
                        {
                            "steps_completed": ["data_upload", "quality_control", "preprocessing", "dim_reduction", "clustering"],
                            "clustering_parameters": {
                                "method": clustering_method,
                                "resolution": resolution,
                                "cluster_key": cluster_key
                            }
                        }
                    )
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during clustering: {str(e)}")
        else:
            st.info("Click 'Run Clustering' to identify cell clusters")
            
            # Show current clustering status
            existing_clusters = [col for col in adata.obs.columns if 'leiden' in col or 'louvain' in col]
            if existing_clusters:
                st.subheader("Existing Clustering Results")
                for cluster_col in existing_clusters:
                    n_clusters = adata.obs[cluster_col].nunique()
                    st.write(f"• **{cluster_col}**: {n_clusters} clusters")
    
    with tab2:
        st.header("Clustering Results")
        
        # Check if clustering has been run
        cluster_columns = [col for col in adata.obs.columns if 'leiden' in col or 'louvain' in col]
        
        if not cluster_columns:
            st.info("Run clustering first to see results")
            st.stop()
        
        # Select which clustering result to display
        selected_cluster = st.selectbox(
            "Select clustering result to display",
            options=cluster_columns,
            index=0
        )
        
        # Cluster statistics
        cluster_counts = adata.obs[selected_cluster].value_counts().sort_index()
        n_clusters = len(cluster_counts)
        
        st.subheader(f"Cluster Distribution ({n_clusters} clusters)")
        
        # Display cluster sizes
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Clusters", n_clusters)
        with col2:
            st.metric("Largest Cluster", f"{cluster_counts.max():,} cells")
        with col3:
            st.metric("Smallest Cluster", f"{cluster_counts.min():,} cells")
        
        # Cluster size table
        st.subheader("Cluster Sizes")
        cluster_df = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Cells': cluster_counts.values,
            'Percentage': (cluster_counts.values / adata.n_obs * 100).round(2)
        })
        st.dataframe(cluster_df, use_container_width=True)
        
        # Visualization
        st.subheader("Cluster Visualization")
        
        # Select embedding for visualization
        embedding_options = []
        if 'X_umap' in adata.obsm:
            embedding_options.append('umap')
        if 'X_tsne' in adata.obsm:
            embedding_options.append('tsne')
        if 'X_pca' in adata.obsm:
            embedding_options.append('pca')
        
        if embedding_options:
            selected_embedding = st.selectbox(
                "Select embedding for visualization",
                options=embedding_options,
                index=0
            )
            
            # 2D plot
            col1, col2 = st.columns(2)
            
            with col1:
                fig_2d = Plotting.plot_embedding(adata, basis=selected_embedding, color=selected_cluster)
                st.plotly_chart(fig_2d, use_container_width=True)
                st.caption(f"2D {selected_embedding.upper()} colored by {selected_cluster}")
            
            with col2:
                # 3D plot if available
                if f'X_{selected_embedding}' in adata.obsm and adata.obsm[f'X_{selected_embedding}'].shape[1] >= 3:
                    fig_3d = Plotting.plot_embedding(adata, basis=selected_embedding, color=selected_cluster, use_3d=True)
                    st.plotly_chart(fig_3d, use_container_width=True)
                    st.caption(f"3D {selected_embedding.upper()} colored by {selected_cluster}")
                else:
                    st.info(f"3D {selected_embedding.upper()} not available")
        
        # Cluster quality metrics
        st.subheader("Cluster Quality")
        
        # Compute silhouette score if possible
        try:
            from sklearn.metrics import silhouette_score
            if 'X_pca' in adata.obsm:
                silhouette_avg = silhouette_score(adata.obsm['X_pca'][:, :10], adata.obs[selected_cluster])
                st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
        except:
            st.info("Silhouette score computation requires scikit-learn")
    
    with tab3:
        st.header("Cluster Comparison")
        
        cluster_columns = [col for col in adata.obs.columns if 'leiden' in col or 'louvain' in col]
        
        if len(cluster_columns) < 2:
            st.info("Run clustering with different parameters to compare results")
            st.stop()
        
        # Select two clustering results to compare
        col1, col2 = st.columns(2)
        
        with col1:
            cluster1 = st.selectbox("First clustering", cluster_columns, key="cluster1")
        with col2:
            cluster2 = st.selectbox("Second clustering", 
                                  [c for c in cluster_columns if c != cluster1], 
                                  key="cluster2")
        
        if cluster1 and cluster2:
            # Create contingency table
            contingency = pd.crosstab(adata.obs[cluster1], adata.obs[cluster2])
            
            st.subheader("Cluster Correspondence")
            st.dataframe(contingency, use_container_width=True)
            
            # Compute adjusted rand index
            try:
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(adata.obs[cluster1], adata.obs[cluster2])
                st.metric("Adjusted Rand Index", f"{ari:.3f}")
            except:
                st.info("ARI computation requires scikit-learn")
        
        # Next steps
        st.divider()
        st.success("Clustering completed! Proceed to **Differential Expression** to find marker genes.")
        
        if st.button("Go to Differential Expression →", use_container_width=True):
            st.switch_page("pages/8_Differential_Expression.py")

if __name__ == "__main__":
    main()