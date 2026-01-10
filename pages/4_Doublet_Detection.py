import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.doublet_detection import DoubletDetection
from utils.plotting import Plotting
from utils.workspace_manager import WorkspaceManager

# Configure matplotlib for publication-quality white background
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'text.color': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': '#e0e0e0',
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'font.size': 11,
    'axes.titleweight': 'bold'
})

def main():
    st.title("👥 Doublet Detection")
    st.markdown("Identify and remove technical doublets using Scanpy's doublet detection methods.")
    
    # Check if data is loaded
    if st.session_state.get('adata') is None:
        st.error("Please load your data first!")
        st.stop()
    
    adata = st.session_state.adata
    
    # Save current state for undo functionality
    if 'adata_history' not in st.session_state:
        st.session_state.adata_history = []
    
    if st.button("💾 Save Current State", key="save_state"):
        st.session_state.adata_history.append(adata.copy())
        st.success("State saved! You can undo changes later.")
    
    st.info(f"Current dataset: **{adata.n_obs} cells** × **{adata.n_vars} genes**")
    
    # Doublet detection parameters
    st.sidebar.header("Doublet Detection Parameters")
    
    expected_doublet_rate = st.sidebar.slider(
        "Expected Doublet Rate", 
        min_value=0.01, max_value=0.2, value=0.05, step=0.01,
        help="Expected fraction of transcriptomes that are doublets"
    )
    
    n_neighbors = st.sidebar.slider(
        "Number of Neighbors",
        min_value=5, max_value=100, value=50, step=5,
        help="Number of neighbors used for doublet detection"
    )
    
    sim_doublet_ratio = st.sidebar.slider(
        "Simulated Doublet Ratio",
        min_value=1, max_value=5, value=2, step=1,
        help="Number of doublets to simulate relative to the number of observed transcriptomes"
    )
    
    # Doublet detection workflow
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Detect Doublets", "📊 Results", "📈 Visualization", "⚙️ Filtering"])
    
    with tab1:
        st.header("Doublet Detection")
        
        if st.button("🚀 Run Doublet Detection", type="primary"):
            with st.spinner("Detecting doublets..."):
                try:
                    adata_with_doublets = DoubletDetection.detect_doublets_scanpy(
                        adata,
                        expected_doublet_rate=expected_doublet_rate,
                        n_neighbors=n_neighbors,
                        sim_doublet_ratio=sim_doublet_ratio
                    )
                    
                    st.session_state.adata = adata_with_doublets
                    st.success("Doublet detection completed!")
                    
                except Exception as e:
                    st.error(f"Error during doublet detection: {str(e)}")
        
        # Show current doublet status
        if 'doublet_score' in adata.obs:
            st.subheader("Current Doublet Status")
            n_doublets = adata.obs['predicted_doublets'].sum() if 'predicted_doublets' in adata.obs else 0
            pct_doublets = (n_doublets / adata.n_obs) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Doublets", n_doublets)
            with col2:
                st.metric("Doublet Rate", f"{pct_doublets:.1f}%")
            with col3:
                st.metric("Expected Rate", f"{expected_doublet_rate*100:.1f}%")
    
    with tab2:
        st.header("Doublet Detection Results")
        
        if 'doublet_score' not in adata.obs:
            st.info("Run doublet detection first to see results")
        else:
            # Doublet summary
            st.subheader("Doublet Summary")
            results_df = DoubletDetection.get_doublet_summary(adata)
            st.dataframe(results_df, use_container_width=True)
            
            # Score distribution
            st.subheader("Doublet Score Distribution")
            fig_dist = DoubletDetection.plot_doublet_score_distribution(adata)
            st.pyplot(fig_dist)
            st.caption("""
            **Interpretation**: This histogram shows the distribution of doublet scores across all cells. 
            A bimodal distribution (with a high-score peak) often indicates the presence of doublets. 
            The vertical dashed line marks the threshold used to classify cells as doublets.
            """)

            # Threshold analysis
            st.subheader("Threshold Analysis")
            fig_threshold = DoubletDetection.plot_threshold_analysis(adata)
            st.pyplot(fig_threshold)
            st.caption("""
            **Interpretation**: Left panel shows the fraction of cells exceeding a given score threshold. 
            Right panel shows how many cells would be called as doublets at different thresholds. 
            The current threshold (blue dashed line) is chosen to match the expected doublet rate.
            """)
    
    with tab3:
        st.header("Doublet Visualization")
        
        if 'doublet_score' not in adata.obs:
            st.info("Run doublet detection first to visualize results")
        else:
            # Interactive UMAP/tSNE with doublet highlighting
            st.subheader("Doublets on Embedding")
            
            embedding_options = []
            if 'X_umap' in adata.obsm:
                embedding_options.append('umap')
            if 'X_tsne' in adata.obsm:
                embedding_options.append('tsne')
            if 'X_pca' in adata.obsm:
                embedding_options.append('pca')
            
            if embedding_options:
                selected_embedding = st.selectbox("Select embedding", embedding_options)
                
                color_option = st.radio(
                    "Color by",
                    ['doublet_score', 'predicted_doublets'],
                    horizontal=True
                )
                
                fig = Plotting.plot_embedding(adata, basis=selected_embedding, color=color_option)
                st.plotly_chart(fig, use_container_width=True)
            
            # Doublet score vs QC metrics
            st.subheader("Doublet Score vs QC Metrics")
            metric_options = ['n_genes_by_counts', 'total_counts']
            if 'pct_counts_mt' in adata.obs:
                metric_options.append('pct_counts_mt')
            
            selected_metric = st.selectbox("Select QC metric", metric_options)
            
            fig_scatter = DoubletDetection.plot_doublet_vs_metric(adata, selected_metric)
            st.pyplot(fig_scatter)
            st.caption(f"""
            **Interpretation**: Doublets often have higher total counts and detected genes due to containing RNA from two cells. 
            Cells with high doublet scores that also show elevated **{selected_metric.replace('_', ' ').title()}** are more likely to be true doublets.
            """)
    
    with tab4:
        st.header("Doublet Filtering")
        
        if 'predicted_doublets' not in adata.obs:
            st.info("Run doublet detection first to filter doublets")
        else:
            n_doublets = adata.obs['predicted_doublets'].sum()
            
            st.subheader("Filter Doublets")
            st.write(f"**{n_doublets}** doublets detected ({n_doublets/adata.n_obs*100:.1f}% of cells)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Remove Doublets", type="primary"):
                    st.session_state.adata_history.append(adata.copy())
                    adata_filtered = adata[~adata.obs['predicted_doublets']].copy()
                    st.session_state.adata = adata_filtered
                    st.success(f"Removed {n_doublets} doublets. Remaining cells: {adata_filtered.n_obs:,}")
                    st.rerun()
            
            with col2:
                if st.session_state.adata_history:
                    if st.button("↩️ Undo Last Action"):
                        if st.session_state.adata_history:
                            st.session_state.adata = st.session_state.adata_history.pop()
                            st.success("Last action undone!")
                            st.rerun()
                else:
                    st.button("↩️ Undo Last Action", disabled=True)
            
            if st.checkbox("Show filtered data preview"):
                st.subheader("Filtered Data Preview")
                st.write(f"**Cells after filtering:** {adata[~adata.obs['predicted_doublets']].n_obs:,}")
                st.write(f"**Genes after filtering:** {adata[~adata.obs['predicted_doublets']].n_vars:,}")
    
    # Next steps
    st.divider()
    st.subheader("🎯 Next Steps")
    st.info("""
    After doublet detection and filtering, proceed to **Filtering** to remove low-quality cells 
    based on QC metrics, then continue to **Batch Correction** if you have multiple batches.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Quality Control"):
            st.switch_page("pages/3_Quality_Control.py")
    with col2:
        if st.button("Proceed to Filtering →"):
            st.switch_page("pages/5_Preprocessing.py")

if __name__ == "__main__":
    main()