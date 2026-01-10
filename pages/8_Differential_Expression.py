import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
from utils.de import DifferentialExpression
from utils.plotting import Plotting
from utils.workspace_manager import WorkspaceManager

def main():
    st.title("Differential Expression")
    st.markdown("Identify marker genes and perform differential expression analysis between clusters.")
    
    # Check if clustering is done
    if st.session_state.get('adata') is None:
        st.error("Please complete clustering first!")
        st.stop()
    
    adata = st.session_state.adata
    workspace_manager = st.session_state.workspace_manager
    
    st.info(f"Current dataset: **{adata.n_obs} cells** × **{adata.n_vars} genes**")
    
    # Get clustering results
    cluster_columns = [col for col in adata.obs.columns if any(x in col for x in ['leiden', 'louvain', 'cluster'])]
    
    if not cluster_columns:
        st.error("No clustering results found. Please run clustering first.")
        st.stop()
    
    # Sidebar for DE parameters
    st.sidebar.header("Differential Expression Parameters")
    
    # Select clustering result
    cluster_key = st.sidebar.selectbox(
        "Clustering result",
        options=cluster_columns,
        index=0
    )
    
    # DE method
    de_method = st.sidebar.selectbox(
        "DE Test Method",
        options=['t-test', 'wilcoxon', 'logreg'],
        index=1,
        help="Wilcoxon rank-sum test is recommended for most cases"
    )
    
    # Multiple testing correction
    corr_method = st.sidebar.selectbox(
        "Multiple Testing Correction",
        options=['benjamini-hochberg', 'bonferroni'],
        index=0
    )
    
    # Fold change threshold
    log2fc_threshold = st.sidebar.slider(
        "log2 Fold Change Threshold",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1
    )
    
    # P-value threshold
    pval_threshold = st.sidebar.slider(
        "P-value Threshold",
        min_value=0.001,
        max_value=0.05,
        value=0.05,
        step=0.001,
        format="%.3f"
    )
    
    # Number of top genes
    n_top_genes = st.sidebar.slider(
        "Number of top genes per cluster",
        min_value=5,
        max_value=50,
        value=10
    )
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["DE Analysis", "Results", "Visualization", "Marker Genes"])
    
    with tab1:
        st.header("Differential Expression Analysis")
        
        # Get cluster information
        clusters = sorted(adata.obs[cluster_key].unique())
        
        st.subheader("Comparison Setup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            comparison_type = st.radio(
                "Comparison type",
                options=['cluster_vs_rest', 'cluster_vs_cluster'],
                index=0
            )
        
        with col2:
            if comparison_type == 'cluster_vs_rest':
                selected_cluster = st.selectbox(
                    "Select cluster to compare",
                    options=clusters
                )
                reference = 'rest'
            else:
                col1, col2 = st.columns(2)
                with col1:
                    cluster1 = st.selectbox("Cluster 1", clusters, key="cluster1")
                with col2:
                    cluster2 = st.selectbox("Cluster 2", 
                                          [c for c in clusters if c != cluster1], 
                                          key="cluster2")
                selected_cluster = cluster1
                reference = cluster2
        
        # Run DE analysis
        if st.button("Run Differential Expression", type="primary"):
            with st.spinner("Running differential expression analysis..."):
                try:
                    if comparison_type == 'cluster_vs_rest':
                        de_results = DifferentialExpression.cluster_vs_rest(
                            adata,
                            cluster_key=cluster_key,
                            cluster=selected_cluster,
                            method=de_method,
                            corr_method=corr_method
                        )
                    else:
                        de_results = DifferentialExpression.cluster_vs_cluster(
                            adata,
                            cluster_key=cluster_key,
                            cluster1=cluster1,
                            cluster2=cluster2,
                            method=de_method,
                            corr_method=corr_method
                        )
                    
                    # Store results in session state
                    st.session_state.de_results = de_results
                    st.session_state.de_params = {
                        'cluster_key': cluster_key,
                        'selected_cluster': selected_cluster,
                        'reference': reference,
                        'method': de_method,
                        'comparison_type': comparison_type
                    }
                    
                    st.success(f"Differential expression analysis completed! Found {len(de_results)} significant genes.")
                    
                    # Update workspace
                    workspace_manager.update_workspace_metadata(
                        st.session_state.workspace,
                        {
                            "steps_completed": ["data_upload", "quality_control", "preprocessing", "dim_reduction", "clustering", "de"],
                            "de_parameters": {
                                "cluster_key": cluster_key,
                                "selected_cluster": selected_cluster,
                                "reference": reference,
                                "method": de_method,
                                "corr_method": corr_method,
                                "comparison_type": comparison_type
                            }
                        }
                    )
                    
                except Exception as e:
                    st.error(f"Error during differential expression analysis: {str(e)}")
        else:
            st.info("Configure the comparison above and click 'Run Differential Expression'")
    
    with tab2:
        st.header("Differential Expression Results")
        
        if 'de_results' not in st.session_state:
            st.info("Run differential expression analysis first")
            st.stop()
        
        de_results = st.session_state.de_results
        de_params = st.session_state.de_params
        
        # Filter results
        significant_results = de_results[
            (de_results['logfoldchanges'].abs() >= log2fc_threshold) & 
            (de_results['pvals'] <= pval_threshold)
        ]
        
        st.subheader("Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Genes", len(de_results))
        with col2:
            st.metric("Significant Genes", len(significant_results))
        with col3:
            st.metric("Up-regulated", len(significant_results[significant_results['logfoldchanges'] > 0]))
        with col4:
            st.metric("Down-regulated", len(significant_results[significant_results['logfoldchanges'] < 0]))
        
        # Volcano plot
        st.subheader("Volcano Plot")
        fig_volcano = Plotting.plot_volcano(de_results, log2fc_threshold, pval_threshold)
        st.plotly_chart(fig_volcano, use_container_width=True)
        
        # Results table
        st.subheader("Differential Expression Results")
        
        # Sort results
        sort_by = st.selectbox("Sort by", ['pvals', 'logfoldchanges', 'scores'], index=0)
        sort_ascending = sort_by == 'pvals'  # p-values should be ascending, others descending
        
        sorted_results = significant_results.sort_values(sort_by, ascending=sort_ascending)
        
        # Display table
        st.dataframe(sorted_results.head(100), use_container_width=True)
        
        # Download results
        csv = sorted_results.to_csv(index=False)
        st.download_button(
            label="Download DE Results (CSV)",
            data=csv,
            file_name=f"de_results_{de_params['selected_cluster']}_vs_{de_params['reference']}.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.header("Gene Expression Visualization")
        
        if 'de_results' not in st.session_state:
            st.info("Run differential expression analysis first")
            st.stop()
        
        de_results = st.session_state.de_results
        de_params = st.session_state.de_params
        
        # Select top genes for visualization
        top_genes = de_results.nlargest(n_top_genes, 'scores')['names'].tolist()
        
        st.subheader(f"Top {n_top_genes} Marker Genes")
        
        # Gene selection
        selected_gene = st.selectbox("Select gene to visualize", options=top_genes)
        
        if selected_gene:
            # Visualization options
            col1, col2 = st.columns(2)
            
            with col1:
                plot_type = st.selectbox("Plot type", ['violin', 'umap', 'dotplot'])
            
            with col2:
                if plot_type == 'umap':
                    embedding = st.selectbox("Embedding", ['umap', 'tsne', 'pca'])
            
            # Generate plot
            if plot_type == 'violin':
                fig = DifferentialExpression.plot_gene_violin(adata, selected_gene, de_params['cluster_key'])
                st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == 'umap':
                fig_2d = Plotting.plot_gene_expression(adata, selected_gene, basis=embedding)
                st.plotly_chart(fig_2d, use_container_width=True)
                
                # 3D if available
                if f'X_{embedding}' in adata.obsm and adata.obsm[f'X_{embedding}'].shape[1] >= 3:
                    fig_3d = Plotting.plot_gene_expression(adata, selected_gene, basis=embedding, use_3d=True)
                    st.plotly_chart(fig_3d, use_container_width=True)
            
            elif plot_type == 'dotplot':
                # Use top genes for dotplot
                genes_for_dotplot = top_genes[:min(10, len(top_genes))]
                fig = DifferentialExpression.plot_dotplot(adata, genes_for_dotplot, de_params['cluster_key'])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Marker Gene Analysis")
        
        if 'de_results' not in st.session_state:
            st.info("Run differential expression analysis first")
            st.stop()
        
        de_results = st.session_state.de_results
        
        # Find markers for all clusters
        if st.button("Find Markers for All Clusters"):
            with st.spinner("Finding marker genes for all clusters..."):
                try:
                    all_markers = DifferentialExpression.find_all_markers(
                        adata,
                        cluster_key=cluster_key,
                        method=de_method,
                        n_genes=n_top_genes
                    )
                    
                    st.session_state.all_markers = all_markers
                    st.success("Marker analysis completed for all clusters!")
                    
                except Exception as e:
                    st.error(f"Error in marker analysis: {str(e)}")
        
        # Display all markers if available
        if 'all_markers' in st.session_state:
            all_markers = st.session_state.all_markers
            
            st.subheader("Top Marker Genes per Cluster")
            
            # Create a table of top markers
            marker_table = []
            for cluster in sorted(all_markers.keys()):
                cluster_markers = all_markers[cluster].head(5)
                for _, row in cluster_markers.iterrows():
                    marker_table.append({
                        'Cluster': cluster,
                        'Gene': row['names'],
                        'log2FC': f"{row['logfoldchanges']:.2f}",
                        'P-value': f"{row['pvals']:.2e}",
                        'Score': f"{row['scores']:.2f}"
                    })
            
            marker_df = pd.DataFrame(marker_table)
            st.dataframe(marker_df, use_container_width=True)
            
            # Heatmap of top markers
            st.subheader("Marker Gene Heatmap")
            
            if st.button("Generate Marker Heatmap"):
                with st.spinner("Generating heatmap..."):
                    try:
                        # Get top markers for heatmap
                        heatmap_genes = []
                        for cluster in sorted(all_markers.keys()):
                            top_genes = all_markers[cluster].head(3)['names'].tolist()
                            heatmap_genes.extend(top_genes)
                        
                        # Remove duplicates
                        heatmap_genes = list(dict.fromkeys(heatmap_genes))
                        
                        # Create heatmap
                        fig = DifferentialExpression.plot_marker_heatmap(adata, heatmap_genes, cluster_key)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error generating heatmap: {str(e)}")

if __name__ == "__main__":
    main()