import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
from utils.plotting import Plotting
from utils.workspace_manager import WorkspaceManager

def main():
    st.title("Advanced Visualization")
    st.markdown("Explore your single-cell data with interactive visualizations and custom plots.")
    
    # Check if data is loaded
    if st.session_state.get('adata') is None:
        st.error("Please load your data first!")
        st.stop()
    
    adata = st.session_state.adata
    
    st.info(f"Current dataset: **{adata.n_obs} cells** × **{adata.n_vars} genes**")
    
    # Sidebar for visualization options
    st.sidebar.header("Visualization Settings")
    
    # Available embeddings
    embeddings = []
    if 'X_umap' in adata.obsm:
        embeddings.append('umap')
    if 'X_tsne' in adata.obsm:
        embeddings.append('tsne')
    if 'X_pca' in adata.obsm:
        embeddings.append('pca')
    
    selected_embedding = st.sidebar.selectbox(
        "Select embedding",
        options=embeddings,
        index=0
    )
    
    # 3D option
    use_3d = st.sidebar.checkbox("3D visualization", value=False)
    
    # Color options
    color_options = ['None'] + list(adata.obs.columns)
    selected_color = st.sidebar.selectbox(
        "Color by",
        options=color_options
    )
    
    # Gene expression
    st.sidebar.subheader("Gene Expression")
    gene_input = st.sidebar.text_input(
        "Enter gene name",
        placeholder="e.g., CD3E, MS4A1"
    )
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Embeddings", "Gene Expression", "Multi-Gene", "Custom Plots"])
    
    with tab1:
        st.header("Embedding Visualizations")
        
        if not embeddings:
            st.error("No embeddings found. Please run dimensionality reduction first.")
            st.stop()
        
        # Embedding plot
        if selected_color != 'None':
            fig_embedding = Plotting.plot_embedding(
                adata, 
                basis=selected_embedding, 
                color=selected_color,
                use_3d=use_3d
            )
        else:
            fig_embedding = Plotting.plot_embedding(
                adata,
                basis=selected_embedding,
                use_3d=use_3d
            )
        
        st.plotly_chart(fig_embedding, use_container_width=True)
        
        # Embedding statistics
        st.subheader("Embedding Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Embedding", selected_embedding.upper())
        with col2:
            st.metric("Dimensions", "3D" if use_3d else "2D")
        with col3:
            if selected_color != 'None':
                st.metric("Colored by", selected_color)
            else:
                st.metric("Color", "Uniform")
        
        # Multiple embeddings comparison
        if len(embeddings) > 1:
            st.subheader("Compare Embeddings")
            
            comparison_embedding = st.selectbox(
                "Compare with",
                [e for e in embeddings if e != selected_embedding]
            )
            
            if comparison_embedding:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig1 = Plotting.plot_embedding(adata, basis=selected_embedding, color=selected_color)
                    st.plotly_chart(fig1, use_container_width=True)
                    st.caption(f"{selected_embedding.upper()}")
                
                with col2:
                    fig2 = Plotting.plot_embedding(adata, basis=comparison_embedding, color=selected_color)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption(f"{comparison_embedding.upper()}")
    
    with tab2:
        st.header("Gene Expression Visualization")
        
        if gene_input:
            genes = [g.strip() for g in gene_input.split(',')]
            valid_genes = [g for g in genes if g in adata.var_names]
            
            if not valid_genes:
                st.error(f"None of the specified genes found in dataset. Available genes: {len(adata.var_names):,}")
                st.stop()
            
            # Show available genes info
            st.success(f"Found {len(valid_genes)} gene(s): {', '.join(valid_genes)}")
            
            # Single gene visualization
            if len(valid_genes) == 1:
                gene = valid_genes[0]
                
                st.subheader(f"Expression of {gene}")
                
                # Expression statistics
                if adata.raw is not None and gene in adata.raw.var_names:
                    expr = adata.raw[:, gene].X.flatten()
                else:
                    expr = adata[:, gene].X.flatten()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Expression", f"{np.mean(expr):.3f}")
                with col2:
                    st.metric("Max Expression", f"{np.max(expr):.3f}")
                with col3:
                    st.metric("Detection Rate", f"{(expr > 0).sum() / len(expr):.2%}")
                with col4:
                    st.metric("Gene", gene)
                
                # Visualization options
                viz_method = st.radio(
                    "Visualization method",
                    options=['Embedding', 'Violin Plot', 'Both'],
                    horizontal=True
                )
                
                if viz_method in ['Embedding', 'Both']:
                    # Gene expression on embedding
                    fig_gene = Plotting.plot_gene_expression(
                        adata, gene, basis=selected_embedding, use_3d=use_3d
                    )
                    st.plotly_chart(fig_gene, use_container_width=True)
                
                if viz_method in ['Violin Plot', 'Both']:
                    # Violin plot by cluster if available
                    cluster_cols = [col for col in adata.obs.columns if any(x in col for x in ['leiden', 'louvain'])]
                    if cluster_cols:
                        cluster_key = st.selectbox("Group by", cluster_cols)
                        
                        # Create violin plot
                        import plotly.express as px
                        plot_df = pd.DataFrame({
                            'Expression': expr,
                            'Cluster': adata.obs[cluster_key]
                        })
                        
                        fig_violin = px.violin(
                            plot_df, 
                            x='Cluster', 
                            y='Expression',
                            title=f'{gene} Expression by Cluster',
                            box=True
                        )
                        st.plotly_chart(fig_violin, use_container_width=True)
            
            # Multiple genes
            else:
                st.subheader(f"Multiple Genes: {', '.join(valid_genes)}")
                
                # Dot plot if clusters available
                cluster_cols = [col for col in adata.obs.columns if any(x in col for x in ['leiden', 'louvain'])]
                if cluster_cols:
                    cluster_key = st.selectbox("Group by", cluster_cols, key="multi_gene_cluster")
                    
                    # Create dot plot
                    import plotly.graph_objects as go
                    
                    # Calculate mean expression per cluster
                    mean_expr = []
                    frac_cells = []
                    
                    for gene in valid_genes:
                        if adata.raw is not None and gene in adata.raw.var_names:
                            expr = adata.raw[:, gene].X
                        else:
                            expr = adata[:, gene].X
                        
                        for cluster in sorted(adata.obs[cluster_key].unique()):
                            cluster_mask = adata.obs[cluster_key] == cluster
                            cluster_expr = expr[cluster_mask].flatten()
                            
                            mean_expr.append(np.mean(cluster_expr))
                            frac_cells.append((cluster_expr > 0).mean())
                    
                    # Create dot plot data
                    clusters = sorted(adata.obs[cluster_key].unique())
                    n_genes = len(valid_genes)
                    n_clusters = len(clusters)
                    
                    fig = go.Figure()
                    
                    # Add dots
                    for i, gene in enumerate(valid_genes):
                        for j, cluster in enumerate(clusters):
                            idx = i * n_clusters + j
                            fig.add_trace(go.Scatter(
                                x=[cluster],
                                y=[gene],
                                marker=dict(
                                    size=frac_cells[idx] * 20 + 5,
                                    color=mean_expr[idx],
                                    colorscale='Viridis',
                                    showscale=True,
                                    colorbar=dict(title="Mean Expression")
                                ),
                                mode='markers',
                                hovertemplate=f"Cluster: {cluster}<br>Gene: {gene}<br>Mean Expression: {mean_expr[idx]:.3f}<br>Fraction Cells: {frac_cells[idx]:.2f}<extra></extra>"
                            ))
                    
                    fig.update_layout(
                        title="Gene Expression Dot Plot",
                        xaxis_title="Cluster",
                        yaxis_title="Gene",
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Enter a gene name in the sidebar to visualize its expression")
    
    with tab3:
        st.header("Multi-Gene Analysis")
        
        # Gene set analysis
        st.subheader("Gene Set Visualization")
        
        # Predefined gene sets
        predefined_sets = {
            "T Cell Markers": ["CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B"],
            "B Cell Markers": ["CD19", "CD79A", "CD79B", "MS4A1"],
            "NK Cell Markers": ["NKG7", "GNLY", "KLRD1", "KLRF1"],
            "Monocyte Markers": ["CD14", "FCGR3A", "FCER1G", "CST3"],
            "Stem Cell Markers": ["PROM1", "THY1", "KIT", "SOX2"],
            "Custom": []
        }
        
        gene_set_choice = st.selectbox("Select gene set", list(predefined_sets.keys()))
        
        if gene_set_choice == "Custom":
            custom_genes = st.text_area(
                "Enter custom gene set (one gene per line)",
                placeholder="CD3E\nCD4\nCD8A\n..."
            )
            if custom_genes:
                gene_set = [g.strip() for g in custom_genes.split('\n') if g.strip()]
            else:
                gene_set = []
        else:
            gene_set = predefined_sets[gene_set_choice]
        
        # Filter to available genes
        available_genes = [g for g in gene_set if g in adata.var_names]
        missing_genes = [g for g in gene_set if g not in adata.var_names]
        
        if available_genes:
            st.success(f"Found {len(available_genes)} genes from set")
            if missing_genes:
                st.warning(f"Missing genes: {', '.join(missing_genes)}")
            
            # Calculate module score
            if st.button("Calculate Gene Module Score"):
                with st.spinner("Calculating module score..."):
                    try:
                        # Add module score to adata
                        sc.tl.score_genes(adata, available_genes, score_name='module_score')
                        
                        # Plot module score
                        fig_module = Plotting.plot_gene_expression(
                            adata, 'module_score', basis=selected_embedding, use_3d=use_3d
                        )
                        st.plotly_chart(fig_module, use_container_width=True)
                        
                        st.success("Module score calculated and visualized!")
                        
                    except Exception as e:
                        st.error(f"Error calculating module score: {str(e)}")
        
        else:
            st.info("Select a gene set or enter custom genes to analyze")
    
    with tab4:
        st.header("Custom Visualization")
        
        st.subheader("Interactive Plot Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # X-axis selection
            x_options = list(adata.obs.columns) + ['pseudotime']
            x_axis = st.selectbox("X-axis", x_options)
            
            # Plot type
            plot_type = st.selectbox(
                "Plot type",
                ['scatter', 'violin', 'box', 'histogram']
            )
        
        with col2:
            # Y-axis selection (for scatter/box/violin)
            if plot_type in ['scatter', 'violin', 'box']:
                y_options = list(adata.obs.columns)
                y_axis = st.selectbox("Y-axis", y_options)
            else:
                y_axis = None
            
            # Color by
            color_by = st.selectbox("Color by", ['None'] + list(adata.obs.columns))
        
        # Generate custom plot
        if st.button("Generate Custom Plot"):
            try:
                import plotly.express as px
                
                plot_data = adata.obs.copy()
                
                if plot_type == 'scatter':
                    if color_by != 'None':
                        fig = px.scatter(plot_data, x=x_axis, y=y_axis, color=color_by,
                                       title=f"{y_axis} vs {x_axis}")
                    else:
                        fig = px.scatter(plot_data, x=x_axis, y=y_axis,
                                       title=f"{y_axis} vs {x_axis}")
                
                elif plot_type == 'violin':
                    if color_by != 'None':
                        fig = px.violin(plot_data, x=x_axis, y=y_axis, color=color_by,
                                      title=f"{y_axis} by {x_axis}")
                    else:
                        fig = px.violin(plot_data, x=x_axis, y=y_axis,
                                      title=f"{y_axis} by {x_axis}")
                
                elif plot_type == 'box':
                    if color_by != 'None':
                        fig = px.box(plot_data, x=x_axis, y=y_axis, color=color_by,
                                   title=f"{y_axis} by {x_axis}")
                    else:
                        fig = px.box(plot_data, x=x_axis, y=y_axis,
                                   title=f"{y_axis} by {x_axis}")
                
                elif plot_type == 'histogram':
                    if color_by != 'None':
                        fig = px.histogram(plot_data, x=x_axis, color=color_by,
                                         title=f"Distribution of {x_axis}")
                    else:
                        fig = px.histogram(plot_data, x=x_axis,
                                         title=f"Distribution of {x_axis}")
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")
        
        # Download plot data
        st.subheader("Export Visualization Data")
        
        if st.button("Download Plot Data (CSV)"):
            plot_data = adata.obs.copy()
            
            # Add embedding coordinates if available
            if selected_embedding in adata.obsm:
                embedding_coords = adata.obsm[f'X_{selected_embedding}']
                if embedding_coords.shape[1] >= 2:
                    plot_data[f'{selected_embedding}_1'] = embedding_coords[:, 0]
                    plot_data[f'{selected_embedding}_2'] = embedding_coords[:, 1]
                if embedding_coords.shape[1] >= 3:
                    plot_data[f'{selected_embedding}_3'] = embedding_coords[:, 2]
            
            csv = plot_data.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="visualization_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()