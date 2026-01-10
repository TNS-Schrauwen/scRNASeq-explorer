import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
import os
import json
import shutil
import zipfile
from datetime import datetime
from io import BytesIO
from utils.workspace_manager import WorkspaceManager
from utils.plotting import Plotting
import plotly.io as pio


def main():
    st.title("💾 Export Results")
    st.markdown("Export your analysis results, plots, and reproducible code.")
    
    # Check if workspace is set
    if not st.session_state.get('workspace'):
        st.error("Please create or load a workspace first!")
        st.stop()
    
    if st.session_state.get('adata') is None:
        st.error("Please load and analyze your data first!")
        st.stop()
    
    adata = st.session_state.adata
    workspace_manager = st.session_state.workspace_manager
    workspace_path = st.session_state.workspace_path
    
    st.info(f"Current workspace: **{st.session_state.workspace}**")
    st.info(f"Current dataset: **{adata.n_obs} cells** × **{adata.n_vars} genes**")
    
    # Analysis summary
    st.header("Analysis Summary")
    
    # Get workspace metadata
    try:
        metadata = workspace_manager.get_workspace_metadata(st.session_state.workspace)
        steps_completed = metadata.get('steps_completed', [])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Workspace", st.session_state.workspace)
        with col2:
            st.metric("Cells", adata.n_obs)
        with col3:
            st.metric("Genes", adata.n_vars)
        with col4:
            st.metric("Steps Completed", len(steps_completed))
        
        # Steps completion
        st.subheader("Analysis Steps Completed")
        all_steps = ["data_upload", "quality_control", "preprocessing", "dim_reduction", "clustering", "de", "visualization"]
        
        for step in all_steps:
            status = "✅" if step in steps_completed else "❌"
            step_name = step.replace('_', ' ').title()
            st.write(f"{status} {step_name}")
    
    except Exception as e:
        st.error(f"Error loading workspace metadata: {str(e)}")
    
    # Export options
    st.header("Export Options")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Export", "📊 Plots Export", "📜 Reproducible Code", "📋 Analysis Report"])
    
    with tab1:
        st.subheader("Export Processed Data")
        
        export_formats = st.multiselect(
            "Select export formats",
            options=['AnnData (.h5ad)', 'CSV Matrix', 'Loom (.loom)', 'Metadata Table'],
            default=['AnnData (.h5ad)']
        )
        
        if st.button("Export Data", type="primary"):
            with st.spinner("Exporting data..."):
                try:
                    export_dir = os.path.join(workspace_path, "exports")
                    os.makedirs(export_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    for format in export_formats:
                        if format == 'AnnData (.h5ad)':
                            filename = f"processed_data_{timestamp}.h5ad"
                            filepath = os.path.join(export_dir, filename)
                            adata.write_h5ad(filepath)
                            st.success(f"Exported AnnData: {filename}")
                        
                        elif format == 'CSV Matrix':
                            # Export count matrix
                            filename = f"count_matrix_{timestamp}.csv"
                            filepath = os.path.join(export_dir, filename)
                            
                            if adata.raw is not None:
                                count_matrix = pd.DataFrame(
                                    adata.raw.X.toarray() if hasattr(adata.raw.X, 'toarray') else adata.raw.X,
                                    index=adata.obs_names,
                                    columns=adata.raw.var_names
                                )
                            else:
                                count_matrix = pd.DataFrame(
                                    adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
                                    index=adata.obs_names,
                                    columns=adata.var_names
                                )
                            
                            count_matrix.T.to_csv(filepath)  # Genes as rows, cells as columns
                            st.success(f"Exported count matrix: {filename}")
                        
                        elif format == 'Loom (.loom)':
                            filename = f"processed_data_{timestamp}.loom"
                            filepath = os.path.join(export_dir, filename)
                            adata.write_loom(filepath)
                            st.success(f"Exported Loom file: {filename}")
                        
                        elif format == 'Metadata Table':
                            filename = f"cell_metadata_{timestamp}.csv"
                            filepath = os.path.join(export_dir, filename)
                            adata.obs.to_csv(filepath)
                            st.success(f"Exported metadata: {filename}")
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error exporting data: {str(e)}")
        
        # Download individual files
        st.subheader("Quick Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Processed data
            if st.button("Download Processed AnnData"):
                try:
                    buffer = BytesIO()
                    adata.write_h5ad(buffer)
                    st.download_button(
                        label="Download .h5ad",
                        data=buffer.getvalue(),
                        file_name=f"processed_data_{st.session_state.workspace}.h5ad",
                        mime="application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"Error preparing download: {str(e)}")
        
        with col2:
            # Cell metadata
            if st.button("Download Cell Metadata"):
                try:
                    csv = adata.obs.to_csv()
                    st.download_button(
                        label="Download Metadata CSV",
                        data=csv,
                        file_name=f"cell_metadata_{st.session_state.workspace}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error preparing metadata download: {str(e)}")
    
    with tab2:
        st.subheader("Export Plots and Visualizations")
        
        # Available embeddings for export
        embeddings = []
        if 'X_umap' in adata.obsm:
            embeddings.append('umap')
        if 'X_tsne' in adata.obsm:
            embeddings.append('tsne')
        if 'X_pca' in adata.obsm:
            embeddings.append('pca')
        
        st.write("**Available embeddings for export:**")
        for emb in embeddings:
            st.write(f"• {emb.upper()}")
        
        # Plot export options
        export_plots = st.multiselect(
            "Select plots to export",
            options=['UMAP 2D', 'UMAP 3D', 't-SNE', 'PCA', 'Cluster Visualization', 'DE Results'],
            default=['UMAP 2D', 'Cluster Visualization']
        )
        
        export_format = st.selectbox(
            "Export format",
            options=['PNG', 'SVG', 'PDF', 'HTML'],
            index=0
        )
        
        if st.button("Export Plots"):
            with st.spinner("Generating and exporting plots..."):
                try:
                    plots_dir = os.path.join(workspace_path, "exports", "plots")
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # UMAP plots
                    if 'UMAP 2D' in export_plots and 'umap' in embeddings:
                        # Color by cluster if available
                        cluster_cols = [col for col in adata.obs.columns if any(x in col for x in ['leiden', 'louvain'])]
                        if cluster_cols:
                            fig = Plotting.plot_embedding(adata, basis='umap', color=cluster_cols[0])
                        else:
                            fig = Plotting.plot_embedding(adata, basis='umap')
                        
                        filename = f"umap_2d_{timestamp}.{export_format.lower()}"
                        filepath = os.path.join(plots_dir, filename)
                        pio.write_image(fig, filepath)
                        st.success(f"Exported UMAP 2D: {filename}")
                    
                    if 'UMAP 3D' in export_plots and 'umap' in embeddings and adata.obsm['X_umap'].shape[1] >= 3:
                        fig = Plotting.plot_embedding(adata, basis='umap', use_3d=True)
                        filename = f"umap_3d_{timestamp}.html"  # 3D plots are best as HTML
                        filepath = os.path.join(plots_dir, filename)
                        fig.write_html(filepath)
                        st.success(f"Exported UMAP 3D: {filename}")
                    
                    # Other embeddings
                    for emb in ['tsne', 'pca']:
                        if emb.upper() in export_plots and emb in embeddings:
                            fig = Plotting.plot_embedding(adata, basis=emb)
                            filename = f"{emb}_{timestamp}.{export_format.lower()}"
                            filepath = os.path.join(plots_dir, filename)
                            pio.write_image(fig, filepath)
                            st.success(f"Exported {emb.upper()}: {filename}")
                    
                    st.success("Plot export completed!")
                    
                except Exception as e:
                    st.error(f"Error exporting plots: {str(e)}")
    
    with tab3:
        st.subheader("Reproducible Analysis Code")
        
        st.info("""
        Generate a Python script that reproduces your complete analysis workflow.
        This script can be run independently to recreate your results.
        """)
        
        # Code generation options
        st.subheader("Code Generation Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_comments = st.checkbox("Include detailed comments", value=True)
            include_plots = st.checkbox("Include plotting code", value=True)
        
        with col2:
            use_raw = st.checkbox("Use raw data if available", value=True)
            save_outputs = st.checkbox("Include save commands", value=True)
        
        if st.button("Generate Reproducible Script"):
            try:
                # Get analysis parameters from metadata
                metadata = workspace_manager.get_workspace_metadata(st.session_state.workspace)
                
                # Generate Python script
                script_lines = [
                    "#!/usr/bin/env python",
                    "# Reproducible scRNA-seq analysis script",
                    f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"# Workspace: {st.session_state.workspace}",
                    "",
                    "import scanpy as sc",
                    "import pandas as pd",
                    "import numpy as np",
                    "",
                    "# Load your data",
                    "# adata = sc.read_h5ad('your_data.h5ad')  # Uncomment and modify path",
                    "",
                    "# Analysis parameters from the web application:",
                    ""
                ]
                
                # Add QC parameters if available
                if 'qc_parameters' in metadata:
                    qc_params = metadata['qc_parameters']
                    script_lines.extend([
                        "# Quality Control Parameters",
                        f"min_genes = {qc_params.get('min_genes', 200)}",
                        f"max_genes = {qc_params.get('max_genes', 2500)}",
                        f"max_mito = {qc_params.get('max_mito', 5.0)}",
                        "",
                        "# Calculate QC metrics",
                        "sc.pp.calculate_qc_metrics(adata, inplace=True)",
                        "",
                        "# Filter cells",
                        f"sc.pp.filter_cells(adata, min_genes=min_genes)",
                        f"adata = adata[adata.obs.n_genes_by_counts <= max_genes, :]",
                        f"if 'pct_counts_mt' in adata.obs:",
                        f"    adata = adata[adata.obs.pct_counts_mt <= max_mito, :]",
                        ""
                    ])
                
                # Add preprocessing parameters
                if 'preprocessing_parameters' in metadata:
                    pp_params = metadata['preprocessing_parameters']
                    script_lines.extend([
                        "# Preprocessing Parameters",
                        f"target_sum = {pp_params.get('target_sum', 10000)}",
                        f"n_top_genes = {pp_params.get('n_top_genes', 2000)}",
                        f"hvg_method = '{pp_params.get('hvg_method', 'seurat')}'",
                        "",
                        "# Normalization",
                        "sc.pp.normalize_total(adata, target_sum=target_sum)",
                        "sc.pp.log1p(adata)",
                        "",
                        "# Highly variable genes",
                        f"sc.pp.highly_variable_genes(adata, flavor=hvg_method, n_top_genes=n_top_genes)",
                        "adata = adata[:, adata.var.highly_variable]",
                        "",
                        "# Scaling",
                        "sc.pp.scale(adata, max_value=10)",
                        ""
                    ])
                
                # Add dimensionality reduction parameters
                if 'dim_reduction_parameters' in metadata:
                    dr_params = metadata['dim_reduction_parameters']
                    script_lines.extend([
                        "# Dimensionality Reduction Parameters",
                        f"n_pcs = {dr_params.get('n_pcs', 50)}",
                        f"n_neighbors = {dr_params.get('n_neighbors', 15)}",
                        f"umap_min_dist = {dr_params.get('umap_min_dist', 0.5)}",
                        "",
                        "# PCA",
                        f"sc.pp.pca(adata, n_comps=n_pcs)",
                        "",
                        "# Neighborhood graph",
                        f"sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)",
                        "",
                        "# UMAP",
                        f"sc.tl.umap(adata, min_dist=umap_min_dist)",
                        ""
                    ])
                
                # Add clustering parameters
                if 'clustering_parameters' in metadata:
                    cluster_params = metadata['clustering_parameters']
                    script_lines.extend([
                        "# Clustering Parameters",
                        f"resolution = {cluster_params.get('resolution', 1.0)}",
                        f"method = '{cluster_params.get('method', 'leiden')}'",
                        "",
                        "# Clustering",
                        f"if method == 'leiden':",
                        f"    sc.tl.leiden(adata, resolution=resolution)",
                        f"else:",
                        f"    sc.tl.louvain(adata, resolution=resolution)",
                        ""
                    ])
                
                # Add differential expression if available
                if 'de_parameters' in metadata:
                    de_params = metadata['de_parameters']
                    script_lines.extend([
                        "# Differential Expression",
                        f"cluster_key = '{de_params.get('cluster_key', 'leiden')}'",
                        f"de_method = '{de_params.get('method', 'wilcoxon')}'",
                        "",
                        "# Example: Find markers for cluster '0'",
                        f"sc.tl.rank_genes_groups(adata, cluster_key, method=de_method)",
                        ""
                    ])
                
                if include_plots:
                    script_lines.extend([
                        "# Visualization",
                        "import matplotlib.pyplot as plt",
                        "",
                        "# UMAP colored by clusters",
                        f"sc.pl.umap(adata, color=cluster_key, show=False)",
                        "plt.title('UMAP - Cell Clusters')",
                        "plt.show()",
                        ""
                    ])
                
                if save_outputs:
                    script_lines.extend([
                        "# Save processed data",
                        "# adata.write_h5ad('processed_data.h5ad')",
                        ""
                    ])
                
                script_content = "\n".join(script_lines)
                
                # Display and provide download
                st.subheader("Generated Python Script")
                st.code(script_content, language='python')
                
                st.download_button(
                    label="Download Python Script",
                    data=script_content,
                    file_name=f"reproducible_analysis_{st.session_state.workspace}.py",
                    mime="text/x-python"
                )
                
            except Exception as e:
                st.error(f"Error generating script: {str(e)}")
    
    with tab4:
        st.subheader("Analysis Report")
        
        # Generate comprehensive report
        if st.button("Generate Analysis Report"):
            with st.spinner("Generating analysis report..."):
                try:
                    # Create report content
                    report_lines = [
                        f"# scRNA-seq Analysis Report",
                        f"**Workspace:** {st.session_state.workspace}",
                        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        f"**Dataset:** {adata.n_obs} cells × {adata.n_vars} genes",
                        "",
                        "## Analysis Summary",
                        ""
                    ]
                    
                    # Add analysis steps
                    metadata = workspace_manager.get_workspace_metadata(st.session_state.workspace)
                    steps_completed = metadata.get('steps_completed', [])
                    
                    report_lines.append("### Steps Completed")
                    for step in steps_completed:
                        report_lines.append(f"- ✅ {step.replace('_', ' ').title()}")
                    report_lines.append("")
                    
                    # Dataset statistics
                    report_lines.append("### Dataset Statistics")
                    report_lines.append(f"- **Cells:** {adata.n_obs:,}")
                    report_lines.append(f"- **Genes:** {adata.n_vars:,}")
                    
                    if 'total_counts' in adata.obs:
                        report_lines.append(f"- **Total counts:** {adata.obs['total_counts'].sum():,}")
                        report_lines.append(f"- **Mean counts per cell:** {adata.obs['total_counts'].mean():.1f}")
                    
                    if 'n_genes_by_counts' in adata.obs:
                        report_lines.append(f"- **Mean genes per cell:** {adata.obs['n_genes_by_counts'].mean():.1f}")
                    
                    report_lines.append("")
                    
                    # Clustering information
                    cluster_cols = [col for col in adata.obs.columns if any(x in col for x in ['leiden', 'louvain'])]
                    if cluster_cols:
                        report_lines.append("### Clustering Results")
                        for cluster_col in cluster_cols:
                            n_clusters = adata.obs[cluster_col].nunique()
                            report_lines.append(f"- **{cluster_col}:** {n_clusters} clusters")
                        report_lines.append("")
                    
                    # Key findings
                    report_lines.append("### Key Findings")
                    report_lines.append("- Analysis completed successfully using standard scRNA-seq workflow")
                    report_lines.append("- Data processed through quality control, normalization, and clustering")
                    report_lines.append("- Results available for further biological interpretation")
                    report_lines.append("")
                    
                    report_lines.append("### Notes")
                    report_lines.append("Add your biological interpretations and conclusions here...")
                    
                    report_content = "\n".join(report_lines)
                    
                    # Display report
                    st.subheader("Generated Report")
                    st.markdown(report_content)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="Download Report (Markdown)",
                            data=report_content,
                            file_name=f"analysis_report_{st.session_state.workspace}.md",
                            mime="text/markdown"
                        )
                    
                    with col2:
                        # Convert to PDF (simplified)
                        st.info("PDF export requires additional setup")
                
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")
        
        # Workspace backup
        st.subheader("Workspace Backup")
        
        if st.button("Create Workspace Backup"):
            try:
                backup_dir = os.path.join(workspace_path, "..", "backups")
                os.makedirs(backup_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_filename = f"{st.session_state.workspace}_backup_{timestamp}.zip"
                backup_path = os.path.join(backup_dir, backup_filename)
                
                # Create zip file
                with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(workspace_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, os.path.join(workspace_path, '..'))
                            zipf.write(file_path, arcname)
                
                st.success(f"Workspace backup created: {backup_filename}")
                
                # Provide download link for backup
                with open(backup_path, 'rb') as f:
                    backup_data = f.read()
                
                st.download_button(
                    label="Download Workspace Backup",
                    data=backup_data,
                    file_name=backup_filename,
                    mime="application/zip"
                )
                
            except Exception as e:
                st.error(f"Error creating workspace backup: {str(e)}")


if __name__ == "__main__":
    main()