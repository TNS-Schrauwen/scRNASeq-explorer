import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.qc_metrics import QCMetrics
from utils.plotting import Plotting
from utils.workspace_manager import WorkspaceManager
import plotly.express as px

# Publication-quality settings: white background, clean fonts
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'text.color': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'grid.color': '#e0e0e0',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

sns.set_style("whitegrid")

def main():
    st.title("Quality Control")
    st.markdown("""
    Comprehensive quality control for single-cell RNA sequencing data.  
    Identify low-quality cells, doublets, and technical artifacts using multiple metrics.
    """)

    if st.session_state.get('adata') is None:
        st.error("Please load your data first in the 'Data Upload' page!")
        st.stop()

    adata = st.session_state.adata

    st.info(f"Current dataset: **{adata.n_obs:,} cells** × **{adata.n_vars:,} genes**")

    # Calculate QC metrics if not present
    if 'total_counts' not in adata.obs or 'n_genes_by_counts' not in adata.obs:
        with st.spinner("Calculating QC metrics (including mitochondrial & ribosomal genes)..."):
            adata = QCMetrics.calculate_qc_metrics(adata)
            st.session_state.adata = adata
            st.success("QC metrics calculated successfully!")

    # Sidebar controls
    st.sidebar.header("QC Settings")
    mt_pattern = st.sidebar.text_input("Mitochondrial gene pattern", value="^MT-|^mt-")
    ribo_pattern = st.sidebar.text_input("Ribosomal gene pattern", value="^RPS|^RPL|^Rps|^Rpl")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Key Metrics", "Static Plots", "Interactive Plots", "Summary Report"])

    with tab1:
        st.header("Key Quality Control Metrics")
        qc_results = QCMetrics.get_comprehensive_metrics(adata)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cells", f"{qc_results['n_cells']:,}")
            st.metric("Total Genes", f"{qc_results['n_genes']:,}")
        with col2:
            st.metric("Median Genes/Cell", f"{qc_results['median_genes_per_cell']:.0f}")
            st.metric("Median UMI/Cell", f"{qc_results['median_umi_per_cell']:.0f}")
        with col3:
            if qc_results['median_pct_mito'] > 0:
                st.metric("Median MT %", f"{qc_results['median_pct_mito']:.1f}%")
            if qc_results['median_pct_ribo'] > 0:
                st.metric("Median Ribosomal %", f"{qc_results['median_pct_ribo']:.1f}%")
        with col4:
            st.metric("Data Sparsity", f"{qc_results['sparsity']:.1%}")
            st.metric("Zero-Count Genes", f"{qc_results['zero_count_genes']:,}")

        st.subheader("Detailed Cell-Level Statistics")
        metrics_df = QCMetrics.get_detailed_metrics_table(adata)
        st.dataframe(metrics_df, use_container_width=True)

    with tab2:
        st.header("Publication-Quality QC Plots")

        # Row 1: Violin + Top Genes Scatter (as in sc-best-practices)
        col1, col2 = st.columns(2)
        with col1:
            fig_violin = QCMetrics.plot_qc_violin(adata)
            st.pyplot(fig_violin)
            st.caption("""
            **Violin plots** of key QC metrics. Wide or bimodal distributions suggest subpopulations or quality issues. 
            High mitochondrial % often indicates stressed or dying cells.
            """)

            # Export buttons
            col1a, col1b = st.columns(2)
            with col1a:
                if st.button("📥 Save Violin as PNG", key="violin_png"):
                    fig_violin.savefig("qc_violin.png", dpi=300, bbox_inches='tight')
                    st.success("Saved: qc_violin.png")
            with col1b:
                if st.button("📥 Save as SVG", key="violin_svg"):
                    fig_violin.savefig("qc_violin.svg", bbox_inches='tight')
                    st.success("Saved: qc_violin.svg")

        with col2:
            fig_top_genes = QCMetrics.plot_highest_expressed_genes(adata)
            st.pyplot(fig_top_genes)
            st.caption("""
            **Top 20 most highly expressed genes**. Dominance by mitochondrial, hemoglobin, or ribosomal genes 
            can indicate biological stress or contamination.
            """)

        # Row 2: UMI vs Genes Scatter + Distributions
        st.subheader("Cell Complexity and Count Distributions")
        col3, col4 = st.columns(2)
        with col3:
            fig_scatter = QCMetrics.plot_umi_gene_scatter(adata)
            st.pyplot(fig_scatter)
            st.caption("""
            **UMI counts vs. detected genes per cell**, colored by mitochondrial content. 
            Cells with low counts/genes and high MT% are likely dying or low-quality.
            """)

            col3a, col3b = st.columns(2)
            with col3a:
                if st.button("📥 Save Scatter as PNG", key="scatter_png"):
                    fig_scatter.savefig("qc_scatter.png", dpi=300, bbox_inches='tight')
                    st.success("Saved: qc_scatter.png")
            with col3b:
                if st.button("📥 Save as SVG", key="scatter_svg"):
                    fig_scatter.savefig("qc_scatter.svg", bbox_inches='tight')
                    st.success("Saved: qc_scatter.svg")

        with col4:
            fig_genes_dist = QCMetrics.plot_gene_distribution(adata)
            st.pyplot(fig_genes_dist)
            st.caption("**Distribution of detected genes per cell**. Left-skewed tail indicates low-quality cells.")

            fig_umi_dist = QCMetrics.plot_umi_distribution(adata)
            st.pyplot(fig_umi_dist)
            st.caption("**Distribution of total UMI counts per cell**. Low-count cells are often poor quality.")

    with tab3:
        st.header("Interactive Visualizations")

        if any(key in adata.obsm for key in ['X_umap', 'X_tsne', 'X_pca']):
            st.subheader("QC Metrics on Embeddings")
            embedding_options = []
            if 'X_umap' in adata.obsm: embedding_options.append('UMAP')
            if 'X_tsne' in adata.obsm: embedding_options.append('t-SNE')
            if 'X_pca' in adata.obsm: embedding_options.append('PCA')

            selected = st.selectbox("Select embedding", embedding_options)
            basis = selected.lower().replace('-', '')

            color_opts = ['n_genes_by_counts', 'total_counts']
            if 'pct_counts_mt' in adata.obs: color_opts.append('pct_counts_mt')
            if 'pct_counts_ribo' in adata.obs: color_opts.append('pct_counts_ribo')

            color_by = st.selectbox("Color by metric", color_opts)

            fig = Plotting.plot_embedding(adata, basis=basis, color=color_by)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("QC Metrics Correlation Heatmap")
        if st.button("Generate Correlation Heatmap"):
            with st.spinner("Computing correlations..."):
                fig_corr = QCMetrics.plot_qc_correlation(adata)
                st.pyplot(fig_corr)
                st.caption("""
                **Pearson correlation** between QC metrics. Strong negative correlation between gene/UMI counts 
                and mitochondrial % is a hallmark of dying cells.
                """)

    with tab4:
        st.header("QC Summary Report")
        report = QCMetrics.generate_qc_report(adata)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Overall Quality Assessment", report['quality_assessment'])
            st.metric("Recommended Action", report['recommended_action'])
        with col2:
            st.metric("Potential Issues Found", len(report['potential_issues']))
            st.metric("Quality Score", f"{report['completeness_score']:.1%}")

        if report['potential_issues']:
            st.subheader("⚠️ Detected Potential Issues")
            for issue in report['potential_issues']:
                st.warning(issue)

        st.subheader("Recommended Next Steps")
        st.info("""
        - Use **MAD-based filtering** (robust to outliers) on `n_genes_by_counts`, `total_counts`, and `pct_counts_mt`.
        - Typical thresholds: >200–500 genes, <10–20% MT (tissue-dependent).
        - Proceed to doublet detection before final filtering to avoid removing rare populations.
        """)

        if st.button("Proceed to Doublet Detection →"):
            st.switch_page("pages/4_Doublet_Detection.py")

if __name__ == "__main__":
    main()