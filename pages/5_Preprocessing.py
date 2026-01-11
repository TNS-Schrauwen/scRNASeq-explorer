import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
from utils.preprocessing import Preprocessing
from utils.workspace_manager import WorkspaceManager
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import matplotlib as mpl


def main():
    st.title("Preprocessing")
    st.markdown("Normalize, find variable genes, and scale your single-cell data.")
    
    # Check if data is loaded and QC is done
    if st.session_state.get('adata') is None:
        st.error("Please load your data first!")
        st.stop()
    
    adata = st.session_state.adata
    workspace_manager = st.session_state.workspace_manager
    
    st.info(f"Current dataset: **{adata.n_obs} cells** × **{adata.n_vars} genes**")
    
    # Sidebar for preprocessing parameters
    st.sidebar.header("Preprocessing Parameters")
    
    # Normalization
    st.sidebar.subheader("Normalization")
    target_sum = st.sidebar.number_input("Target sum for normalization", 
                                        value=10000.0, min_value=1000.0, max_value=100000.0)
    log_transform = st.sidebar.checkbox("Log transform (log1p)", value=True)
    
    # Regression
    st.sidebar.subheader("Regression")
    regress_out = st.sidebar.multiselect(
        "Regress out variables",
        options=['total_counts', 'pct_counts_mt', 'n_genes_by_counts'],
        default=[],
        help="Select variables to regress out"
    )
    
    # Highly Variable Genes
    st.sidebar.subheader("Highly Variable Genes")
    hvg_method = st.sidebar.selectbox(
        "Method",
        options=['seurat', 'cell_ranger', 'seurat_v3'],
        index=0
    )
    n_top_genes = st.sidebar.slider("Number of HVGs", 
                                   min_value=500, max_value=5000, value=2000, step=100)
    min_mean = st.sidebar.number_input("Minimum mean", value=0.0125, format="%f")
    max_mean = st.sidebar.number_input("Maximum mean", value=3.0, format="%f")
    min_disp = st.sidebar.number_input("Minimum dispersion", value=0.5, format="%f")
    
    # Scaling
    st.sidebar.subheader("Scaling")
    scale_data = st.sidebar.checkbox("Scale to unit variance", value=True)
    max_value = st.sidebar.number_input("Max value for scaling", value=10.0)
    
    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Normalization", "Variable Genes", "Scaling", "Results"])
    
    with tab1:
        st.header("Normalization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Before Normalization")
            if 'total_counts' in adata.obs:
                counts_before = adata.obs['total_counts']
                # Remove any infinite or NaN values for plotting
                counts_before_clean = counts_before[np.isfinite(counts_before)]
                
                # Set plot style: white bg, bold black text
                mpl.rcParams.update({
                    'axes.facecolor': 'white',
                    'axes.edgecolor': 'black',
                    'axes.labelcolor': 'black',
                    'axes.labelweight': 'bold',
                    'xtick.color': 'black',
                    'xtick.labelsize': 'large',
                    'ytick.color': 'black',
                    'ytick.labelsize': 'large',
                    'text.color': 'black',
                    'font.weight': 'bold',
                    'grid.color': 'lightgray'
                })
                
                fig_before, ax_before = plt.subplots()
                ax_before.hist(counts_before_clean, bins=50, color='dodgerblue', edgecolor='black')
                ax_before.set_title('Counts Distribution (Before)', fontweight='bold', color='black')
                ax_before.set_xlabel('Total Counts', fontweight='bold', color='black')
                ax_before.set_ylabel('Frequency', fontweight='bold', color='black')
                st.pyplot(fig_before)
                
                st.metric("Mean Counts", f"{counts_before.mean():.0f}")
                st.metric("Median Counts", f"{counts_before.median():.0f}")
        
        with col2:
            st.subheader("After Normalization (Preview)")
            # Show what normalization would do
            if st.button("Preview Normalization"):
                with st.spinner("Calculating preview..."):
                    adata_preview = Preprocessing.normalize_data(
                        adata, target_sum=target_sum, log_transform=log_transform
                    )
                    
                    if log_transform:
                        # Show normalized counts - ensure it's 1D array and clean
                        if issparse(adata_preview.X):
                            normalized_counts = np.array(adata_preview.X.sum(axis=1)).flatten()
                        else:
                            normalized_counts = np.sum(adata_preview.X, axis=1).flatten()
                        
                        # Remove infinite and NaN values
                        normalized_counts_clean = normalized_counts[np.isfinite(normalized_counts)]
                        
                        if len(normalized_counts_clean) > 0:
                            # Set plot style
                            mpl.rcParams.update({
                                'axes.facecolor': 'white',
                                'axes.edgecolor': 'black',
                                'axes.labelcolor': 'black',
                                'axes.labelweight': 'bold',
                                'xtick.color': 'black',
                                'xtick.labelsize': 'large',
                                'ytick.color': 'black',
                                'ytick.labelsize': 'large',
                                'text.color': 'black',
                                'font.weight': 'bold',
                                'grid.color': 'lightgray'
                            })
                            
                            fig_after, ax_after = plt.subplots()
                            ax_after.hist(normalized_counts_clean, bins=50, color='dodgerblue', edgecolor='black')
                            ax_after.set_title('Normalized Counts Distribution', fontweight='bold', color='black')
                            ax_after.set_xlabel('Normalized Counts', fontweight='bold', color='black')
                            ax_after.set_ylabel('Frequency', fontweight='bold', color='black')
                            st.pyplot(fig_after)
                            
                            st.metric("Mean Normalized", f"{np.mean(normalized_counts_clean):.0f}")
                            st.metric("Median Normalized", f"{np.median(normalized_counts_clean):.0f}")
                        else:
                            st.warning("No valid data points for plotting after cleaning infinite/NaN values")
                    else:
                        st.info("Enable log transform to see normalized distribution")
        
        st.subheader("Normalization Parameters")
        st.write(f"**Target sum:** {target_sum:,.0f}")
        st.write(f"**Log transform:** {log_transform}")
        st.write(f"**Regress out:** {', '.join(regress_out) if regress_out else 'None'}")
    
    with tab2:
        st.header("Highly Variable Genes")
        
        # Explain differences between methods
        with st.expander("Differences Between HVG Methods"):
            st.markdown("""
            - **Seurat**: Computes normalized dispersion (variance/mean) for each gene, normalized via loess fit. Good for identifying genes with high variability relative to expression level. Expects logarithmized data.
            - **Cell Ranger**: Bins genes by mean expression and selects top dispersed genes per bin. Ensures even selection across expression ranges. Expects logarithmized data.
            - **Seurat_v3**: Uses variance stabilization (standardized variance after clipping). Ranks genes by variance; better for large datasets with batch effects. Expects raw count data (not logged).
            
            Choose **Seurat** or **Cell Ranger** for standard dispersion-based selection; **Seurat_v3** for variance-based ranking in raw data. Adjust based on dataset size and preprocessing.
            """)
        
        # Preview HVG selection
        if st.button("Find Variable Genes (Preview)"):
            with st.spinner("Finding highly variable genes..."):
                try:
                    # Prepare data based on flavor
                    adata_copy = adata.copy()
                    if hvg_method in ['seurat', 'cell_ranger']:
                        # Normalize and log for these flavors
                        adata_copy = Preprocessing.normalize_data(
                            adata_copy, target_sum=target_sum, log_transform=True
                        )
                    # For seurat_v3, use raw (no norm/log)
                    
                    adata_hvg = Preprocessing.find_variable_genes(
                        adata_copy,
                        method=hvg_method,
                        n_top_genes=n_top_genes,
                        min_mean=min_mean,
                        max_mean=max_mean,
                        min_disp=min_disp
                    )
                    
                    # Plot HVG with custom style
                    mpl.rcParams.update({
                        'axes.facecolor': 'white',
                        'axes.edgecolor': 'black',
                        'axes.labelcolor': 'black',
                        'axes.labelweight': 'bold',
                        'xtick.color': 'black',
                        'xtick.labelsize': 'large',
                        'ytick.color': 'black',
                        'ytick.labelsize': 'large',
                        'text.color': 'black',
                        'font.weight': 'bold',
                        'grid.color': 'lightgray',
                        'scatter.edgecolors': 'black'
                    })
                    
                    # Custom plot for HVGs
                    fig, ax = plt.subplots(figsize=(8, 6))
                    df = adata_hvg.var
                    
                    # Determine Y-axis metric
                    if 'dispersions_norm' in df.columns:
                        y_col = 'dispersions_norm'
                        y_label = 'Normalized Dispersion'
                    elif 'variances_norm' in df.columns:
                        y_col = 'variances_norm'
                        y_label = 'Normalized Variance'
                    else:
                        y_col = 'dispersions'
                        y_label = 'Dispersion'
                    
                    # Plot non-HVG
                    ax.scatter(df.loc[~df['highly_variable'], 'means'], 
                             df.loc[~df['highly_variable'], y_col],
                             c='lightgray', s=20, label='Non-variable', alpha=0.6, edgecolors='none')
                    
                    # Plot HVG
                    ax.scatter(df.loc[df['highly_variable'], 'means'], 
                             df.loc[df['highly_variable'], y_col],
                             c='#d62728', s=30, label='Highly Variable', alpha=0.9, edgecolors='white', linewidth=0.5)
                    
                    ax.set_xlabel('Mean Expression')
                    ax.set_ylabel(y_label)
                    ax.set_title('Highly Variable Genes Selection')
                    ax.legend(frameon=True, facecolor='white', edgecolor='black')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    
                    if hvg_method == 'seurat_v3':
                        ax.set_xscale('log')
                        
                    st.pyplot(fig)
                    
                    st.caption("""
                    **Plot Interpretation:**
                    * **X-axis (Mean expression):** Average expression level of each gene.
                    * **Y-axis (Dispersion/Variance):** Measure of variability. Higher values indicate genes that vary more than expected.
                    * **Red points:** Genes selected as 'highly variable' (HVGs). These carry the most biological information.
                    * **Gray points:** Genes with stable expression across cells (noise or housekeeping).
                    """)
                    
                    # HVG statistics
                    n_hvg = adata_hvg.var['highly_variable'].sum()
                    st.success(f"Found **{n_hvg}** highly variable genes")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("HVG Count", n_hvg)
                    with col2:
                        st.metric("Percentage", f"{(n_hvg / adata.n_vars * 100):.1f}%")
                    with col3:
                        st.metric("Method", hvg_method.upper())
                    
                    # Show top variable genes
                    with st.expander("Top Highly Variable Genes"):
                        if hvg_method == 'seurat_v3':
                            hvg_df = adata_hvg.var[adata_hvg.var['highly_variable']].nlargest(20, 'variances_norm')
                            st.dataframe(hvg_df[['means', 'variances_norm']], width='stretch')
                        else:
                            hvg_df = adata_hvg.var[adata_hvg.var['highly_variable']].nlargest(20, 'dispersions_norm')
                            st.dataframe(hvg_df[['means', 'dispersions_norm']], width='stretch')
                
                except Exception as e:
                    st.error(f"Error finding variable genes: {str(e)}")
                    st.info("Try adjusting the HVG parameters or check if your data contains infinite values.")
        
        else:
            st.info("Click 'Find Variable Genes' to preview HVG selection")
    
    with tab3:
        st.header("Data Scaling")
        
        if scale_data:
            st.success("Scaling will be applied to normalize gene expression across cells.")
            st.write("**Scaling parameters:**")
            st.write(f"- Scale to unit variance: Yes")
            st.write(f"- Max value: {max_value}")
            st.write(f"- Features: Highly variable genes (if selected) or all genes")
        else:
            st.warning("Scaling is disabled. This may affect downstream analysis.")
        
        # Explain scaling
        with st.expander("About Scaling"):
            st.markdown("""
            **Scaling (z-score normalization)** transforms data so that each gene has:
            - Mean = 0
            - Standard deviation = 1
            
            This ensures that all genes contribute equally to distance calculations in downstream analysis.
            
            **Why scale?**
            - Prevents highly expressed genes from dominating the analysis
            - Improves performance of dimensionality reduction and clustering
            - Standard practice in single-cell analysis workflows
            """)
    
    with tab4:
        st.header("Run Complete Preprocessing")
        
        st.subheader("Processing Steps")
        
        steps = [
            f"✅ Quality control filtering",
            f"🔲 Normalize total counts to {target_sum:,.0f}",
            f"🔲 {'Apply log1p transform' if log_transform else 'Skip log transform'}",
            f"🔲 Regress out: {', '.join(regress_out) if regress_out else 'None'}",
            f"🔲 Find {n_top_genes} highly variable genes ({hvg_method} method)",
            f"🔲 {'Scale data' if scale_data else 'Skip scaling'}"
        ]
        
        for step in steps:
            st.write(step)
        
        # Data validation before processing
        st.subheader("Data Validation")
        
        # Check for infinite values
        if adata.X is not None:
            if issparse(adata.X):
                # For sparse, check if any values are problematic
                has_inf = False
                has_nan = False
            else:
                # For dense matrices, check for infinite and NaN values
                has_inf = np.any(~np.isfinite(adata.X))
                has_nan = np.any(np.isnan(adata.X))
            
            if has_inf or has_nan:
                st.warning("⚠️ Dataset contains infinite or NaN values. These will be handled during processing.")
            else:
                st.success("✅ Data looks good for processing!")
        
        # Run complete preprocessing
        if st.button("Run Complete Preprocessing", type="primary", use_container_width=True):
            with st.spinner("Running preprocessing pipeline..."):
                try:
                    progress_bar = st.progress(0)
                    
                    # Step 1: Normalization
                    st.write("**1. Normalizing data...**")
                    adata_processed = Preprocessing.normalize_data(
                        adata, target_sum=target_sum, log_transform=log_transform, regress_out=regress_out
                    )
                    progress_bar.progress(25)
                    
                    # Step 2: Highly variable genes with error handling
                    st.write("**2. Finding highly variable genes...**")
                    try:
                        adata_processed = Preprocessing.find_variable_genes(
                            adata_processed,
                            method=hvg_method,
                            n_top_genes=n_top_genes,
                            min_mean=min_mean,
                            max_mean=max_mean,
                            min_disp=min_disp
                        )
                    except Exception as e:
                        st.error(f"Error in HVG selection: {str(e)}")
                        st.info("Trying alternative HVG parameters...")
                        # Try with default parameters
                        adata_processed = Preprocessing.find_variable_genes(
                            adata_processed,
                            method=hvg_method,
                            n_top_genes=min(2000, adata_processed.n_vars),
                            min_mean=0.0125,
                            max_mean=3,
                            min_disp=0.5
                        )
                    
                    progress_bar.progress(50)
                    
                    # Step 3: Subset to HVG
                    st.write("**3. Subsetting to highly variable genes...**")
                    if 'highly_variable' in adata_processed.var and adata_processed.var['highly_variable'].sum() > 0:
                        adata_processed = adata_processed[:, adata_processed.var.highly_variable]
                    else:
                        st.warning("No highly variable genes found. Using all genes.")
                    progress_bar.progress(75)
                    
                    # Step 4: Scaling
                    if scale_data:
                        st.write("**4. Scaling data...**")
                        adata_processed = Preprocessing.scale_data(adata_processed, max_value=max_value)
                    progress_bar.progress(100)
                    
                    # Update session state
                    st.session_state.adata = adata_processed
                    
                    # Update workspace
                    workspace_manager.update_workspace_metadata(
                        st.session_state.workspace,
                        {
                            "steps_completed": ["data_upload", "quality_control", "preprocessing"],
                            "preprocessing_parameters": {
                                "target_sum": target_sum,
                                "log_transform": log_transform,
                                "regress_out": regress_out,
                                "hvg_method": hvg_method,
                                "n_top_genes": n_top_genes,
                                "min_mean": min_mean,
                                "max_mean": max_mean,
                                "min_disp": min_disp,
                                "scale_data": scale_data,
                                "max_value": max_value
                            }
                        }
                    )
                    
                    st.success("Preprocessing completed successfully!")
                    
                    # Show results
                    st.subheader("Results Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Cells", adata_processed.n_obs)
                    with col2:
                        st.metric("Genes (after HVG)", adata_processed.n_vars)
                    with col3:
                        n_hvg = adata_processed.var['highly_variable'].sum() if 'highly_variable' in adata_processed.var else 0
                        st.metric("HVG Count", n_hvg)
                    with col4:
                        st.metric("Scaling", "Applied" if scale_data else "Skipped")
                    
                    # Show HVG plot if available
                    if 'highly_variable' in adata_processed.var:
                        mpl.rcParams.update({
                            'axes.facecolor': 'white',
                            'axes.edgecolor': 'black',
                            'axes.labelcolor': 'black',
                            'axes.labelweight': 'bold',
                            'xtick.color': 'black',
                            'xtick.labelsize': 'large',
                            'ytick.color': 'black',
                            'ytick.labelsize': 'large',
                            'text.color': 'black',
                            'font.weight': 'bold',
                            'grid.color': 'lightgray',
                            'scatter.edgecolors': 'black'
                        })
                        
                        # Custom plot for HVGs
                        fig, ax = plt.subplots(figsize=(8, 6))
                        df = adata_processed.var
                        
                        # Determine Y-axis metric
                        if 'dispersions_norm' in df.columns:
                            y_col = 'dispersions_norm'
                            y_label = 'Normalized Dispersion'
                        elif 'variances_norm' in df.columns:
                            y_col = 'variances_norm'
                            y_label = 'Normalized Variance'
                        else:
                            y_col = 'dispersions'
                            y_label = 'Dispersion'
                        
                        # Plot non-HVG
                        ax.scatter(df.loc[~df['highly_variable'], 'means'], 
                                 df.loc[~df['highly_variable'], y_col],
                                 c='lightgray', s=20, label='Non-variable', alpha=0.6, edgecolors='none')
                        
                        # Plot HVG
                        ax.scatter(df.loc[df['highly_variable'], 'means'], 
                                 df.loc[df['highly_variable'], y_col],
                                 c='#d62728', s=30, label='Highly Variable', alpha=0.9, edgecolors='white', linewidth=0.5)
                        
                        ax.set_xlabel('Mean Expression')
                        ax.set_ylabel(y_label)
                        ax.set_title('Highly Variable Genes Selection')
                        ax.legend(frameon=True, facecolor='white', edgecolor='black')
                        ax.grid(True, linestyle='--', alpha=0.3)
                        
                        if hvg_method == 'seurat_v3':
                            ax.set_xscale('log')
                            
                        st.pyplot(fig)
                    
                    st.caption("""
                    **Plot Interpretation:**
                    * **X-axis (Mean expression):** Average expression level of each gene.
                    * **Y-axis (Dispersion/Variance):** Measure of variability. Higher values indicate genes that vary more than expected.
                    * **Red points:** Genes selected as 'highly variable' (HVGs). These carry the most biological information.
                    * **Gray points:** Genes with stable expression across cells (noise or housekeeping).
                    """)
                    
                    # Next steps
                    st.divider()
                    st.success("Preprocessing completed! Proceed to **Dimensionality Reduction**.")
                    
                    if st.button("Go to Dimensionality Reduction →", use_container_width=True):
                        st.switch_page("pages/6_Dimensionality_Reduction.py")
                
                except Exception as e:
                    st.error(f"Error during preprocessing: {str(e)}")
        
        else:
            st.info("Click the button above to run the complete preprocessing pipeline")


if __name__ == "__main__":
    main()