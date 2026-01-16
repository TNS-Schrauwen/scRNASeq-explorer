import scanpy as sc
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import issparse


class DifferentialExpression:
    """Handles differential expression analysis"""
    
    @staticmethod
    def _get_expression(adata: sc.AnnData, gene: str) -> np.ndarray:
        """Get gene expression array handling both sparse and dense matrices"""
        if adata.raw is not None and gene in adata.raw.var_names:
            expr_data = adata.raw[:, gene].X
        else:
            expr_data = adata[:, gene].X
        
        # Convert sparse to dense if necessary
        if issparse(expr_data):
            return expr_data.toarray().flatten()
        else:
            return expr_data.flatten()
    
    @staticmethod
    def cluster_vs_rest(adata: sc.AnnData,
                       cluster_key: str,
                       cluster: str,
                       method: str = 'wilcoxon',
                       corr_method: str = 'benjamini-hochberg',
                       **kwargs) -> pd.DataFrame:
        """Perform differential expression for one cluster vs all others"""
        adata_copy = adata.copy()
        
        # Ensure cluster is categorical
        if not pd.api.types.is_categorical_dtype(adata_copy.obs[cluster_key]):
            adata_copy.obs[cluster_key] = pd.Categorical(adata_copy.obs[cluster_key])
        
        # Perform DE analysis
        sc.tl.rank_genes_groups(
            adata_copy, 
            groupby=cluster_key, 
            groups=[cluster],
            reference='rest',
            method=method,
            corr_method=corr_method,
            **kwargs
        )
        
        # Extract results
        de_results = sc.get.rank_genes_groups_df(adata_copy, group=cluster)
        
        return de_results
    
    @staticmethod
    def cluster_vs_cluster(adata: sc.AnnData,
                          cluster_key: str,
                          cluster1: str,
                          cluster2: str,
                          method: str = 'wilcoxon',
                          corr_method: str = 'benjamini-hochberg',
                          **kwargs) -> pd.DataFrame:
        """Perform differential expression between two specific clusters"""
        adata_copy = adata.copy()
        
        # Ensure cluster is categorical
        if not pd.api.types.is_categorical_dtype(adata_copy.obs[cluster_key]):
            adata_copy.obs[cluster_key] = pd.Categorical(adata_copy.obs[cluster_key])
        
        # Perform DE analysis
        sc.tl.rank_genes_groups(
            adata_copy, 
            groupby=cluster_key, 
            groups=[cluster1],
            reference=cluster2,
            method=method,
            corr_method=corr_method,
            **kwargs
        )
        
        # Extract results
        de_results = sc.get.rank_genes_groups_df(adata_copy, group=cluster1)
        
        return de_results
    
    @staticmethod
    def find_all_markers(adata: sc.AnnData,
                        cluster_key: str,
                        method: str = 'wilcoxon',
                        n_genes: int = 10,
                        **kwargs) -> Dict[str, pd.DataFrame]:
        """Find marker genes for all clusters"""
        adata_copy = adata.copy()
        
        # Ensure cluster is categorical
        if not pd.api.types.is_categorical_dtype(adata_copy.obs[cluster_key]):
            adata_copy.obs[cluster_key] = pd.Categorical(adata_copy.obs[cluster_key])
        
        # Perform DE analysis for all clusters
        sc.tl.rank_genes_groups(
            adata_copy,
            groupby=cluster_key,
            method=method,
            **kwargs
        )
        
        # Extract results for each cluster
        all_markers = {}
        clusters = adata_copy.obs[cluster_key].cat.categories
        
        for cluster in clusters:
            try:
                markers = sc.get.rank_genes_groups_df(adata_copy, group=cluster)
                # Sort by significance and take top n_genes
                markers_sorted = markers.sort_values('pvals').head(n_genes)
                all_markers[cluster] = markers_sorted
            except Exception as e:
                st.warning(f"Could not get markers for cluster {cluster}: {e}")
        
        return all_markers
    
    @staticmethod
    def plot_gene_violin(adata: sc.AnnData, gene: str, cluster_key: str) -> go.Figure:
        """Create violin plot of gene expression across clusters"""
        if gene not in adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in dataset")
        
        # Get expression data using helper method
        expr = DifferentialExpression._get_expression(adata, gene)
        
        # Create plot data
        plot_data = pd.DataFrame({
            'Expression': expr,
            'Cluster': adata.obs[cluster_key]
        })
        
        # Create violin plot
        fig = px.violin(
            plot_data,
            x='Cluster',
            y='Expression',
            title=f'{gene} Expression by Cluster',
            box=True,
            points=False
        )
        
        fig.update_layout(
            xaxis_title='Cluster',
            yaxis_title='Expression Level',
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_dotplot(adata: sc.AnnData, genes: List[str], cluster_key: str) -> go.Figure:
        """Create dot plot showing gene expression across clusters"""
        # Filter to available genes
        available_genes = [g for g in genes if g in adata.var_names]
        
        if not available_genes:
            raise ValueError("None of the specified genes found in dataset")
        
        # Calculate mean expression and fraction expressed per cluster
        clusters = sorted(adata.obs[cluster_key].unique())
        
        mean_expr_data = []
        frac_expr_data = []
        
        for gene in available_genes:
            # Get expression using helper method
            expr = DifferentialExpression._get_expression(adata, gene)
            
            for cluster in clusters:
                cluster_mask = adata.obs[cluster_key] == cluster
                cluster_expr = expr[cluster_mask]
                
                mean_expr = np.mean(cluster_expr)
                frac_expr = (cluster_expr > 0).mean()
                
                mean_expr_data.append({
                    'Gene': gene,
                    'Cluster': cluster,
                    'MeanExpression': mean_expr
                })
                
                frac_expr_data.append({
                    'Gene': gene,
                    'Cluster': cluster,
                    'FractionExpressed': frac_expr
                })
        
        mean_expr_df = pd.DataFrame(mean_expr_data)
        frac_expr_df = pd.DataFrame(frac_expr_data)
        
        # Create dot plot
        fig = go.Figure()
        
        for cluster in clusters:
            cluster_mean = mean_expr_df[mean_expr_df['Cluster'] == cluster]
            cluster_frac = frac_expr_df[frac_expr_df['Cluster'] == cluster]
            
            fig.add_trace(go.Scatter(
                x=cluster_mean['Gene'],
                y=[cluster] * len(cluster_mean),
                marker=dict(
                    size=cluster_frac['FractionExpressed'] * 30 + 5,
                    color=cluster_mean['MeanExpression'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Mean Expression")
                ),
                mode='markers',
                name=str(cluster),
                hovertemplate=(
                    "Gene: %{x}<br>"
                    "Cluster: %{y}<br>"
                    "Mean Expression: %{marker.color:.3f}<br>"
                    "Fraction Expressed: %{marker.size:.2f}"
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title="Gene Expression Dot Plot",
            xaxis_title="Gene",
            yaxis_title="Cluster",
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_marker_heatmap(adata: sc.AnnData, genes: List[str], cluster_key: str) -> go.Figure:
        """Create heatmap of marker gene expression"""
        # Filter to available genes
        available_genes = [g for g in genes if g in adata.var_names]
        
        if not available_genes:
            raise ValueError("None of the specified genes found in dataset")
        
        # Calculate mean expression per cluster
        clusters = sorted(adata.obs[cluster_key].unique())
        
        heatmap_data = []
        for gene in available_genes:
            # Get expression using helper method
            expr = DifferentialExpression._get_expression(adata, gene)
            
            row_data = {'Gene': gene}
            for cluster in clusters:
                cluster_mask = adata.obs[cluster_key] == cluster
                cluster_expr = expr[cluster_mask]
                row_data[cluster] = np.mean(cluster_expr)
            
            heatmap_data.append(row_data)
        
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index('Gene')
        
        # Create heatmap
        fig = px.imshow(
            heatmap_df,
            title="Marker Gene Expression Heatmap",
            color_continuous_scale='Viridis',
            aspect="auto"
        )
        
        fig.update_layout(
            xaxis_title="Cluster",
            yaxis_title="Gene"
        )
        
        return fig