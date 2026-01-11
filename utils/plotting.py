import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
import scanpy as sc
from typing import Optional, List, Dict, Any
import streamlit as st
from scipy.sparse import issparse


class Plotting:
    """Handles all interactive plotting using Plotly"""
    
    # Professional vibrant color schemes
    COLOR_SCHEMES = {
        'qualitative': px.colors.qualitative.Vivid,
        'sequential': px.colors.sequential.Viridis,
        'diverging': px.colors.diverging.RdYlBu,
        'clusters': px.colors.qualitative.Set3,
        'expression': px.colors.sequential.Plasma
    }

    # Publication-quality layout base
    PUBLICATION_LAYOUT = dict(
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", size=12, color="black"),
        title=dict(font=dict(size=18, color="black", family="Arial Black")),
        xaxis=dict(
            title_font=dict(family="Arial Black", size=14, color="black"),
            tickfont=dict(color="black"),
            linewidth=1.5,
            linecolor="black",
            mirror=True,
            showgrid=False
        ),
        yaxis=dict(
            title_font=dict(family="Arial Black", size=14, color="black"),
            tickfont=dict(color="black"),
            linewidth=1.5,
            linecolor="black",
            mirror=True,
            showgrid=False
        ),
        legend=dict(
            title_font=dict(color="black"),
            font=dict(color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        ),
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    
    @staticmethod
    def _apply_publication_style(fig: go.Figure, title: Optional[str] = None) -> go.Figure:
        """Apply consistent publication-quality styling"""
        fig.update_layout(**Plotting.PUBLICATION_LAYOUT)
        if title:
            fig.update_layout(title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18, color="black", family="Arial")))
        
        # Remove marker borders and adjust opacity/size
        fig.update_traces(
            marker=dict(
                line=dict(width=0),  # No borders
            ),
            selector=dict(mode='markers')
        )
        return fig
    
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
    def plot_qc_violin(adata: sc.AnnData) -> go.Figure:
        """Create violin plots for QC metrics"""
        qc_metrics = ['n_genes_by_counts', 'total_counts']
        if 'pct_counts_mt' in adata.obs:
            qc_metrics.append('pct_counts_mt')
        
        fig = sp.make_subplots(
            rows=1, 
            cols=len(qc_metrics),
            subplot_titles=[metric.replace('_', ' ').title() for metric in qc_metrics]
        )
        
        colors = Plotting.COLOR_SCHEMES['qualitative']
        
        for i, metric in enumerate(qc_metrics):
            if metric in adata.obs:
                fig.add_trace(
                    go.Violin(
                        y=adata.obs[metric],
                        name=metric.replace('_', ' ').title(),
                        box_visible=True,
                        meanline_visible=True,
                        line_color=colors[i % len(colors)],
                        fillcolor=colors[i % len(colors)],
                        opacity=0.7
                    ),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title="Quality Control Metrics",
            height=400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    @staticmethod
    def plot_gene_count_scatter(adata: sc.AnnData) -> go.Figure:
        """Create scatter plot of genes vs counts"""
        if 'n_genes_by_counts' in adata.obs and 'total_counts' in adata.obs:
            if 'pct_counts_mt' in adata.obs:
                fig = px.scatter(
                    adata.obs,
                    x='total_counts',
                    y='n_genes_by_counts',
                    color='pct_counts_mt',
                    title='Genes vs Counts per Cell (colored by MT%)',
                    labels={
                        'total_counts': 'Total Counts',
                        'n_genes_by_counts': 'Number of Genes',
                        'pct_counts_mt': 'Mitochondrial %'
                    },
                    color_continuous_scale=Plotting.COLOR_SCHEMES['sequential'],
                    template='plotly_white'
                )
            else:
                fig = px.scatter(
                    adata.obs,
                    x='total_counts',
                    y='n_genes_by_counts',
                    title='Genes vs Counts per Cell',
                    labels={
                        'total_counts': 'Total Counts',
                        'n_genes_by_counts': 'Number of Genes'
                    },
                    color_discrete_sequence=[Plotting.COLOR_SCHEMES['qualitative'][0]],
                    template='plotly_white'
                )
            
            return fig
        return go.Figure()
    
    @staticmethod
    def plot_hvg(adata: sc.AnnData) -> go.Figure:
        """Plot highly variable genes"""
        if 'highly_variable' not in adata.var:
            return go.Figure()
        
        fig = px.scatter(
            adata.var,
            x='means',
            y='dispersions',
            color='highly_variable',
            title='Highly Variable Genes',
            labels={
                'means': 'Mean Expression',
                'dispersions': 'Dispersion',
                'highly_variable': 'Highly Variable'
            },
            color_discrete_map={
                True: Plotting.COLOR_SCHEMES['qualitative'][1],
                False: Plotting.COLOR_SCHEMES['qualitative'][0]
            },
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_pca_variance(adata: sc.AnnData, n_pcs: int = 50) -> go.Figure:
        """Plot PCA variance explained with elbow point detection"""
        if 'pca' not in adata.uns:
            return go.Figure()
        
        variance_ratio = adata.uns['pca']['variance_ratio'][:n_pcs]
        components = list(range(1, len(variance_ratio) + 1))
        
        # Calculate cumulative variance
        cumulative_variance = np.cumsum(variance_ratio)
        
        # Find elbow point
        elbow_point = Plotting._find_elbow_point(variance_ratio)
        
        fig = go.Figure()
        
        # Cumulative variance
        fig.add_trace(go.Scatter(
            x=components,
            y=cumulative_variance,
            mode='lines+markers',
            name='Cumulative Variance',
            line=dict(color=Plotting.COLOR_SCHEMES['qualitative'][0], width=4),
            marker=dict(size=8)
        ))
        
        # Individual variance
        fig.add_trace(go.Scatter(
            x=components,
            y=variance_ratio,
            mode='lines+markers',
            name='Variance per PC',
            line=dict(color=Plotting.COLOR_SCHEMES['qualitative'][3], width=2, dash='dot'),
            marker=dict(size=6)
        ))
        
        # Add elbow point annotation
        if elbow_point < len(components):
            fig.add_vline(
                x=elbow_point + 1, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Elbow: PC{elbow_point + 1}",
                annotation_position="top right"
            )
        
        fig = Plotting._apply_publication_style(fig, title="PCA Variance Explained")
        fig.update_xaxes(title="Principal Component")
        fig.update_yaxes(title="Variance Explained")
        
        return fig
    
    @staticmethod
    def _find_elbow_point(variance_ratio, min_components=5):
        """Find the elbow point in PCA variance using maximum curvature method"""
        if len(variance_ratio) <= min_components:
            return min_components - 1
        
        # Calculate cumulative variance
        cumulative = np.cumsum(variance_ratio)
        
        # Normalize
        x = np.arange(len(variance_ratio))
        y = cumulative
        
        # Calculate distances from each point to line between first and last point
        first_point = np.array([x[0], y[0]])
        last_point = np.array([x[-1], y[-1]])
        
        line_vec = last_point - first_point
        line_len = np.linalg.norm(line_vec)
        line_vec_norm = line_vec / line_len
        
        distances = []
        for i in range(len(x)):
            point_vec = np.array([x[i], y[i]]) - first_point
            # Project point onto line
            proj_length = np.dot(point_vec, line_vec_norm)
            proj_point = first_point + proj_length * line_vec_norm
            # Calculate distance
            distance = np.linalg.norm(np.array([x[i], y[i]]) - proj_point)
            distances.append(distance)
        
        # Find point with maximum distance (elbow point)
        elbow_idx = np.argmax(distances[min_components:]) + min_components
        
        return min(elbow_idx, len(variance_ratio) - 1)
    
    @staticmethod
    def plot_embedding(adata: sc.AnnData, 
                      basis: str = 'umap',
                      color: Optional[str] = None,
                      use_3d: bool = False,
                      custom_color_map: Optional[Dict[Any, str]] = None,
                      marker_opacity: float = 0.8,
                      marker_size: int = 5) -> go.Figure:
        """Plot 2D or 3D embeddings (UMAP, t-SNE, PCA) with custom color support"""
        if f'X_{basis}' not in adata.obsm:
            return go.Figure()
        
        embedding_data = adata.obsm[f'X_{basis}']
        dims = embedding_data.shape[1]
        
        # Determine color sequence
        if custom_color_map and color:
            color_sequence = [custom_color_map[val] for val in adata.obs[color]]
            use_custom = True
        else:
            use_custom = False
            color_sequence = Plotting.COLOR_SCHEMES['clusters'] if color else [Plotting.COLOR_SCHEMES['qualitative'][0]]

        title = f"{basis.upper()} Projection"
        if color:
            title += f" - Colored by {color}"

        if use_3d and dims >= 3:
            if use_custom:
                fig = go.Figure(data=go.Scatter3d(
                    x=embedding_data[:, 0],
                    y=embedding_data[:, 1],
                    z=embedding_data[:, 2],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=color_sequence,
                        opacity=0.55,  # Translucent for better depth perception
                        line=dict(width=0)
                    ),
                    text=[str(t) for t in adata.obs[color]] if color else None,
                    hoverinfo='text' if color else 'skip'
                ))
            elif color and pd.api.types.is_categorical_dtype(adata.obs[color]):
                fig = px.scatter_3d(
                    x=embedding_data[:, 0],
                    y=embedding_data[:, 1],
                    z=embedding_data[:, 2],
                    color=adata.obs[color],
                    color_discrete_sequence=color_sequence,
                    labels={'x': f'{basis.upper()}1', 'y': f'{basis.upper()}2', 'z': f'{basis.upper()}3'},
                    template='simple_white'
                )
            else:
                fig = px.scatter_3d(
                    x=embedding_data[:, 0],
                    y=embedding_data[:, 1],
                    z=embedding_data[:, 2],
                    color=adata.obs[color] if color else None,
                    color_continuous_scale=Plotting.COLOR_SCHEMES['expression'],
                    labels={'x': f'{basis.upper()}1', 'y': f'{basis.upper()}2', 'z': f'{basis.upper()}3'},
                    template='simple_white'
                )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title=f'{basis.upper()}1',
                    yaxis_title=f'{basis.upper()}2',
                    zaxis_title=f'{basis.upper()}3',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
                    aspectmode='cube',
                    bgcolor='white'
                )
            )
        else:
            if use_custom:
                fig = go.Figure(data=go.Scatter(
                    x=embedding_data[:, 0],
                    y=embedding_data[:, 1],
                    mode='markers',
                    marker=dict(
                        size=marker_size,
                        color=color_sequence,
                        opacity=marker_opacity,
                        line=dict(width=0)
                    ),
                    text=[str(t) for t in adata.obs[color]] if color else None,
                    hoverinfo='text' if color else 'skip'
                ))
            elif color and pd.api.types.is_categorical_dtype(adata.obs[color]):
                fig = px.scatter(
                    x=embedding_data[:, 0],
                    y=embedding_data[:, 1],
                    color=adata.obs[color],
                    color_discrete_sequence=color_sequence,
                    labels={'x': f'{basis.upper()}1', 'y': f'{basis.upper()}2'},
                    template='simple_white'
                )
            else:
                fig = px.scatter(
                    x=embedding_data[:, 0],
                    y=embedding_data[:, 1],
                    color=adata.obs[color] if color else None,
                    color_discrete_sequence=color_sequence,
                    color_continuous_scale=Plotting.COLOR_SCHEMES['expression'],
                    labels={'x': f'{basis.upper()}1', 'y': f'{basis.upper()}2'},
                    template='simple_white'
                )

        fig = Plotting._apply_publication_style(fig, title=title)
        
        # Axis labels
        if use_3d and dims >= 3:
            fig.update_scenes(
                xaxis_title=f'{basis.upper()}1',
                yaxis_title=f'{basis.upper()}2',
                zaxis_title=f'{basis.upper()}3'
            )
        else:
            fig.update_xaxes(title=f'{basis.upper()}1')
            fig.update_yaxes(title=f'{basis.upper()}2')

        return fig
    
    @staticmethod
    def plot_pca_scatter(adata: sc.AnnData, 
                        color: Optional[str] = None,
                        pc1: int = 1, 
                        pc2: int = 2,
                        custom_color_map: Optional[Dict[Any, str]] = None) -> go.Figure:
        """Plot PCA scatter plot for specific PCs with custom color support"""
        if 'X_pca' not in adata.obsm:
            return go.Figure()
        
        pca_data = adata.obsm['X_pca']
        
        pc1_idx = pc1 - 1
        pc2_idx = pc2 - 1
        
        if pc1_idx >= pca_data.shape[1] or pc2_idx >= pca_data.shape[1]:
            st.error(f"Requested PCs ({pc1}, {pc2}) exceed available PCs ({pca_data.shape[1]})")
            return go.Figure()
        
        title = f"PCA: PC{pc1} vs PC{pc2}"
        if color:
            title += f" - Colored by {color}"

        if custom_color_map and color:
            color_sequence = [custom_color_map[val] for val in adata.obs[color]]
            fig = go.Figure(data=go.Scatter(
                x=pca_data[:, pc1_idx],
                y=pca_data[:, pc2_idx],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color_sequence,
                    opacity=0.8,
                    line=dict(width=0)
                ),
                text=[str(t) for t in adata.obs[color]],
                hoverinfo='text'
            ))
        elif color and pd.api.types.is_categorical_dtype(adata.obs[color]):
            fig = px.scatter(
                x=pca_data[:, pc1_idx],
                y=pca_data[:, pc2_idx],
                color=adata.obs[color],
                color_discrete_sequence=Plotting.COLOR_SCHEMES['clusters'],
                labels={'x': f'PC{pc1}', 'y': f'PC{pc2}'},
                template='simple_white'
            )
        else:
            fig = px.scatter(
                x=pca_data[:, pc1_idx],
                y=pca_data[:, pc2_idx],
                color=adata.obs[color] if color else None,
                color_discrete_sequence=[Plotting.COLOR_SCHEMES['qualitative'][0]],
                color_continuous_scale=Plotting.COLOR_SCHEMES['expression'],
                labels={'x': f'PC{pc1}', 'y': f'PC{pc2}'},
                template='simple_white'
            )

        fig = Plotting._apply_publication_style(fig, title=title)
        fig.update_xaxes(title=f"PC{pc1}")
        fig.update_yaxes(title=f"PC{pc2}")

        return fig

    # --- Remaining functions unchanged (gene expression, volcano, etc.) ---

    @staticmethod
    def plot_gene_expression(adata: sc.AnnData, 
                           gene: str,
                           basis: str = 'umap',
                           use_3d: bool = False) -> go.Figure:
        """Plot gene expression on embedding"""
        if gene not in adata.var_names:
            return go.Figure()
        
        if f'X_{basis}' not in adata.obsm:
            return go.Figure()
        
        # Get gene expression using the helper method
        expr = Plotting._get_expression(adata, gene)
        
        embedding_data = adata.obsm[f'X_{basis}']
        
        if use_3d and embedding_data.shape[1] >= 3:
            fig = px.scatter_3d(
                x=embedding_data[:, 0],
                y=embedding_data[:, 1],
                z=embedding_data[:, 2],
                color=expr,
                title=f'{gene} Expression on {basis.upper()} 3D',
                color_continuous_scale=Plotting.COLOR_SCHEMES['expression'],
                labels={'x': f'{basis.upper()}1', 'y': f'{basis.upper()}2', 'z': f'{basis.upper()}3', 'color': 'Expression'},
                template='plotly_white'
            )
        else:
            fig = px.scatter(
                x=embedding_data[:, 0],
                y=embedding_data[:, 1],
                color=expr,
                title=f'{gene} Expression on {basis.upper()}',
                color_continuous_scale=Plotting.COLOR_SCHEMES['expression'],
                labels={'x': f'{basis.upper()}1', 'y': f'{basis.upper()}2', 'color': 'Expression'},
                template='plotly_white'
            )
        
        # Improve marker appearance
        fig.update_traces(
            marker=dict(
                size=5,
                line=dict(width=0.3, color='white'),
                opacity=0.8
            ),
            selector=dict(mode='markers')
        )
        
        return fig
    
    @staticmethod
    def plot_volcano(de_results: pd.DataFrame, 
                    log2fc_threshold: float = 1.0,
                    pval_threshold: float = 0.05) -> go.Figure:
        """Create volcano plot from DE results"""
        if de_results.empty:
            return go.Figure()
        
        # Create significance categories
        de_results = de_results.copy()
        de_results['-log10_pvals'] = -np.log10(de_results['pvals'])
        
        conditions = [
            (de_results['logfoldchanges'].abs() > log2fc_threshold) & (de_results['pvals'] < pval_threshold),
            (de_results['logfoldchanges'] > 0) & (de_results['pvals'] >= pval_threshold),
            (de_results['logfoldchanges'] < 0) & (de_results['pvals'] >= pval_threshold),
            (de_results['logfoldchanges'].abs() <= log2fc_threshold) & (de_results['pvals'] < pval_threshold)
        ]
        
        categories = ['Significant Up', 'Non-sig Up', 'Non-sig Down', 'Significant Down']
        colors = [
            Plotting.COLOR_SCHEMES['qualitative'][1],  # Significant Up - Green
            Plotting.COLOR_SCHEMES['qualitative'][0],  # Non-sig Up - Blue
            Plotting.COLOR_SCHEMES['qualitative'][2],  # Non-sig Down - Red
            Plotting.COLOR_SCHEMES['qualitative'][3]   # Significant Down - Purple
        ]
        
        de_results['category'] = np.select(conditions, categories, default='Other')
        
        fig = px.scatter(
            de_results,
            x='logfoldchanges',
            y='-log10_pvals',
            color='category',
            title='Volcano Plot - Differential Expression',
            labels={
                'logfoldchanges': 'log2(Fold Change)',
                '-log10_pvals': '-log10(p-value)',
                'category': 'Significance'
            },
            color_discrete_map=dict(zip(categories, colors)),
            hover_data=['names'],
            template='plotly_white'
        )
        
        # Add threshold lines
        fig.add_vline(x=log2fc_threshold, line_dash="dash", line_color="red", opacity=0.7)
        fig.add_vline(x=-log2fc_threshold, line_dash="dash", line_color="red", opacity=0.7)
        fig.add_hline(y=-np.log10(pval_threshold), line_dash="dash", line_color="red", opacity=0.7)
        
        # Improve marker appearance
        fig.update_traces(
            marker=dict(
                size=6,
                line=dict(width=0.5, color='white'),
                opacity=0.7
            ),
            selector=dict(mode='markers')
        )
        
        return fig