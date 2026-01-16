import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
from typing import Dict, Any, Tuple
import warnings

# Set a clean, publication-ready style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'figure.figsize': (8, 6),
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'legend.framealpha': 0.9
})

class DoubletDetection:
    """Doublet detection using Scanpy's methods"""
    
    @staticmethod
    def detect_doublets_scanpy(adata: sc.AnnData, 
                              expected_doublet_rate: float = 0.05,
                              n_neighbors: int = 50,
                              sim_doublet_ratio: int = 2) -> sc.AnnData:
        """
        Detect doublets using Scanpy's doublet detection approach (primarily Scrublet)
        """
        adata_copy = adata.copy()
        
        # Ensure we have normalized data for doublet detection
        if 'log1p' not in adata_copy.uns_keys():
            sc.pp.normalize_total(adata_copy, target_sum=1e4)
            sc.pp.log1p(adata_copy)
        
        # Ensure we have PCA for neighborhood graph
        if 'X_pca' not in adata_copy.obsm:
            sc.pp.pca(adata_copy, n_comps=min(50, adata_copy.n_vars-1))
        
        # Calculate doublet scores using neighborhood-based approach
        try:
            # Use Scanpy's Scrublet integration
            sc.external.pp.scrublet(adata_copy, 
                                  expected_doublet_rate=expected_doublet_rate,
                                  sim_doublet_ratio=sim_doublet_ratio)
            
            # Standardize column names
            if 'doublet_score' in adata_copy.obs:
                adata_copy.obs['doublet_score'] = adata_copy.obs['doublet_score']
            if 'predicted_doublet' in adata_copy.obs:
                adata_copy.obs['predicted_doublets'] = adata_copy.obs['predicted_doublet'].astype(bool)
                
        except Exception as e:
            warnings.warn(f"Scrublet failed: {e}. Falling back to manual doublet detection.")
            adata_copy = DoubletDetection._manual_doublet_detection(
                adata_copy, expected_doublet_rate, n_neighbors
            )
        
        return adata_copy
    
    @staticmethod
    def _manual_doublet_detection(adata: sc.AnnData, 
                                 expected_doublet_rate: float,
                                 n_neighbors: int) -> sc.AnnData:
        """Fallback manual doublet detection using local density in PCA space"""
        adata_copy = adata.copy()
        
        if 'neighbors' not in adata_copy.uns:
            sc.pp.neighbors(adata_copy, n_neighbors=n_neighbors)
        
        from sklearn.neighbors import NearestNeighbors
        
        X_pca = adata_copy.obsm['X_pca'][:, :10]
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_pca)
        distances, _ = nbrs.kneighbors(X_pca)
        
        mean_distances = distances.mean(axis=1)
        doublet_scores = (mean_distances - mean_distances.mean()) / mean_distances.std()
        
        from scipy import stats
        doublet_scores = stats.norm.cdf(doublet_scores)
        
        adata_copy.obs['doublet_score'] = doublet_scores
        
        threshold = np.percentile(doublet_scores, 100 * (1 - expected_doublet_rate))
        adata_copy.obs['predicted_doublets'] = doublet_scores > threshold
        
        return adata_copy
    
    @staticmethod
    def get_doublet_summary(adata: sc.AnnData) -> pd.DataFrame:
        """Get comprehensive doublet detection summary"""
        if 'predicted_doublets' not in adata.obs:
            return pd.DataFrame()
        
        n_doublets = adata.obs['predicted_doublets'].sum()
        n_cells = adata.n_obs
        doublet_rate = n_doublets / n_cells
        
        summary_data = {
            'Metric': [
                'Total Cells',
                'Predicted Doublets', 
                'Doublet Rate',
                'Doublet Score Mean',
                'Doublet Score Std',
                'Doublet Score Max'
            ],
            'Value': [
                f"{n_cells:,}",
                f"{n_doublets:,}",
                f"{doublet_rate:.2%}",
                f"{adata.obs['doublet_score'].mean():.3f}",
                f"{adata.obs['doublet_score'].std():.3f}",
                f"{adata.obs['doublet_score'].max():.3f}"
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def plot_doublet_score_distribution(adata: sc.AnnData) -> plt.Figure:
        """Plot distribution of doublet scores with publication-quality styling"""
        if 'doublet_score' not in adata.obs:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No doublet scores available", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return fig
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(adata.obs['doublet_score'], bins=60, color='#1f77b4', alpha=0.8, 
                edgecolor='black', linewidth=0.7)
        
        if 'predicted_doublets' in adata.obs and adata.obs['predicted_doublets'].any():
            threshold = adata.obs['doublet_score'][adata.obs['predicted_doublets']].min()
            ax.axvline(threshold, color='#d62728', linestyle='--', linewidth=2.5,
                      label=f'Threshold = {threshold:.3f}')
            ax.legend(fontsize=11)
        
        ax.set_xlabel('Doublet Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Cells', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Doublet Scores', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_threshold_analysis(adata: sc.AnnData) -> plt.Figure:
        """Plot threshold analysis: cumulative fraction and doublet count vs threshold"""
        if 'doublet_score' not in adata.obs:
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.text(0.5, 0.5, "No doublet scores available", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return fig
        
        doublet_scores = adata.obs['doublet_score'].values
        sorted_scores = np.sort(doublet_scores)
        cumulative_fraction_above = 1 - np.arange(len(sorted_scores)) / len(sorted_scores)
        
        thresholds = np.linspace(doublet_scores.min(), doublet_scores.max(), 200)
        n_doublets_at_thresh = np.array([np.sum(doublet_scores > t) for t in thresholds])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left: Cumulative fraction above threshold
        ax1.plot(sorted_scores, cumulative_fraction_above, color='#2ca02c', linewidth=2.5)
        ax1.set_xlabel('Doublet Score', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Fraction of Cells Above Threshold', fontsize=12, fontweight='bold')
        ax1.set_title('Cumulative Distribution of Doublet Scores', fontsize=14, fontweight='bold')
        
        # Right: Number of predicted doublets vs threshold
        ax2.plot(thresholds, n_doublets_at_thresh, color='#d62728', linewidth=2.5)
        ax2.set_xlabel('Doublet Score Threshold', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Predicted Doublets', fontsize=12, fontweight='bold')
        ax2.set_title('Threshold vs Predicted Doublet Count', fontsize=14, fontweight='bold')
        
        # Mark current threshold if available
        if 'predicted_doublets' in adata.obs and adata.obs['predicted_doublets'].any():
            current_thresh = doublet_scores[adata.obs['predicted_doublets']].min()
            current_n = adata.obs['predicted_doublets'].sum()
            
            ax1.axvline(current_thresh, color='#ff7f0e', linestyle=':', linewidth=2,
                       label=f'Current threshold')
            ax2.axvline(current_thresh, color='#ff7f0e', linestyle=':', linewidth=2,
                       label=f'{current_n:,} doublets')
            ax1.legend(fontsize=11)
            ax2.legend(fontsize=11)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_doublet_vs_metric(adata: sc.AnnData, metric: str) -> plt.Figure:
        """Scatter plot of doublet score vs a QC metric, highlighting predicted doublets"""
        if 'doublet_score' not in adata.obs or metric not in adata.obs:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Required data not available", ha='center', va='center', transform=ax.transAxes)
            ax.set_axis_off()
            return fig
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        singlets = adata.obs[~adata.obs.get('predicted_doublets', pd.Series([False]*adata.n_obs))]
        doublets = adata.obs[adata.obs.get('predicted_doublets', pd.Series([False]*adata.n_obs))]
        
        # Plot singlets
        ax.scatter(singlets[metric], singlets['doublet_score'],
                   color='#1f77b4', alpha=0.6, s=25, label='Singlets', edgecolors='none')
        
        # Plot predicted doublets on top
        if len(doublets) > 0:
            ax.scatter(doublets[metric], doublets['doublet_score'],
                       color='#d62728', alpha=0.9, s=40, label='Predicted Doublets',
                       edgecolors='black', linewidth=0.8)
        
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Doublet Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Doublet Score vs {metric.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        
        if len(doublets) > 0:
            ax.legend(fontsize=11, loc='upper left')
        
        plt.tight_layout()
        return fig