import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import issparse
import re

class QCMetrics:
    @staticmethod
    def calculate_qc_metrics(adata, mt_pattern="^MT-|^mt-", ribo_pattern="^RPS|^RPL|^Rps|^Rpl"):
        adata = adata.copy()
        sc.pp.calculate_qc_metrics(adata, inplace=True)

        adata.var['mt'] = adata.var_names.str.contains(mt_pattern, case=False)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

        adata.var['ribo'] = adata.var_names.str.contains(ribo_pattern, case=False)
        sc.pp.calculate_qc_metrics(adata, qc_vars=['ribo'], inplace=True)

        return adata

    @staticmethod
    def get_comprehensive_metrics(adata):
        metrics = {
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'median_genes_per_cell': adata.obs['n_genes_by_counts'].median(),
            'median_umi_per_cell': adata.obs['total_counts'].median(),
            'median_pct_mito': adata.obs['pct_counts_mt'].median() if 'pct_counts_mt' in adata.obs else 0,
            'median_pct_ribo': adata.obs['pct_counts_ribo'].median() if 'pct_counts_ribo' in adata.obs else 0,
            'sparsity': 1 - (adata.X.nnz / (adata.n_obs * adata.n_vars)) if issparse(adata.X) else 1 - np.count_nonzero(adata.X)/(adata.n_obs * adata.n_vars),
            'zero_count_genes': int(np.sum(np.ravel(adata.X.sum(axis=0)) == 0)),
        }
        return metrics

    @staticmethod
    def get_detailed_metrics_table(adata):
        metrics_data = []
        cell_metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo']
        for metric in cell_metrics:
            if metric in adata.obs:
                stats = {
                    'Metric': metric.replace('_', ' ').title(),
                    'Mean': adata.obs[metric].mean(),
                    'Median': adata.obs[metric].median(),
                    'Std': adata.obs[metric].std(),
                    'Min': adata.obs[metric].min(),
                    'Max': adata.obs[metric].max(),
                    'Q1': adata.obs[metric].quantile(0.25),
                    'Q3': adata.obs[metric].quantile(0.75),
                }
                metrics_data.append(stats)
        return pd.DataFrame(metrics_data)

    @staticmethod
    def plot_qc_violin(adata):
        metrics = ['total_counts', 'n_genes_by_counts', 'pct_counts_mt', 'pct_counts_ribo']
        metrics = [m for m in metrics if m in adata.obs]
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 6))
        if len(metrics) == 1: axes = [axes]
        palette = sns.color_palette("muted", len(metrics))
        for ax, metric, color in zip(axes, metrics, palette):
            sns.violinplot(y=adata.obs[metric], ax=ax, color=color, inner="quartile")
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel("Value")
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_highest_expressed_genes(adata, top_n=20):
        # As in sc-best-practices tutorial
        sc.pl.highest_expr_genes(adata, n_top=top_n, show=False)
        fig = plt.gcf()
        fig.suptitle("Top 20 Most Highly Expressed Genes", fontsize=14, y=0.98)
        return fig

    @staticmethod
    def plot_umi_gene_scatter(adata):
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(adata.obs['total_counts'], adata.obs['n_genes_by_counts'],
                             c=adata.obs.get('pct_counts_mt', None), cmap='viridis', alpha=0.7, s=12)
        if 'pct_counts_mt' in adata.obs:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Mitochondrial %')
        ax.set_xlabel('Total UMI Counts')
        ax.set_ylabel('Detected Genes per Cell')
        ax.set_title('UMI Counts vs. Genes per Cell')
        return fig

    @staticmethod
    def plot_gene_distribution(adata):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(adata.obs['n_genes_by_counts'], kde=True, ax=ax, color='#1f77b4', bins=60)
        median = adata.obs['n_genes_by_counts'].median()
        ax.axvline(median, color='red', linestyle='--', label=f'Median: {median:.0f}')
        ax.legend()
        ax.set_xlabel('Genes per Cell')
        ax.set_title('Distribution of Detected Genes per Cell')
        return fig

    @staticmethod
    def plot_umi_distribution(adata):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(adata.obs['total_counts'], kde=True, ax=ax, color='#ff7f0e', bins=60)
        median = adata.obs['total_counts'].median()
        ax.axvline(median, color='red', linestyle='--', label=f'Median: {median:.0f}')
        ax.legend()
        ax.set_xlabel('UMI Counts per Cell')
        ax.set_title('Distribution of UMI Counts per Cell')
        return fig

    @staticmethod
    def plot_qc_correlation(adata):
        metrics = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_ribo']
        metrics = [m for m in metrics if m in adata.obs]
        corr = adata.obs[metrics].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax, fmt='.2f', linewidths=0.5)
        ax.set_title('Correlation Between QC Metrics')
        return fig

    @staticmethod
    def generate_qc_report(adata):
        metrics = QCMetrics.get_comprehensive_metrics(adata)
        report = {
            'quality_assessment': 'Good',
            'recommended_action': 'Proceed with filtering',
            'potential_issues': [],
            'completeness_score': 1.0
        }
        issues = []
        score_factors = []

        if metrics['median_genes_per_cell'] < 500:
            issues.append("Low median genes per cell (<500) — possible empty droplets or poor capture")
            score_factors.append(0.4)
        else:
            score_factors.append(1.0)

        if metrics['median_pct_mito'] > 15:
            issues.append("High mitochondrial content (>15%) — likely apoptotic or dying cells")
            score_factors.append(0.3)
        elif metrics['median_pct_mito'] > 8:
            issues.append("Elevated mitochondrial content (8–15%) — monitor for stressed cells")
            score_factors.append(0.7)
        else:
            score_factors.append(1.0)

        if metrics['sparsity'] > 0.95:
            issues.append("High data sparsity (>95%) — may need deeper sequencing")
            score_factors.append(0.7)
        else:
            score_factors.append(1.0)

        report['potential_issues'] = issues
        report['completeness_score'] = np.mean(score_factors) if score_factors else 1.0

        if report['completeness_score'] < 0.6:
            report['quality_assessment'] = 'Poor'
            report['recommended_action'] = 'Re-process or aggressive filtering required'
        elif report['completeness_score'] < 0.85:
            report['quality_assessment'] = 'Moderate'
            report['recommended_action'] = 'Apply standard filtering (e.g., MAD-based)'

        return report