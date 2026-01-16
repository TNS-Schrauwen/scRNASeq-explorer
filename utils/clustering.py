import scanpy as sc
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import streamlit as st


class Clustering:
    """Handles clustering operations for scRNA-seq data"""
    
    @staticmethod
    def compute_neighbors(adata: sc.AnnData, 
                         n_neighbors: int = 15,
                         n_pcs: int = 50) -> sc.AnnData:
        """Compute neighborhood graph"""
        adata_copy = adata.copy()
        
        if 'X_pca' not in adata_copy.obsm:
            st.warning("PCA not computed. Computing PCA first...")
            sc.pp.pca(adata_copy, n_comps=min(50, adata_copy.n_vars - 1))
        
        sc.pp.neighbors(adata_copy, n_neighbors=n_neighbors, n_pcs=n_pcs)
        return adata_copy
    
    @staticmethod
    def leiden_clustering(adata: sc.AnnData, 
                         resolution: float = 1.0,
                         key_added: str = 'leiden') -> sc.AnnData:
        """Perform Leiden clustering"""
        adata_copy = adata.copy()
        
        if 'neighbors' not in adata_copy.uns:
            st.warning("Neighborhood graph not computed. Computing neighbors first...")
            adata_copy = Clustering.compute_neighbors(adata_copy)
        
        sc.tl.leiden(adata_copy, resolution=resolution, key_added=key_added)
        return adata_copy
    
    @staticmethod
    def louvain_clustering(adata: sc.AnnData, 
                          resolution: float = 1.0,
                          key_added: str = 'louvain') -> sc.AnnData:
        """Perform Louvain clustering"""
        adata_copy = adata.copy()
        
        if 'neighbors' not in adata_copy.uns:
            st.warning("Neighborhood graph not computed. Computing neighbors first...")
            adata_copy = Clustering.compute_neighbors(adata_copy)
        
        sc.tl.louvain(adata_copy, resolution=resolution, key_added=key_added)
        return adata_copy
    
    @staticmethod
    def get_cluster_stats(adata: sc.AnnData, cluster_key: str) -> Dict[str, Any]:
        """Get statistics for clustering results"""
        if cluster_key not in adata.obs:
            raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")
        
        cluster_counts = adata.obs[cluster_key].value_counts().sort_index()
        
        stats = {
            'n_clusters': len(cluster_counts),
            'cluster_sizes': cluster_counts.to_dict(),
            'largest_cluster': cluster_counts.max(),
            'smallest_cluster': cluster_counts.min(),
            'median_cluster_size': cluster_counts.median(),
            'total_cells': cluster_counts.sum()
        }
        
        return stats
    
    @staticmethod
    def compare_clusterings(adata: sc.AnnData, 
                           cluster_key1: str, 
                           cluster_key2: str) -> Dict[str, Any]:
        """Compare two different clustering results"""
        if cluster_key1 not in adata.obs or cluster_key2 not in adata.obs:
            raise ValueError("Both cluster keys must be present in adata.obs")
        
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        ari = adjusted_rand_score(adata.obs[cluster_key1], adata.obs[cluster_key2])
        nmi = normalized_mutual_info_score(adata.obs[cluster_key1], adata.obs[cluster_key2])
        
        contingency = pd.crosstab(adata.obs[cluster_key1], adata.obs[cluster_key2])
        
        comparison = {
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi,
            'contingency_table': contingency,
            'cluster_key1': cluster_key1,
            'cluster_key2': cluster_key2,
            'n_clusters1': adata.obs[cluster_key1].nunique(),
            'n_clusters2': adata.obs[cluster_key2].nunique()
        }
        
        return comparison