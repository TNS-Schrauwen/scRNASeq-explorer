import scanpy as sc
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import streamlit as st
from scipy.sparse import issparse
import scipy


class Preprocessing:
    """Handles all preprocessing steps for scRNA-seq data"""
    
    @staticmethod
    def calculate_qc_metrics(adata: sc.AnnData) -> sc.AnnData:
        """Calculate quality control metrics"""
        adata_copy = adata.copy()
        
        # Calculate basic QC metrics
        sc.pp.calculate_qc_metrics(adata_copy, inplace=True)
        
        # Calculate mitochondrial genes percentage
        if 'mt' not in adata_copy.var_keys():
            # Try to identify mitochondrial genes
            mt_genes = adata_copy.var_names.str.startswith('MT-') | adata_copy.var_names.str.startswith('mt-')
            if mt_genes.any():
                adata_copy.var['mt'] = mt_genes
            else:
                # Fallback: look for common mitochondrial gene patterns
                mt_patterns = ['^MT-', '^mt-', '^MT.', '^mt.', 'MTRNR', 'mtrnr']
                for pattern in mt_patterns:
                    mt_genes = adata_copy.var_names.str.contains(pattern, case=False, regex=True)
                    if mt_genes.any():
                        adata_copy.var['mt'] = mt_genes
                        break
                else:
                    adata_copy.var['mt'] = False
        
        if 'mt' in adata_copy.var_keys() and adata_copy.var['mt'].any():
            sc.pp.calculate_qc_metrics(adata_copy, qc_vars=['mt'], inplace=True)
        
        return adata_copy
    
    @staticmethod
    def filter_cells(adata: sc.AnnData, 
                    min_genes: int = 200, 
                    max_genes: int = 2500,
                    min_counts: Optional[int] = None,
                    max_counts: Optional[int] = None,
                    max_mito: Optional[float] = 5.0) -> sc.AnnData:
        """Filter cells based on QC metrics"""
        adata_copy = adata.copy()
        
        # Apply filters
        if min_genes is not None:
            sc.pp.filter_cells(adata_copy, min_genes=min_genes)
        
        if max_genes is not None and 'n_genes_by_counts' in adata_copy.obs:
            adata_copy = adata_copy[adata_copy.obs.n_genes_by_counts <= max_genes, :]
        
        if min_counts is not None:
            sc.pp.filter_cells(adata_copy, min_counts=min_counts)
        
        if max_counts is not None and 'total_counts' in adata_copy.obs:
            adata_copy = adata_copy[adata_copy.obs.total_counts <= max_counts, :]
        
        if max_mito is not None and 'pct_counts_mt' in adata_copy.obs:
            adata_copy = adata_copy[adata_copy.obs.pct_counts_mt <= max_mito, :]
        
        return adata_copy
    
    @staticmethod
    def normalize_data(adata: sc.AnnData, 
                      target_sum: float = 1e4,
                      log_transform: bool = True,
                      regress_out: Optional[list] = None) -> sc.AnnData:
        """Normalize and transform data"""
        adata_copy = adata.copy()
        
        # Clean data before normalization
        Preprocessing._clean_data(adata_copy)
        
        # Normalize total counts
        sc.pp.normalize_total(adata_copy, target_sum=target_sum)
        
        # Log transform
        if log_transform:
            sc.pp.log1p(adata_copy)
        
        # Regress out variables if specified
        if regress_out and all(var in adata_copy.obs.columns for var in regress_out):
            sc.pp.regress_out(adata_copy, regress_out)
        
        return adata_copy
    
    @staticmethod
    def find_variable_genes(adata: sc.AnnData,
                           method: str = 'seurat',
                           n_top_genes: int = 2000,
                           min_mean: float = 0.0125,
                           max_mean: float = 3,
                           min_disp: float = 0.5) -> sc.AnnData:
        """Find highly variable genes"""
        adata_copy = adata.copy()
        
        # Enhanced cleaning: Filter genes with zero counts to prevent inf/NaN
        sc.pp.filter_genes(adata_copy, min_counts=1)
        
        # Clean data before finding variable genes
        Preprocessing._clean_data(adata_copy)
        
        # Find highly variable genes
        sc.pp.highly_variable_genes(
            adata_copy,
            flavor=method,
            n_top_genes=n_top_genes,
            min_mean=min_mean,
            max_mean=max_mean,
            min_disp=min_disp
        )
        
        return adata_copy
    
    @staticmethod
    def scale_data(adata: sc.AnnData, max_value: float = 10) -> sc.AnnData:
        """Scale data to unit variance and zero mean"""
        adata_copy = adata.copy()
        
        # Clean data before scaling
        Preprocessing._clean_data(adata_copy)
        
        sc.pp.scale(adata_copy, max_value=max_value)
        return adata_copy
    
    @staticmethod
    def _clean_data(adata: sc.AnnData):
        """Clean data by removing infinite and NaN values"""
        if adata.X is not None:
            if issparse(adata.X):
                # For sparse, convert to dense temporarily for cleaning (efficient for scRNA)
                X_dense = adata.X.toarray()
                X_dense[~np.isfinite(X_dense)] = 0
                adata.X = scipy.sparse.csr_matrix(X_dense)
            else:
                # For dense
                adata.X[~np.isfinite(adata.X)] = 0
    
    @staticmethod
    def get_normalized_counts(adata: sc.AnnData) -> np.ndarray:
        """Get normalized counts as 1D array for plotting"""
        if adata.X is None:
            return np.array([])
        
        # Calculate total counts per cell
        if issparse(adata.X):
            counts = np.array(adata.X.sum(axis=1)).flatten()
        else:
            counts = adata.X.sum(axis=1).flatten()
        
        # Clean the counts
        counts = counts[np.isfinite(counts)]
        
        return counts