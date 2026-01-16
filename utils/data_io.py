# ============================
# data_io.py
# ============================
import scanpy as sc
import tempfile
import os
import scipy.sparse as sp
import numpy as np
import pandas as pd
import warnings
from typing import List, Dict


class DataIO:
    """Handles loading and merging multiple 10x HDF5 files or single .h5ad with unique sample IDs"""

    @staticmethod
    def _save_temp_file(uploaded_file, suffix='.h5'):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(uploaded_file.getbuffer())
        tmp_file.close()
        return tmp_file.name

    @staticmethod
    def validate_uploaded_files(uploaded_files: List, file_type: str = None) -> tuple[bool, str]:
        if not uploaded_files:
            return False, "No files uploaded."

        for file in uploaded_files:
            ext = os.path.splitext(file.name)[1].lower()
            if file_type == 'h5' and ext != '.h5':
                return False, f"Only .h5 files are supported for this mode. Invalid: {file.name}"
            elif file_type == 'h5ad' and ext != '.h5ad':
                return False, f"Only .h5ad files are supported for this mode. Invalid: {file.name}"
            elif not file_type and ext not in ['.h5', '.h5ad']:
                return False, f"Only .h5 or .h5ad files are supported. Invalid: {file.name}"

        return True, f"{len(uploaded_files)} valid files detected."

    @staticmethod
    def load_multiple_10x_h5(uploaded_files: List, sample_to_group: Dict[str, str]) -> sc.AnnData:
        """
        Load multiple .h5 files, ensure unique sample IDs, concatenate, and add group metadata.
        sample_to_group: {sample_id: group_name}
        """
        adata_list = []
        base_sample_ids = []
        seen = {}

        for uploaded_file in uploaded_files:
            base_name = os.path.splitext(uploaded_file.name)[0]
            # Make unique sample ID
            if base_name not in seen:
                unique_id = base_name
                seen[base_name] = 1
            else:
                count = seen[base_name] + 1
                seen[base_name] = count
                unique_id = f"{base_name}_{count}"

            temp_path = DataIO._save_temp_file(uploaded_file, '.h5')
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message="Variable names are not unique")
                    adata = sc.read_10x_h5(temp_path)
                adata.var_names_make_unique()  # Safe merging

                adata.obs['sample'] = unique_id
                group = sample_to_group.get(unique_id, "unspecified")
                adata.obs['group'] = group

                adata_list.append(adata)
                base_sample_ids.append(unique_id)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        if not adata_list:
            raise ValueError("No data loaded.")

        # Concat with unique keys
        combined_adata = sc.concat(
            adata_list,
            join="outer",
            keys=base_sample_ids,      # Guaranteed unique
            index_unique="-"
        )

        # Final categorical cleanup
        combined_adata.obs['group'] = combined_adata.obs['group'].astype('category')
        combined_adata.obs['sample'] = combined_adata.obs['sample'].astype('category')

        return combined_adata

    @staticmethod
    def load_single_h5ad(uploaded_file) -> sc.AnnData:
        """
        Load a single .h5ad file, add default metadata if missing.
        """
        temp_path = DataIO._save_temp_file(uploaded_file, '.h5ad')
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="Variable names are not unique")
                adata = sc.read_h5ad(temp_path)
            adata.var_names_make_unique()

            # Add default 'sample' if missing
            if 'sample' not in adata.obs:
                adata.obs['sample'] = 'default_sample'
            adata.obs['sample'] = adata.obs['sample'].astype('category')

            # Add default 'group' if missing
            if 'group' not in adata.obs:
                adata.obs['group'] = 'unspecified'
            adata.obs['group'] = adata.obs['group'].astype('category')

            return adata
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def get_dataset_summary(adata: sc.AnnData) -> dict:
        X = adata.X
        if sp.issparse(X):
            sparsity = 1.0 - (X.nnz / (X.shape[0] * X.shape[1]))
        else:
            sparsity = 1.0 - (np.count_nonzero(X) / X.size)

        n_samples = adata.obs['sample'].nunique() if 'sample' in adata.obs else 1
        n_groups = adata.obs['group'].nunique() if 'group' in adata.obs else 0

        return {
            'n_cells': adata.n_obs,
            'n_genes': adata.n_vars,
            'n_samples': n_samples,
            'n_groups': n_groups,
            'sparsity': sparsity,
        }