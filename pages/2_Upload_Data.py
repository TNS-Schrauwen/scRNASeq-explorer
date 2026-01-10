# ============================
# upload_data.py 
# ============================
import streamlit as st
import os
import pandas as pd
from utils.data_io import DataIO
from utils.workspace_manager import WorkspaceManager


def main():
    st.title("Upload Single-Cell Data (Multiple Samples or Single File)")
    st.markdown("""
    **Step 1**: Define your experimental groups (e.g., Control, Treated) if uploading .h5 files.  
    **Step 2**: Upload either multiple 10x HDF5 files (.h5) or a single .h5ad file.  
    This ensures correct metadata for downstream comparisons.
    """)

    if not st.session_state.get('workspace'):
        st.error("Please create or load a workspace first!")
        st.stop()

    workspace_manager = st.session_state.workspace_manager
    workspace_path = st.session_state.workspace_path

    with st.expander("Supported Formats"):
        st.markdown("""
        - **10x Genomics HDF5 (.h5)** – Multiple samples allowed (one file per sample).
        - **AnnData HDF5 (.h5ad)** – Single file only (pre-merged dataset).
        """)

    # === Upload Files ===
    st.subheader("Upload Files")
    st.info("Upload either multiple .h5 files (one per sample) or a single .h5ad file.")

    uploaded_files = st.file_uploader(
        "Upload files (.h5 or .h5ad)",
        type=['h5', 'h5ad'],
        accept_multiple_files=True,
        help="Multiple .h5 files for individual samples or one .h5ad for a pre-merged dataset."
    )

    if uploaded_files:
        file_types = set(os.path.splitext(f.name)[1].lower() for f in uploaded_files)
        n_files = len(uploaded_files)

        if len(file_types) > 1:
            st.error("Mixed file types uploaded. Please upload either all .h5 or a single .h5ad.")
            st.stop()

        if '.h5ad' in file_types:
            if n_files != 1:
                st.error("Only one .h5ad file is supported.")
                st.stop()
            # Handle single .h5ad
            handle_single_h5ad(uploaded_files[0], workspace_manager, workspace_path)
        elif '.h5' in file_types:
            # Handle multiple .h5 (or single .h5 as special case)
            handle_multiple_h5(uploaded_files, workspace_manager, workspace_path)
        else:
            st.error("Unsupported file type.")
            st.stop()

    # Preview after load (common for both paths)
    if st.session_state.get('adata'):
        st.divider()
        st.header("Combined Dataset Summary")
        adata = st.session_state.adata
        summary = DataIO.get_dataset_summary(adata)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Cells", f"{summary['n_cells']:,}")
        col2.metric("Genes/Features", f"{summary['n_genes']:,}")
        col3.metric("Samples", summary['n_samples'])
        col4.metric("Groups", summary['n_groups'])
        col5.metric("Sparsity", f"{summary['sparsity']:.2%}")

        with st.expander("🔍 Preview"):
            st.subheader("Dataset Contents")
            st.write("**Obs Columns (Cell Metadata):**", adata.obs.columns.tolist())
            st.write("**Var Columns (Gene Metadata):**", adata.var.columns.tolist())
            st.write("**Layers:**", list(adata.layers.keys()))
            st.write("**Unstructured Annotations (uns):**", list(adata.uns.keys()))
            st.write("**Obsm Keys (Embeddings):**", list(adata.obsm.keys()))
            st.write("**Varm Keys:**", list(adata.varm.keys()))

            st.subheader("Cell Distribution by Sample & Group")
            if 'sample' in adata.obs and 'group' in adata.obs:
                dist = adata.obs[['sample', 'group']].value_counts().reset_index(name="Cell Count")
                st.dataframe(dist.sort_values(['group', 'sample']), use_container_width=True)
            else:
                st.write("No 'sample' or 'group' metadata available.")

            st.subheader("Obs Metadata (First 1000)")
            st.dataframe(adata.obs.head(1000), use_container_width=True)

            st.subheader("Var Metadata (First 1000)")
            st.dataframe(adata.var.head(1000), use_container_width=True)

        if st.button("Go to Quality Control →", type="primary", use_container_width=True):
            st.switch_page("pages/3_Quality_Control.py")


def handle_multiple_h5(uploaded_files, workspace_manager, workspace_path):
    # === STEP 1: Define Groups ===
    st.subheader("1. Define Groups")
    group_input = st.text_input("Enter group names (comma-separated)", "Control, Treated")
    groups = [g.strip() for g in group_input.split(",") if g.strip()]
    if not groups:
        st.warning("Please enter at least one group name.")
        st.stop()

    st.success(f"Defined groups: **{', '.join(groups)}**")

    # === STEP 2: Assign Files to Groups ===
    st.subheader("2. Assign Files to Groups")

    is_valid, message = DataIO.validate_uploaded_files(uploaded_files, file_type='h5')
    if not is_valid:
        st.error(f"Invalid: {message}")
        st.stop()

    n_files = len(uploaded_files)
    st.success(f"✅ {n_files} .h5 files uploaded")

    # Auto-generate sample IDs (will be made unique internally)
    sample_ids = []
    for i, f in enumerate(uploaded_files):
        base = os.path.splitext(f.name)[0]
        sample_ids.append(base if i == 0 or base not in sample_ids else f"{base}_{i+1}")

    # Assignment table
    assignment_df = pd.DataFrame({
        "Sample ID": sample_ids,
        "Filename": [f.name for f in uploaded_files],
        "Group": [""] * n_files
    })

    st.write("Assign each sample to a group:")
    edited_df = st.data_editor(
        assignment_df,
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Sample ID": st.column_config.TextColumn(disabled=True),
            "Filename": st.column_config.TextColumn(disabled=True),
            "Group": st.column_config.SelectboxColumn(
                "Group",
                options=groups,
                required=True
            )
        }
    )

    # Validate assignments
    if edited_df["Group"].isin(groups).all() and not edited_df["Group"].isnull().any():
        sample_to_group = dict(zip(edited_df["Sample ID"], edited_df["Group"]))
    else:
        st.error("All samples must be assigned to one of the defined groups.")
        st.stop()

    # Group summary
    group_counts = edited_df["Group"].value_counts().sort_index()
    st.write("**Assignment Summary:**")
    st.dataframe(group_counts.to_frame(name="Samples Assigned"))

    if st.button("Load & Merge Data", type="primary", use_container_width=True):
        with st.spinner("Merging samples..."):
            try:
                adata = DataIO.load_multiple_10x_h5(uploaded_files, sample_to_group)

                st.session_state.adata = adata

                data_dir = os.path.join(workspace_path, "data")
                os.makedirs(data_dir, exist_ok=True)
                save_path = os.path.join(data_dir, "raw_data.h5ad")
                adata.write_h5ad(save_path)

                workspace_manager.update_workspace_metadata(
                    st.session_state.workspace,
                    {
                        "data_loaded": True,
                        "data_format": "multiple_10x_h5",
                        "data_shape": [adata.n_obs, adata.n_vars],
                        "n_samples": adata.obs['sample'].nunique(),
                        "groups": sorted(adata.obs['group'].cat.categories.tolist()),
                        "steps_completed": ["data_upload"],
                    },
                )

                st.success("All samples merged successfully!")
                st.balloons()

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)


def handle_single_h5ad(uploaded_file, workspace_manager, workspace_path):
    is_valid, message = DataIO.validate_uploaded_files([uploaded_file], file_type='h5ad')
    if not is_valid:
        st.error(f"Invalid: {message}")
        st.stop()

    st.success(f"✅ 1 .h5ad file uploaded: {uploaded_file.name}")

    if st.button("Load Data", type="primary", use_container_width=True):
        with st.spinner("Loading .h5ad file..."):
            try:
                adata = DataIO.load_single_h5ad(uploaded_file)

                st.session_state.adata = adata

                data_dir = os.path.join(workspace_path, "data")
                os.makedirs(data_dir, exist_ok=True)
                save_path = os.path.join(data_dir, "raw_data.h5ad")
                adata.write_h5ad(save_path)

                # Detect groups and samples
                groups = sorted(adata.obs['group'].cat.categories.tolist()) if 'group' in adata.obs else []
                n_samples = adata.obs['sample'].nunique() if 'sample' in adata.obs else 1

                workspace_manager.update_workspace_metadata(
                    st.session_state.workspace,
                    {
                        "data_loaded": True,
                        "data_format": "single_h5ad",
                        "data_shape": [adata.n_obs, adata.n_vars],
                        "n_samples": n_samples,
                        "groups": groups,
                        "steps_completed": ["data_upload"],
                    },
                )

                st.success("Dataset loaded successfully!")
                st.balloons()

                # Show immediate info
                st.info(f"This .h5ad file contains {n_samples} sample(s) and {len(groups)} group(s): {', '.join(groups) if groups else 'None defined'}.")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()