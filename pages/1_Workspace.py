import streamlit as st
import os
import yaml
from utils.workspace_manager import WorkspaceManager

def main():
    st.title("🏠 Workspace Management")
    st.markdown("Create or load a workspace to organize your single-cell analysis projects.")
    
    # Initialize workspace manager
    if 'workspace_manager' not in st.session_state:
        config = st.session_state.get('config', {})
        base_dir = config.get('workspace', {}).get('base_dir', './workspaces')
        st.session_state.workspace_manager = WorkspaceManager(base_dir)
    
    workspace_manager = st.session_state.workspace_manager
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Create New Workspace")
        
        with st.form("create_workspace"):
            workspace_name = st.text_input("Workspace Name", 
                                         placeholder="e.g., PBMC_analysis")
            workspace_description = st.text_area("Description", 
                                               placeholder="Describe your project...")
            
            if st.form_submit_button("Create Workspace", use_container_width=True):
                if workspace_name:
                    try:
                        workspace_path = workspace_manager.create_workspace(
                            workspace_name, workspace_description
                        )
                        st.session_state.workspace = workspace_name
                        st.session_state.workspace_path = workspace_path
                        st.session_state.adata = None
                        st.success(f"Workspace '{workspace_name}' created successfully!")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.error("Please enter a workspace name.")
    
    with col2:
        st.header("Load Existing Workspace")
        
        existing_workspaces = workspace_manager.list_workspaces()
        
        if existing_workspaces:
            selected_workspace = st.selectbox(
                "Select Workspace",
                existing_workspaces,
                index=None,
                placeholder="Choose a workspace..."
            )
            
            if st.button("Load Workspace", use_container_width=True):
                if selected_workspace:
                    try:
                        workspace_path = workspace_manager.load_workspace(selected_workspace)
                        st.session_state.workspace = selected_workspace
                        st.session_state.workspace_path = workspace_path
                        
                        # Try to load existing AnnData if available
                        adata_path = os.path.join(workspace_path, "data", "processed_data.h5ad")
                        if os.path.exists(adata_path):
                            import scanpy as sc
                            st.session_state.adata = sc.read_h5ad(adata_path)
                            st.success(f"Workspace '{selected_workspace}' loaded with existing data!")
                        else:
                            st.session_state.adata = None
                            st.success(f"Workspace '{selected_workspace}' loaded!")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading workspace: {str(e)}")
                else:
                    st.error("Please select a workspace.")
        else:
            st.info("No existing workspaces found. Create a new workspace to get started.")
    
    # Current workspace info
    if st.session_state.get('workspace'):
        st.divider()
        st.header("Current Workspace")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Workspace", st.session_state.workspace)
        
        with col2:
            status = "Data Loaded" if st.session_state.get('adata') is not None else "No Data"
            st.metric("Data Status", status)
        
        with col3:
            if st.session_state.get('adata') is not None:
                st.metric("Cells", f"{st.session_state.adata.n_obs:,}")
                st.metric("Genes", f"{st.session_state.adata.n_vars:,}")
            else:
                st.metric("Next Step", "Upload Data")
        
        # Workspace metadata
        try:
            metadata = workspace_manager.get_workspace_metadata(st.session_state.workspace)
            with st.expander("Workspace Details"):
                st.json(metadata)
                
                if st.button("Delete Workspace", type="secondary"):
                    if st.checkbox("I understand this will permanently delete the workspace and all its data"):
                        workspace_manager.delete_workspace(st.session_state.workspace)
                        st.session_state.workspace = None
                        st.session_state.workspace_path = None
                        st.session_state.adata = None
                        st.success("Workspace deleted successfully!")
                        st.rerun()
        
        except Exception as e:
            st.error(f"Error loading workspace metadata: {str(e)}")
    
    # Quick start guide
    st.divider()
    st.header("Getting Started")
    
    steps = [
        "1. **Create or Load** a workspace to organize your analysis",
        "2. **Upload Data** in the next page (supports .h5ad, .loom, 10X, CSV/TSV)",
        "3. **Follow the workflow** through quality control, preprocessing, and analysis",
        "4. **Export** your results and reproducible code"
    ]
    
    for step in steps:
        st.markdown(step)

if __name__ == "__main__":
    main()