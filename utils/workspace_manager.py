import os
import json
import shutil
from datetime import datetime
from typing import Optional, Dict, Any
import streamlit as st


class WorkspaceManager:
    """Manages workspace creation, loading, and metadata storage"""
    
    def __init__(self, base_dir: str = "./workspaces"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def create_workspace(self, name: str, description: str = "") -> str:
        """Create a new workspace directory structure"""
        workspace_path = os.path.join(self.base_dir, name)
        
        if os.path.exists(workspace_path):
            raise ValueError(f"Workspace '{name}' already exists")
        
        # Create directory structure
        os.makedirs(workspace_path)
        os.makedirs(os.path.join(workspace_path, "data"))
        os.makedirs(os.path.join(workspace_path, "results"))
        os.makedirs(os.path.join(workspace_path, "plots"))
        os.makedirs(os.path.join(workspace_path, "logs"))
        
        # Create metadata file
        metadata = {
            "name": name,
            "description": description,
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "steps_completed": [],
            "parameters": {}
        }
        
        with open(os.path.join(workspace_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return workspace_path
    
    def load_workspace(self, name: str) -> str:
        """Load an existing workspace"""
        workspace_path = os.path.join(self.base_dir, name)
        
        if not os.path.exists(workspace_path):
            raise ValueError(f"Workspace '{name}' does not exist")
        
        return workspace_path
    
    def list_workspaces(self) -> list:
        """List all available workspaces"""
        workspaces = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, "metadata.json")
                if os.path.exists(metadata_path):
                    workspaces.append(item)
        return sorted(workspaces)
    
    def get_workspace_metadata(self, name: str) -> Dict[str, Any]:
        """Get workspace metadata"""
        workspace_path = os.path.join(self.base_dir, name)
        metadata_path = os.path.join(workspace_path, "metadata.json")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def update_workspace_metadata(self, name: str, updates: Dict[str, Any]):
        """Update workspace metadata"""
        workspace_path = os.path.join(self.base_dir, name)
        metadata_path = os.path.join(workspace_path, "metadata.json")
        
        metadata = self.get_workspace_metadata(name)
        metadata.update(updates)
        metadata["modified"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def delete_workspace(self, name: str):
        """Delete a workspace"""
        workspace_path = os.path.join(self.base_dir, name)
        if os.path.exists(workspace_path):
            shutil.rmtree(workspace_path)