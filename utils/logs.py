import json
import os
from datetime import datetime
from typing import Dict, Any, List
import streamlit as st


class AnalysisLogger:
    """Handles logging of analysis steps and parameters"""
    
    def __init__(self, workspace_path: str):
        self.workspace_path = workspace_path
        self.logs_dir = os.path.join(workspace_path, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
    
    def log_step(self, step_name: str, parameters: Dict[str, Any], status: str = "completed"):
        """Log an analysis step with parameters"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'status': status,
            'parameters': parameters
        }
        
        # Append to main log file
        main_log_path = os.path.join(self.logs_dir, "analysis_log.jsonl")
        with open(main_log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Also update step-specific log
        step_log_path = os.path.join(self.logs_dir, f"{step_name}.json")
        step_data = {
            'last_run': datetime.now().isoformat(),
            'status': status,
            'parameters': parameters
        }
        
        with open(step_log_path, 'w') as f:
            json.dump(step_data, f, indent=2)
    
    def get_step_history(self, step_name: str) -> List[Dict[str, Any]]:
        """Get history of a specific step"""
        main_log_path = os.path.join(self.logs_dir, "analysis_log.jsonl")
        history = []
        
        if os.path.exists(main_log_path):
            with open(main_log_path, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry['step'] == step_name:
                        history.append(entry)
        
        return history
    
    def get_complete_log(self) -> List[Dict[str, Any]]:
        """Get complete analysis log"""
        main_log_path = os.path.join(self.logs_dir, "analysis_log.jsonl")
        log_entries = []
        
        if os.path.exists(main_log_path):
            with open(main_log_path, 'r') as f:
                for line in f:
                    log_entries.append(json.loads(line.strip()))
        
        return log_entries
    
    def generate_workflow_summary(self) -> Dict[str, Any]:
        """Generate a summary of the analysis workflow"""
        log_entries = self.get_complete_log()
        
        completed_steps = []
        failed_steps = []
        parameters = {}
        
        for entry in log_entries:
            if entry['status'] == 'completed':
                completed_steps.append(entry['step'])
            elif entry['status'] == 'failed':
                failed_steps.append(entry['step'])
            
            parameters[entry['step']] = entry['parameters']
        
        summary = {
            'total_steps': len(log_entries),
            'completed_steps': completed_steps,
            'failed_steps': failed_steps,
            'completion_rate': len(completed_steps) / len(log_entries) if log_entries else 0,
            'parameters': parameters,
            'last_update': datetime.now().isoformat()
        }
        
        return summary
    
    def export_log(self, format: str = 'json') -> str:
        """Export log in specified format"""
        log_entries = self.get_complete_log()
        
        if format == 'json':
            return json.dumps(log_entries, indent=2)
        elif format == 'text':
            text_lines = ["Analysis Log", "============", ""]
            
            for entry in log_entries:
                text_lines.append(f"Step: {entry['step']}")
                text_lines.append(f"Time: {entry['timestamp']}")
                text_lines.append(f"Status: {entry['status']}")
                text_lines.append("Parameters:")
                
                for key, value in entry['parameters'].items():
                    text_lines.append(f"  {key}: {value}")
                
                text_lines.append("")
            
            return "\n".join(text_lines)
        else:
            raise ValueError(f"Unsupported format: {format}")