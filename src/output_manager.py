"""
Output manager for saving and loading system outputs
"""
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any

class OutputManager:
    def __init__(self, output_dir: str = "outputs"):
        """Initialize output manager with directory paths"""
        self.output_dir = Path(output_dir)
        self.summaries_dir = self.output_dir / "summaries"
        self.indexes_dir = self.output_dir / "indexes"
        self.metrics_dir = self.output_dir / "metrics"
        
        # Create directories if they don't exist
        for dir_path in [self.summaries_dir, self.indexes_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_code_summaries(self, summaries: Dict[str, str], filename: str = "code_summaries.json"):
        """Save code summaries, handling both new and pre-generated formats"""
        try:
            # First try to load existing summaries to merge
            existing_summaries = {}
            try:
                with open(self.summaries_dir / "summaries.json", 'r', encoding='utf-8') as f:
                    existing_summaries = json.load(f)
            except FileNotFoundError:
                pass
            
            # Merge with new summaries, new ones take precedence
            merged_summaries = {**existing_summaries, **summaries}
            
            # Save to both files for compatibility
            output_paths = [
                self.summaries_dir / "summaries.json",
                self.summaries_dir / filename
            ]
            
            for output_path in output_paths:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_summaries, f, indent=2)
                print(f"Saved code summaries to {output_path}")
                
        except Exception as e:
            print(f"Warning: Error saving code summaries: {str(e)}")
    
    def save_requirement_summaries(self, summaries: Dict[str, str], filename: str = "requirement_summaries.json"):
        """Save requirement summaries"""
        try:
            output_path = self.summaries_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summaries, f, indent=2)
            print(f"Saved requirement summaries to {output_path}")
        except Exception as e:
            print(f"Warning: Error saving requirement summaries: {str(e)}")
    
    def save_index_data(self, index_name: str, data: Any):
        """Save index data (vectors, matrices, etc.)"""
        try:
            output_path = self.indexes_dir / f"{index_name}.pkl"
            pd.to_pickle(data, output_path)
            print(f"Saved index data to {output_path}")
        except Exception as e:
            print(f"Warning: Error saving index data: {str(e)}")
    
    def save_evaluation_metrics(self, metrics: Dict[str, float], filename: str = "evaluation_metrics.json"):
        """Save evaluation metrics"""
        try:
            output_path = self.metrics_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            print(f"Saved evaluation metrics to {output_path}")
        except Exception as e:
            print(f"Warning: Error saving evaluation metrics: {str(e)}")
            
    def save_traceability_links(self, links: List[tuple], filename: str):
        """Save generated traceability links"""
        try:
            output_path = self.output_dir / "traces" / filename
            output_path.parent.mkdir(exist_ok=True)
            
            # Convert links to dictionary format
            links_dict = {
                "requirement_id": [link[0] for link in links],
                "code_id": [link[1] for link in links]
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(links_dict, f, indent=2)
            print(f"Saved traceability links to {output_path}")
        except Exception as e:
            print(f"Warning: Error saving traceability links: {str(e)}") 