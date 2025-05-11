import os
import re
from typing import Dict, List, Set, Tuple
import json
from pathlib import Path

class AdaDataProcessor:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.code_files: Dict[str, str] = {}
        self.requirements: Dict[str, str] = {}
        self.solution_links: Dict[str, Set[str]] = {}
        
    def load_code_files(self) -> Dict[str, str]:
        """Load all Ada source files from the dataset."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(('.ada', '.adb', '.ads')):
                    file_path = os.path.join(root, file)
                    content = None
                    
                    # Try different encodings
                    for encoding in encodings:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                                break  # If successful, break the encoding loop
                        except UnicodeDecodeError:
                            continue
                        except Exception as e:
                            print(f"Error reading {file_path} with {encoding}: {e}")
                            continue
                    
                    if content is not None:
                        # Store relative path as key
                        rel_path = os.path.relpath(file_path, self.dataset_path)
                        self.code_files[rel_path] = content
                    else:
                        print(f"Warning: Could not read {file_path} with any encoding")
        
        print(f"Successfully loaded {len(self.code_files)} Ada files")
        return self.code_files
    
    def extract_requirements(self) -> Dict[str, str]:
        """Extract requirements from Ada comments and documentation."""
        requirements = {}
        req_id = 1
        
        for file_path, content in self.code_files.items():
            # Look for package documentation in .ads files
            if file_path.endswith('.ads'):
                # Extract package documentation
                pkg_match = re.search(r'package\s+(\w+)', content)
                if pkg_match:
                    pkg_name = pkg_match.group(1)
                    # Look for comments before package declaration
                    comment_match = re.search(r'--\s*(.*?)(?=\n\s*package)', content, re.DOTALL)
                    if comment_match:
                        req_text = comment_match.group(1).strip()
                        if req_text:
                            requirements[f"REQ{req_id}"] = f"Package {pkg_name}: {req_text}"
                            req_id += 1
                
                # Extract procedure/function documentation
                proc_matches = re.finditer(r'procedure\s+(\w+).*?(?=--|$)', content, re.DOTALL)
                for match in proc_matches:
                    proc_name = match.group(1)
                    # Look for comments before procedure
                    comment_match = re.search(r'--\s*(.*?)(?=\n\s*procedure)', content[:match.start()], re.DOTALL)
                    if comment_match:
                        req_text = comment_match.group(1).strip()
                        if req_text:
                            requirements[f"REQ{req_id}"] = f"Procedure {proc_name}: {req_text}"
                            req_id += 1
                
                func_matches = re.finditer(r'function\s+(\w+).*?(?=--|$)', content, re.DOTALL)
                for match in func_matches:
                    func_name = match.group(1)
                    # Look for comments before function
                    comment_match = re.search(r'--\s*(.*?)(?=\n\s*function)', content[:match.start()], re.DOTALL)
                    if comment_match:
                        req_text = comment_match.group(1).strip()
                        if req_text:
                            requirements[f"REQ{req_id}"] = f"Function {func_name}: {req_text}"
                            req_id += 1
        
        print(f"Extracted {len(requirements)} requirements")
        self.requirements = requirements
        return requirements
    
    def generate_ground_truth(self) -> Dict[str, Set[str]]:
        """Generate ground truth links based on Ada package dependencies."""
        solution_links = {}
        
        for req_id in self.requirements.keys():
            solution_links[req_id] = set()
            
            # Find files that contain the requirement text
            req_text = self.requirements[req_id].lower()
            for file_path, content in self.code_files.items():
                if req_text in content.lower():
                    solution_links[req_id].add(file_path)
        
        self.solution_links = solution_links
        return solution_links
    
    def save_summaries(self, output_dir: str):
        """Save code and requirement summaries."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save code summaries
        code_summaries = {}
        for file_path, content in self.code_files.items():
            summary = self._generate_code_summary(file_path, content)
            code_summaries[file_path] = summary
        
        with open(os.path.join(output_dir, 'code_summaries.json'), 'w') as f:
            json.dump(code_summaries, f, indent=2)
        
        # Save requirement summaries
        with open(os.path.join(output_dir, 'requirement_summaries.json'), 'w') as f:
            json.dump(self.requirements, f, indent=2)
    
    def _generate_code_summary(self, file_path: str, content: str) -> Dict:
        """Generate a summary for an Ada source file."""
        summary = {
            'file_path': file_path,
            'package_name': '',
            'dependencies': set(),
            'procedures': [],
            'functions': [],
            'types': [],
            'constants': []
        }
        
        # Extract package name
        if file_path.endswith('.ads'):
            pkg_match = re.search(r'package\s+(\w+)', content)
            if pkg_match:
                summary['package_name'] = pkg_match.group(1)
        
        # Extract dependencies
        with_matches = re.finditer(r'with\s+([\w\.]+)', content)
        for match in with_matches:
            summary['dependencies'].add(match.group(1))
        
        # Extract procedures
        proc_matches = re.finditer(r'procedure\s+(\w+)', content)
        for match in proc_matches:
            summary['procedures'].append(match.group(1))
        
        # Extract functions
        func_matches = re.finditer(r'function\s+(\w+)', content)
        for match in func_matches:
            summary['functions'].append(match.group(1))
        
        # Extract types
        type_matches = re.finditer(r'type\s+(\w+)', content)
        for match in type_matches:
            summary['types'].append(match.group(1))
        
        # Extract constants
        const_matches = re.finditer(r'(\w+)\s*:\s*constant', content)
        for match in const_matches:
            summary['constants'].append(match.group(1))
        
        # Convert set to list for JSON serialization
        summary['dependencies'] = list(summary['dependencies'])
        
        return summary 