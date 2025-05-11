"""
Data Processor Module for loading and preprocessing the iTrust dataset.
"""
import os
import json
from typing import Dict, List, Tuple, Set
import networkx as nx
from pathlib import Path

class iTrustDataProcessor:
    def __init__(self, dataset_path: str):
        """
        Initialize the iTrust data processor.
        
        Args:
            dataset_path: Path to the iTrust dataset folder
        """
        self.dataset_path = Path(dataset_path)
        self.requirements: Dict[str, str] = {}
        self.code_files: Dict[str, str] = {}
        self.call_graph: nx.DiGraph = nx.DiGraph()
        self.solution_links: List[Tuple[str, str]] = []
        
    def load_requirements(self) -> Dict[str, str]:
        """Load requirements from the req folder"""
        try:
            # Check if cached requirements exist
            cache_path = self.dataset_path / 'cache' / 'requirements.json'
            if cache_path.exists():
                print("Loading cached requirements...")
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.requirements = json.load(f)
                print(f"Loaded {len(self.requirements)} cached requirements")
                return self.requirements

            req_dir = self.dataset_path / 'req'
            if not req_dir.exists():
                print(f"Warning: Requirements directory not found at {req_dir}")
                return self.requirements
                
            for req_file in req_dir.glob('*.txt'):
                try:
                    with open(req_file, 'r', encoding='utf-8') as f:
                        # Use full filename to match solution links
                        file_id = req_file.name
                        self.requirements[file_id] = f.read().strip()
                except Exception as e:
                    print(f"Warning: Failed to read requirement file {req_file}: {str(e)}")
            
            # Cache the requirements
            cache_path.parent.mkdir(exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.requirements, f)
            
            return self.requirements
        except Exception as e:
            print(f"Error loading requirements: {str(e)}")
            return self.requirements
    
    def load_code_files(self) -> Dict[str, str]:
        """Load Java code files"""
        try:
            # Check if cached code files exist
            cache_path = self.dataset_path / 'cache' / 'code_files.json'
            if cache_path.exists():
                print("Loading cached code files...")
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.code_files = json.load(f)
                print(f"Loaded {len(self.code_files)} cached code files")
                return self.code_files

            code_dir = self.dataset_path / 'code'
            if not code_dir.exists():
                print(f"Warning: Code directory not found at {code_dir}")
                return self.code_files
            
            # Create a mapping from filename to full path
            file_mapping = {}
            for root, _, files in os.walk(code_dir):
                for file in files:
                    if file.endswith('.java'):
                        file_mapping[file] = os.path.join(root, file)
            
            # Load code files using just the filename as key
            for filename, filepath in file_mapping.items():
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.code_files[filename] = f.read().strip()
                except Exception as e:
                    print(f"Warning: Failed to read code file {filepath}: {str(e)}")
            
            # Cache the code files
            cache_path.parent.mkdir(exist_ok=True)
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.code_files, f)
            
            print(f"Successfully loaded {len(self.code_files)} code files")
            return self.code_files
        except Exception as e:
            print(f"Error loading code files: {str(e)}")
            return self.code_files
    
    def load_call_graph(self) -> nx.DiGraph:
        """Load and parse the call graph from raw call graph file"""
        try:
            call_graph_path = self.dataset_path / 'itrust_raw_callgraph.txt'
            print(f"\nAttempting to load call graph from: {call_graph_path}")
            print(f"Absolute path: {call_graph_path.absolute()}")
            print(f"File exists: {call_graph_path.exists()}")
            
            if not call_graph_path.exists():
                print(f"Warning: Call graph file not found at {call_graph_path}")
                return self.call_graph
            
            print("Found call graph file, loading relationships...")
            
            # Updated relationship weights
            relationship_weights = {
                'inherits': 0.95,    # Increased inheritance weight
                'implements': 0.95,   # Interface implementation equally important
                'imports': 0.70,      # Reduced import weight
                'calls': 0.85,        # Direct method calls more important
                'references': 0.60,   # General references less important
                'contains': 0.75,     # Package relationships more significant
                'overrides': 0.90,    # Method overrides highly relevant
                'uses': 0.80         # Direct usage relationships
            }
            
            with open(call_graph_path, 'r', encoding='utf-8') as f:
                line_count = 0
                relationship_count = 0
                for line in f:
                    try:
                        line = line.strip()
                        if not line:
                            continue
                            
                        line_count += 1
                        if line_count == 1:
                            print(f"First line sample: {line}")
                            
                        # Parse relationship type and classes
                        if not line.startswith('C:'):
                            continue  # Skip non-class relationships
                            
                        # Remove the 'C:' prefix and split the remaining parts
                        parts = line[2:].strip().split()
                        if len(parts) < 2:
                            print(f"Warning: Invalid class parts in line {line_count}: {line}")
                            continue
                            
                        source = parts[0]
                        target = parts[1]
                        
                        # Extract class names without package
                        source_class = source.split('.')[-1]
                        target_class = target.split('.')[-1]
                        
                        # Determine relationship type with more granular analysis
                        rel_type = 'references'  # default type
                        
                        if target == 'java.lang.Object':
                            rel_type = 'inherits'
                        elif '[L' in target:  # Array type
                            rel_type = 'uses'
                        elif target.startswith('java.'):
                            rel_type = 'imports'
                        elif source.split('.')[-2] == target.split('.')[-2]:  # Same package
                            rel_type = 'contains'
                        else:
                            rel_type = 'calls'
                        
                        # Add nodes with full package info
                        self.call_graph.add_node(source_class, 
                                               package='.'.join(source.split('.')[:-1]),
                                               is_interface='Interface' in source)
                        self.call_graph.add_node(target_class, 
                                               package='.'.join(target.split('.')[:-1]),
                                               is_interface='Interface' in target)
                        
                        # Add edge with relationship type and weight
                        weight = relationship_weights.get(rel_type, 0.5)
                        
                        # Boost weight for same-package relationships
                        if source.split('.')[-2] == target.split('.')[-2]:
                            weight *= 1.2
                        
                        self.call_graph.add_edge(
                            source_class,
                            target_class,
                            type=rel_type,
                            weight=weight
                        )
                        relationship_count += 1
                        
                    except Exception as e:
                        print(f"Warning: Failed to parse call graph line {line_count}: {str(e)}")
                        continue
                    
                print(f"\nProcessed {line_count} lines")
                print(f"Added {len(self.call_graph.nodes)} nodes and {len(self.call_graph.edges)} edges")
                print(f"Created {relationship_count} relationships")
                
                return self.call_graph
        except Exception as e:
            print(f"Error loading call graph: {str(e)}")
            return self.call_graph
    
    def load_solution_links(self) -> List[Tuple[str, str]]:
        """Load solution links from the solution links file"""
        try:
            solution_links_path = self.dataset_path / 'itrust_solution_links_english.txt'
            if not solution_links_path.exists():
                print(f"Warning: Solution links file not found at {solution_links_path}")
                return self.solution_links
            
            print("Loading solution links...")
            with open(solution_links_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Handle both Windows and Unix line endings
                        line = line.strip().replace('\r', '')
                        if not line:
                            continue
                            
                        # Split on colon and clean up
                        req_id, code_id = line.split(':')
                        req_id = req_id.strip()
                        code_id = code_id.strip()
                        
                        # Add to solution links
                        self.solution_links.append((req_id, code_id))
                    except Exception as e:
                        print(f"Warning: Failed to parse solution link line: {str(e)}")
                        continue
            
            print(f"Loaded {len(self.solution_links)} solution links")
            return self.solution_links
        except Exception as e:
            print(f"Error loading solution links: {str(e)}")
            return self.solution_links
    
    def extract_class_info(self, code: str) -> Dict[str, any]:
        """
        Extract class information from Java code
        
        Args:
            code: Java source code
            
        Returns:
            Dictionary containing class name, methods, and other metadata
        """
        try:
            lines = code.split('\n')
            class_info = {
                'name': '',
                'package': '',
                'superclass': '',
                'interfaces': [],
                'methods': [],
                'fields': [],
                'imports': [],
                'api_calls': set(),
                'visibility': '',
                'is_abstract': False
            }
            
            in_method = False
            current_method = {'name': '', 'params': [], 'return_type': '', 'visibility': ''}
            
            for line in lines:
                try:
                    line = line.strip()
                    
                    # Extract package
                    if line.startswith('package '):
                        class_info['package'] = line[8:].rstrip(';')
                        
                    # Extract imports
                    elif line.startswith('import '):
                        import_stmt = line[7:].rstrip(';')
                        class_info['imports'].append(import_stmt)
                        # Track API usage
                        if 'java.' in import_stmt or 'javax.' in import_stmt:
                            class_info['api_calls'].add(import_stmt)
                    
                    # Extract class declaration
                    elif 'class ' in line and (line.startswith('public ') or line.startswith('private ') or line.startswith('protected ')):
                        parts = line.split(' ')
                        class_info['visibility'] = parts[0]
                        class_info['is_abstract'] = 'abstract' in parts
                        
                        # Get class name and inheritance
                        class_idx = parts.index('class')
                        class_info['name'] = parts[class_idx + 1].split('{')[0]
                        
                        # Extract superclass and interfaces
                        if 'extends' in parts:
                            extends_idx = parts.index('extends')
                            class_info['superclass'] = parts[extends_idx + 1].split('{')[0]
                        
                        if 'implements' in parts:
                            impl_idx = parts.index('implements')
                            interfaces = parts[impl_idx + 1].split('{')[0].split(',')
                            class_info['interfaces'] = [i.strip() for i in interfaces]
                    
                    # Extract methods
                    elif any(modifier in line for modifier in ['public', 'private', 'protected']) and '(' in line and ')' in line:
                        if not in_method:
                            in_method = True
                            method_parts = line.split('(')
                            declaration = method_parts[0].split()
                            
                            current_method = {
                                'name': declaration[-1],
                                'visibility': declaration[0],
                                'return_type': declaration[-2] if len(declaration) > 2 else 'void',
                                'params': [],
                                'throws': []
                            }
                            
                            # Extract parameters
                            params = method_parts[1].split(')')[0].split(',')
                            current_method['params'] = [p.strip() for p in params if p.strip()]
                            
                            # Extract throws clause
                            if 'throws' in line:
                                throws_part = line.split('throws')[1].split('{')[0]
                                current_method['throws'] = [t.strip() for t in throws_part.split(',')]
                            
                            class_info['methods'].append(current_method)
                    
                    # Method end
                    elif in_method and line.startswith('}'):
                        in_method = False
                        current_method = {'name': '', 'params': [], 'return_type': '', 'visibility': ''}
                    
                    # Extract fields
                    elif any(modifier in line for modifier in ['public', 'private', 'protected']) and ';' in line and '(' not in line:
                        field_parts = line.rstrip(';').split()
                        if len(field_parts) >= 3:
                            field = {
                                'name': field_parts[-1],
                                'type': field_parts[-2],
                                'visibility': field_parts[0],
                                'is_static': 'static' in field_parts,
                                'is_final': 'final' in field_parts
                            }
                            class_info['fields'].append(field)
                            
                except Exception as e:
                    print(f"Warning: Failed to parse line: {line}: {str(e)}")
                    continue
                    
            # Convert api_calls to list for JSON serialization
            class_info['api_calls'] = list(class_info['api_calls'])
            return class_info
            
        except Exception as e:
            print(f"Error extracting class info: {str(e)}")
            return {
                'name': '',
                'package': '',
                'superclass': '',
                'interfaces': [],
                'methods': [],
                'fields': [],
                'imports': [],
                'api_calls': [],
                'visibility': '',
                'is_abstract': False
            }
    
    def get_related_classes(self, class_name: str, depth: int = 1) -> Set[str]:
        """
        Get related classes from the call graph up to specified depth
        
        Args:
            class_name: Name of the class to find relations for
            depth: How many levels deep to traverse the call graph
            
        Returns:
            Set of related class names
        """
        try:
            related = set()
            current = {class_name}
            
            for _ in range(depth):
                next_level = set()
                for node in current:
                    try:
                        # Add successors (called by this class)
                        next_level.update(self.call_graph.successors(node))
                        # Add predecessors (classes calling this class)
                        next_level.update(self.call_graph.predecessors(node))
                    except Exception as e:
                        print(f"Warning: Failed to get relations for node {node}: {str(e)}")
                        continue
                related.update(current)
                current = next_level
                
            return related
        except Exception as e:
            print(f"Error getting related classes: {str(e)}")
            return set()
    
    def load_all(self):
        """Load all dataset components"""
        print("Loading requirements...")
        self.load_requirements()
        
        print("Loading code files...")
        self.load_code_files()
        
        print("Loading call graph...")
        self.load_call_graph()
        
        print("Loading solution links...")
        self.load_solution_links()
        
        # Print summary
        print(f"\nDataset Summary:")
        print(f"- Requirements loaded: {len(self.requirements)}")
        print(f"- Code files loaded: {len(self.code_files)}")
        print(f"- Call graph nodes: {len(self.call_graph.nodes)}")
        print(f"- Solution links loaded: {len(self.solution_links)}") 