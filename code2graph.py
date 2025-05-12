"""
Module for parsing Ada files to extract entities and relationships.
This module will create nodes and relationships for a graph database.
"""
import os
import re
import time
import json
import pickle
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Set

from constants import ADA_CODE_DIR, ADA_EXTENSIONS, OUTPUT_DIR

# Create checkpoints directory
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

def print_timestamp(message):
    """Print a message with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """Print a progress bar to the console."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

class AdaNode:
    """Represents a node in the Ada code graph."""
    def __init__(self, name: str, node_type: str, file_path: str, line_number: int, 
                 body: str = "", parent: str = None):
        self.id = f"{node_type.lower()}-{name}"
        self.name = name
        self.type = node_type
        self.file_path = file_path
        self.line_number = line_number
        self.body = body
        self.parent = parent
        self.children = []
        self.references = []
        self.summary = ""
    
    def __repr__(self):
        return f"{self.type} {self.name} at {self.file_path}:{self.line_number}"

def save_checkpoint(data, checkpoint_name):
    """Save checkpoint to avoid reprocessing."""
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{checkpoint_name}.pickle")
    try:
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        print_timestamp(f"✅ Saved checkpoint: {checkpoint_path}")
        
        # Also save a small JSON metadata file with timestamp for visibility
        metadata = {
            "checkpoint": checkpoint_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "items": len(data) if isinstance(data, (list, dict)) else "N/A"
        }
        with open(os.path.join(CHECKPOINTS_DIR, f"{checkpoint_name}_meta.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        print_timestamp(f"⚠️ Failed to save checkpoint: {str(e)}")

def load_checkpoint(checkpoint_name):
    """Load checkpoint if it exists."""
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{checkpoint_name}.pickle")
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            print_timestamp(f"✅ Loaded checkpoint: {checkpoint_path}")
            return data
        except Exception as e:
            print_timestamp(f"⚠️ Failed to load checkpoint: {str(e)}")
    return None

def find_ada_files(directory: Path) -> List[Path]:
    """Find all Ada files in a directory and its subdirectories."""
    # Try to load from checkpoint first
    checkpoint_data = load_checkpoint("ada_files")
    if checkpoint_data:
        return checkpoint_data
    
    print_timestamp(f"🔍 STARTED: Searching for Ada files in {directory}...")
    
    if not os.path.exists(directory):
        print_timestamp(f"⚠️ Directory {directory} does not exist")
        return []
    
    # First, count how many files we have in total for progress tracking
    total_files = 0
    for root, _, files in os.walk(directory):
        total_files += len(files)
    
    print_timestamp(f"📊 Found {total_files} total files to check for Ada extensions")
    
    # Now find the Ada files with progress tracking
    ada_files = []
    files_checked = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            files_checked += 1
            if files_checked % 100 == 0 or files_checked == total_files:
                print_progress_bar(files_checked, total_files, prefix=f'Checking files: {files_checked}/{total_files}', suffix='Complete', length=40)
            
            if any(file.endswith(ext) for ext in ADA_EXTENSIONS):
                ada_files.append(Path(os.path.join(root, file)))
    
    print_timestamp(f"✅ COMPLETED: Found {len(ada_files)} Ada files out of {total_files} total files")
    
    # Print some example files
    if ada_files:
        print_timestamp("📋 Sample Ada files:")
        for i, file in enumerate(ada_files[:5]):
            print(f"  - {file}")
        if len(ada_files) > 5:
            print(f"  - ... and {len(ada_files) - 5} more files")
    
    # Save to checkpoint
    save_checkpoint(ada_files, "ada_files")
    
    return ada_files

def extract_package_info(file_path: Path) -> Tuple[str, str]:
    """Extract package name and body from a file."""
    print_timestamp(f"📄 Processing file: {file_path}")
    
    try:
        file_size_kb = os.path.getsize(file_path) / 1024
        start_time = time.time()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        read_time = time.time() - start_time
        print_timestamp(f"  📖 Read {file_size_kb:.1f} KB in {read_time:.3f} seconds")
        
        # Extract package name
        package_match = re.search(r'package\s+(\w+(\.\w+)*)', content, re.IGNORECASE)
        package_name = package_match.group(1) if package_match else os.path.basename(file_path).split('.')[0]
        
        print_timestamp(f"  📦 Found package: {package_name}")
        return package_name, content
    
    except Exception as e:
        print_timestamp(f"⚠️ Error reading file {file_path}: {str(e)}")
        return os.path.basename(file_path).split('.')[0], ""

def extract_procedures(content: str, file_path: str, package_name: str) -> List[AdaNode]:
    """Extract procedure definitions from Ada content."""
    start_time = time.time()
    procedure_nodes = []
    
    # Find procedures
    procedure_matches = list(re.finditer(
        r'procedure\s+(\w+)(\s*\([^)]*\))?\s*(?:is|;)', 
        content, 
        re.IGNORECASE
    ))
    
    if procedure_matches:
        print_timestamp(f"  🔍 Found {len(procedure_matches)} procedure matches, extracting details...")
    
    for i, match in enumerate(procedure_matches):
        if i % 20 == 0 and len(procedure_matches) > 50:
            print_progress_bar(i, len(procedure_matches), prefix=f'   Processing procedures: {i}/{len(procedure_matches)}', suffix='Complete', length=30)
            
        proc_name = match.group(1)
        line_number = content[:match.start()].count('\n') + 1
        
        # Try to extract the procedure body
        proc_start = match.start()
        body = ""
        
        # If this is a spec file (.ads), there may not be a body
        if file_path.endswith('.adb'):
            # Try to find the end of the procedure
            proc_end_match = re.search(
                r'end\s+' + proc_name + r'\s*;', 
                content[proc_start:], 
                re.IGNORECASE
            )
            
            if proc_end_match:
                proc_end = proc_start + proc_end_match.end()
                body = content[proc_start:proc_end]
        
        node = AdaNode(
            name=proc_name,
            node_type="PROCEDURE",
            file_path=file_path,
            line_number=line_number,
            body=body,
            parent=package_name
        )
        procedure_nodes.append(node)
    
    if len(procedure_matches) > 50:
        print_progress_bar(len(procedure_matches), len(procedure_matches), prefix=f'   Processing procedures: {len(procedure_matches)}/{len(procedure_matches)}', suffix='Complete', length=30)
    
    end_time = time.time()
    if procedure_nodes:
        print_timestamp(f"  ✅ Extracted {len(procedure_nodes)} procedures in {end_time - start_time:.3f} seconds")
    
    return procedure_nodes

def extract_functions(content: str, file_path: str, package_name: str) -> List[AdaNode]:
    """Extract function definitions from Ada content."""
    start_time = time.time()
    function_nodes = []
    
    # Find functions
    function_matches = list(re.finditer(
        r'function\s+(\w+)(\s*\([^)]*\))?\s*return\s+\w+(?:\s+is|\s*;)', 
        content, 
        re.IGNORECASE
    ))
    
    if function_matches:
        print_timestamp(f"  🔍 Found {len(function_matches)} function matches, extracting details...")
    
    for i, match in enumerate(function_matches):
        if i % 20 == 0 and len(function_matches) > 50:
            print_progress_bar(i, len(function_matches), prefix=f'   Processing functions: {i}/{len(function_matches)}', suffix='Complete', length=30)
            
        func_name = match.group(1)
        line_number = content[:match.start()].count('\n') + 1
        
        # Try to extract the function body
        func_start = match.start()
        body = ""
        
        # If this is a spec file (.ads), there may not be a body
        if file_path.endswith('.adb'):
            # Try to find the end of the function
            func_end_match = re.search(
                r'end\s+' + func_name + r'\s*;', 
                content[func_start:], 
                re.IGNORECASE
            )
            
            if func_end_match:
                func_end = func_start + func_end_match.end()
                body = content[func_start:func_end]
        
        node = AdaNode(
            name=func_name,
            node_type="FUNCTION",
            file_path=file_path,
            line_number=line_number,
            body=body,
            parent=package_name
        )
        function_nodes.append(node)
    
    if len(function_matches) > 50:
        print_progress_bar(len(function_matches), len(function_matches), prefix=f'   Processing functions: {len(function_matches)}/{len(function_matches)}', suffix='Complete', length=30)
    
    end_time = time.time()
    if function_nodes:
        print_timestamp(f"  ✅ Extracted {len(function_nodes)} functions in {end_time - start_time:.3f} seconds")
    
    return function_nodes

def extract_types(content: str, file_path: str, package_name: str) -> List[AdaNode]:
    """Extract type definitions from Ada content."""
    start_time = time.time()
    type_nodes = []
    
    # Find types
    type_matches = list(re.finditer(
        r'type\s+(\w+)\s+is', 
        content, 
        re.IGNORECASE
    ))
    
    if type_matches:
        print_timestamp(f"  🔍 Found {len(type_matches)} type matches, extracting details...")
    
    for i, match in enumerate(type_matches):
        if i % 20 == 0 and len(type_matches) > 50:
            print_progress_bar(i, len(type_matches), prefix=f'   Processing types: {i}/{len(type_matches)}', suffix='Complete', length=30)
            
        type_name = match.group(1)
        line_number = content[:match.start()].count('\n') + 1
        
        # Get the type definition
        type_start = match.start()
        type_end = content.find(';', type_start)
        if type_end != -1:
            body = content[type_start:type_end + 1]
        else:
            body = content[type_start:type_start + 100] + "..."  # Just get the first 100 chars if we can't find the end
        
        node = AdaNode(
            name=type_name,
            node_type="TYPE",
            file_path=file_path,
            line_number=line_number,
            body=body,
            parent=package_name
        )
        type_nodes.append(node)
    
    if len(type_matches) > 50:
        print_progress_bar(len(type_matches), len(type_matches), prefix=f'   Processing types: {len(type_matches)}/{len(type_matches)}', suffix='Complete', length=30)
    
    end_time = time.time()
    if type_nodes:
        print_timestamp(f"  ✅ Extracted {len(type_nodes)} types in {end_time - start_time:.3f} seconds")
    
    return type_nodes

def extract_references(nodes: List[AdaNode], max_nodes_to_analyze=None, min_node_name_length=3) -> Dict[str, Set[str]]:
    """
    Extract references between nodes based on name usage.
    
    Args:
        nodes: List of Ada nodes to analyze
        max_nodes_to_analyze: Maximum number of nodes to analyze (None for all)
        min_node_name_length: Minimum length of node names to look for references (avoid short names like "a", "i")
    """
    # Try to load from checkpoint first
    checkpoint_data = load_checkpoint("node_references")
    if checkpoint_data:
        return checkpoint_data
    
    start_time = time.time()
    print_timestamp(f"🔄 STARTED: Analyzing references between {len(nodes)} nodes...")
    
    # Performance optimization: limit number of nodes to analyze if specified
    if max_nodes_to_analyze and max_nodes_to_analyze < len(nodes):
        print_timestamp(f"⚠️ PERFORMANCE: Limiting analysis to {max_nodes_to_analyze} nodes (out of {len(nodes)})")
        nodes_to_process = nodes[:max_nodes_to_analyze]
    else:
        nodes_to_process = nodes
    
    # Filter nodes with very short names (likely to cause false positives)
    eligible_nodes = [n for n in nodes if len(n.name) >= min_node_name_length]
    print_timestamp(f"ℹ️ Using {len(eligible_nodes)} nodes with names of at least {min_node_name_length} characters for reference detection")
    
    references = {}
    reference_count = 0
    
    # Create a node name lookup for faster reference checks
    node_lookup = {node.name: node.id for node in eligible_nodes}
    
    # Process in batches to enable periodic checkpoint saving
    BATCH_SIZE = 100
    total_nodes = len(nodes_to_process)
    
    for batch_start in range(0, total_nodes, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_nodes)
        batch = nodes_to_process[batch_start:batch_end]
        
        for i, node in enumerate(batch):
            global_i = batch_start + i
            if global_i % 100 == 0 or global_i == total_nodes - 1:
                print_progress_bar(global_i+1, total_nodes, prefix=f'Analyzing node references: {global_i+1}/{total_nodes}', suffix=f'({reference_count} refs found)', length=40)
                # Save checkpoint every 500 nodes
                if global_i > 0 and global_i % 500 == 0:
                    save_checkpoint(references, "node_references")
            
            if node.body:
                references[node.id] = set()
                
                # Only check for references if body has substantial content
                if len(node.body) > 10:
                    # For efficiency, do a quick check for node names using sets
                    body_words = set(re.findall(r'\b\w+\b', node.body))
                    
                    for other_name, other_id in node_lookup.items():
                        # Skip self-references
                        if other_id == node.id:
                            continue
                            
                        # Only do regex search if the name appears in the body text
                        if other_name in body_words:
                            # Confirm with proper word boundary regex
                            if re.search(r'\b' + re.escape(other_name) + r'\b', node.body):
                                references[node.id].add(other_id)
                                reference_count += 1
    
    # Save final checkpoint
    save_checkpoint(references, "node_references")
    
    duration = time.time() - start_time
    print_timestamp(f"✅ COMPLETED: Found {reference_count} references between nodes in {duration:.2f} seconds ({reference_count/duration:.1f} refs/sec)")
    
    # Print distribution of references
    if references:
        ref_counts = [len(refs) for refs in references.values() if refs]
        if ref_counts:
            avg_refs = sum(ref_counts) / len(ref_counts)
            max_refs = max(ref_counts) if ref_counts else 0
            print_timestamp(f"📊 Reference stats: avg {avg_refs:.1f} refs per node, max {max_refs} refs")
    
    return references

def parse_ada_files() -> Tuple[List[AdaNode], Dict]:
    """Parse all Ada files in the code directory and extract nodes and relationships."""
    # Try to load from checkpoint first
    all_nodes = load_checkpoint("all_nodes")
    if all_nodes:
        # If we have nodes checkpoint, try to load references as well
        references = load_checkpoint("node_references")
        if references:
            print_timestamp("✅ Loaded all data from checkpoints")
            return all_nodes, references
    
    print_timestamp("🔄 STARTED: Ada code parsing...")
    total_start_time = time.time()
    
    ada_files = find_ada_files(ADA_CODE_DIR)
    
    if not ada_files:
        print_timestamp("⚠️ No Ada files found. Creating an empty graph.")
        return [], {}
    
    all_nodes = []
    package_nodes = {}
    
    # First pass: extract package nodes
    print_timestamp("\n📦 STARTED: First pass - Extracting package nodes...")
    first_pass_start = time.time()
    
    for i, file_path in enumerate(ada_files):
        if i % 10 == 0 or i == len(ada_files) - 1:
            print_progress_bar(i+1, len(ada_files), prefix=f'Processing files (pass 1): {i+1}/{len(ada_files)}', suffix='Complete', length=40)
        
        package_name, content = extract_package_info(file_path)
        str_path = str(file_path)
        
        if package_name not in package_nodes:
            package_nodes[package_name] = AdaNode(
                name=package_name,
                node_type="PACKAGE",
                file_path=str_path,
                line_number=1,
                body=content
            )
            all_nodes.append(package_nodes[package_name])
    
    first_pass_duration = time.time() - first_pass_start
    print_timestamp(f"✅ COMPLETED: First pass - Extracted {len(package_nodes)} package nodes in {first_pass_duration:.2f} seconds")
    
    # Save checkpoint after first pass
    save_checkpoint(all_nodes, "package_nodes")
    
    # Second pass: extract procedures, functions, and types
    print_timestamp("\n📦 STARTED: Second pass - Extracting procedures, functions, and types...")
    second_pass_start = time.time()
    proc_count, func_count, type_count = 0, 0, 0
    
    for i, file_path in enumerate(ada_files):
        if i % 5 == 0 or i == len(ada_files) - 1:
            print_progress_bar(i+1, len(ada_files), prefix=f'Processing files (pass 2): {i+1}/{len(ada_files)}', suffix=f'P:{proc_count} F:{func_count} T:{type_count}', length=40)
            # Save checkpoint every 20 files
            if i > 0 and i % 20 == 0:
                save_checkpoint(all_nodes, "all_nodes")
        
        package_name, content = extract_package_info(file_path)
        str_path = str(file_path)
        
        # Extract procedures, functions, and types
        procedures = extract_procedures(content, str_path, package_name)
        functions = extract_functions(content, str_path, package_name)
        types = extract_types(content, str_path, package_name)
        
        proc_count += len(procedures)
        func_count += len(functions)
        type_count += len(types)
        
        # Add to all nodes
        all_nodes.extend(procedures)
        all_nodes.extend(functions)
        all_nodes.extend(types)
        
        # Add as children to package node
        if package_name in package_nodes:
            package_nodes[package_name].children.extend([node.id for node in procedures + functions + types])
    
    second_pass_duration = time.time() - second_pass_start
    
    print_timestamp(f"\n📊 Entity extraction summary:")
    print_timestamp(f"  - {len(package_nodes)} packages")
    print_timestamp(f"  - {proc_count} procedures")
    print_timestamp(f"  - {func_count} functions")
    print_timestamp(f"  - {type_count} types")
    print_timestamp(f"  - {len(all_nodes)} total Ada entities")
    print_timestamp(f"  - Extraction rate: {len(all_nodes)/second_pass_duration:.1f} entities/second")
    
    # Save checkpoint of all nodes
    save_checkpoint(all_nodes, "all_nodes")
    
    # Extract references with performance optimizations - limit to 1000 nodes for large codebases
    # and only look for references to names at least 3 characters long
    max_nodes = 1000 if len(all_nodes) > 2000 else None
    references = extract_references(all_nodes, max_nodes_to_analyze=max_nodes, min_node_name_length=3)
    
    total_duration = time.time() - total_start_time
    print_timestamp(f"✅ COMPLETED: Ada code parsing completed in {total_duration:.2f} seconds")
    
    return all_nodes, references

def build_code_graph():
    """Main function to parse Ada files and build a graph of code entities."""
    print_timestamp("🏗️ STARTED: Building Ada code graph...")
    start_time = time.time()
    
    nodes, references = parse_ada_files()
    
    # Add references to nodes
    print_timestamp("📝 Adding references to nodes...")
    reference_count = 0
    
    for i, (node_id, ref_ids) in enumerate(references.items()):
        if i % 500 == 0 or i == len(references) - 1:
            print_progress_bar(i+1, len(references), prefix=f'Adding references: {i+1}/{len(references)}', suffix=f'({reference_count} refs)', length=40)
        
        for node in nodes:
            if node.id == node_id:
                node.references = list(ref_ids)
                reference_count += len(ref_ids)
                break
    
    # Save final checkpoint of nodes with references
    save_checkpoint(nodes, "nodes_with_references")
    
    end_time = time.time()
    print_timestamp(f"📊 Code graph statistics:")
    print_timestamp(f"  - {len(nodes)} nodes")
    print_timestamp(f"  - {reference_count} references")
    print_timestamp(f"✅ COMPLETED: Built Ada code graph in {end_time - start_time:.2f} seconds")
    
    return nodes

if __name__ == "__main__":
    overall_start_time = time.time()
    print_timestamp("🚀 STARTED: Ada code analysis")
    
    nodes = build_code_graph()
    
    overall_end_time = time.time()
    duration = overall_end_time - overall_start_time
    
    print_timestamp(f"\n✅ COMPLETED: Extracted {len(nodes)} nodes from Ada code in {duration:.2f} seconds")
    
    # Print node type distribution
    node_types = {}
    for node in nodes:
        if node.type not in node_types:
            node_types[node.type] = 0
        node_types[node.type] += 1
    
    print_timestamp("\n📊 Node type distribution:")
    for node_type, count in node_types.items():
        print_timestamp(f"  - {node_type}: {count} nodes")
    
    # Print some example nodes
    print_timestamp("\n📋 Sample nodes:")
    for i, node in enumerate(nodes[:10]):
        print(f"{i+1}. {node.type}: {node.name}")
        if node.children:
            print(f"   Children: {len(node.children)}")
        if node.references:
            print(f"   References: {len(node.references)}") 