"""
Module for parsing requirements files and creating requirement nodes.
"""
import os
import time
from pathlib import Path
from typing import Dict, List

from constants import REQUIREMENTS_DIR

class RequirementNode:
    """Represents a requirement node in the graph."""
    def __init__(self, req_id: str, text: str, file_path: str):
        self.id = f"requirement-{req_id}"
        self.req_id = req_id
        self.text = text
        self.file_path = file_path
        self.embedding = None
        self.related_nodes = []
    
    def __repr__(self):
        return f"Requirement {self.req_id}: {self.text[:50]}..."

def read_requirement_file(file_path: Path) -> str:
    """Read a requirement file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content
    except Exception as e:
        print(f"⚠️ Error reading requirement file {file_path}: {str(e)}")
        return f"Error reading requirement: {str(e)}"

def extract_requirement_id(file_name: str) -> str:
    """Extract the requirement ID from the file name."""
    # Assuming file names are in the format 'reqXX.txt'
    req_id = file_name.replace('req', '').replace('.txt', '')
    return req_id

def parse_requirements() -> List[RequirementNode]:
    """Parse all requirement files and create requirement nodes."""
    print("🔄 Starting requirements parsing...")
    start_time = time.time()
    
    requirement_nodes = []
    
    # Check if directory exists
    if not os.path.exists(REQUIREMENTS_DIR):
        print(f"⚠️ Requirements directory {REQUIREMENTS_DIR} does not exist")
        return requirement_nodes
    
    # Find all requirement files
    print(f"🔍 Searching for requirement files in {REQUIREMENTS_DIR}...")
    requirement_files = []
    for root, _, files in os.walk(REQUIREMENTS_DIR):
        for file in files:
            if file.startswith('req') and file.endswith('.txt'):
                requirement_files.append(Path(os.path.join(root, file)))
    
    if not requirement_files:
        print("⚠️ No requirement files found")
        return requirement_nodes
    
    print(f"✅ Found {len(requirement_files)} requirement files")
    
    # Print some example files
    if requirement_files:
        print("📋 Sample requirement files:")
        for i, file in enumerate(requirement_files[:5]):
            print(f"  - {file}")
        if len(requirement_files) > 5:
            print(f"  - ... and {len(requirement_files) - 5} more files")
    
    # Parse each requirement file
    print("\n📝 Parsing requirement files...")
    for i, file_path in enumerate(requirement_files):
        # Print progress every 10 files or for the first/last file
        if i % 10 == 0 or i == len(requirement_files) - 1:
            print(f"📄 Processing requirement file {i+1}/{len(requirement_files)}: {file_path.name}")
        
        req_id = extract_requirement_id(file_path.name)
        text = read_requirement_file(file_path)
        
        node = RequirementNode(
            req_id=req_id,
            text=text,
            file_path=str(file_path)
        )
        requirement_nodes.append(node)
        
        # Print sample content for the first few requirements
        if i < 3:
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"  📋 Requirement {req_id}: {preview}")
    
    end_time = time.time()
    print(f"✅ Parsed {len(requirement_nodes)} requirements in {end_time - start_time:.2f} seconds")
    
    # Sort requirements by ID for consistent ordering
    requirement_nodes.sort(key=lambda x: x.req_id)
    
    return requirement_nodes

def get_requirement_by_id(nodes: List[RequirementNode], req_id: str) -> RequirementNode:
    """Get a requirement node by its ID."""
    for node in nodes:
        if node.req_id == req_id:
            return node
    return None

if __name__ == "__main__":
    start_time = time.time()
    requirements = parse_requirements()
    end_time = time.time()
    
    print(f"\n✅ Extracted {len(requirements)} requirements in {end_time - start_time:.2f} seconds")
    
    # Print requirements with stats
    print("\n📊 Requirements statistics:")
    print(f"  - Total requirements: {len(requirements)}")
    
    # Calculate average length
    if requirements:
        avg_length = sum(len(req.text) for req in requirements) / len(requirements)
        print(f"  - Average requirement length: {avg_length:.1f} characters")
        
        # Find longest and shortest requirements
        longest = max(requirements, key=lambda x: len(x.text))
        shortest = min(requirements, key=lambda x: len(x.text))
        print(f"  - Longest requirement: Req {longest.req_id} ({len(longest.text)} chars)")
        print(f"  - Shortest requirement: Req {shortest.req_id} ({len(shortest.text)} chars)")
    
    # Print some requirements for debugging
    print("\n📋 Sample requirements:")
    for i, req in enumerate(requirements[:10]):
        print(f"{i+1}. Req {req.req_id}: {req.text[:100]}{'...' if len(req.text) > 100 else ''}") 