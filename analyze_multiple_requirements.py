#!/usr/bin/env python3
"""
Script to analyze multiple requirements and load them all into Neo4j.
This provides a complete view of requirements traceability across the entire codebase.
"""
import os
import sys
import json
import time
from pathlib import Path
import argparse

from tqdm import tqdm
from graph_database import Neo4jConnector, print_timestamp
from analyze_single_req import Req2CodeAnalyzer
from code2graph import build_code_graph, load_checkpoint
from req2nodes import RequirementNode, parse_requirements

# Configuration
OUTPUT_DIR = os.path.join("outputs", "saved_analysis", "multiple_reqs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_multiple_requirements(requirements, output_dir=OUTPUT_DIR, top_n=10):
    """
    Analyze multiple requirements against the codebase.
    
    Args:
        requirements: List of (req_id, req_text) tuples
        output_dir: Directory to save results
        top_n: Number of top results to keep for each requirement
    
    Returns:
        Path to the combined analysis file
    """
    print_timestamp(f"Analyzing {len(requirements)} requirements...")
    
    # Initialize analyzer
    analyzer = Req2CodeAnalyzer()
    
    all_results = {
        "requirements": [],
        "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_requirements": len(requirements)
    }
    
    # Process each requirement
    for i, (req_id, req_text) in enumerate(tqdm(requirements, desc="Analyzing requirements")):
        print_timestamp(f"Analyzing requirement {i+1}/{len(requirements)}: {req_id}")
        
        # Analyze requirement
        results = analyzer.analyze_requirement(req_text, req_id, top_n)
        
        # Save individual results
        req_output_file = os.path.join(output_dir, f"req_{req_id}_analysis.json")
        with open(req_output_file, 'w') as f:
            json.dump({
                "requirement_id": req_id,
                "requirement_text": req_text,
                "top_files": results,
                "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }, f, indent=2)
        
        # Add to combined results
        all_results["requirements"].append({
            "requirement_id": req_id,
            "requirement_text": req_text,
            "top_files": results
        })
        
        print_timestamp(f"Completed analysis for {req_id}")
    
    # Save combined results
    combined_output_file = os.path.join(output_dir, "combined_analysis.json")
    with open(combined_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print_timestamp(f"Analysis complete. Results saved to {combined_output_file}")
    
    return combined_output_file

def load_requirements_from_file(file_path):
    """Load requirements from a file."""
    requirements = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                req_id = f"REQ-{i+1:02d}"
                requirements.append((req_id, line))
    return requirements

def load_requirements_to_neo4j(combined_analysis_file):
    """
    Load all requirements from the combined analysis into Neo4j.
    
    Args:
        combined_analysis_file: Path to the combined analysis file
    """
    print_timestamp(f"Loading multiple requirements into Neo4j...")
    
    # Load combined analysis
    with open(combined_analysis_file, 'r') as f:
        analysis = json.load(f)
    
    # Connect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    # Clear the database
    if input("Clear existing Neo4j database? (y/n): ").lower() == 'y':
        connector.clear_database()
    
    # Create constraints
    connector.create_constraints()
    
    # Load code nodes
    code_nodes = load_checkpoint("all_nodes")
    if not code_nodes:
        print_timestamp("No code nodes found in checkpoint. Please run build_code_graph() first.")
        return
    
    # Create a map of file paths to nodes
    file_to_nodes = {}
    for node in code_nodes:
        if node.file_path not in file_to_nodes:
            file_to_nodes[node.file_path] = []
        file_to_nodes[node.file_path].append(node)
    
    # Create a list to store unique files for reuse
    unique_files = {}
    
    # Process each requirement and its files
    for req_data in tqdm(analysis["requirements"], desc="Processing requirements"):
        req_id = req_data["requirement_id"]
        req_text = req_data["requirement_text"]
        
        # Create requirement node
        req_node = RequirementNode(req_id, req_text, "manual_input.txt")
        connector._execute_query("""
        MERGE (r:Requirement {id: $id})
        SET r.text = $text,
            r.req_id = $req_id
        """, {
            "id": req_node.id,
            "text": req_node.text,
            "req_id": req_node.req_id
        })
        print_timestamp(f"Created requirement node: {req_node.req_id}")
        
        # Process files for this requirement
        for i, file_info in enumerate(req_data["top_files"]):
            file_path = file_info["file_path"]
            score = file_info["score"]
            rank = i + 1
            
            # Create file node if it doesn't exist yet
            file_node_id = f"file-{Path(file_path).name}"
            if file_node_id not in unique_files:
                connector._execute_query("""
                MERGE (f:File {id: $id})
                SET f.path = $path,
                    f.name = $name
                """, {
                    "id": file_node_id,
                    "path": file_path,
                    "name": Path(file_path).name
                })
                unique_files[file_node_id] = file_path
                
                # Add nodes from the file if it's the first time we're seeing it
                if file_path in file_to_nodes:
                    nodes = file_to_nodes[file_path]
                    print_timestamp(f"Adding {len(nodes)} nodes from file: {file_path}")
                    
                    for node in nodes:
                        connector.create_ada_node(node)
                        
                        # Create relationship between file and node
                        connector._execute_query("""
                        MATCH (f:File {id: $file_id})
                        MATCH (n:AdaNode {id: $node_id})
                        MERGE (f)-[rel:CONTAINS]->(n)
                        """, {
                            "file_id": file_node_id,
                            "node_id": node.id
                        })
            
            # Create relationship between requirement and file
            connector._execute_query("""
            MATCH (r:Requirement {id: $req_id})
            MATCH (f:File {id: $file_id})
            MERGE (r)-[rel:RELATED_TO]->(f)
            SET rel.score = $score,
                rel.rank = $rank
            """, {
                "req_id": req_node.id,
                "file_id": file_node_id,
                "score": score,
                "rank": rank
            })
            
            # Add relationships between req and nodes
            if file_path in file_to_nodes:
                for node in file_to_nodes[file_path]:
                    connector._execute_query("""
                    MATCH (r:Requirement {id: $req_id})
                    MATCH (n:AdaNode {id: $node_id})
                    MERGE (r)-[rel:TRACED_TO]->(n)
                    SET rel.score = $score,
                        rel.file_rank = $rank
                    """, {
                        "req_id": req_node.id,
                        "node_id": node.id,
                        "score": score,
                        "rank": rank
                    })
    
    # Build relationships between nodes
    print_timestamp("Building parent-child relationships...")
    connector.build_parent_child_relationships(code_nodes)
    
    # Build reference relationships
    print_timestamp("Building reference relationships...")
    connector.build_references_relationships(code_nodes)
    
    # Create a special node to identify this as a complete analysis
    connector._execute_query("""
    MERGE (a:Analysis {id: 'complete-analysis'})
    SET a.timestamp = $timestamp,
        a.num_requirements = $num_requirements,
        a.analysis_file = $analysis_file
    """, {
        "timestamp": analysis["analysis_timestamp"],
        "num_requirements": analysis["num_requirements"],
        "analysis_file": combined_analysis_file
    })
    
    print_timestamp("All requirements loaded successfully into Neo4j")
    
    print("\nUseful Cypher queries to explore the data:")
    print("1. View all requirements and their related files:")
    print("   MATCH (r:Requirement)-[rel:RELATED_TO]->(f:File) RETURN r, rel, f")
    print("2. View files that satisfy multiple requirements (meeting point):")
    print("   MATCH (r:Requirement)-[rel:RELATED_TO]->(f:File)")
    print("   WITH f, count(r) as req_count, collect(r) as requirements")
    print("   WHERE req_count > 1")
    print("   RETURN f, requirements")
    print("3. View requirements that trace to the same code elements:")
    print("   MATCH (r1:Requirement)-[:TRACED_TO]->(n:AdaNode)<-[:TRACED_TO]-(r2:Requirement)")
    print("   WHERE r1.id < r2.id")
    print("   RETURN r1, r2, n")

def create_sample_requirements_file():
    """Create a sample requirements file with multiple requirements."""
    sample_file = os.path.join("outputs", "saved_analysis", "sample_requirements.txt")
    
    requirements = [
        "The system must provide a secure authentication mechanism for users to log in with their credentials. The authentication process should validate user credentials against stored data and enforce password policies including minimum length and complexity requirements. Failed login attempts should be recorded and after a configurable number of failed attempts, the account should be temporarily locked.",
        "The system must provide user account management features that allow administrators to create, modify, and delete user accounts. The system should maintain a record of all changes made to user accounts for audit purposes.",
        "The system must implement access control mechanisms that restrict user access to authorized functions and data based on their role and permissions in the system.",
        "The system must provide a secure method for users to reset their forgotten passwords through email verification or security questions.",
        "The system must maintain an audit log of all security-related events including login attempts, password changes, permission changes, and administrator actions."
    ]
    
    with open(sample_file, 'w') as f:
        for req in requirements:
            f.write(req + "\n\n")
    
    print_timestamp(f"Created sample requirements file at {sample_file}")
    return sample_file

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze multiple requirements and load them into Neo4j")
    
    parser.add_argument("--req-file", type=str, 
                        help="File containing requirements (one per line)")
    
    parser.add_argument("--create-sample", action="store_true",
                        help="Create a sample requirements file")
    
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top results to keep for each requirement")
    
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create sample file if requested
    if args.create_sample:
        sample_file = create_sample_requirements_file()
        if not args.req_file:
            args.req_file = sample_file
    
    # Ensure we have a requirements file
    if not args.req_file:
        print("Error: Please specify a requirements file with --req-file or use --create-sample")
        return 1
    
    # Load and analyze requirements
    requirements = load_requirements_from_file(args.req_file)
    
    if not requirements:
        print("Error: No requirements found in the file")
        return 1
    
    print_timestamp(f"Loaded {len(requirements)} requirements from {args.req_file}")
    
    # Analyze requirements
    combined_analysis_file = analyze_multiple_requirements(
        requirements, args.output_dir, args.top_n
    )
    
    # Load results into Neo4j
    if input("Load results into Neo4j? (y/n): ").lower() == 'y':
        load_requirements_to_neo4j(combined_analysis_file)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 