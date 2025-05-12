#!/usr/bin/env python3
"""
Script to analyze all requirements from the ada_requirements directory,
ensuring proper connections with commits.
"""
import os
import sys
import json
import time
from pathlib import Path
from tqdm import tqdm

from graph_database import Neo4jConnector, print_timestamp
from analyze_single_req import Req2CodeAnalyzer
from visualize_traceability import rebuild_commit_relationships

def process_all_requirements():
    """Process all requirements in the ada_requirements directory."""
    print_timestamp("Processing all requirements in ada_requirements directory...")
    
    # Initialize analyzer
    analyzer = Req2CodeAnalyzer()
    
    # Connect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    # Check if ada_requirements directory exists
    if not os.path.exists("ada_requirements"):
        print_timestamp("Error: ada_requirements directory not found.")
        return False
    
    # Get all text files in ada_requirements directory
    req_files = []
    for root, _, files in os.walk("ada_requirements"):
        for file in files:
            if file.endswith(".txt"):
                req_files.append(os.path.join(root, file))
    
    if not req_files:
        print_timestamp("Error: No requirement files (.txt) found in ada_requirements directory.")
        return False
    
    print_timestamp(f"Found {len(req_files)} requirement files.")
    
    # Process each requirement file
    for i, req_file in enumerate(tqdm(req_files, desc="Processing requirements")):
        # Generate requirement ID from filename
        req_id = os.path.splitext(os.path.basename(req_file))[0]
        req_id = f"REQ-{i+1:02d}" if req_id.startswith("req") else req_id
        
        # Read requirement text
        with open(req_file, 'r') as f:
            req_text = f.read().strip()
        
        if not req_text:
            print_timestamp(f"Warning: Empty requirement file {req_file}. Skipping.")
            continue
        
        # Analyze requirement
        print_timestamp(f"Analyzing requirement {req_id}: {req_text[:50]}...")
        results = analyzer.analyze_requirement(req_text, req_id, 10)
        
        # Create requirement node
        req_node_id = f"req-{req_id}"
        
        connector._execute_query("""
        MERGE (r:Requirement {id: $id})
        SET r.text = $text,
            r.req_id = $req_id,
            r.file_path = $file_path,
            r.timestamp = $timestamp
        """, {
            "id": req_node_id,
            "text": req_text,
            "req_id": req_id,
            "file_path": req_file,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        })
        
        # Connect to files
        for j, result in enumerate(results):
            file_path = result['file_path']
            file_name = Path(file_path).name
            
            connector._execute_query("""
            MERGE (f:File {name: $name})
            SET f.path = $path
            WITH f
            MATCH (r:Requirement {id: $req_id})
            MERGE (r)-[rel:RELATED_TO]->(f)
            SET rel.score = $score,
                rel.rank = $rank
            """, {
                "name": file_name,
                "path": file_path,
                "req_id": req_node_id,
                "score": result['score'],
                "rank": j + 1
            })
    
    # Create a batch file for each requirement
    print_timestamp("Creating batch results file...")
    
    # Query for all requirements and their related files
    requirements = connector._execute_query("""
    MATCH (r:Requirement)
    OPTIONAL MATCH (r)-[rel:RELATED_TO]->(f:File)
    RETURN r.req_id as id, r.text as text, collect({name: f.name, path: f.path, score: rel.score}) as files
    ORDER BY r.req_id
    """)
    
    # Create batch results file
    with open("requirements_analysis_results.json", 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "requirements_count": len(requirements),
            "requirements": requirements
        }, f, indent=2)
    
    print_timestamp(f"Processed {len(req_files)} requirements. Results saved to requirements_analysis_results.json")
    
    # Rebuild commit relationships to ensure proper connections
    print_timestamp("Rebuilding commit relationships...")
    rebuild_commit_relationships()
    
    # Create visual relationships
    print_timestamp("Creating visual relationships...")
    os.system("python visualize_traceability.py")
    
    return True

def main():
    """Main function."""
    if process_all_requirements():
        print_timestamp("All requirements processed and connected successfully!")
        print("\nTo visualize the relationships, open Neo4j Browser and try these queries:")
        print("\n1. View all requirements and their files:")
        print("   MATCH p=(r:Requirement)-[rel:RELATED_TO]->(f:File) RETURN p LIMIT 100")
        print("\n2. View requirements connected to commits:")
        print("   MATCH p=(r:Requirement)-[rel:SATISFIES]->(c:Commit) RETURN p LIMIT 50")
        print("\n3. View the complete traceability chain:")
        print("   MATCH p=(r:Requirement)-[:RELATED_TO]->(f:File)<-[:CHANGES]-(c:Commit) RETURN p LIMIT 25")
        
        return 0
    else:
        print_timestamp("Error processing requirements.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 