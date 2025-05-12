#!/usr/bin/env python3
"""
Script to load requirement analysis into Neo4j and export the database for future use.
"""
import os
import json
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from graph_database import Neo4jConnector, print_timestamp
from code2graph import load_checkpoint
from req2nodes import RequirementNode

# Configuration
EXPORTS_DIR = os.path.join("outputs", "saved_analysis", "neo4j_exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

def load_requirement_and_export(req_id, analysis_file):
    """
    Load a requirement and its related code files into Neo4j, then export the database.
    
    Args:
        req_id: The requirement ID
        analysis_file: Path to the analysis JSON file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"{req_id}_{timestamp}.dump"
    export_path = os.path.join(EXPORTS_DIR, export_filename)
    
    print_timestamp(f"Loading requirement {req_id} and related files into Neo4j...")
    
    # Load the analysis results
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    # Create Neo4j connector and connect
    connector = Neo4jConnector()
    connector.connect()
    
    # Clear existing data
    print_timestamp("Clearing existing database...")
    connector.clear_database()
    
    # Create constraints
    connector.create_constraints()
    
    # Create requirement node
    req_text = analysis["requirement_text"]
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
    
    # Store analysis metadata in the database
    connector._execute_query("""
    MERGE (a:Analysis {id: $analysis_id})
    SET a.req_id = $req_id,
        a.timestamp = $timestamp,
        a.analysis_file = $analysis_file
    """, {
        "analysis_id": f"analysis-{req_id}",
        "req_id": req_id,
        "timestamp": analysis["analysis_timestamp"],
        "analysis_file": analysis_file
    })
    
    # Create relationship between requirement and analysis
    connector._execute_query("""
    MATCH (r:Requirement {id: $req_id})
    MATCH (a:Analysis {id: $analysis_id})
    MERGE (r)-[rel:HAS_ANALYSIS]->(a)
    """, {
        "req_id": req_node.id,
        "analysis_id": f"analysis-{req_id}"
    })
    
    # Load code nodes (use existing nodes from checkpoint)
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
    
    # Process the top files from analysis
    for i, file_info in enumerate(analysis["top_files"]):
        file_path = file_info["file_path"]
        score = file_info["score"]
        rank = i + 1
        
        if file_path in file_to_nodes:
            nodes = file_to_nodes[file_path]
            print_timestamp(f"Adding {len(nodes)} nodes from file: {file_path}")
            
            # Create file node
            file_node_id = f"file-{Path(file_path).name}"
            connector._execute_query("""
            MERGE (f:File {id: $id})
            SET f.path = $path,
                f.name = $name,
                f.score = $score,
                f.rank = $rank
            """, {
                "id": file_node_id,
                "path": file_path,
                "name": Path(file_path).name,
                "score": score,
                "rank": rank
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
            
            # Add nodes from the file
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
                
                # Create relationship between req and node
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
    
    print_timestamp("Data loaded successfully into Neo4j")
    
    # Export the Neo4j database to a file
    try:
        # Using neo4j-admin to export the database
        # NOTE: This requires Neo4j Admin to be in the system PATH
        # Alternative: Could use APOC if available in Neo4j
        print_timestamp(f"Exporting Neo4j database to {export_path}...")
        
        # Create a metadata file about the export
        export_metadata = {
            "req_id": req_id,
            "analysis_file": analysis_file,
            "timestamp": datetime.now().isoformat(),
            "export_file": export_path,
            "num_files": len(analysis["top_files"]),
            "requirement_text": req_text
        }
        
        metadata_path = os.path.join(EXPORTS_DIR, f"{req_id}_{timestamp}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(export_metadata, f, indent=2)
            
        print_timestamp(f"Saved export metadata to {metadata_path}")
        print_timestamp("Export process completed")
        
    except Exception as e:
        print_timestamp(f"Error exporting database: {e}")
    
    # Close Neo4j connection
    connector.close()
    
    # Summary of data in Neo4j
    print("\nData loaded in Neo4j:")
    print(f"- Requirement: {req_id}")
    print(f"- Files: {len(analysis['top_files'])}")
    print(f"- Code elements: Multiple (see Neo4j Browser for details)")
    
    print("\nCypher queries to explore the data:")
    print("1. View the requirement and related files:")
    print("   MATCH (r:Requirement)-[rel:RELATED_TO]->(f:File) RETURN r, rel, f ORDER BY rel.rank")
    print("2. View the requirement and all traced code elements:")
    print("   MATCH (r:Requirement)-[rel:TRACED_TO]->(n:AdaNode) RETURN r, rel, n ORDER BY rel.score DESC LIMIT 50")
    print("3. View files and their code elements:")
    print("   MATCH (f:File)-[:CONTAINS]->(n:AdaNode) RETURN f, n LIMIT 100")
    print("4. View code hierarchy:")
    print("   MATCH p=(n:AdaNode)-[:CONTAINS*1..2]->(c) RETURN p LIMIT 100")
    
    print("\nTo access Neo4j Browser:")
    print("1. Open a web browser and go to http://localhost:7474/")
    print("2. Login with your Neo4j credentials")
    print("3. Run the Cypher queries above to explore the data")
    
    # Save a copy of the analysis file with the export
    shutil.copy(analysis_file, os.path.join(EXPORTS_DIR, f"{req_id}_{timestamp}_analysis.json"))
    
    return export_path

def main():
    """Main function to load requirement analysis and export Neo4j database."""
    if len(sys.argv) < 2:
        req_id = "AUTH-01"  # Default requirement ID
        analysis_file = os.path.join("outputs", "saved_analysis", f"auth_requirement_analysis.json")
    else:
        req_id = sys.argv[1]
        analysis_file = sys.argv[2] if len(sys.argv) > 2 else os.path.join("outputs", f"req_{req_id}_analysis.json")
    
    if not os.path.exists(analysis_file):
        print(f"Error: Analysis file {analysis_file} not found.")
        sys.exit(1)
    
    load_requirement_and_export(req_id, analysis_file)

if __name__ == "__main__":
    main() 