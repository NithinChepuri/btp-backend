#!/usr/bin/env python3
"""
Script to analyze a requirement given as a string.
This script:
1. Takes a requirement text as input
2. Analyzes it against the codebase
3. Connects it to files, commits, and issues
4. Generates visualizations in Neo4j
5. Outputs an analysis file
"""
import os
import sys
import json
import time
import uuid
from pathlib import Path
from datetime import datetime

from graph_database import Neo4jConnector, print_timestamp
from analyze_single_req import Req2CodeAnalyzer

def analyze_requirement_string(req_text, output_dir="analysis_reports"):
    """
    Analyze a requirement given as a string.
    
    Args:
        req_text: The requirement text to analyze
        output_dir: Directory to store analysis reports
    
    Returns:
        The requirement ID created for this analysis
    """
    print_timestamp(f"Analyzing requirement: {req_text[:50]}...")
    
    # Generate a unique ID for this requirement
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    req_id = f"REQ-{timestamp}"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize analyzer
    analyzer = Req2CodeAnalyzer()
    
    # Connect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    # Analyze the requirement
    print_timestamp("Running analysis against codebase...")
    results = analyzer.analyze_requirement(req_text, req_id, top_n=10)
    
    # Create analysis report file
    report_file = os.path.join(output_dir, f"{req_id}_analysis.txt")
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"REQUIREMENT ANALYSIS REPORT: {req_id}\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Requirement Text:\n{req_text}\n\n")
        f.write("-"*80 + "\n")
        f.write("FILES BY RELEVANCE:\n")
        f.write("-"*80 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['file_path']} (Score: {result['score']:.4f})\n")
            
            # Extract file content for analysis
            file_path = result['file_path']
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file_content:
                        content = file_content.read()
                        
                        # Truncate if too large
                        if len(content) > 2000:
                            content = content[:2000] + "...\n[Content truncated for brevity]"
                            
                        f.write("\nFile Content:\n")
                        f.write("```\n")
                        f.write(content + "\n")
                        f.write("```\n")
                        
                        # Basic file analysis
                        f.write("\nFile Analysis:\n")
                        
                        # File type
                        ext = os.path.splitext(file_path)[1]
                        f.write(f"- Type: {ext} file\n")
                        
                        # Size
                        size = os.path.getsize(file_path)
                        f.write(f"- Size: {size} bytes\n")
                        
                        # Lines count
                        lines = content.count('\n') + 1
                        f.write(f"- Lines: {lines}\n")
                        
                        # Ada-specific analysis
                        if ext in ['.ads', '.adb']:
                            # Count procedures, functions, packages
                            procedures = content.lower().count("procedure ")
                            functions = content.lower().count("function ")
                            packages = content.lower().count("package ")
                            
                            f.write(f"- Packages: {packages}\n")
                            f.write(f"- Procedures: {procedures}\n")
                            f.write(f"- Functions: {functions}\n")
                            
                            # Check for key Ada features
                            if "generic" in content.lower():
                                f.write("- Contains generic components\n")
                            if "tagged" in content.lower():
                                f.write("- Contains tagged types (OOP)\n")
                            if "access" in content.lower():
                                f.write("- Uses access types (pointers)\n")
                        
                        f.write("\n")
                except Exception as e:
                    f.write(f"\nError reading file: {e}\n\n")
            else:
                f.write("\nFile not found on disk.\n\n")
            
            f.write("-"*60 + "\n\n")
    
    print_timestamp(f"Analysis report written to {report_file}")
    
    # Create requirement node in Neo4j
    print_timestamp("Creating requirement node in Neo4j...")
    req_node_id = f"req-{req_id}"
    
    connector._execute_query("""
    MERGE (r:Requirement {id: $id})
    SET r.text = $text,
        r.req_id = $req_id,
        r.timestamp = $timestamp,
        r.is_string_analysis = true
    """, {
        "id": req_node_id,
        "text": req_text,
        "req_id": req_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
    })
    
    # Connect to files
    for i, result in enumerate(results):
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
            "rank": i + 1
        })
    
    # Clear Neo4j cache
    connector._execute_query("MATCH (n) REMOVE n.highlighted, n.selected, n.focused")
    
    # Highlight the requirement
    connector._execute_query(
        "MATCH (r:Requirement {req_id: $req_id}) SET r.highlighted = true, r.focused = true", 
        {"req_id": req_id}
    )
    
    # Highlight files
    connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[rel:RELATED_TO]->(f:File)
        SET f.highlighted = true, f.selected = true
        """, 
        {"req_id": req_id}
    )
    
    # Connect to commits
    print_timestamp("Connecting to relevant commits...")
    connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[:RELATED_TO]->(f:File)<-[:CHANGES]-(c:Commit)
        SET c.highlighted = true
        MERGE (r)-[:SATISFIES {through: f.name, strength: 0.8, width: 7, color: 'green', visible: true}]->(c)
        """, 
        {"req_id": req_id}
    )
    
    # Connect to issues
    print_timestamp("Connecting to relevant issues...")
    connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[:RELATED_TO]->(f:File)<-[:IMPACTS]-(i:Issue)
        SET i.highlighted = true
        MERGE (r)-[:SATISFIES {through: f.name, strength: 0.8, width: 7, color: 'green', visible: true}]->(i)
        """, 
        {"req_id": req_id}
    )
    
    # Count connections
    files_count = connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[:RELATED_TO]->(f:File)
        RETURN count(f) as count
        """, 
        {"req_id": req_id}
    )[0]["count"]
    
    commits_count = connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[:RELATED_TO]->(f:File)<-[:CHANGES]-(c:Commit)
        RETURN count(DISTINCT c) as count
        """, 
        {"req_id": req_id}
    )[0]["count"]
    
    issues_count = connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[:RELATED_TO]->(f:File)<-[:IMPACTS]-(i:Issue)
        RETURN count(DISTINCT i) as count
        """, 
        {"req_id": req_id}
    )[0]["count"]
    
    print("\n" + "="*80)
    print(f"REQUIREMENT ANALYZED AS: {req_id}")
    print("="*80)
    print(f"Text: {req_text[:200]}...")
    print("\nConnections created:")
    print(f"- Files: {files_count}")
    print(f"- Commits: {commits_count}")
    print(f"- Issues: {issues_count}")
    print("\nAnalysis report: " + report_file)
    print("\nTo view in Neo4j Browser, use this query:")
    print("MATCH (n) WHERE n.highlighted = true RETURN n")
    
    # Update the report with connection counts
    with open(report_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("NEO4J CONNECTIONS\n")
        f.write("="*80 + "\n")
        f.write(f"Files connected: {files_count}\n")
        f.write(f"Commits connected: {commits_count}\n")
        f.write(f"Issues connected: {issues_count}\n\n")
        f.write("Neo4j query to view:\n")
        f.write("MATCH (n) WHERE n.highlighted = true RETURN n\n")
    
    return req_id

def main():
    """Main function."""
    if len(sys.argv) <= 1:
        print("Error: Missing requirement text.")
        print("Usage: python analyze_requirement_string.py \"The system must provide authentication for users\"")
        print("   or: python analyze_requirement_string.py --file requirement.txt")
        return 1
    
    # Check if input is from file
    if sys.argv[1] == "--file" and len(sys.argv) >= 3:
        file_path = sys.argv[2]
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return 1
        
        with open(file_path, 'r') as f:
            req_text = f.read().strip()
    else:
        # Get requirement text from command line
        req_text = sys.argv[1]
    
    # Analyze the requirement
    req_id = analyze_requirement_string(req_text)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 