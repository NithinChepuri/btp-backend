#!/usr/bin/env python3
"""
Script to visualize a single requirement in Neo4j.
This script:
1. Clears the Neo4j cache
2. Focuses on the specific requirement
3. Shows all connections (files, commits, issues)
4. Prints the related files
"""
import os
import sys
import json
import time
from pathlib import Path

from graph_database import Neo4jConnector, print_timestamp

def view_requirement(req_id):
    """
    View a single requirement with all its connections.
    Clears cache and focuses the Neo4j view on just this requirement.
    """
    print_timestamp(f"Visualizing requirement {req_id} in Neo4j...")
    
    # Connect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    # Step 1: Clear Neo4j browser state/cache
    print_timestamp("Clearing Neo4j cache and previous state...")
    connector._execute_query("MATCH (n) REMOVE n.highlighted, n.selected, n.focused")
    
    # First check if requirement exists
    req_exists = connector._execute_query(
        "MATCH (r:Requirement {req_id: $req_id}) RETURN count(r) as count", 
        {"req_id": req_id}
    )[0]["count"] > 0
    
    if not req_exists:
        print_timestamp(f"Error: Requirement {req_id} does not exist in the database.")
        print("\nAvailable requirements:")
        result = connector._execute_query(
            "MATCH (r:Requirement) RETURN r.req_id as req_id ORDER BY r.req_id"
        )
        for r in result:
            print(f"- {r['req_id']}")
        return False
    
    # Step 2: Highlight target requirement
    connector._execute_query(
        "MATCH (r:Requirement {req_id: $req_id}) SET r.highlighted = true, r.focused = true", 
        {"req_id": req_id}
    )
    
    # Step 3: Get and highlight files directly related to requirement
    files = connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[rel:RELATED_TO]->(f:File)
        SET f.highlighted = true, f.selected = true
        RETURN f.name as name, f.path as path, rel.score as score
        ORDER BY rel.score DESC
        """, 
        {"req_id": req_id}
    )
    
    # Step 4: Get and highlight commits
    commits = connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[:RELATED_TO]->(f:File)<-[:CHANGES]-(c:Commit)
        SET c.highlighted = true
        RETURN DISTINCT c.sha as sha, c.commit.message as message
        """, 
        {"req_id": req_id}
    )
    
    # Step 5: Get and highlight issues
    issues = connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[:RELATED_TO]->(f:File)<-[:IMPACTS]-(i:Issue)
        SET i.highlighted = true
        RETURN DISTINCT i.number as number, i.title as title
        """, 
        {"req_id": req_id}
    )
    
    # Create direct SATISFIES relationships for better visualization
    connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[:RELATED_TO]->(f:File)<-[:CHANGES]-(c:Commit)
        WHERE NOT (r)-[:SATISFIES]->(c)
        MERGE (r)-[:SATISFIES {through: f.name, strength: 0.8, width: 7, color: 'green', visible: true}]->(c)
        """, 
        {"req_id": req_id}
    )
    
    connector._execute_query(
        """
        MATCH (r:Requirement {req_id: $req_id})-[:RELATED_TO]->(f:File)<-[:IMPACTS]-(i:Issue)
        WHERE NOT (r)-[:SATISFIES]->(i)
        MERGE (r)-[:SATISFIES {through: f.name, strength: 0.8, width: 7, color: 'green', visible: true}]->(i)
        """, 
        {"req_id": req_id}
    )
    
    # Print requirement details
    req_info = connector._execute_query(
        "MATCH (r:Requirement {req_id: $req_id}) RETURN r.text as text", 
        {"req_id": req_id}
    )[0]
    
    # Print summary
    print("\n" + "="*80)
    print(f"REQUIREMENT: {req_id}")
    print("="*80)
    print(f"Text: {req_info['text'][:200]}...")
    print("-"*80)
    
    # Print files
    print(f"\nFILES ({len(files)}):")
    if files:
        for i, file in enumerate(files, 1):
            print(f"{i}. {file['name']} - Score: {file['score']:.3f}")
            print(f"   Path: {file['path']}")
    else:
        print("No files connected to this requirement.")
    
    # Print commits
    print(f"\nCOMMITS ({len(commits)}):")
    if commits:
        for i, commit in enumerate(commits, 1):
            message = commit['message'].split('\n')[0][:60] if commit['message'] else "No message"
            print(f"{i}. {commit['sha'][:8]} - {message}...")
    else:
        print("No commits connected to this requirement.")
    
    # Print issues
    print(f"\nISSUES ({len(issues)}):")
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"{i}. Issue #{issue['number']} - {issue['title']}")
    else:
        print("No issues connected to this requirement.")
    
    print("\n" + "="*80)
    print("NEO4J VISUALIZATION")
    print("="*80)
    print("To view this requirement in Neo4j Browser, use this query:")
    print("MATCH (n) WHERE n.highlighted = true RETURN n")
    print("\nAlternative detailed queries:")
    print(f"1. Files only: MATCH p=(r:Requirement {{req_id: \"{req_id}\"}})-[:RELATED_TO]->(f:File) RETURN p")
    print(f"2. Commits: MATCH p=(r:Requirement {{req_id: \"{req_id}\"}})-[:SATISFIES]->(c:Commit) RETURN p")
    print(f"3. Issues: MATCH p=(r:Requirement {{req_id: \"{req_id}\"}})-[:SATISFIES]->(i:Issue) RETURN p")
    print(f"4. Complete view: MATCH p=(r:Requirement {{req_id: \"{req_id}\"}}) WHERE r.highlighted=true MATCH (n) WHERE n.highlighted=true RETURN n")

    return True

def main():
    """Main function."""
    if len(sys.argv) <= 1:
        print("Error: Missing requirement ID.")
        print("Usage: python view_single_requirement.py REQ-01")
        
        # Show available requirements
        connector = Neo4jConnector()
        connector.connect()
        print("\nAvailable requirements:")
        result = connector._execute_query(
            "MATCH (r:Requirement) RETURN r.req_id as req_id ORDER BY r.req_id"
        )
        for r in result:
            print(f"- {r['req_id']}")
        return 1
    
    # Get requirement ID from command line
    req_id = sys.argv[1]
    
    # View the requirement
    if view_requirement(req_id):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 