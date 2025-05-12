#!/usr/bin/env python3
"""
Script to enhance connections between issues, commits, and code files in Neo4j.
This script specifically focuses on making relationships more visible in the graph.
"""
import os
import sys
import json
import re
from pathlib import Path
import argparse

from tqdm import tqdm
from graph_database import Neo4jConnector, print_timestamp

def enhance_neo4j_connections():
    """Create more direct and visible connections in the Neo4j database."""
    print_timestamp("Enhancing connections in Neo4j...")
    
    # Connect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    # Check if database has data
    result = connector._execute_query("MATCH (n) RETURN count(n) as count")
    if result[0]["count"] == 0:
        print_timestamp("Error: Neo4j database is empty. Please run analyze_ada_requirements.py first.")
        return
    
    # Create direct and more visible relationships between issues/commits and files
    print_timestamp("Creating direct LINKED_TO relationships...")
    
    # Create LINKED_TO relationship from issues to files (combines all other relationships)
    connector._execute_query("""
    MATCH (i:Issue)-[r:MENTIONS|MIGHT_RELATE_TO]->(f:File)
    MERGE (i)-[link:LINKED_TO]->(f)
    SET link.type = type(r),
        link.created_at = timestamp()
    """)
    
    # Create LINKED_TO relationship from commits to files (combines all other relationships)
    connector._execute_query("""
    MATCH (c:Commit)-[r:MODIFIES|REFERENCES|MIGHT_RELATE_TO]->(f:File)
    MERGE (c)-[link:LINKED_TO]->(f)
    SET link.type = type(r),
        link.created_at = timestamp()
    """)
    
    # Create direct links between requirements and issues/commits
    print_timestamp("Creating direct IMPLEMENTS relationships...")
    
    # Connect requirements directly to issues
    connector._execute_query("""
    MATCH (r:Requirement)-[:RELATED_TO]->(f:File)<-[:LINKED_TO]-(i:Issue)
    MERGE (i)-[rel:IMPLEMENTS]->(r)
    SET rel.created_at = timestamp()
    """)
    
    # Connect requirements directly to commits
    connector._execute_query("""
    MATCH (r:Requirement)-[:RELATED_TO]->(f:File)<-[:LINKED_TO]-(c:Commit)
    MERGE (c)-[rel:IMPLEMENTS]->(r)
    SET rel.created_at = timestamp()
    """)
    
    # Connect issues and commits directly
    print_timestamp("Creating direct connections between issues and commits...")
    
    # Connect commits that mention issues (by number)
    connector._execute_query("""
    MATCH (c:Commit), (i:Issue)
    WHERE c.message CONTAINS ('#' + toString(i.number)) OR
          c.message CONTAINS ('issue ' + toString(i.number)) OR
          c.message CONTAINS ('Issue ' + toString(i.number)) OR
          c.message CONTAINS ('fixes #' + toString(i.number)) OR
          c.message CONTAINS ('closes #' + toString(i.number))
    MERGE (c)-[rel:RESOLVES]->(i)
    SET rel.created_at = timestamp()
    """)
    
    # Connect issues and commits that modify the same files
    connector._execute_query("""
    MATCH (i:Issue)-[:LINKED_TO]->(f:File)<-[:LINKED_TO]-(c:Commit)
    MERGE (c)-[rel:RELATED_TO_ISSUE]->(i)
    SET rel.file_path = f.path,
        rel.created_at = timestamp()
    """)
    
    # Create relationships between AdaNodes and Issues/Commits
    print_timestamp("Creating relationships between code nodes and issues/commits...")
    
    # Connect AdaNodes to issues that reference their files
    connector._execute_query("""
    MATCH (i:Issue)-[:LINKED_TO]->(f:File)-[:CONTAINS]->(n:AdaNode)
    MERGE (i)-[rel:REFERENCES_NODE]->(n)
    SET rel.created_at = timestamp()
    """)
    
    # Connect AdaNodes to commits that modify their files
    connector._execute_query("""
    MATCH (c:Commit)-[:LINKED_TO]->(f:File)-[:CONTAINS]->(n:AdaNode)
    MERGE (c)-[rel:CHANGES_NODE]->(n)
    SET rel.created_at = timestamp()
    """)
    
    print_timestamp("Creating highlights to make connections more visible...")
    
    # Highlight key nodes to make them more visible in the graph
    connector._execute_query("""
    MATCH (f:File)<-[:LINKED_TO]-(n)
    WHERE n:Issue OR n:Commit
    WITH f, count(distinct n) as connections
    WHERE connections > 1
    SET f.highlight = true,
        f.connections = connections
    """)
    
    # Highlight important requirements
    connector._execute_query("""
    MATCH (r:Requirement)<-[:IMPLEMENTS]-(n)
    WITH r, count(n) as implementations
    WHERE implementations > 0
    SET r.highlight = true,
        r.implementations = implementations
    """)
    
    print_timestamp("Enhancement complete. Neo4j connections have been enhanced.")
    
    print("\nUseful Cypher queries to explore the enhanced connections:")
    print("1. View direct connections between issues, commits, and requirements:")
    print("   MATCH p=(:Issue)-[:IMPLEMENTS]->(:Requirement) RETURN p LIMIT 10")
    print("   MATCH p=(:Commit)-[:IMPLEMENTS]->(:Requirement) RETURN p LIMIT 10")
    print("2. View issues and commits related to the same files:")
    print("   MATCH p=(:Issue)<-[:RELATED_TO_ISSUE]-(:Commit) RETURN p LIMIT 10")
    print("3. View code elements referenced by issues or changed by commits:")
    print("   MATCH p=(:Issue)-[:REFERENCES_NODE]->(:AdaNode) RETURN p LIMIT 10")
    print("   MATCH p=(:Commit)-[:CHANGES_NODE]->(:AdaNode) RETURN p LIMIT 10")
    
def analyze_single_requirement(req_id):
    """
    Analyze a single requirement's connections in Neo4j.
    
    Args:
        req_id: The requirement ID to analyze
    """
    print_timestamp(f"Analyzing requirement {req_id} in Neo4j...")
    
    # Connect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    # Check if requirement exists
    result = connector._execute_query("""
    MATCH (r:Requirement {req_id: $req_id})
    RETURN count(r) as count
    """, {"req_id": req_id})
    
    if result[0]["count"] == 0:
        print(f"Error: Requirement {req_id} not found in Neo4j. Make sure you've run analyze_ada_requirements.py first.")
        return
    
    # Get requirement text
    result = connector._execute_query("""
    MATCH (r:Requirement {req_id: $req_id})
    RETURN r.text as text
    """, {"req_id": req_id})
    
    req_text = result[0]["text"]
    
    print("\n===========================")
    print(f"Requirement {req_id}: {req_text}")
    print("===========================\n")
    
    # Get related files
    print("Files implementing this requirement:")
    results = connector._execute_query("""
    MATCH (r:Requirement {req_id: $req_id})-[rel:RELATED_TO]->(f:File)
    RETURN f.name as name, f.path as path, rel.score as score
    ORDER BY rel.score DESC
    """, {"req_id": req_id})
    
    for i, row in enumerate(results):
        print(f"{i+1}. {row['name']} ({row['path']}) - Score: {row['score']:.4f}")
    
    # Get related issues
    print("\nIssues related to this requirement:")
    results = connector._execute_query("""
    MATCH (r:Requirement {req_id: $req_id})<-[:IMPLEMENTS]-(i:Issue)
    RETURN i.number as number, i.title as title, i.state as state
    """, {"req_id": req_id})
    
    if not results:
        print("No directly related issues found.")
    else:
        for i, row in enumerate(results):
            print(f"{i+1}. Issue #{row['number']}: {row['title']} ({row['state']})")
    
    # Get related commits
    print("\nCommits related to this requirement:")
    results = connector._execute_query("""
    MATCH (r:Requirement {req_id: $req_id})<-[:IMPLEMENTS]-(c:Commit)
    RETURN c.sha as sha, c.message as message, c.author as author, c.date as date
    """, {"req_id": req_id})
    
    if not results:
        print("No directly related commits found.")
    else:
        for i, row in enumerate(results):
            # Truncate commit message to first line
            message = row['message'].split('\n')[0]
            print(f"{i+1}. {row['sha'][:7]}: {message} (by {row['author']})")
    
    # Get related code nodes
    print("\nCode elements implementing this requirement:")
    results = connector._execute_query("""
    MATCH (r:Requirement {req_id: $req_id})-[:TRACED_TO]->(n:AdaNode)
    RETURN n.name as name, n.type as type, n.file_path as file_path
    ORDER BY n.type, n.name
    LIMIT 20
    """, {"req_id": req_id})
    
    if not results:
        print("No directly traced code elements found.")
    else:
        for i, row in enumerate(results):
            file_name = os.path.basename(row['file_path']) if row['file_path'] else "Unknown"
            print(f"{i+1}. {row['type']} {row['name']} in {file_name}")
    
    print("\n===========================")
    print(f"Neo4j queries for req {req_id}:")
    print("===========================")
    print(f"MATCH (r:Requirement {{req_id: '{req_id}'}})-[rel:RELATED_TO]->(f:File) RETURN r, rel, f")
    print(f"MATCH p=(r:Requirement {{req_id: '{req_id}'}})<-[:IMPLEMENTS]-(i:Issue) RETURN p")
    print(f"MATCH p=(r:Requirement {{req_id: '{req_id}'}})<-[:IMPLEMENTS]-(c:Commit) RETURN p")
    
def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhance Neo4j connections and analyze single requirements")
    
    parser.add_argument("--enhance", action="store_true",
                        help="Enhance connections in Neo4j database")
    
    parser.add_argument("--analyze-req", type=str,
                        help="Analyze a single requirement by ID (e.g., 'req8')")
    
    args = parser.parse_args()
    
    if args.enhance:
        enhance_neo4j_connections()
    
    if args.analyze_req:
        analyze_single_requirement(args.analyze_req)
    
    if not args.enhance and not args.analyze_req:
        print("Please specify either --enhance to improve connections or --analyze-req REQ_ID to analyze a requirement")
        print("Example: python enhance_neo4j_connections.py --enhance")
        print("Example: python enhance_neo4j_connections.py --analyze-req req8")
        print("Example: python enhance_neo4j_connections.py --enhance --analyze-req req8")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 