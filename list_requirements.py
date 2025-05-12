#!/usr/bin/env python3
"""
Script to list available requirements in Neo4j.
"""
import sys
from graph_database import Neo4jConnector, print_timestamp

def list_requirements():
    """List all requirements available in Neo4j."""
    print_timestamp("Listing requirements in Neo4j...")
    
    # Connect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    # Get all requirements
    results = connector._execute_query("""
    MATCH (r:Requirement)
    RETURN r.req_id as req_id, r.text as text
    ORDER BY r.req_id
    """)
    
    if not results:
        print("No requirements found in Neo4j. Make sure you've run analyze_ada_requirements.py first.")
        return
    
    print(f"\nFound {len(results)} requirements in Neo4j:\n")
    
    for i, row in enumerate(results):
        # Truncate long text
        text = row['text']
        if len(text) > 50:
            text = text[:47] + "..."
        
        print(f"{i+1}. {row['req_id']}: {text}")
    
    # Get most connected requirements
    print("\n\nTop requirements by number of connected files:")
    results = connector._execute_query("""
    MATCH (r:Requirement)-[:RELATED_TO]->(f:File)
    WITH r, count(f) as file_count
    RETURN r.req_id as req_id, r.text as text, file_count
    ORDER BY file_count DESC
    LIMIT 10
    """)
    
    for i, row in enumerate(results):
        # Truncate long text
        text = row['text']
        if len(text) > 50:
            text = text[:47] + "..."
        
        print(f"{i+1}. {row['req_id']} ({row['file_count']} files): {text}")
    
    # Get requirements with issues and commits
    print("\n\nRequirements with related issues/commits:")
    results = connector._execute_query("""
    MATCH (r:Requirement)<-[:IMPLEMENTS]-(n)
    WHERE n:Issue OR n:Commit
    WITH r, count(distinct n) as connections
    RETURN r.req_id as req_id, connections
    ORDER BY connections DESC
    LIMIT 10
    """)
    
    if not results:
        print("No requirements with connected issues/commits found.")
    else:
        for i, row in enumerate(results):
            print(f"{i+1}. {row['req_id']}: {row['connections']} connections")
    
    print("\nTo analyze a specific requirement, run:")
    print("python enhance_neo4j_connections.py --analyze-req <req_id>")
    print("Example: python enhance_neo4j_connections.py --analyze-req req1")

if __name__ == "__main__":
    list_requirements()
    sys.exit(0) 