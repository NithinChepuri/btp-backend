import os
import json
import pickle
import argparse
import sys
from pathlib import Path

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Error: Neo4j driver not installed. Please install it with:")
    print("pip install neo4j")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Export graph data to Neo4j database')
    parser.add_argument('--uri', default="neo4j://localhost:7687", help='Neo4j URI (default: neo4j://localhost:7687)')
    parser.add_argument('--user', default="neo4j", help='Neo4j username (default: neo4j)')
    parser.add_argument('--password', required=True, help='Neo4j password')
    parser.add_argument('--graph-file', default="outputs/indexes/graph_index.pkl", help='Path to graph index file')
    parser.add_argument('--no-clear', action='store_true', help='Do not clear existing data in Neo4j')
    args = parser.parse_args()

    # Check if graph file exists
    if not os.path.exists(args.graph_file):
        print(f"Error: Graph file {args.graph_file} not found.")
        sys.exit(1)
    
    # Load the graph from file
    print(f"Loading graph from {args.graph_file}...")
    try:
        with open(args.graph_file, 'rb') as f:
            graph = pickle.load(f)
        print(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    except Exception as e:
        print(f"Error loading graph: {str(e)}")
        sys.exit(1)
    
    # Connect to Neo4j
    print(f"Connecting to Neo4j at {args.uri}...")
    try:
        driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            if test_value != 1:
                print("Error: Failed to connect to Neo4j database.")
                sys.exit(1)
        
        print("Connected to Neo4j successfully")
    except Exception as e:
        print(f"Error connecting to Neo4j: {str(e)}")
        sys.exit(1)
    
    # Clear existing data if requested
    if not args.no_clear:
        print("Clearing existing Neo4j data...")
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Cleared existing Neo4j data")
    
    # Create constraints for faster lookups
    with driver.session() as session:
        # Create constraint for file nodes
        try:
            session.run("CREATE CONSTRAINT file_id IF NOT EXISTS FOR (f:File) REQUIRE f.id IS UNIQUE")
        except Exception:
            # Older Neo4j versions have different syntax
            try:
                session.run("CREATE CONSTRAINT ON (f:File) ASSERT f.id IS UNIQUE")
            except Exception as e:
                print(f"Warning: Could not create file constraint: {str(e)}")
        
        # Create constraint for package nodes
        try:
            session.run("CREATE CONSTRAINT package_id IF NOT EXISTS FOR (p:Package) REQUIRE p.id IS UNIQUE")
        except Exception:
            try:
                session.run("CREATE CONSTRAINT ON (p:Package) ASSERT p.id IS UNIQUE")
            except Exception as e:
                print(f"Warning: Could not create package constraint: {str(e)}")
        
        # Create constraint for procedure nodes
        try:
            session.run("CREATE CONSTRAINT procedure_id IF NOT EXISTS FOR (p:Procedure) REQUIRE p.id IS UNIQUE")
        except Exception:
            try:
                session.run("CREATE CONSTRAINT ON (p:Procedure) ASSERT p.id IS UNIQUE")
            except Exception as e:
                print(f"Warning: Could not create procedure constraint: {str(e)}")
        
        # Create constraint for function nodes
        try:
            session.run("CREATE CONSTRAINT function_id IF NOT EXISTS FOR (f:Function) REQUIRE f.id IS UNIQUE")
        except Exception:
            try:
                session.run("CREATE CONSTRAINT ON (f:Function) ASSERT f.id IS UNIQUE")
            except Exception as e:
                print(f"Warning: Could not create function constraint: {str(e)}")
    
    # Export nodes
    print("Exporting nodes to Neo4j...")
    node_count = 0
    
    # Batch nodes by type for efficient insertion
    file_nodes = []
    package_nodes = []
    procedure_nodes = []
    function_nodes = []
    other_nodes = []
    
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs.get('type', 'unknown')
        
        if node_type == 'file':
            file_nodes.append({
                'id': node_id,
                'name': attrs.get('name', ''),
                'package': attrs.get('package', ''),
                'file_type': attrs.get('file_type', '')
            })
        elif node_type == 'package':
            package_nodes.append({
                'id': node_id,
                'name': attrs.get('name', '')
            })
        elif node_type == 'procedure':
            procedure_nodes.append({
                'id': node_id,
                'name': attrs.get('name', ''),
                'package': attrs.get('package', '')
            })
        elif node_type == 'function':
            function_nodes.append({
                'id': node_id,
                'name': attrs.get('name', ''),
                'package': attrs.get('package', '')
            })
        else:
            # Any other node types
            other_nodes.append({
                'id': node_id,
                'type': node_type,
                **attrs
            })
    
    # Create nodes in batches
    with driver.session() as session:
        # Create file nodes
        if file_nodes:
            print(f"Creating {len(file_nodes)} File nodes...")
            batch_size = 100
            for i in range(0, len(file_nodes), batch_size):
                batch = file_nodes[i:i+batch_size]
                session.run("""
                    UNWIND $nodes AS node
                    CREATE (f:File {id: node.id, name: node.name, package: node.package, fileType: node.file_type})
                """, {'nodes': batch})
                node_count += len(batch)
                print(f"  Progress: {min(i+batch_size, len(file_nodes))}/{len(file_nodes)}")
        
        # Create package nodes
        if package_nodes:
            print(f"Creating {len(package_nodes)} Package nodes...")
            batch_size = 100
            for i in range(0, len(package_nodes), batch_size):
                batch = package_nodes[i:i+batch_size]
                session.run("""
                    UNWIND $nodes AS node
                    CREATE (p:Package {id: node.id, name: node.name})
                """, {'nodes': batch})
                node_count += len(batch)
                print(f"  Progress: {min(i+batch_size, len(package_nodes))}/{len(package_nodes)}")
        
        # Create procedure nodes
        if procedure_nodes:
            print(f"Creating {len(procedure_nodes)} Procedure nodes...")
            batch_size = 100
            for i in range(0, len(procedure_nodes), batch_size):
                batch = procedure_nodes[i:i+batch_size]
                session.run("""
                    UNWIND $nodes AS node
                    CREATE (p:Procedure {id: node.id, name: node.name, package: node.package})
                """, {'nodes': batch})
                node_count += len(batch)
                print(f"  Progress: {min(i+batch_size, len(procedure_nodes))}/{len(procedure_nodes)}")
        
        # Create function nodes
        if function_nodes:
            print(f"Creating {len(function_nodes)} Function nodes...")
            batch_size = 100
            for i in range(0, len(function_nodes), batch_size):
                batch = function_nodes[i:i+batch_size]
                session.run("""
                    UNWIND $nodes AS node
                    CREATE (f:Function {id: node.id, name: node.name, package: node.package})
                """, {'nodes': batch})
                node_count += len(batch)
                print(f"  Progress: {min(i+batch_size, len(function_nodes))}/{len(function_nodes)}")
        
        # Create other nodes
        if other_nodes:
            print(f"Creating {len(other_nodes)} Other nodes...")
            batch_size = 100
            for i in range(0, len(other_nodes), batch_size):
                batch = other_nodes[i:i+batch_size]
                session.run("""
                    UNWIND $nodes AS node
                    CREATE (o:Other {id: node.id, type: node.type})
                """, {'nodes': batch})
                node_count += len(batch)
                print(f"  Progress: {min(i+batch_size, len(other_nodes))}/{len(other_nodes)}")
    
    print(f"Exported {node_count} nodes to Neo4j")
    
    # Export relationships
    print("Exporting relationships to Neo4j...")
    edge_count = 0
    
    # Group relationships by type for batch insertion
    from collections import defaultdict
    relationships = defaultdict(list)
    
    for source, target, attrs in graph.edges(data=True):
        rel_type = attrs.get('type', 'RELATED_TO').upper()
        relationships[rel_type].append({'source': source, 'target': target})
    
    # Create relationships in batches
    with driver.session() as session:
        for rel_type, edges in relationships.items():
            print(f"Creating {len(edges)} {rel_type} relationships...")
            batch_size = 500
            for i in range(0, len(edges), batch_size):
                batch = edges[i:i+batch_size]
                cypher_query = f"""
                    UNWIND $edges AS edge
                    MATCH (source {{id: edge.source}})
                    MATCH (target {{id: edge.target}})
                    CREATE (source)-[:{rel_type}]->(target)
                """
                session.run(cypher_query, {'edges': batch})
                edge_count += len(batch)
                print(f"  Progress: {min(i+batch_size, len(edges))}/{len(edges)}")
    
    print(f"Exported {edge_count} relationships to Neo4j")
    
    # Close the driver
    driver.close()
    print("Graph successfully exported to Neo4j")
    
    # Print some example queries
    print("\nExample Neo4j queries you can run in the Neo4j Browser:")
    print("1. View all nodes: MATCH (n) RETURN n LIMIT 100")
    print("2. View all packages: MATCH (p:Package) RETURN p")
    print("3. View files in a package: MATCH (p:Package {name: 'your_package_name'})-[:CONTAINS]->(f:File) RETURN p, f")
    print("4. View procedures in files: MATCH (p:Procedure)-[:DEFINED_IN]->(f:File) RETURN p, f LIMIT 20")
    print("5. Explore package dependencies: MATCH (p1:Package)-[r:DEPENDS_ON]->(p2:Package) RETURN p1, r, p2")

if __name__ == "__main__":
    main() 