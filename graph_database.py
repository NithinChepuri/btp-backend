"""
Module for building and querying the Neo4j graph database.
"""
import os
import json
import time
import datetime
from typing import Dict, List, Any, Tuple

from neo4j import GraphDatabase, basic_auth
from tqdm import tqdm

from constants import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OUTPUT_DIR,
    IMPLEMENTS, DEPENDS_ON, REFERENCES, SIMILAR_TO,
    COMMITS_TO, ISSUES_RELATED
)
from code2graph import AdaNode, load_checkpoint, save_checkpoint
from req2nodes import RequirementNode
from github_integration import GitHubIssue, GitCommit
from embeddings import Node

# Path for graph database status checkpoint
GRAPH_STATUS_PATH = os.path.join(OUTPUT_DIR, "checkpoints", "graph_status.json")

def print_timestamp(message):
    """Print a message with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

class Neo4jConnector:
    """Connector for the Neo4j graph database."""
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        """Initialize the Neo4j connector."""
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.graph_status = self._load_graph_status()
    
    def _load_graph_status(self) -> Dict:
        """Load graph status from checkpoint file."""
        if os.path.exists(GRAPH_STATUS_PATH):
            try:
                with open(GRAPH_STATUS_PATH, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading graph status: {e}")
        return {
            "constraints_created": False,
            "nodes_created": {
                "AdaNode": 0,
                "Requirement": 0,
                "Issue": 0,
                "Commit": 0
            },
            "relationships_created": {
                DEPENDS_ON: 0,
                REFERENCES: 0,
                IMPLEMENTS: 0,
                ISSUES_RELATED: 0,
                COMMITS_TO: 0
            },
            "last_updated": ""
        }
    
    def _save_graph_status(self):
        """Save graph status to checkpoint file."""
        self.graph_status["last_updated"] = datetime.datetime.now().isoformat()
        try:
            os.makedirs(os.path.dirname(GRAPH_STATUS_PATH), exist_ok=True)
            with open(GRAPH_STATUS_PATH, 'w') as f:
                json.dump(self.graph_status, f, indent=2)
        except Exception as e:
            print(f"Error saving graph status: {e}")
    
    def connect(self) -> None:
        """Connect to the Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))
            print_timestamp(f"✅ Connected to Neo4j at {self.uri}")
        except Exception as e:
            print_timestamp(f"❌ Error connecting to Neo4j: {e}")
    
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
            print_timestamp("Neo4j connection closed")
    
    def clear_database(self) -> None:
        """Clear all nodes and relationships from the database."""
        print_timestamp("🧹 Clearing database...")
        query = "MATCH (n) DETACH DELETE n"
        self._execute_query(query)
        
        # Reset graph status
        self.graph_status = {
            "constraints_created": False,
            "nodes_created": {
                "AdaNode": 0,
                "Requirement": 0,
                "Issue": 0,
                "Commit": 0
            },
            "relationships_created": {
                DEPENDS_ON: 0,
                REFERENCES: 0,
                IMPLEMENTS: 0,
                ISSUES_RELATED: 0,
                COMMITS_TO: 0
            },
            "last_updated": datetime.datetime.now().isoformat()
        }
        self._save_graph_status()
        print_timestamp("✅ Database cleared")
    
    def create_constraints(self) -> None:
        """Create constraints for faster lookups."""
        if self.graph_status["constraints_created"]:
            print_timestamp("✅ Database constraints already exist")
            return
            
        print_timestamp("🔧 Creating database constraints...")
        constraints = [
            "CREATE CONSTRAINT ada_node_id IF NOT EXISTS FOR (n:AdaNode) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT requirement_id IF NOT EXISTS FOR (n:Requirement) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT issue_id IF NOT EXISTS FOR (n:Issue) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT commit_id IF NOT EXISTS FOR (n:Commit) REQUIRE n.id IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                self._execute_query(constraint)
            except Exception as e:
                print_timestamp(f"⚠️ Error creating constraint: {e}")
        
        self.graph_status["constraints_created"] = True
        self._save_graph_status()
        print_timestamp("✅ Database constraints created")
    
    def _execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a Cypher query."""
        if not self.driver:
            print_timestamp("❌ Not connected to Neo4j")
            return []
        
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    
    def _execute_batch_query(self, query: str, params_list: List[Dict], batch_size: int = 100) -> None:
        """Execute a Cypher query with a list of parameter dicts in batches."""
        if not self.driver:
            print_timestamp("❌ Not connected to Neo4j")
            return
        
        total_batches = (len(params_list) + batch_size - 1) // batch_size
        
        with tqdm(total=len(params_list), desc=f"Processing {len(params_list)} items") as pbar:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(params_list))
                batch = params_list[start_idx:end_idx]
                
                with self.driver.session() as session:
                    for params in batch:
                        session.run(query, params)
                        pbar.update(1)
    
    def create_ada_node(self, node: AdaNode) -> None:
        """Create an Ada node in the database."""
        query = """
        MERGE (n:AdaNode {id: $id})
        SET n.name = $name,
            n.type = $type,
            n.file_path = $file_path,
            n.line_number = $line_number,
            n.summary = $summary
        """
        
        params = {
            "id": node.id,
            "name": node.name,
            "type": node.type,
            "file_path": node.file_path,
            "line_number": node.line_number,
            "summary": node.summary if hasattr(node, "summary") and node.summary else ""
        }
        
        self._execute_query(query, params)
        self.graph_status["nodes_created"]["AdaNode"] += 1
    
    def create_requirement_node(self, node: RequirementNode) -> None:
        """Create a requirement node in the database."""
        query = """
        MERGE (n:Requirement {id: $id})
        SET n.req_id = $req_id,
            n.text = $text,
            n.file_path = $file_path
        """
        
        params = {
            "id": node.id,
            "req_id": node.req_id,
            "text": node.text,
            "file_path": node.file_path
        }
        
        self._execute_query(query, params)
        self.graph_status["nodes_created"]["Requirement"] += 1
    
    def create_issue_node(self, node: GitHubIssue) -> None:
        """Create a GitHub issue node in the database."""
        query = """
        MERGE (n:Issue {id: $id})
        SET n.issue_id = $issue_id,
            n.title = $title,
            n.body = $body,
            n.created_at = $created_at,
            n.updated_at = $updated_at,
            n.state = $state
        """
        
        params = {
            "id": node.id,
            "issue_id": node.issue_id,
            "title": node.title,
            "body": node.body,
            "created_at": node.created_at,
            "updated_at": node.updated_at,
            "state": node.state
        }
        
        self._execute_query(query, params)
        self.graph_status["nodes_created"]["Issue"] += 1
    
    def create_commit_node(self, node: GitCommit) -> None:
        """Create a Git commit node in the database."""
        query = """
        MERGE (n:Commit {id: $id})
        SET n.commit_hash = $commit_hash,
            n.author = $author,
            n.date = $date,
            n.message = $message,
            n.changed_files = $changed_files
        """
        
        params = {
            "id": node.id,
            "commit_hash": node.commit_hash,
            "author": node.author,
            "date": node.date,
            "message": node.message,
            "changed_files": node.changed_files
        }
        
        self._execute_query(query, params)
        self.graph_status["nodes_created"]["Commit"] += 1
    
    def create_relationship(self, source_id: str, target_id: str, rel_type: str, properties: Dict = None) -> None:
        """Create a relationship between two nodes."""
        query = f"""
        MATCH (source), (target)
        WHERE source.id = $source_id AND target.id = $target_id
        MERGE (source)-[r:{rel_type}]->(target)
        """
        
        if properties:
            query += "SET " + ", ".join(f"r.{key} = ${key}" for key in properties.keys())
        
        params = {
            "source_id": source_id,
            "target_id": target_id,
            **(properties or {})
        }
        
        self._execute_query(query, params)
        
        # Update relationship count
        if rel_type in self.graph_status["relationships_created"]:
            self.graph_status["relationships_created"][rel_type] += 1
    
    def create_batch_relationships(self, relationships: List[Tuple[str, str, str, Dict]], batch_size: int = 100) -> None:
        """Create a batch of relationships."""
        if not relationships:
            return
            
        # Group by relationship type for efficiency
        rel_groups = {}
        for source_id, target_id, rel_type, properties in relationships:
            if rel_type not in rel_groups:
                rel_groups[rel_type] = []
            rel_groups[rel_type].append((source_id, target_id, properties or {}))
        
        # Process each relationship type
        for rel_type, rels in rel_groups.items():
            query = f"""
            MATCH (source), (target)
            WHERE source.id = $source_id AND target.id = $target_id
            MERGE (source)-[r:{rel_type}]->(target)
            """
            
            # Add property updates if properties exist in any relationship
            has_props = any(props for _, _, props in rels)
            if has_props:
                query += "SET " + ", ".join(f"r.{key} = ${key}" for key in rels[0][2].keys())
            
            # Convert to params list
            params_list = []
            for source_id, target_id, props in rels:
                params = {
                    "source_id": source_id, 
                    "target_id": target_id,
                    **props
                }
                params_list.append(params)
            
            # Execute in batches
            print_timestamp(f"🔗 Creating {len(params_list)} {rel_type} relationships in batches...")
            self._execute_batch_query(query, params_list, batch_size)
            
            # Update relationship count
            if rel_type in self.graph_status["relationships_created"]:
                self.graph_status["relationships_created"][rel_type] += len(params_list)
        
        # Save status after batch
        self._save_graph_status()
    
    def build_parent_child_relationships(self, nodes: List[AdaNode]) -> None:
        """Build parent-child relationships between Ada nodes."""
        print_timestamp("🔄 Building parent-child relationships...")
        
        # Create a mapping of node IDs to nodes
        node_map = {node.id: node for node in nodes}
        
        # Build relationships
        relationships = []
        for node in tqdm(nodes, desc="Processing nodes"):
            if hasattr(node, "parent") and node.parent:
                # Find parent node
                parent_nodes = [n for n in nodes if n.name == node.parent and n.type == "PACKAGE"]
                
                if parent_nodes:
                    parent = parent_nodes[0]
                    relationships.append((parent.id, node.id, DEPENDS_ON, {}))
            
            if hasattr(node, "children") and node.children:
                for child_id in node.children:
                    if child_id in node_map:
                        relationships.append((node.id, child_id, DEPENDS_ON, {}))
        
        # Create relationships in batches
        print_timestamp(f"🔄 Creating {len(relationships)} parent-child relationships...")
        self.create_batch_relationships(relationships)
        print_timestamp(f"✅ Created {len(relationships)} parent-child relationships")
    
    def build_references_relationships(self, nodes: List[AdaNode]) -> None:
        """Build references relationships between Ada nodes."""
        print_timestamp("🔄 Building reference relationships...")
        
        relationships = []
        for node in tqdm(nodes, desc="Processing references"):
            if hasattr(node, "references") and node.references:
                for ref_id in node.references:
                    relationships.append((node.id, ref_id, REFERENCES, {}))
        
        # Create relationships in batches
        print_timestamp(f"🔄 Creating {len(relationships)} reference relationships...")
        self.create_batch_relationships(relationships)
        print_timestamp(f"✅ Created {len(relationships)} reference relationships")
    
    def build_similarity_relationships(
        self, 
        source_nodes: List[Node], 
        target_nodes: List[Node],
        links: Dict[str, List[Tuple[str, float]]],
        rel_type: str = SIMILAR_TO
    ) -> None:
        """Build similarity relationships between nodes."""
        print_timestamp(f"🔄 Building {rel_type} relationships...")
        
        relationships = []
        for source_id, targets in links.items():
            for target_id, similarity in targets:
                properties = {"similarity": similarity}
                relationships.append((source_id, target_id, rel_type, properties))
        
        # Create relationships in batches
        print_timestamp(f"🔄 Creating {len(relationships)} {rel_type} relationships...")
        self.create_batch_relationships(relationships)
        print_timestamp(f"✅ Created {len(relationships)} {rel_type} relationships")
    
    def build_commit_relationships(self, commits: List[GitCommit], code_nodes: List[AdaNode]) -> None:
        """Build relationships between commits and code nodes."""
        print_timestamp("🔄 Building commit relationships...")
        
        # Create file path to node ID mapping for faster lookups
        file_path_to_nodes = {}
        for node in code_nodes:
            file_path = node.file_path
            if file_path not in file_path_to_nodes:
                file_path_to_nodes[file_path] = []
            file_path_to_nodes[file_path].append(node.id)
        
        relationships = []
        for commit in tqdm(commits, desc="Processing commits"):
            for file_path in commit.changed_files:
                # Find code nodes in the changed file
                if file_path in file_path_to_nodes:
                    for node_id in file_path_to_nodes[file_path]:
                        relationships.append((commit.id, node_id, COMMITS_TO, {}))
        
        # Create relationships in batches
        print_timestamp(f"🔄 Creating {len(relationships)} commit relationships...")
        self.create_batch_relationships(relationships)
        print_timestamp(f"✅ Created {len(relationships)} commit relationships")
    
    def query_node_by_id(self, node_id: str) -> Dict:
        """Query a node by ID."""
        query = """
        MATCH (n {id: $id})
        RETURN n
        """
        
        params = {"id": node_id}
        results = self._execute_query(query, params)
        
        if results:
            return results[0]["n"]
        
        return None
    
    def query_related_nodes(self, node_id: str, rel_type: str = None, max_hops: int = 1) -> List[Dict]:
        """Query nodes related to a given node."""
        if rel_type:
            query = f"""
            MATCH (n {{id: $id}})-[r:{rel_type}*1..{max_hops}]-(related)
            RETURN related
            """
        else:
            query = f"""
            MATCH (n {{id: $id}})-[r*1..{max_hops}]-(related)
            RETURN related
            """
        
        params = {"id": node_id}
        results = self._execute_query(query, params)
        
        return [result["related"] for result in results]
    
    def query_requirements_with_code(self) -> List[Dict]:
        """Query requirements that are implemented in code."""
        query = """
        MATCH (r:Requirement)-[rel:IMPLEMENTS]->(code:AdaNode)
        RETURN r.req_id AS req_id, count(code) AS code_count
        """
        
        return self._execute_query(query)
    
    def query_code_with_requirements(self) -> List[Dict]:
        """Query code that implements requirements."""
        query = """
        MATCH (r:Requirement)-[rel:IMPLEMENTS]->(code:AdaNode)
        RETURN code.name AS code_name, code.type AS code_type, count(r) AS req_count
        """
        
        return self._execute_query(query)
    
    def query_issue_impact(self) -> List[Dict]:
        """Query issues and their impact on code."""
        query = """
        MATCH (i:Issue)-[rel:ISSUES_RELATED]->(code:AdaNode)
        RETURN i.issue_id AS issue_id, i.title AS title, count(code) AS code_count
        """
        
        return self._execute_query(query)

def build_graph_database(
    code_nodes: List[AdaNode],
    req_nodes: List[RequirementNode],
    issues: List[GitHubIssue],
    commits: List[GitCommit],
    req_code_links: Dict[str, List[Tuple[str, float]]],
    issue_code_links: Dict[str, List[Tuple[str, float]]]
) -> None:
    """Build the Neo4j graph database from code nodes, requirements, issues, and commits."""
    print_timestamp("🚀 STARTED: Building Neo4j graph database")
    start_time = time.time()
    
    # Connect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    try:
        # Check if we have existing data in the graph
        if os.path.exists(GRAPH_STATUS_PATH):
            with open(GRAPH_STATUS_PATH, 'r') as f:
                graph_status = json.load(f)
                
            # Ask if we should reuse existing graph or rebuild
            if sum(graph_status["nodes_created"].values()) > 0:
                print_timestamp(f"ℹ️ Found existing graph with {sum(graph_status['nodes_created'].values())} nodes.")
                print_timestamp(f"ℹ️ Last updated: {graph_status['last_updated']}")
                
                # For this script, we'll just rebuild - in real use, might prompt user
                print_timestamp("🔄 Rebuilding graph database...")
                connector.clear_database()
        
        # Create constraints for faster lookups
        connector.create_constraints()
        
        # Add Ada nodes to the database
        print_timestamp(f"🔄 Adding {len(code_nodes)} Ada nodes to the database...")
        added = 0
        for node in tqdm(code_nodes, desc="Creating Ada nodes"):
            connector.create_ada_node(node)
            added += 1
            if added % 500 == 0:
                connector._save_graph_status()
        
        # Add requirement nodes to the database
        print_timestamp(f"🔄 Adding {len(req_nodes)} requirement nodes to the database...")
        for node in tqdm(req_nodes, desc="Creating requirement nodes"):
            connector.create_requirement_node(node)
        
        # Add issue nodes to the database
        print_timestamp(f"🔄 Adding {len(issues)} issue nodes to the database...")
        for node in tqdm(issues, desc="Creating issue nodes"):
            connector.create_issue_node(node)
        
        # Add commit nodes to the database
        print_timestamp(f"🔄 Adding {len(commits)} commit nodes to the database...")
        for node in tqdm(commits, desc="Creating commit nodes"):
            connector.create_commit_node(node)
        
        # Save node creation status
        connector._save_graph_status()
        
        # Build relationships
        connector.build_parent_child_relationships(code_nodes)
        connector.build_references_relationships(code_nodes)
        
        # Build similarity relationships
        connector.build_similarity_relationships(req_nodes, code_nodes, req_code_links, IMPLEMENTS)
        connector.build_similarity_relationships(issues, code_nodes, issue_code_links, ISSUES_RELATED)
        
        # Build commit relationships
        connector.build_commit_relationships(commits, code_nodes)
        
        # Final save of status
        connector._save_graph_status()
        
        end_time = time.time()
        duration = end_time - start_time
        print_timestamp(f"✅ COMPLETED: Built Neo4j graph database in {duration:.2f} seconds")
        
        # Print database statistics
        status = connector.graph_status
        print_timestamp(f"📊 Database statistics:")
        for node_type, count in status["nodes_created"].items():
            print_timestamp(f"  - {node_type} nodes: {count}")
        
        for rel_type, count in status["relationships_created"].items():
            print_timestamp(f"  - {rel_type} relationships: {count}")
        
    finally:
        connector.close()

if __name__ == "__main__":
    print_timestamp("This module should be imported, not run directly.") 