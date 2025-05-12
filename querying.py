"""
Module for querying the traceability relationships in the knowledge graph.
This module provides tools for exploring relationships between requirements, code, issues, and commits.
"""
import json
import os
from typing import Dict, List, Any, Optional

from neo4j import GraphDatabase, basic_auth
from tabulate import tabulate

from constants import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OUTPUT_DIR
from graph_database import Neo4jConnector

class TraceabilityQuery:
    """Class for querying traceability relationships in the knowledge graph."""
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        """Initialize the traceability query tool."""
        self.connector = Neo4jConnector(uri, user, password)
        self.connector.connect()
    
    def close(self):
        """Close the Neo4j connection."""
        self.connector.close()
    
    def get_requirements_coverage(self) -> List[Dict]:
        """Get all requirements and their code coverage."""
        query = """
        MATCH (r:Requirement)
        OPTIONAL MATCH (r)-[rel:IMPLEMENTS]->(code:AdaNode)
        WITH r, collect({
            code_id: code.id,
            code_name: code.name,
            code_type: code.type,
            similarity: rel.similarity
        }) as implementations
        RETURN r.req_id as req_id, 
               r.text as text, 
               size(implementations) as implementation_count,
               implementations
        ORDER BY r.req_id
        """
        
        results = self.connector._execute_query(query)
        return results
    
    def get_code_requirements(self) -> List[Dict]:
        """Get all code nodes and their related requirements."""
        query = """
        MATCH (code:AdaNode)
        OPTIONAL MATCH (r:Requirement)-[rel:IMPLEMENTS]->(code)
        WITH code, collect({
            req_id: r.req_id,
            text: r.text,
            similarity: rel.similarity
        }) as requirements
        RETURN code.id as code_id,
               code.name as code_name,
               code.type as code_type,
               size(requirements) as requirement_count,
               requirements
        ORDER BY code.type, code.name
        """
        
        results = self.connector._execute_query(query)
        return results
    
    def get_issues_impact(self) -> List[Dict]:
        """Get all issues and their impact on code."""
        query = """
        MATCH (i:Issue)
        OPTIONAL MATCH (i)-[rel:ISSUES_RELATED]->(code:AdaNode)
        WITH i, collect({
            code_id: code.id,
            code_name: code.name,
            code_type: code.type,
            similarity: rel.similarity
        }) as impacted_code
        RETURN i.issue_id as issue_id,
               i.title as title,
               i.state as state,
               size(impacted_code) as impact_count,
               impacted_code
        ORDER BY i.issue_id
        """
        
        results = self.connector._execute_query(query)
        return results
    
    def get_commit_changes(self) -> List[Dict]:
        """Get all commits and their code changes."""
        query = """
        MATCH (c:Commit)
        OPTIONAL MATCH (c)-[rel:COMMITS_TO]->(code:AdaNode)
        WITH c, collect({
            code_id: code.id,
            code_name: code.name,
            code_type: code.type
        }) as changed_code
        RETURN c.commit_hash as commit_hash,
               c.author as author,
               c.message as message,
               c.date as date,
               size(changed_code) as change_count,
               changed_code
        ORDER BY c.date DESC
        """
        
        results = self.connector._execute_query(query)
        return results
    
    def trace_requirement_to_code(self, req_id: str) -> List[Dict]:
        """Trace a requirement to its implementing code."""
        query = """
        MATCH (r:Requirement {req_id: $req_id})-[rel:IMPLEMENTS]->(code:AdaNode)
        RETURN code.id as code_id,
               code.name as code_name,
               code.type as code_type,
               code.file_path as file_path,
               code.line_number as line_number,
               rel.similarity as similarity
        ORDER BY rel.similarity DESC
        """
        
        params = {"req_id": req_id}
        results = self.connector._execute_query(query, params)
        return results
    
    def trace_code_to_requirements(self, code_name: str, code_type: Optional[str] = None) -> List[Dict]:
        """Trace code to its related requirements."""
        if code_type:
            query = """
            MATCH (r:Requirement)-[rel:IMPLEMENTS]->(code:AdaNode {name: $code_name, type: $code_type})
            RETURN r.req_id as req_id,
                   r.text as text,
                   rel.similarity as similarity
            ORDER BY rel.similarity DESC
            """
            params = {"code_name": code_name, "code_type": code_type}
        else:
            query = """
            MATCH (r:Requirement)-[rel:IMPLEMENTS]->(code:AdaNode {name: $code_name})
            RETURN r.req_id as req_id,
                   r.text as text,
                   code.type as code_type,
                   rel.similarity as similarity
            ORDER BY rel.similarity DESC
            """
            params = {"code_name": code_name}
        
        results = self.connector._execute_query(query, params)
        return results
    
    def find_code_without_requirements(self) -> List[Dict]:
        """Find code nodes that are not linked to any requirements."""
        query = """
        MATCH (code:AdaNode)
        WHERE NOT EXISTS { MATCH (r:Requirement)-[:IMPLEMENTS]->(code) }
        RETURN code.id as code_id,
               code.name as code_name,
               code.type as code_type,
               code.file_path as file_path
        ORDER BY code.type, code.name
        """
        
        results = self.connector._execute_query(query)
        return results
    
    def find_requirements_without_code(self) -> List[Dict]:
        """Find requirements that are not linked to any code."""
        query = """
        MATCH (r:Requirement)
        WHERE NOT EXISTS { MATCH (r)-[:IMPLEMENTS]->(:AdaNode) }
        RETURN r.req_id as req_id,
               r.text as text
        ORDER BY r.req_id
        """
        
        results = self.connector._execute_query(query)
        return results
    
    def trace_issue_to_requirements(self, issue_id: int) -> List[Dict]:
        """Trace an issue to affected requirements through code."""
        query = """
        MATCH (i:Issue {issue_id: $issue_id})-[:ISSUES_RELATED]->(code:AdaNode)<-[rel:IMPLEMENTS]-(r:Requirement)
        RETURN DISTINCT r.req_id as req_id,
                        r.text as text
        ORDER BY r.req_id
        """
        
        params = {"issue_id": issue_id}
        results = self.connector._execute_query(query, params)
        return results
    
    def get_traceability_stats(self) -> Dict:
        """Get statistics about the traceability coverage."""
        stats = {}
        
        # Count of requirements
        query = "MATCH (r:Requirement) RETURN count(r) as count"
        result = self.connector._execute_query(query)
        stats["total_requirements"] = result[0]["count"] if result else 0
        
        # Count of requirements with code
        query = """
        MATCH (r:Requirement)-[:IMPLEMENTS]->(:AdaNode)
        RETURN count(DISTINCT r) as count
        """
        result = self.connector._execute_query(query)
        stats["requirements_with_code"] = result[0]["count"] if result else 0
        
        # Count of code nodes
        query = "MATCH (c:AdaNode) RETURN count(c) as count"
        result = self.connector._execute_query(query)
        stats["total_code_nodes"] = result[0]["count"] if result else 0
        
        # Count of code nodes with requirements
        query = """
        MATCH (c:AdaNode)<-[:IMPLEMENTS]-(:Requirement)
        RETURN count(DISTINCT c) as count
        """
        result = self.connector._execute_query(query)
        stats["code_with_requirements"] = result[0]["count"] if result else 0
        
        # Count of issues
        query = "MATCH (i:Issue) RETURN count(i) as count"
        result = self.connector._execute_query(query)
        stats["total_issues"] = result[0]["count"] if result else 0
        
        # Count of commits
        query = "MATCH (c:Commit) RETURN count(c) as count"
        result = self.connector._execute_query(query)
        stats["total_commits"] = result[0]["count"] if result else 0
        
        # Calculate percentages
        if stats["total_requirements"] > 0:
            stats["requirements_coverage_percentage"] = (stats["requirements_with_code"] / stats["total_requirements"]) * 100
        else:
            stats["requirements_coverage_percentage"] = 0
            
        if stats["total_code_nodes"] > 0:
            stats["code_coverage_percentage"] = (stats["code_with_requirements"] / stats["total_code_nodes"]) * 100
        else:
            stats["code_coverage_percentage"] = 0
        
        return stats
    
    def print_traceability_report(self):
        """Print a comprehensive traceability report."""
        stats = self.get_traceability_stats()
        
        print("\n===== Traceability Report =====")
        print(f"\nRequirements Coverage: {stats['requirements_coverage_percentage']:.2f}% ({stats['requirements_with_code']}/{stats['total_requirements']})")
        print(f"Code Coverage: {stats['code_coverage_percentage']:.2f}% ({stats['code_with_requirements']}/{stats['total_code_nodes']})")
        print(f"Total Issues: {stats['total_issues']}")
        print(f"Total Commits: {stats['total_commits']}")
        
        # Get requirements without code
        uncovered_reqs = self.find_requirements_without_code()
        if uncovered_reqs:
            print(f"\nRequirements without code implementations ({len(uncovered_reqs)}):")
            table_data = [(r["req_id"], r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"]) for r in uncovered_reqs[:5]]
            print(tabulate(table_data, headers=["Req ID", "Text"]))
            if len(uncovered_reqs) > 5:
                print(f"... and {len(uncovered_reqs) - 5} more")
        
        # Get code without requirements
        uncovered_code = self.find_code_without_requirements()
        if uncovered_code:
            print(f"\nCode without requirements ({len(uncovered_code)}):")
            table_data = [(c["code_name"], c["code_type"], c["file_path"]) for c in uncovered_code[:5]]
            print(tabulate(table_data, headers=["Name", "Type", "File Path"]))
            if len(uncovered_code) > 5:
                print(f"... and {len(uncovered_code) - 5} more")

def export_traceability_json(output_file: Optional[str] = None):
    """Export all traceability data to a JSON file."""
    query_tool = TraceabilityQuery()
    
    try:
        # Collect all data
        data = {
            "stats": query_tool.get_traceability_stats(),
            "requirements": query_tool.get_requirements_coverage(),
            "code": query_tool.get_code_requirements(),
            "issues": query_tool.get_issues_impact(),
            "commits": query_tool.get_commit_changes(),
            "uncovered_requirements": query_tool.find_requirements_without_code(),
            "uncovered_code": query_tool.find_code_without_requirements()
        }
        
        # Save to file
        if output_file is None:
            output_file = os.path.join(OUTPUT_DIR, "traceability_data.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported traceability data to {output_file}")
        
    finally:
        query_tool.close()

if __name__ == "__main__":
    # Create the query tool
    query_tool = TraceabilityQuery()
    
    try:
        # Print a comprehensive report
        query_tool.print_traceability_report()
        
        # Export data to JSON
        export_traceability_json()
        
    finally:
        query_tool.close() 