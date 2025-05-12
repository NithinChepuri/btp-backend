#!/usr/bin/env python3
"""
Script to analyze existing Ada requirements, issues, and commits for a comprehensive traceability view.
"""
import os
import sys
import json
import glob
import time
import re
from pathlib import Path
import argparse

from tqdm import tqdm
from graph_database import Neo4jConnector, print_timestamp
from analyze_single_req import Req2CodeAnalyzer
from code2graph import build_code_graph, load_checkpoint
from req2nodes import RequirementNode, parse_requirements

# Configuration
OUTPUT_DIR = os.path.join("outputs", "saved_analysis", "ada_requirements")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_ada_requirements():
    """Load existing Ada requirements from the ada_requirements directory."""
    req_dir = os.path.join("datasets", "ada-awa", "ada_requirements")
    requirements = []
    
    req_files = glob.glob(os.path.join(req_dir, "req*.txt"))
    for req_file in sorted(req_files):
        req_id = os.path.basename(req_file).replace(".txt", "")
        try:
            with open(req_file, 'r') as f:
                req_text = f.read().strip()
                if req_text:
                    requirements.append((req_id, req_text))
        except Exception as e:
            print_timestamp(f"Error reading {req_file}: {e}")
    
    return requirements

def load_issues():
    """Load GitHub issues from issues.json."""
    if os.path.exists("issues.json"):
        try:
            with open("issues.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            print_timestamp(f"Error loading issues.json: {e}")
    
    return []

def load_commits():
    """Load Git commits from commits.json."""
    if os.path.exists("commits.json"):
        try:
            with open("commits.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            print_timestamp(f"Error loading commits.json: {e}")
    
    return []

def analyze_requirements(requirements, output_dir=OUTPUT_DIR, top_n=10, max_reqs=None):
    """
    Analyze Ada requirements against the codebase.
    
    Args:
        requirements: List of (req_id, req_text) tuples
        output_dir: Directory to save results
        top_n: Number of top results to keep for each requirement
        max_reqs: Maximum number of requirements to analyze (for testing)
    
    Returns:
        Path to the combined analysis file
    """
    if max_reqs and max_reqs > 0:
        requirements = requirements[:max_reqs]
        
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
        req_output_file = os.path.join(output_dir, f"{req_id}_analysis.json")
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

def extract_ada_files_from_text(text):
    """
    Extract potential Ada file references from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of potential file names
    """
    # Common Ada file patterns
    file_pattern = r'\b[a-zA-Z0-9_\-.]+\.(ads|adb)\b'
    matches = re.findall(file_pattern, text)
    
    # Also look for package references that might be files
    package_pattern = r'\b(AWA|Ada|awa)\.[a-zA-Z0-9_]+\b'
    package_matches = re.findall(package_pattern, text)
    
    # Convert package matches to potential file names
    for pkg in package_matches:
        pkg_parts = pkg.split('.')
        if len(pkg_parts) >= 2:
            file_base = pkg_parts[-2] + '-' + pkg_parts[-1]
            matches.append(file_base.lower() + '.ads')
            matches.append(file_base.lower() + '.adb')
    
    return list(set(matches))

def find_file_matches(file_name, unique_files):
    """
    Find matches for a file name in the unique_files dictionary.
    
    Args:
        file_name: File name to match
        unique_files: Dictionary of file node IDs to file paths
        
    Returns:
        List of matching file node IDs
    """
    matches = []
    
    # Try exact match first
    exact_match_id = f"file-{file_name}"
    if exact_match_id in unique_files:
        matches.append(exact_match_id)
        return matches
    
    # Try base name match (ignore path)
    for file_id, file_path in unique_files.items():
        if file_id.endswith(f"-{file_name}"):
            matches.append(file_id)
            
    # If no matches, try fuzzy match
    if not matches:
        for file_id, file_path in unique_files.items():
            # Get just the filename part
            base_name = os.path.basename(file_path)
            # Check if file_name is similar to base_name
            if file_name.replace('_', '-') in base_name.replace('_', '-') or \
               base_name.replace('_', '-') in file_name.replace('_', '-'):
                matches.append(file_id)
    
    return matches

def load_all_to_neo4j(combined_analysis_file, issues, commits):
    """
    Load requirements, issues, and commits into Neo4j for a comprehensive traceability view.
    
    Args:
        combined_analysis_file: Path to the combined requirements analysis file
        issues: List of GitHub issues
        commits: List of Git commits
    """
    print_timestamp(f"Loading comprehensive traceability data into Neo4j...")
    
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
        req_node = RequirementNode(req_id, req_text, f"{req_id}.txt")
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
                    
                    for node in file_to_nodes[file_path]:
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
    
    # Extract all unique ada files for better matching
    all_ada_files = set()
    for file_id, file_path in unique_files.items():
        file_name = os.path.basename(file_path)
        all_ada_files.add(file_name)
    
    print_timestamp(f"Found {len(all_ada_files)} unique Ada files for matching")
    
    # Process GitHub issues - create Issue nodes and link to files/code nodes
    if issues:
        print_timestamp(f"Adding {len(issues)} GitHub issues...")
        
        for issue in tqdm(issues, desc="Processing issues"):
            issue_id = f"issue-{issue['number']}"
            issue_title = issue["title"]
            issue_body = issue.get("body", "") or ""
            
            # Create issue node
            connector._execute_query("""
            MERGE (i:Issue {id: $id})
            SET i.number = $number,
                i.title = $title,
                i.state = $state,
                i.body = $body,
                i.created_at = $created_at,
                i.url = $url
            """, {
                "id": issue_id,
                "number": issue["number"],
                "title": issue_title,
                "state": issue["state"],
                "body": issue_body,
                "created_at": issue["created_at"],
                "url": issue["html_url"]
            })
            
            # Analyze issue title and body to find potential file mentions
            potential_files = extract_ada_files_from_text(issue_title)
            potential_files.extend(extract_ada_files_from_text(issue_body))
            
            if potential_files:
                for file_name in potential_files:
                    # Find matching files in our unique_files dictionary
                    matches = find_file_matches(file_name, unique_files)
                    
                    for file_id in matches:
                        connector._execute_query("""
                        MATCH (i:Issue {id: $issue_id})
                        MATCH (f:File {id: $file_id})
                        MERGE (i)-[rel:MENTIONS]->(f)
                        """, {
                            "issue_id": issue_id,
                            "file_id": file_id
                        })
            
            # If no specific files were found, look for broader package mentions
            # and link to all files in that package
            if not potential_files:
                # Look for package mentions
                package_pattern = r'\b(awa|ada)[._-][a-zA-Z0-9_]+\b'
                package_matches = re.findall(package_pattern, issue_title.lower() + " " + issue_body.lower())
                
                for package in package_matches:
                    # Find files that might belong to this package
                    for file_id, file_path in unique_files.items():
                        if package.replace('_', '-').replace('.', '-') in file_path.lower():
                            connector._execute_query("""
                            MATCH (i:Issue {id: $issue_id})
                            MATCH (f:File {id: $file_id})
                            MERGE (i)-[rel:MIGHT_RELATE_TO]->(f)
                            """, {
                                "issue_id": issue_id,
                                "file_id": file_id
                            })
    
    # Process Git commits - create Commit nodes and link to files
    if commits:
        print_timestamp(f"Adding {len(commits)} Git commits...")
        
        for commit in tqdm(commits, desc="Processing commits"):
            commit_id = f"commit-{commit['sha'][:8]}"
            commit_message = commit["commit"]["message"]
            
            # Create commit node
            connector._execute_query("""
            MERGE (c:Commit {id: $id})
            SET c.sha = $sha,
                c.message = $message,
                c.author = $author,
                c.date = $date,
                c.url = $url
            """, {
                "id": commit_id,
                "sha": commit["sha"],
                "message": commit_message,
                "author": commit["commit"]["author"]["name"],
                "date": commit["commit"]["author"]["date"],
                "url": commit["html_url"]
            })
            
            # Extract files from commit message
            potential_files = extract_ada_files_from_text(commit_message)
            
            # Link commits to files from commit.files if available
            files_processed = False
            if "files" in commit:
                for file_change in commit["files"]:
                    file_name = os.path.basename(file_change["filename"])
                    if file_name.endswith(('.ads', '.adb')):
                        files_processed = True
                        # Find matching files
                        matches = find_file_matches(file_name, unique_files)
                        
                        for file_id in matches:
                            connector._execute_query("""
                            MATCH (c:Commit {id: $commit_id})
                            MATCH (f:File {id: $file_id})
                            MERGE (c)-[rel:MODIFIES]->(f)
                            SET rel.changes = $changes,
                                rel.additions = $additions,
                                rel.deletions = $deletions
                            """, {
                                "commit_id": commit_id,
                                "file_id": file_id,
                                "changes": file_change.get("changes", 0),
                                "additions": file_change.get("additions", 0),
                                "deletions": file_change.get("deletions", 0)
                            })
            
            # If no files were found in the commit.files, try to extract from message
            if not files_processed and potential_files:
                for file_name in potential_files:
                    matches = find_file_matches(file_name, unique_files)
                    
                    for file_id in matches:
                        connector._execute_query("""
                        MATCH (c:Commit {id: $commit_id})
                        MATCH (f:File {id: $file_id})
                        MERGE (c)-[rel:REFERENCES]->(f)
                        """, {
                            "commit_id": commit_id,
                            "file_id": file_id
                        })
            
            # If still no files found, look for package mentions
            if not files_processed and not potential_files:
                # Look for package mentions
                package_pattern = r'\b(awa|ada)[._-][a-zA-Z0-9_]+\b'
                package_matches = re.findall(package_pattern, commit_message.lower())
                
                for package in package_matches:
                    # Find files that might belong to this package
                    for file_id, file_path in unique_files.items():
                        if package.replace('_', '-').replace('.', '-') in file_path.lower():
                            connector._execute_query("""
                            MATCH (c:Commit {id: $commit_id})
                            MATCH (f:File {id: $file_id})
                            MERGE (c)-[rel:MIGHT_RELATE_TO]->(f)
                            """, {
                                "commit_id": commit_id,
                                "file_id": file_id
                            })
    
    # Connect issues to commits (based on issue number references in commit messages)
    print_timestamp("Connecting issues to commits...")
    connector._execute_query("""
    MATCH (c:Commit)
    MATCH (i:Issue)
    WHERE c.message CONTAINS ('#' + toString(i.number)) OR
          c.message CONTAINS ('issue ' + toString(i.number)) OR
          c.message CONTAINS ('Issue ' + toString(i.number)) OR
          c.message CONTAINS ('fixes #' + toString(i.number)) OR
          c.message CONTAINS ('closes #' + toString(i.number))
    MERGE (c)-[rel:RESOLVES]->(i)
    """)
    
    # Build relationships between nodes
    print_timestamp("Building parent-child relationships...")
    connector.build_parent_child_relationships(code_nodes)
    
    # Build reference relationships
    print_timestamp("Building reference relationships...")
    connector.build_references_relationships(code_nodes)
    
    # Create direct links between requirements and issues/commits
    print_timestamp("Creating direct links between requirements and issues/commits...")
    connector._execute_query("""
    MATCH (r:Requirement)-[:RELATED_TO]->(f:File)<-[:MENTIONS|MIGHT_RELATE_TO]-(i:Issue)
    MERGE (r)-[rel:RELATED_ISSUE]->(i)
    """)
    
    connector._execute_query("""
    MATCH (r:Requirement)-[:RELATED_TO]->(f:File)<-[:MODIFIES|REFERENCES|MIGHT_RELATE_TO]-(c:Commit)
    MERGE (r)-[rel:RELATED_COMMIT]->(c)
    """)
    
    # Create a special node to identify this as a complete analysis
    connector._execute_query("""
    MERGE (a:Analysis {id: 'complete-ada-analysis'})
    SET a.timestamp = $timestamp,
        a.num_requirements = $num_requirements,
        a.num_issues = $num_issues,
        a.num_commits = $num_commits,
        a.analysis_file = $analysis_file
    """, {
        "timestamp": analysis["analysis_timestamp"],
        "num_requirements": analysis["num_requirements"],
        "num_issues": len(issues),
        "num_commits": len(commits),
        "analysis_file": combined_analysis_file
    })
    
    print_timestamp("Complete traceability data loaded successfully into Neo4j")
    
    print("\nUseful Cypher queries to explore the data:")
    print("1. View all requirements and their related files:")
    print("   MATCH (r:Requirement)-[rel:RELATED_TO]->(f:File) RETURN r, rel, f")
    print("2. View files that satisfy multiple requirements:")
    print("   MATCH (r:Requirement)-[rel:RELATED_TO]->(f:File)")
    print("   WITH f, count(r) as req_count, collect(r) as requirements")
    print("   WHERE req_count > 1")
    print("   RETURN f, requirements")
    print("3. View issues that relate to requirements:")
    print("   MATCH (r:Requirement)-[rel:RELATED_ISSUE]->(i:Issue)")
    print("   RETURN r, rel, i")
    print("4. View commits that relate to requirements:")
    print("   MATCH (r:Requirement)-[rel:RELATED_COMMIT]->(c:Commit)")
    print("   RETURN r, rel, c")
    print("5. View the complete traceability chain:")
    print("   MATCH p=(r:Requirement)-[:RELATED_TO]->(f:File)<-[:MENTIONS|MODIFIES]-(x)")
    print("   WHERE x:Issue OR x:Commit")
    print("   RETURN p")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze Ada requirements and load them into Neo4j")
    
    parser.add_argument("--max-reqs", type=int, default=0,
                        help="Maximum number of requirements to analyze (for testing)")
    
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top results to keep for each requirement")
    
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory to save results")
    
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip analysis and use existing combined analysis file")
    
    args = parser.parse_args()
    
    # Load requirements, issues, and commits
    requirements = load_ada_requirements()
    issues = load_issues()
    commits = load_commits()
    
    print_timestamp(f"Loaded {len(requirements)} requirements")
    print_timestamp(f"Loaded {len(issues)} GitHub issues")
    print_timestamp(f"Loaded {len(commits)} Git commits")
    
    # Analyze requirements or use existing analysis
    combined_analysis_file = os.path.join(args.output_dir, "combined_analysis.json")
    
    if not args.skip_analysis or not os.path.exists(combined_analysis_file):
        combined_analysis_file = analyze_requirements(
            requirements, args.output_dir, args.top_n, args.max_reqs
        )
    else:
        print_timestamp(f"Using existing analysis file: {combined_analysis_file}")
    
    # Load results into Neo4j
    if input("Load results into Neo4j? (y/n): ").lower() == 'y':
        load_all_to_neo4j(combined_analysis_file, issues, commits)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 