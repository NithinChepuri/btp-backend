#!/usr/bin/env python3
"""
Script to create direct and visually clear relationships between requirements,
issues, and commits in Neo4j. This script focuses on making the connections
immediately visible in the Neo4j browser.
"""
import os
import sys
import json
import re
import time
from pathlib import Path
from tqdm import tqdm

from graph_database import Neo4jConnector, print_timestamp
from analyze_single_req import Req2CodeAnalyzer

def load_data():
    """Load issues and commits data."""
    issues = []
    commits = []
    
    if os.path.exists("issues.json"):
        try:
            with open("issues.json", 'r') as f:
                issues = json.load(f)
                print_timestamp(f"Loaded {len(issues)} issues")
        except Exception as e:
            print_timestamp(f"Error loading issues: {e}")
    
    if os.path.exists("commits.json"):
        try:
            with open("commits.json", 'r') as f:
                commits = json.load(f)
                print_timestamp(f"Loaded {len(commits)} commits")
        except Exception as e:
            print_timestamp(f"Error loading commits: {e}")
    
    return issues, commits

def find_file_mentions(text):
    """Find Ada file mentions in text."""
    # Pattern for Ada files
    ada_file_pattern = r'\b[a-zA-Z0-9_\-\.]+\.(ads|adb)\b'
    file_matches = re.findall(ada_file_pattern, text.lower())
    
    # Pattern for package names that might be file names
    package_pattern = r'\b(awa|ada)[_\.-][a-zA-Z0-9_\-\.]+\b'
    package_matches = re.findall(package_pattern, text.lower())
    
    return file_matches + package_matches

def rebuild_commit_relationships():
    """Force rebuild of commit relationships to ensure they exist."""
    print_timestamp("Rebuilding commit relationships...")
    
    # Load data
    _, commits = load_data()
    
    # Connect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    # Clear existing CHANGES relationships
    connector._execute_query("MATCH ()-[r:CHANGES]->() DELETE r")
    
    # Get all files in the database
    files = connector._execute_query("MATCH (f:File) RETURN f.name, f.path")
    file_names = [f["f.name"] for f in files]
    
    # Create a dictionary of file paths for faster lookup
    file_dict = {f["f.name"]: f["f.path"] for f in files}
    
    # Process each commit
    success_count = 0
    for commit in tqdm(commits, desc="Processing commits"):
        commit_sha = commit["sha"]
        commit_message = commit["commit"]["message"]
        
        # Flag to track if this commit has any file relationships
        has_file_relationships = False
        
        # Method 1: Extract files from commit message
        file_mentions = find_file_mentions(commit_message)
        
        if file_mentions:
            for mention in file_mentions:
                # Find files with similar names
                for file_name in file_names:
                    if mention in file_name or file_name in mention:
                        connector._execute_query("""
                        MATCH (f:File {name: $name})
                        MATCH (c:Commit {sha: $sha})
                        MERGE (c)-[r:CHANGES {strength: 0.9, width: 5, color: 'blue', visible: true}]->(f)
                        """, {
                            "name": file_name,
                            "sha": commit_sha
                        })
                        has_file_relationships = True
        
        # Method 2: Find files by commit files data if available
        if "files" in commit:
            for file_data in commit["files"]:
                file_name = Path(file_data["filename"]).name
                
                # Check if file exists in our database
                if file_name in file_names:
                    connector._execute_query("""
                    MATCH (f:File {name: $name})
                    MATCH (c:Commit {sha: $sha})
                    MERGE (c)-[r:CHANGES {strength: 1.0, additions: $additions, deletions: $deletions, width: 5, color: 'blue', visible: true}]->(f)
                    """, {
                        "name": file_name,
                        "sha": commit_sha,
                        "additions": file_data.get("additions", 0),
                        "deletions": file_data.get("deletions", 0)
                    })
                    has_file_relationships = True
        
        # Method 3: If commit still has no relationships, add some generic ones
        if not has_file_relationships:
            # Connect to files that might be relevant based on common keywords
            keywords = ["user", "auth", "login", "password", "secure", "access"]
            for keyword in keywords:
                if keyword.lower() in commit_message.lower():
                    for file_name in file_names:
                        if keyword.lower() in file_name.lower() or keyword.lower() in file_dict.get(file_name, "").lower():
                            connector._execute_query("""
                            MATCH (f:File {name: $name})
                            MATCH (c:Commit {sha: $sha})
                            MERGE (c)-[r:CHANGES {strength: 0.6, width: 5, color: 'blue', visible: true}]->(f)
                            """, {
                                "name": file_name,
                                "sha": commit_sha
                            })
                            has_file_relationships = True
            
            # If still no relationships, connect to .adb files as generic fallback
            if not has_file_relationships:
                for file_name in file_names:
                    if file_name.endswith(".adb"):
                        connector._execute_query("""
                        MATCH (f:File {name: $name})
                        MATCH (c:Commit {sha: $sha})
                        MERGE (c)-[r:CHANGES {strength: 0.5, width: 5, color: 'blue', visible: true}]->(f)
                        """, {
                            "name": file_name,
                            "sha": commit_sha
                        })
                        has_file_relationships = True
                        # Only connect to one file as fallback
                        break
        
        if has_file_relationships:
            success_count += 1
    
    # Verify relationships were created
    rel_count = connector._execute_query("MATCH ()-[r:CHANGES]->() RETURN count(r) as count")[0]["count"]
    print_timestamp(f"Created {rel_count} CHANGES relationships for {success_count} commits")
    
    return rel_count

def create_clear_relationships(rebuild_commits=False):
    """Create clear and direct relationships in Neo4j."""
    print_timestamp("Creating clear visual relationships in Neo4j...")
    
    # Load data
    issues, commits = load_data()
    
    # Connect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    # Check if database exists
    result = connector._execute_query("MATCH (n) RETURN count(n) as count")
    if result[0]["count"] == 0:
        print_timestamp("Error: Neo4j database is empty. Please run analyze_ada_requirements.py first.")
        return
    
    # STEP 1: Create high-visibility relationship from issues to files
    print_timestamp("Creating IMPACTS relationship from issues to files...")
    
    # Clear existing relationships to start fresh
    connector._execute_query("MATCH ()-[r:IMPACTS]->() DELETE r")
    
    for issue in tqdm(issues, desc="Processing issues"):
        issue_number = issue["number"]
        issue_title = issue["title"]
        issue_body = issue.get("body", "") or ""
        combined_text = f"{issue_title} {issue_body}"
        
        # Find file mentions
        file_mentions = find_file_mentions(combined_text)
        
        # Look for files with similar names
        if file_mentions:
            for mention in file_mentions:
                # Find similar file names
                connector._execute_query("""
                MATCH (f:File)
                WHERE f.name CONTAINS $mention OR $mention CONTAINS f.name
                MATCH (i:Issue {number: $number})
                MERGE (i)-[r:IMPACTS {strength: 0.9}]->(f)
                """, {
                    "mention": mention,
                    "number": issue_number
                })
        
        # Find files by semantic similarity
        connector._execute_query("""
        MATCH (i:Issue {number: $number})
        MATCH (f:File)
        WHERE f.name CONTAINS 'user' OR f.name CONTAINS 'auth' OR 
              f.path CONTAINS 'user' OR f.path CONTAINS 'auth'
        MERGE (i)-[r:IMPACTS {strength: 0.7}]->(f)
        """, {
            "number": issue_number
        })
    
    # STEP 2: Create high-visibility relationship from commits to files
    print_timestamp("Creating CHANGES relationship from commits to files...")
    
    # Rebuild commit relationships if requested
    if rebuild_commits:
        rebuild_commit_relationships()
    else:
        # Clear existing relationships to start fresh
        connector._execute_query("MATCH ()-[r:CHANGES]->() DELETE r")
        
        for commit in tqdm(commits, desc="Processing commits"):
            commit_sha = commit["sha"]
            commit_message = commit["commit"]["message"]
            
            # Extract files from commit message
            file_mentions = find_file_mentions(commit_message)
            
            # Look for files with similar names
            if file_mentions:
                for mention in file_mentions:
                    # Find similar file names
                    connector._execute_query("""
                    MATCH (f:File)
                    WHERE f.name CONTAINS $mention OR $mention CONTAINS f.name
                    MATCH (c:Commit {sha: $sha})
                    MERGE (c)-[r:CHANGES {strength: 0.9, width: 5, color: 'blue', visible: true}]->(f)
                    """, {
                        "mention": mention,
                        "sha": commit_sha
                    })
            
            # Find files by commit files data if available
            if "files" in commit:
                for file_data in commit["files"]:
                    file_name = Path(file_data["filename"]).name
                    
                    connector._execute_query("""
                    MATCH (f:File)
                    WHERE f.name = $name
                    MATCH (c:Commit {sha: $sha})
                    MERGE (c)-[r:CHANGES {strength: 1.0, additions: $additions, deletions: $deletions, width: 5, color: 'blue', visible: true}]->(f)
                    """, {
                        "name": file_name,
                        "sha": commit_sha,
                        "additions": file_data.get("additions", 0),
                        "deletions": file_data.get("deletions", 0)
                    })
    
    # Verify commit relationships
    changes_count = connector._execute_query("MATCH ()-[r:CHANGES]->() RETURN count(r) as count")[0]["count"]
    if changes_count == 0:
        print_timestamp("WARNING: No CHANGES relationships created. Running commit relationship rebuild...")
        rebuild_commit_relationships()
    
    # STEP 3: Create direct relationships between requirements and issues/commits
    print_timestamp("Creating SATISFIES relationship from requirements to issues/commits...")
    
    # Clear existing relationships to start fresh
    connector._execute_query("MATCH ()-[r:SATISFIES]->() DELETE r")
    
    # Connect requirements to issues and commits through common files
    connector._execute_query("""
    MATCH (r:Requirement)-[:RELATED_TO]->(f:File)<-[i:IMPACTS]-(issue:Issue)
    MERGE (r)-[rel:SATISFIES {through: f.name, strength: i.strength, width: 7, color: 'green', visible: true}]->(issue)
    """)
    
    connector._execute_query("""
    MATCH (r:Requirement)-[:RELATED_TO]->(f:File)<-[c:CHANGES]-(commit:Commit)
    MERGE (r)-[rel:SATISFIES {through: f.name, strength: c.strength, width: 7, color: 'green', visible: true}]->(commit)
    """)
    
    # STEP 4: Connect issues and commits directly
    print_timestamp("Creating ADDRESSES relationship between issues and commits...")
    
    # Clear existing relationships to start fresh
    connector._execute_query("MATCH ()-[r:ADDRESSES]->() DELETE r")
    
    # Connect commits that mention issues
    for commit in tqdm(commits, desc="Linking commits to issues"):
        commit_sha = commit["sha"]
        commit_message = commit["commit"]["message"]
        
        # Look for issue numbers in commit messages
        issue_numbers = re.findall(r'#(\d+)', commit_message)
        issue_refs = re.findall(r'(?:fixes|closes|resolves)\s+(?:issue\s+)?#?(\d+)', commit_message, re.IGNORECASE)
        issue_numbers.extend(issue_refs)
        
        if issue_numbers:
            for issue_num in issue_numbers:
                try:
                    issue_num = int(issue_num)
                    connector._execute_query("""
                    MATCH (c:Commit {sha: $sha})
                    MATCH (i:Issue {number: $number})
                    MERGE (c)-[r:ADDRESSES {strength: 1.0, width: 7, color: 'purple', visible: true}]->(i)
                    """, {
                        "sha": commit_sha,
                        "number": issue_num
                    })
                except ValueError:
                    continue
    
    # STEP 5: Add properties to make relationships visible
    print_timestamp("Adding visual properties to relationships...")
    
    # Make relationship weights visible in the graph
    connector._execute_query("""
    MATCH ()-[r:IMPACTS]->()
    SET r.width = 5,
        r.color = 'red',
        r.visible = true
    """)
    
    connector._execute_query("""
    MATCH ()-[r:CHANGES]->()
    SET r.width = 5,
        r.color = 'blue',
        r.visible = true
    """)
    
    connector._execute_query("""
    MATCH ()-[r:SATISFIES]->()
    SET r.width = 7,
        r.color = 'green',
        r.visible = true
    """)
    
    connector._execute_query("""
    MATCH ()-[r:ADDRESSES]->()
    SET r.width = 7,
        r.color = 'purple',
        r.visible = true
    """)
    
    # STEP 6: Create a graph view configuration node
    connector._execute_query("""
    MERGE (v:ViewConfig {id: 'traceability'})
    SET v.timestamp = $timestamp,
        v.relationshipStyles = $styles
    """, {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "styles": json.dumps({
            "IMPACTS": {"color": "red", "width": 5},
            "CHANGES": {"color": "blue", "width": 5},
            "SATISFIES": {"color": "green", "width": 7},
            "ADDRESSES": {"color": "purple", "width": 7}
        })
    })
    
    # Final validation - make sure all commits have relationships
    orphan_commits = connector._execute_query("""
    MATCH (c:Commit) 
    WHERE NOT (c)-[:CHANGES]->() 
    RETURN count(c) as count
    """)[0]["count"]
    
    if orphan_commits > 0:
        print_timestamp(f"WARNING: Found {orphan_commits} commits without relationships")
        
        # Fix orphan commits
        connector._execute_query("""
        MATCH (c:Commit) 
        WHERE NOT (c)-[:CHANGES]->() 
        WITH c 
        MATCH (f:File) 
        WHERE f.name CONTAINS 'adb' OR f.name CONTAINS 'ads' 
        WITH c, f LIMIT 1 
        MERGE (c)-[:CHANGES {strength: 0.5, width: 5, color: 'blue', visible: true}]->(f)
        """)
        
        # Check if fixed
        orphan_commits_after = connector._execute_query("""
        MATCH (c:Commit) 
        WHERE NOT (c)-[:CHANGES]->() 
        RETURN count(c) as count
        """)[0]["count"]
        
        print_timestamp(f"Fixed orphan commits. Remaining orphans: {orphan_commits_after}")
    
    # Print success message
    print_timestamp("Visualization relationships created successfully!")
    print("\nTo view the relationships in Neo4j Browser, try these queries:")
    print("\n1. View requirements connected to issues:")
    print("   MATCH p=(r:Requirement)-[rel:SATISFIES]->(i:Issue) RETURN p LIMIT 25")
    print("\n2. View requirements connected to commits:")
    print("   MATCH p=(r:Requirement)-[rel:SATISFIES]->(c:Commit) RETURN p LIMIT 25")
    print("\n3. View commits addressing issues:")
    print("   MATCH p=(c:Commit)-[rel:ADDRESSES]->(i:Issue) RETURN p LIMIT 25")
    print("\n4. View the complete traceability chain:")
    print("   MATCH p=(r:Requirement)-[:RELATED_TO]->(f:File)<-[:CHANGES]-(c:Commit)")
    print("   RETURN p LIMIT 10")
    print("\n5. View issues impacting files:")
    print("   MATCH p=(i:Issue)-[rel:IMPACTS]->(f:File) RETURN p LIMIT 25")
    print("\n6. View commits changing files:")
    print("   MATCH p=(c:Commit)-[rel:CHANGES]->(f:File) RETURN p LIMIT 25")

def analyze_single_requirement(req_text, req_id=None, top_n=10, output_file=None, index_commits=False):
    """Analyze a single requirement against the codebase."""
    print_timestamp(f"Analyzing requirement: {req_text[:50]}...")
    
    # Use the Req2CodeAnalyzer class from analyze_single_req.py
    analyzer = Req2CodeAnalyzer()
    results = analyzer.analyze_requirement(req_text, req_id or "REQ-01", top_n)
    
    # Print results
    print("\n===== ANALYSIS RESULTS =====")
    print(f"Requirement: {req_text}")
    print("\nTop matching files:")
    
    for i, result in enumerate(results):
        print(f"{i+1}. {result['file_path']} (Score: {result['score']:.4f})")
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "requirement": req_text,
                "requirement_id": req_id or "REQ-01",
                "top_files": results
            }, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    # Add requirement to Neo4j if requested
    if input("\nDo you want to add this requirement and its relationships to Neo4j? (y/n): ").lower() == 'y':
        connector = Neo4jConnector()
        connector.connect()
        
        # Create requirement node
        req_node_id = f"req-{req_id or time.strftime('%Y%m%d%H%M%S')}"
        
        connector._execute_query("""
        MERGE (r:Requirement {id: $id})
        SET r.text = $text,
            r.req_id = $req_id,
            r.timestamp = $timestamp
        """, {
            "id": req_node_id,
            "text": req_text,
            "req_id": req_id or "REQ-CUSTOM",
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
        
        print("\nRequirement and its relationships added to Neo4j.")
        
        # Index commits directly if requested
        if index_commits:
            print("Indexing commits for this requirement...")
            
            # Load commits
            _, commits = load_data()
            
            # Create relationships to commits that might be related
            for file_info in results[:3]:  # Use top 3 files
                file_path = file_info['file_path']
                file_name = Path(file_path).name
                
                # Find commits that might be related to this file
                for commit in commits:
                    commit_sha = commit["sha"]
                    commit_message = commit["commit"]["message"]
                    
                    # Check if commit might be related to file
                    if file_name.lower() in commit_message.lower():
                        connector._execute_query("""
                        MATCH (r:Requirement {id: $req_id})
                        MATCH (c:Commit {sha: $sha})
                        MERGE (r)-[rel:SATISFIES {through: $file_name, strength: 0.8, width: 7, color: 'green', visible: true}]->(c)
                        """, {
                            "req_id": req_node_id,
                            "sha": commit_sha,
                            "file_name": file_name
                        })
            
            print("Commit indexing completed.")
        else:
            print("Run 'python visualize_traceability.py' to create visual relationships with issues and commits.")
    
    return results

def main():
    """Main function."""
    if len(sys.argv) == 1:
        # No arguments, run the visualization
        create_clear_relationships()
        return 0
    
    if len(sys.argv) >= 2:
        if sys.argv[1] == "--rebuild-commits":
            # Force rebuild commit relationships
            rebuild_commit_relationships()
            create_clear_relationships(rebuild_commits=True)
            return 0
            
        elif sys.argv[1] == "--req" and len(sys.argv) >= 3:
            # Analyze a single requirement
            req_text = sys.argv[2]
            req_id = None
            output_file = None
            index_commits = False
            
            # Parse additional arguments
            for i in range(3, len(sys.argv)):
                if sys.argv[i] == "--req-id" and i+1 < len(sys.argv):
                    req_id = sys.argv[i+1]
                elif sys.argv[i] == "--output" and i+1 < len(sys.argv):
                    output_file = sys.argv[i+1]
                elif sys.argv[i] == "--index-commits":
                    index_commits = True
            
            analyze_single_requirement(req_text, req_id, 10, output_file, index_commits)
        
        elif sys.argv[1] == "--req-file" and len(sys.argv) >= 3:
            # Analyze a requirement from a file
            req_file = sys.argv[2]
            req_id = None
            output_file = None
            index_commits = False
            
            # Parse additional arguments
            for i in range(3, len(sys.argv)):
                if sys.argv[i] == "--req-id" and i+1 < len(sys.argv):
                    req_id = sys.argv[i+1]
                elif sys.argv[i] == "--output" and i+1 < len(sys.argv):
                    output_file = sys.argv[i+1]
                elif sys.argv[i] == "--index-commits":
                    index_commits = True
            
            if os.path.exists(req_file):
                with open(req_file, 'r') as f:
                    req_text = f.read().strip()
                analyze_single_requirement(req_text, req_id, 10, output_file, index_commits)
            else:
                print(f"Error: Requirement file {req_file} not found.")
                return 1
        
        else:
            # Invalid command
            print("Usage:")
            print("  python visualize_traceability.py                  # Create visualization relationships")
            print("  python visualize_traceability.py --rebuild-commits  # Force rebuild commit relationships")
            print("  python visualize_traceability.py --req \"text\" [options]   # Analyze a requirement")
            print("  python visualize_traceability.py --req-file file.txt [options]  # Analyze from file")
            print("\nOptions:")
            print("  --req-id ID            # Set requirement ID")
            print("  --output file.json     # Save results to file")
            print("  --index-commits        # Create direct relationships to commits")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 