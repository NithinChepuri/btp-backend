"""
Main runner script for the Ada Traceability project.
This script orchestrates the entire pipeline from extraction to graph database building.
"""
import os
import json
import argparse
import time
from pathlib import Path

from constants import OUTPUT_DIR, SUMMARIES_DIR, GRAPH_DIR
from code2graph import build_code_graph
from req2nodes import parse_requirements
from github_integration import extract_github_data
from embeddings import (
    generate_embeddings, 
    link_nodes_by_similarity, 
    apply_summaries_to_nodes, 
    load_summaries
)
from graph_database import build_graph_database
from querying import TraceabilityQuery, export_traceability_json

def ensure_directories_exist():
    """Ensure that all necessary directories exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    print(f"📁 Ensured output directories exist: {OUTPUT_DIR}, {SUMMARIES_DIR}, {GRAPH_DIR}")

def extract_data():
    """Extract data from the code, requirements, and GitHub."""
    print("=== Extracting Data ===")
    
    # Extract code nodes
    print("\n📊 Extracting code nodes...")
    code_nodes = build_code_graph()
    print(f"✅ Extracted {len(code_nodes)} code nodes")
    # Print some example code nodes
    if code_nodes:
        print("📋 Sample code nodes:")
        for i, node in enumerate(code_nodes[:3]):
            print(f"  - {node.type} {node.name} in {os.path.basename(node.file_path)}")
        if len(code_nodes) > 3:
            print(f"  - ... and {len(code_nodes) - 3} more nodes")
    
    # Extract requirement nodes
    print("\n📝 Extracting requirement nodes...")
    req_nodes = parse_requirements()
    print(f"✅ Extracted {len(req_nodes)} requirement nodes")
    # Print some example requirement nodes
    if req_nodes:
        print("📋 Sample requirements:")
        for i, node in enumerate(req_nodes[:3]):
            text = node.text[:50] + "..." if len(node.text) > 50 else node.text
            print(f"  - Req {node.req_id}: {text}")
        if len(req_nodes) > 3:
            print(f"  - ... and {len(req_nodes) - 3} more requirements")
    
    # Extract GitHub data
    print("\n🔍 Extracting GitHub data...")
    issues, commits = extract_github_data()
    print(f"✅ Extracted {len(issues)} GitHub issues")
    print(f"✅ Extracted {len(commits)} Git commits")
    
    # Print some example GitHub data
    if issues:
        print("📋 Sample issues:")
        for i, issue in enumerate(issues[:3]):
            print(f"  - Issue #{issue.issue_id}: {issue.title[:50]}...")
        if len(issues) > 3:
            print(f"  - ... and {len(issues) - 3} more issues")
    
    if commits:
        print("📋 Sample commits:")
        for i, commit in enumerate(commits[:3]):
            print(f"  - Commit {commit.commit_hash[:8]}: {commit.message[:50]}...")
        if len(commits) > 3:
            print(f"  - ... and {len(commits) - 3} more commits")
    
    return code_nodes, req_nodes, issues, commits

def apply_node_summaries(code_nodes):
    """Apply pre-generated summaries to code nodes."""
    print("\n=== Applying Summaries ===")
    summaries = load_summaries()
    print(f"✅ Loaded {len(summaries)} summaries")
    
    apply_summaries_to_nodes(code_nodes, summaries)
    
    # Count nodes with summaries
    nodes_with_summaries = sum(1 for node in code_nodes if hasattr(node, 'summary') and node.summary)
    print(f"✅ Applied summaries to {nodes_with_summaries}/{len(code_nodes)} code nodes")
    
    # Print some examples
    if nodes_with_summaries > 0:
        print("📋 Sample nodes with summaries:")
        count = 0
        for node in code_nodes:
            if hasattr(node, 'summary') and node.summary:
                summary = node.summary[:100] + "..." if len(node.summary) > 100 else node.summary
                print(f"  - {node.type} {node.name}: {summary}")
                count += 1
                if count >= 3:
                    break
        if nodes_with_summaries > 3:
            print(f"  - ... and {nodes_with_summaries - 3} more nodes with summaries")

def generate_all_embeddings(code_nodes, req_nodes, issues, commits):
    """Generate embeddings for all nodes."""
    print("\n=== Generating Embeddings with BERT ===")
    all_nodes = code_nodes + req_nodes + issues + commits
    
    print(f"🔤 Generating embeddings for {len(all_nodes)} nodes")
    print("📊 Node distribution:")
    print(f"  - Code nodes: {len(code_nodes)}")
    print(f"  - Requirement nodes: {len(req_nodes)}")
    print(f"  - Issue nodes: {len(issues)}")
    print(f"  - Commit nodes: {len(commits)}")
    
    start_time = time.time()
    embeddings = generate_embeddings(all_nodes)
    end_time = time.time()
    
    print(f"✅ Generated {len(embeddings)} embeddings in {end_time - start_time:.2f} seconds")
    
    return embeddings

def link_all_nodes(req_nodes, code_nodes, issues, commits):
    """Link all nodes based on embedding similarity."""
    print("\n=== Linking Nodes ===")
    
    # Link requirements to code
    print("\n🔗 Linking requirements to code...")
    start_time = time.time()
    req_code_links = link_nodes_by_similarity(req_nodes, code_nodes, threshold=0.6)
    end_time = time.time()
    req_link_count = sum(len(links) for links in req_code_links.values())
    print(f"✅ Created {req_link_count} links between requirements and code in {end_time - start_time:.2f} seconds")
    
    # Print sample requirement-code links
    if req_link_count > 0:
        print("📋 Sample requirement-code links:")
        count = 0
        for req_id, links in req_code_links.items():
            if links:
                req = next((r for r in req_nodes if r.id == req_id), None)
                if req:
                    print(f"  - Req {req.req_id} linked to:")
                    for target_id, similarity in links[:2]:
                        code = next((c for c in code_nodes if c.id == target_id), None)
                        if code:
                            print(f"    → {code.type} {code.name} (similarity: {similarity:.3f})")
                    count += 1
                    if count >= 3:
                        break
    
    # Link issues to code
    print("\n🔗 Linking issues to code...")
    start_time = time.time()
    issue_code_links = link_nodes_by_similarity(issues, code_nodes, threshold=0.6)
    end_time = time.time()
    issue_link_count = sum(len(links) for links in issue_code_links.values())
    print(f"✅ Created {issue_link_count} links between issues and code in {end_time - start_time:.2f} seconds")
    
    # Print sample issue-code links
    if issue_link_count > 0:
        print("📋 Sample issue-code links:")
        count = 0
        for issue_id, links in issue_code_links.items():
            if links:
                issue = next((i for i in issues if i.id == issue_id), None)
                if issue:
                    print(f"  - Issue #{issue.issue_id} linked to:")
                    for target_id, similarity in links[:2]:
                        code = next((c for c in code_nodes if c.id == target_id), None)
                        if code:
                            print(f"    → {code.type} {code.name} (similarity: {similarity:.3f})")
                    count += 1
                    if count >= 3:
                        break
    
    # Link commits to code (based on file changes, handled in graph_database.py)
    print("\n⏳ Commit-code links will be created during graph database building...")
    
    return req_code_links, issue_code_links

def save_links_to_file(req_code_links, issue_code_links):
    """Save the links to a JSON file for later analysis."""
    links_file = os.path.join(OUTPUT_DIR, "links.json")
    
    links_data = {
        "req_code_links": {k: v for k, v in req_code_links.items()},
        "issue_code_links": {k: v for k, v in issue_code_links.items()},
    }
    
    print(f"💾 Saving links to {links_file}...")
    with open(links_file, 'w', encoding='utf-8') as f:
        json.dump(links_data, f, indent=2)
    
    links_size = os.path.getsize(links_file) / 1024  # Size in KB
    print(f"✅ Saved links to {links_file} ({links_size:.2f} KB)")

def build_graph(code_nodes, req_nodes, issues, commits, req_code_links, issue_code_links):
    """Build the Neo4j graph database."""
    print("\n=== Building Graph Database ===")
    
    start_time = time.time()
    build_graph_database(
        code_nodes,
        req_nodes,
        issues,
        commits,
        req_code_links,
        issue_code_links
    )
    end_time = time.time()
    
    print(f"✅ Built graph database in {end_time - start_time:.2f} seconds")
    print("📊 Created nodes:")
    print(f"  - {len(code_nodes)} code nodes")
    print(f"  - {len(req_nodes)} requirement nodes")
    print(f"  - {len(issues)} issue nodes")
    print(f"  - {len(commits)} commit nodes")
    print("📈 Created relationships:")
    print(f"  - IMPLEMENTS: {sum(len(links) for links in req_code_links.values())} links")
    print(f"  - ISSUES_RELATED: {sum(len(links) for links in issue_code_links.values())} links")
    print(f"  - DEPENDS_ON, REFERENCES, COMMITS_TO: Created during graph building")

def generate_traceability_report():
    """Generate a traceability report and export data to JSON."""
    print("\n=== Generating Traceability Report ===")
    
    start_time = time.time()
    query_tool = TraceabilityQuery()
    try:
        print("\n📊 Generating traceability report...")
        query_tool.print_traceability_report()
        
        print("\n💾 Exporting traceability data to JSON...")
        json_file = os.path.join(OUTPUT_DIR, "traceability_data.json")
        export_traceability_json(json_file)
        json_size = os.path.getsize(json_file) / 1024  # Size in KB
        print(f"✅ Exported traceability data to {json_file} ({json_size:.2f} KB)")
    finally:
        query_tool.close()
    
    end_time = time.time()
    print(f"✅ Generated traceability report in {end_time - start_time:.2f} seconds")

def run_pipeline():
    """Run the complete pipeline."""
    start_time = time.time()
    
    # Ensure directories exist
    ensure_directories_exist()
    
    # Extract data
    code_nodes, req_nodes, issues, commits = extract_data()
    
    # Apply summaries
    apply_node_summaries(code_nodes)
    
    # Generate embeddings
    generate_all_embeddings(code_nodes, req_nodes, issues, commits)
    
    # Link nodes
    req_code_links, issue_code_links = link_all_nodes(req_nodes, code_nodes, issues, commits)
    
    # Save links to file
    save_links_to_file(req_code_links, issue_code_links)
    
    # Build graph database
    build_graph(code_nodes, req_nodes, issues, commits, req_code_links, issue_code_links)
    
    # Generate traceability report
    generate_traceability_report()
    
    end_time = time.time()
    total_minutes = (end_time - start_time) / 60
    print(f"\n🏁 Pipeline completed in {end_time - start_time:.2f} seconds ({total_minutes:.2f} minutes)")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ada Traceability Pipeline")
    
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip generating embeddings")
    parser.add_argument("--skip-graph", action="store_true", help="Skip building graph database")
    parser.add_argument("--report-only", action="store_true", help="Only generate traceability report")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.report_only:
        generate_traceability_report()
    else:
        run_pipeline() 