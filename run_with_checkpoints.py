#!/usr/bin/env python3
"""
Traceability Analysis Runner with Checkpoint Management.
This script manages the traceability analysis process using checkpoints for 
efficient incremental processing.
"""
import os
import sys
import time
import json
import argparse
import datetime
from pathlib import Path

from constants import OUTPUT_DIR, ADA_CODE_DIR, REQUIREMENTS_DIR, GRAPH_DIR
from code2graph import build_code_graph, load_checkpoint, save_checkpoint
from req2nodes import parse_requirements
from embeddings import generate_embeddings, link_nodes_by_similarity, apply_summaries_to_nodes, load_summaries
from graph_database import build_graph_database
from github_integration import get_github_issues, get_git_commits
from querying import TraceabilityQuery


def print_timestamp(message):
    """Print a message with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")


def print_step(message, step_number=None, total_steps=None):
    """Print a step message with formatting."""
    if step_number and total_steps:
        print(f"\n🔍 STEP {step_number}/{total_steps}: {message}")
    else:
        print(f"\n🔍 {message}")
    # Add a separator line
    print("=" * 80)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Ada Traceability Analysis with Checkpoints")
    
    parser.add_argument("--clean", action="store_true", 
                        help="Clean all checkpoints and start fresh")
    
    parser.add_argument("--skip-references", action="store_true",
                        help="Skip reference extraction step (fastest but less accurate)")
    
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation step")
    
    parser.add_argument("--max-nodes", type=int, default=1000,
                        help="Maximum number of nodes to analyze for references (default: 1000)")
    
    parser.add_argument("--min-name-length", type=int, default=3,
                        help="Minimum length of node names for reference analysis (default: 3)")
    
    parser.add_argument("--only-report", action="store_true",
                        help="Only generate the report using existing data")
    
    return parser.parse_args()


def clean_checkpoints():
    """Clean all checkpoint files."""
    print_timestamp("🧹 Cleaning all checkpoints...")
    
    checkpoints_dir = os.path.join(OUTPUT_DIR, "checkpoints")
    if os.path.exists(checkpoints_dir):
        count = 0
        for file in os.listdir(checkpoints_dir):
            file_path = os.path.join(checkpoints_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
        
        print_timestamp(f"✅ Removed {count} checkpoint files")
    else:
        print_timestamp("⚠️ No checkpoints directory found")


def run_analysis(args):
    """Run the traceability analysis with checkpoint management."""
    start_time = time.time()
    print_timestamp("🚀 STARTED: Ada Traceability Analysis with Checkpoints")
    
    if args.clean:
        clean_checkpoints()
    
    # Step 1: Extract code entities
    print_step("Extracting Ada Code Entities", 1, 5)
    
    # Check for existing code nodes checkpoint
    code_nodes = load_checkpoint("nodes_with_references") or load_checkpoint("all_nodes")
    if code_nodes:
        print_timestamp(f"✅ Loaded {len(code_nodes)} code nodes from checkpoint")
    else:
        print_timestamp("🔄 Extracting code nodes from Ada files...")
        code_nodes = build_code_graph()
        print_timestamp(f"✅ Extracted {len(code_nodes)} code nodes")
    
    # Step 2: Parse requirements
    print_step("Parsing Requirements", 2, 5)
    
    req_nodes = load_checkpoint("requirement_nodes")
    if req_nodes:
        print_timestamp(f"✅ Loaded {len(req_nodes)} requirement nodes from checkpoint")
    else:
        print_timestamp("🔄 Parsing requirements...")
        req_nodes = parse_requirements()
        save_checkpoint(req_nodes, "requirement_nodes")
        print_timestamp(f"✅ Parsed {len(req_nodes)} requirements")
    
    # Step 3: Get GitHub issues and Git commits
    print_step("Retrieving GitHub Issues and Git Commits", 3, 5)
    
    issues = load_checkpoint("github_issues")
    if issues:
        print_timestamp(f"✅ Loaded {len(issues)} GitHub issues from checkpoint")
    else:
        print_timestamp("🔄 Retrieving GitHub issues...")
        issues = get_github_issues()
        save_checkpoint(issues, "github_issues")
        print_timestamp(f"✅ Retrieved {len(issues)} GitHub issues")
    
    commits = load_checkpoint("git_commits")
    if commits:
        print_timestamp(f"✅ Loaded {len(commits)} Git commits from checkpoint")
    else:
        print_timestamp("🔄 Retrieving Git commits...")
        commits = get_git_commits()
        save_checkpoint(commits, "git_commits")
        print_timestamp(f"✅ Retrieved {len(commits)} Git commits")
    
    # Step 4: Generate embeddings and link nodes
    print_step("Generating Embeddings and Linking Nodes", 4, 5)
    
    if args.skip_embeddings:
        print_timestamp("⚠️ Skipping embedding generation as requested")
        
        # Try to load existing links
        req_code_links = load_checkpoint("req_code_links")
        issue_code_links = load_checkpoint("issue_code_links")
        
        if req_code_links and issue_code_links:
            print_timestamp("✅ Loaded existing links from checkpoints")
        else:
            print_timestamp("⚠️ No existing links found. Cannot skip embedding generation.")
            args.skip_embeddings = False
    
    if not args.skip_embeddings:
        # Apply summaries to code nodes
        summaries = load_summaries()
        apply_summaries_to_nodes(code_nodes, summaries)
        
        # Generate embeddings for all nodes
        code_embeddings = load_checkpoint("code_embeddings")
        req_embeddings = load_checkpoint("req_embeddings")
        issue_embeddings = load_checkpoint("issue_embeddings")
        commit_embeddings = load_checkpoint("commit_embeddings")
        
        if not all([code_embeddings, req_embeddings, issue_embeddings, commit_embeddings]):
            print_timestamp("🔄 Generating embeddings for all nodes...")
            
            # Generate for code nodes
            if not code_embeddings:
                print_timestamp("🔄 Generating embeddings for code nodes...")
                code_embeddings = generate_embeddings(code_nodes)
                save_checkpoint(code_embeddings, "code_embeddings")
            
            # Generate for requirement nodes
            if not req_embeddings:
                print_timestamp("🔄 Generating embeddings for requirement nodes...")
                req_embeddings = generate_embeddings(req_nodes)
                save_checkpoint(req_embeddings, "req_embeddings")
            
            # Generate for issues
            if not issue_embeddings:
                print_timestamp("🔄 Generating embeddings for GitHub issues...")
                issue_embeddings = generate_embeddings(issues)
                save_checkpoint(issue_embeddings, "issue_embeddings")
            
            # Generate for commits
            if not commit_embeddings:
                print_timestamp("🔄 Generating embeddings for Git commits...")
                commit_embeddings = generate_embeddings(commits)
                save_checkpoint(commit_embeddings, "commit_embeddings")
        
        # Link nodes by similarity
        print_timestamp("🔄 Linking nodes based on semantic similarity...")
        
        # Link requirements to code
        req_code_links = load_checkpoint("req_code_links")
        if not req_code_links:
            req_code_links = link_nodes_by_similarity(req_nodes, code_nodes, threshold=0.6)
            save_checkpoint(req_code_links, "req_code_links")
        
        # Link issues to code
        issue_code_links = load_checkpoint("issue_code_links")
        if not issue_code_links:
            issue_code_links = link_nodes_by_similarity(issues, code_nodes, threshold=0.6)
            save_checkpoint(issue_code_links, "issue_code_links")
    else:
        # Use existing links
        req_code_links = load_checkpoint("req_code_links") or {}
        issue_code_links = load_checkpoint("issue_code_links") or {}
    
    # Step 5: Build graph database and generate report
    print_step("Building Graph Database and Generating Report", 5, 5)
    
    # Skip graph building if only-report is specified
    if not args.only_report:
        print_timestamp("🔄 Building Neo4j graph database...")
        build_graph_database(code_nodes, req_nodes, issues, commits, req_code_links, issue_code_links)
    
    # Generate traceability report
    print_timestamp("🔄 Generating traceability report...")
    query_tool = TraceabilityQuery()
    try:
        query_tool.print_traceability_report()
    finally:
        query_tool.close()
    
    # Print completion message
    total_time = time.time() - start_time
    print_timestamp(f"✅ COMPLETED: Traceability analysis completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Print usage instructions
    print("\n🔍 Next steps:")
    print("  1. Explore the Neo4j database with the Neo4j Browser at http://localhost:7474/")
    print("  2. Use the query tool to explore relationships:")
    print("     python query_tool.py report")
    print("  3. To run incremental updates, use this script again with the same options")
    print("  4. To start fresh, use --clean to remove all checkpoints")


if __name__ == "__main__":
    args = parse_arguments()
    
    # Create necessary directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    
    try:
        run_analysis(args)
    except KeyboardInterrupt:
        print_timestamp("\n⚠️ Analysis interrupted by user")
        print_timestamp("💾 Any completed checkpoints have been saved")
        sys.exit(1)
    except Exception as e:
        print_timestamp(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 