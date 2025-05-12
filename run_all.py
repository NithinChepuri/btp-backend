#!/usr/bin/env python3
"""
Comprehensive runner script for the Ada Traceability System.
This script handles the entire pipeline in a single command.
"""
import os
import time
import shutil
import sys
from pathlib import Path

from constants import (
    ADA_CODE_DIR, REQUIREMENTS_DIR, OUTPUT_DIR, 
    SUMMARIES_DIR, GRAPH_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
)

def print_step(message, step_number=None, total_steps=None):
    """Print a step message with formatting."""
    if step_number and total_steps:
        print(f"\n🔍 STEP {step_number}/{total_steps}: {message}")
    else:
        print(f"\n🔍 {message}")
    # Add a separator line
    print("=" * 80)

def print_progress(message):
    """Print a progress message."""
    print(f"  📋 {message}")

def print_success(message):
    """Print a success message."""
    print(f"  ✅ {message}")

def print_warning(message):
    """Print a warning message."""
    print(f"  ⚠️ {message}")

def print_error(message):
    """Print an error message."""
    print(f"  ❌ {message}")

def check_neo4j_connection():
    """Check if Neo4j database is accessible."""
    print_progress("Testing connection to Neo4j database...")
    try:
        from neo4j import GraphDatabase, basic_auth
        
        driver = GraphDatabase.driver(
            NEO4J_URI, 
            auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD)
        )
        
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j connection successful' as message")
            message = result.single()["message"]
            print_success(f"{message} at {NEO4J_URI}")
        
        driver.close()
        return True
    
    except Exception as e:
        print_error(f"Error connecting to Neo4j: {e}")
        print_warning(f"Please check that Neo4j is running and that your credentials in constants.py are correct:")
        print_warning(f"URI: {NEO4J_URI}")
        print_warning(f"User: {NEO4J_USER}")
        print_warning(f"Password: set in constants.py")
        return False

def check_json_files():
    """Check if issues.json and commits.json files exist."""
    print_progress("Checking for issues.json and commits.json files...")
    
    issues_file = Path("issues.json")
    commits_file = Path("commits.json")
    
    issues_exist = issues_file.exists()
    commits_exist = commits_file.exists()
    
    if issues_exist:
        issue_count = sum(1 for _ in open(issues_file)) // 10
        print_success(f"Found issues.json with approximately {issue_count} issues")
    else:
        print_warning("issues.json not found. Will attempt to use GitHub API or create sample data.")
    
    if commits_exist:
        commit_count = sum(1 for _ in open(commits_file)) // 20
        print_success(f"Found commits.json with approximately {commit_count} commits")
    else:
        print_warning("commits.json not found. Will attempt to use Git history or create sample data.")
    
    return issues_exist and commits_exist

def check_ada_code():
    """Check if Ada code directory exists and has Ada files."""
    print_progress(f"Checking for Ada code files in {ADA_CODE_DIR}...")
    
    if not os.path.exists(ADA_CODE_DIR):
        print_warning(f"Ada code directory {ADA_CODE_DIR} not found")
        print_progress(f"Creating directories...")
        os.makedirs(ADA_CODE_DIR, exist_ok=True)
        os.makedirs(REQUIREMENTS_DIR, exist_ok=True)
        return False
    
    # Check for Ada files
    ada_files = []
    for root, _, files in os.walk(ADA_CODE_DIR):
        for file in files:
            if file.endswith((".ads", ".adb", ".ada")):
                ada_files.append(os.path.join(root, file))
    
    if ada_files:
        print_success(f"Found {len(ada_files)} Ada files in {ADA_CODE_DIR}")
        for i, file in enumerate(ada_files[:5]):
            print_progress(f"File {i+1}: {os.path.basename(file)}")
        if len(ada_files) > 5:
            print_progress(f"... and {len(ada_files) - 5} more")
        return True
    else:
        print_warning(f"No Ada files found in {ADA_CODE_DIR}")
        return False

def check_requirements():
    """Check if requirements directory exists and has requirement files."""
    print_progress(f"Checking for requirement files in {REQUIREMENTS_DIR}...")
    
    if not os.path.exists(REQUIREMENTS_DIR):
        print_warning(f"Requirements directory {REQUIREMENTS_DIR} not found")
        print_progress(f"Creating directory...")
        os.makedirs(REQUIREMENTS_DIR, exist_ok=True)
        return False
    
    # Check for requirement files
    req_files = []
    for root, _, files in os.walk(REQUIREMENTS_DIR):
        for file in files:
            if file.startswith("req") and file.endswith(".txt"):
                req_files.append(os.path.join(root, file))
    
    if req_files:
        print_success(f"Found {len(req_files)} requirement files in {REQUIREMENTS_DIR}")
        for i, file in enumerate(req_files[:5]):
            print_progress(f"Requirement {i+1}: {os.path.basename(file)}")
        if len(req_files) > 5:
            print_progress(f"... and {len(req_files) - 5} more")
        return True
    else:
        print_warning(f"No requirement files found in {REQUIREMENTS_DIR}")
        return False

def setup_environment():
    """Set up the environment for the pipeline."""
    print_step("Setting Up Environment", 1, 7)
    
    # Create output directories
    print_progress("Creating output directories...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SUMMARIES_DIR, exist_ok=True)
    os.makedirs(GRAPH_DIR, exist_ok=True)
    print_success(f"Created output directories:")
    print_success(f"  - {OUTPUT_DIR}")
    print_success(f"  - {SUMMARIES_DIR}")
    print_success(f"  - {GRAPH_DIR}")
    
    # Add sample requirement if none exist
    if not check_requirements():
        sample_req_path = os.path.join(REQUIREMENTS_DIR, "req01.txt")
        print_progress(f"Creating sample requirement file at {sample_req_path}...")
        with open(sample_req_path, 'w', encoding='utf-8') as f:
            f.write("This is a sample requirement for the Ada Traceability System.")
        print_success(f"Created sample requirement file at {sample_req_path}")
    
    # Add sample Ada file if none exist
    if not check_ada_code():
        sample_ada_path = os.path.join(ADA_CODE_DIR, "sample.ads")
        print_progress(f"Creating sample Ada file at {sample_ada_path}...")
        with open(sample_ada_path, 'w', encoding='utf-8') as f:
            f.write("""
-- Sample Ada specification file
package Sample is
   procedure Hello;
   function Add (A, B : Integer) return Integer;
end Sample;
            """)
        print_success(f"Created sample Ada file at {sample_ada_path}")
    
    # Check JSON files
    check_json_files()
    
    # Check Neo4j connection
    neo4j_ok = check_neo4j_connection()
    if not neo4j_ok:
        print_warning("Neo4j connection failed. Please check your configuration.")
        print_warning("You can update Neo4j credentials in constants.py")
        return False
    
    print_success("Environment setup completed successfully!")
    return True

def run_pipeline():
    """Run the complete pipeline with detailed progress tracking."""
    from run import extract_data, apply_node_summaries, generate_all_embeddings
    from run import link_all_nodes, save_links_to_file, build_graph, generate_traceability_report
    
    # Extract data
    print_step("Extracting Data", 2, 7)
    code_nodes, req_nodes, issues, commits = extract_data()
    
    # Apply summaries to code
    print_step("Applying Summaries to Code", 3, 7)
    apply_node_summaries(code_nodes)
    
    # Generate embeddings
    print_step("Generating Embeddings", 4, 7)
    generate_all_embeddings(code_nodes, req_nodes, issues, commits)
    
    # Link nodes
    print_step("Linking Nodes by Similarity", 5, 7)
    req_code_links, issue_code_links = link_all_nodes(req_nodes, code_nodes, issues, commits)
    
    # Save links to file
    print_step("Saving Links to File", 6, 7)
    save_links_to_file(req_code_links, issue_code_links)
    
    # Build graph
    print_step("Building Neo4j Graph Database", 7, 7)
    build_graph(code_nodes, req_nodes, issues, commits, req_code_links, issue_code_links)
    
    # Generate report
    print_step("Generating Traceability Report", "FINAL", 7)
    generate_traceability_report()

def run_all():
    """Run the entire Ada Traceability pipeline."""
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("🚀 Ada Traceability System - Starting Pipeline")
    print("=" * 80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 80 + "\n")
    
    # Check if the necessary dependencies are installed
    try:
        import sentence_transformers
        print_success("Required dependency 'sentence_transformers' is installed")
    except ImportError:
        print_error("Required dependency 'sentence_transformers' is not installed")
        print_warning("Please install it using: pip install sentence-transformers")
        return
    
    try:
        import tabulate
        print_success("Required dependency 'tabulate' is installed")
    except ImportError:
        print_error("Required dependency 'tabulate' is not installed")
        print_warning("Please install it using: pip install tabulate")
        return
    
    # Set up environment
    if not setup_environment():
        print_error("Environment setup failed. Please fix the issues before running again.")
        return
    
    # Run the pipeline
    try:
        run_pipeline()
    except Exception as e:
        print_error(f"Pipeline failed with error: {str(e)}")
        import traceback
        print_error("Traceback:")
        traceback.print_exc()
        return
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print completion message
    print("\n" + "=" * 80)
    print("✅ Ada Traceability System - Pipeline Completed Successfully!")
    print("=" * 80)
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    print("\n🔍 Next steps:")
    print("  1. View the traceability report in the terminal output above")
    print("  2. Explore the Neo4j database with the Neo4j Browser at http://localhost:7474/")
    print("  3. Use the query tool to explore relationships:")
    print("     python query_tool.py report")
    print("  4. Examine the JSON export in the outputs directory")
    print("=" * 80)

if __name__ == "__main__":
    try:
        run_all()
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user")
        sys.exit(1) 