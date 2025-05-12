#!/usr/bin/env python3
"""
Script to restore Neo4j database from a previously saved analysis.
"""
import os
import json
import sys
import glob
from pathlib import Path

from graph_database import Neo4jConnector, print_timestamp

# Configuration
EXPORTS_DIR = os.path.join("outputs", "saved_analysis", "neo4j_exports")

def list_available_exports():
    """List all available exports in the exports directory."""
    print("Available exports:")
    
    # Look for metadata files
    metadata_files = glob.glob(os.path.join(EXPORTS_DIR, "*_metadata.json"))
    
    if not metadata_files:
        print("No exports found. Run load_and_persist_neo4j.py first.")
        return []
    
    exports = []
    for i, metadata_file in enumerate(sorted(metadata_files, reverse=True)):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            req_id = metadata.get("req_id", "unknown")
            timestamp = metadata.get("timestamp", "unknown")
            num_files = metadata.get("num_files", 0)
            
            print(f"{i+1}. {req_id} - {timestamp} ({num_files} files)")
            exports.append(metadata)
        except Exception as e:
            print(f"Error reading metadata file {metadata_file}: {e}")
    
    return exports

def restore_neo4j_data(metadata):
    """
    Restore Neo4j database from a previously saved analysis.
    
    Args:
        metadata: The metadata dictionary for the export to restore
    """
    req_id = metadata.get("req_id", "unknown")
    timestamp = metadata.get("timestamp", "unknown")
    analysis_file = metadata.get("analysis_file")
    
    print_timestamp(f"Restoring Neo4j database for requirement {req_id} from {timestamp}...")
    
    # Check if the analysis file exists
    if not os.path.exists(analysis_file):
        print_timestamp(f"Analysis file {analysis_file} not found.")
        
        # Try to find the exported copy
        export_basename = os.path.basename(metadata_file).replace("_metadata.json", "_analysis.json")
        alternative_file = os.path.join(EXPORTS_DIR, export_basename)
        
        if os.path.exists(alternative_file):
            analysis_file = alternative_file
            print_timestamp(f"Using alternative analysis file: {analysis_file}")
        else:
            print_timestamp("Cannot find analysis file. Restore aborted.")
            return False
    
    # Reconnect to Neo4j
    connector = Neo4jConnector()
    connector.connect()
    
    # Clear existing data
    print_timestamp("Clearing existing database...")
    connector.clear_database()
    
    # Restore from the analysis data
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    # Execute load_requirement_and_related_files, but without exporting
    # Could import and call the function from load_and_persist_neo4j.py instead
    import load_and_persist_neo4j
    load_and_persist_neo4j.load_requirement_and_export(
        req_id=req_id,
        analysis_file=analysis_file
    )
    
    print_timestamp(f"Restored Neo4j database for requirement {req_id}")
    return True

def main():
    """Main function to restore Neo4j data from a previously saved analysis."""
    # List available exports
    exports = list_available_exports()
    
    if not exports:
        return 1
    
    # Ask user to select an export to restore
    selection = None
    while selection is None:
        try:
            idx = int(input("\nEnter number of export to restore (or 0 to exit): "))
            if idx == 0:
                return 0
            if 1 <= idx <= len(exports):
                selection = idx - 1
            else:
                print(f"Invalid selection. Enter a number between 1 and {len(exports)}.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Restore the selected export
    metadata = exports[selection]
    restore_neo4j_data(metadata)
    
    print("\nTo access Neo4j Browser:")
    print("1. Open a web browser and go to http://localhost:7474/")
    print("2. Login with your Neo4j credentials")
    print("3. Explore the data with Cypher queries")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 