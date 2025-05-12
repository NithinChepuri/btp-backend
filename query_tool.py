#!/usr/bin/env python3
"""
Command-line tool for querying the Ada Traceability database.
This script provides an interface for exploring traceability relationships.
"""

import argparse
import json
from tabulate import tabulate

from querying import TraceabilityQuery, export_traceability_json

def print_table(results, headers=None):
    """Print results as a table."""
    if not results:
        print("No results found.")
        return
    
    if not headers:
        headers = results[0].keys()
    
    table_data = []
    for row in results:
        table_row = []
        for key in headers:
            value = row.get(key, "")
            # Truncate long text
            if isinstance(value, str) and len(value) > 100:
                value = value[:97] + "..."
            table_row.append(value)
        table_data.append(table_row)
    
    print(tabulate(table_data, headers=headers))

def find_requirements_without_code(query_tool):
    """Find requirements that don't have code implementations."""
    results = query_tool.find_requirements_without_code()
    print(f"\nFound {len(results)} requirements without code implementations:\n")
    print_table(results, ["req_id", "text"])

def find_code_without_requirements(query_tool):
    """Find code that isn't linked to any requirements."""
    results = query_tool.find_code_without_requirements()
    print(f"\nFound {len(results)} code entities without requirements:\n")
    print_table(results, ["code_name", "code_type", "file_path"])

def trace_requirement(query_tool, req_id):
    """Trace a requirement to its code implementations."""
    results = query_tool.trace_requirement_to_code(req_id)
    print(f"\nCode implementations for requirement {req_id}:\n")
    print_table(results, ["code_name", "code_type", "file_path", "line_number", "similarity"])

def trace_code(query_tool, code_name, code_type=None):
    """Trace code to its requirements."""
    results = query_tool.trace_code_to_requirements(code_name, code_type)
    print(f"\nRequirements for code {code_name}{'(' + code_type + ')' if code_type else ''}:\n")
    print_table(results, ["req_id", "text", "similarity"])

def trace_issue(query_tool, issue_id):
    """Trace an issue to affected requirements."""
    results = query_tool.trace_issue_to_requirements(int(issue_id))
    print(f"\nRequirements affected by issue #{issue_id}:\n")
    print_table(results, ["req_id", "text"])

def show_stats(query_tool):
    """Show traceability statistics."""
    stats = query_tool.get_traceability_stats()
    print("\n===== Traceability Statistics =====\n")
    print(f"Requirements: {stats['requirements_with_code']}/{stats['total_requirements']} " +
          f"({stats['requirements_coverage_percentage']:.2f}%)")
    print(f"Code: {stats['code_with_requirements']}/{stats['total_code_nodes']} " +
          f"({stats['code_coverage_percentage']:.2f}%)")
    print(f"Issues: {stats['total_issues']}")
    print(f"Commits: {stats['total_commits']}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Ada Traceability Query Tool")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate a comprehensive traceability report")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export traceability data to JSON")
    export_parser.add_argument("--output", "-o", help="Output file path")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show traceability statistics")
    
    # Find uncovered requirements command
    uncovered_req_parser = subparsers.add_parser("uncovered-reqs", help="Find requirements without code")
    
    # Find uncovered code command
    uncovered_code_parser = subparsers.add_parser("uncovered-code", help="Find code without requirements")
    
    # Trace requirement command
    trace_req_parser = subparsers.add_parser("trace-req", help="Trace a requirement to code")
    trace_req_parser.add_argument("req_id", help="Requirement ID")
    
    # Trace code command
    trace_code_parser = subparsers.add_parser("trace-code", help="Trace code to requirements")
    trace_code_parser.add_argument("code_name", help="Code name")
    trace_code_parser.add_argument("--type", "-t", help="Code type (PACKAGE, PROCEDURE, FUNCTION, TYPE)")
    
    # Trace issue command
    trace_issue_parser = subparsers.add_parser("trace-issue", help="Trace an issue to requirements")
    trace_issue_parser.add_argument("issue_id", help="Issue ID")
    
    args = parser.parse_args()
    
    # Create query tool
    query_tool = TraceabilityQuery()
    
    try:
        if args.command == "report" or not args.command:
            query_tool.print_traceability_report()
        
        elif args.command == "export":
            export_traceability_json(args.output)
        
        elif args.command == "stats":
            show_stats(query_tool)
        
        elif args.command == "uncovered-reqs":
            find_requirements_without_code(query_tool)
        
        elif args.command == "uncovered-code":
            find_code_without_requirements(query_tool)
        
        elif args.command == "trace-req":
            trace_requirement(query_tool, args.req_id)
        
        elif args.command == "trace-code":
            trace_code(query_tool, args.code_name, args.type)
        
        elif args.command == "trace-issue":
            trace_issue(query_tool, args.issue_id)
    
    finally:
        query_tool.close()

if __name__ == "__main__":
    main() 