#!/usr/bin/env python3
"""
Wrapper script for analyze_single_req.py to make it easier to run.
"""
import os
import sys
import subprocess

def run_analysis():
    """Run analyze_single_req.py with proper arguments."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze_req_wrapper.py <requirement_text>                    # Analyze text directly")
        print("  python analyze_req_wrapper.py --file <file> --id <id> --top <num>   # Analyze from file")
        print("")
        print("Example:")
        print("  python analyze_req_wrapper.py \"The system must provide user authentication\"")
        print("  python analyze_req_wrapper.py --file sample_requirement.txt --id AUTH-01 --top 10")
        return 1
    
    if sys.argv[1] == "--file":
        # Extract parameters
        req_file = None
        req_id = "REQ-01"  # Default ID
        top_n = 10         # Default top N results
        
        for i in range(2, len(sys.argv)):
            if i+1 < len(sys.argv):
                if sys.argv[i] == "--file":
                    req_file = sys.argv[i+1]
                elif sys.argv[i] == "--id":
                    req_id = sys.argv[i+1]
                elif sys.argv[i] == "--top":
                    top_n = int(sys.argv[i+1])
        
        if not req_file:
            print("Error: Missing requirement file path. Use --file <path>")
            return 1
        
        if not os.path.exists(req_file):
            print(f"Error: Requirement file {req_file} not found.")
            return 1
        
        # Build command
        cmd = [
            "python", "analyze_single_req.py",
            "--req-file", req_file,
            "--req-id", req_id,
            "--top-n", str(top_n)
        ]
    else:
        # Direct requirement text
        req_text = sys.argv[1]
        
        # Create a temporary file
        tmp_file = "temp_requirement.txt"
        with open(tmp_file, 'w') as f:
            f.write(req_text)
        
        # Build command
        cmd = [
            "python", "analyze_single_req.py",
            "--req-file", tmp_file,
            "--req-id", "DIRECT-REQ",
            "--top-n", "10"
        ]
    
    # Print the command
    print(f"Running: {' '.join(cmd)}")
    
    # Run the command
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"Error running analyze_single_req.py: {e}")
        return 1
    
    # Clean up temporary file if created
    if "tmp_file" in locals() and os.path.exists(tmp_file):
        os.remove(tmp_file)
    
    return 0

if __name__ == "__main__":
    sys.exit(run_analysis()) 