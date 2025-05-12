#!/usr/bin/env python3
"""
Optimized Ada Traceability Analysis Runner.
This script provides easy command-line options to run the traceability analysis
with performance optimizations and checkpointing.
"""
import os
import sys
import time
import argparse
import datetime
from pathlib import Path

from constants import OUTPUT_DIR
from run_with_checkpoints import run_analysis, clean_checkpoints, print_timestamp

def parse_arguments():
    """Parse command-line arguments with help text."""
    parser = argparse.ArgumentParser(
        description="Optimized Ada Traceability Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--clean", action="store_true", 
                        help="Clean all checkpoints and start fresh")
    
    parser.add_argument("--skip-references", action="store_true",
                        help="Skip reference extraction step (fastest but less accurate)")
    
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding generation step")
    
    parser.add_argument("--max-nodes", type=int, default=1000,
                        help="Maximum number of nodes to analyze for references")
    
    parser.add_argument("--min-name-length", type=int, default=3,
                        help="Minimum length of node names for reference analysis")
    
    parser.add_argument("--only-report", action="store_true",
                        help="Only generate the report using existing data")
    
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for database operations")
    
    parser.add_argument("--mode", choices=["full", "fast", "ultrafast", "report"], default="full",
                        help="Preset performance modes (full=complete analysis, fast=skip some references, ultrafast=minimal processing, report=only generate report)")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    print_timestamp("\n🚀 Ada Traceability Analysis")
    print_timestamp("This script runs the Ada code traceability analysis with optimized performance.")
    
    args = parse_arguments()
    
    # Apply preset modes
    if args.mode == "fast":
        args.max_nodes = 500
        args.min_name_length = 4
        print_timestamp("ℹ️ Running in FAST mode: Limited reference analysis")
    elif args.mode == "ultrafast":
        args.skip_references = True
        args.max_nodes = 100
        print_timestamp("ℹ️ Running in ULTRAFAST mode: Skipping reference analysis")
    elif args.mode == "report":
        args.only_report = True
        print_timestamp("ℹ️ Running in REPORT mode: Only generating report")
    else:
        print_timestamp("ℹ️ Running in FULL mode: Complete analysis")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)
    
    # Run the analysis
    start_time = time.time()
    
    try:
        run_analysis(args)
        
        end_time = time.time()
        duration = end_time - start_time
        print_timestamp(f"\n⏱️ Total execution time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Print usage tips
        print("\n📋 Usage Tips:")
        print("  - For incremental updates, run this script again (checkpoints will be used)")
        print("  - To force a clean run, use: python run_optimized.py --clean")
        print("  - For faster processing with some accuracy trade-offs, use: python run_optimized.py --mode fast")
        print("  - To only generate a report, use: python run_optimized.py --mode report")
        print("  - To see all available options, use: python run_optimized.py --help")
        
    except KeyboardInterrupt:
        print_timestamp("\n⚠️ Analysis interrupted by user")
        print_timestamp("💾 Any completed checkpoints have been saved")
        sys.exit(1)
    except Exception as e:
        print_timestamp(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 