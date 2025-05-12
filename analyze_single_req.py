#!/usr/bin/env python3
"""
Single Requirement Analysis Script.
Analyzes a single requirement and finds the top 10 most relevant Ada code files.
Based on the Requirements2Code approach.
"""
import os
import sys
import time
import json
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from constants import ADA_CODE_DIR, REQUIREMENTS_DIR, OUTPUT_DIR
from code2graph import build_code_graph, load_checkpoint, AdaNode, print_timestamp
from req2nodes import RequirementNode, parse_requirements

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class Req2CodeAnalyzer:
    """Analyzes a single requirement and finds the most relevant Ada code files."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the analyzer with the given embedding model."""
        print_timestamp(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print_timestamp("Model loaded successfully")
        
        # Create output directories
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load or extract code nodes
        self.code_nodes = load_checkpoint("all_nodes")
        if not self.code_nodes:
            print_timestamp("No cached code nodes found. Extracting from Ada files...")
            self.code_nodes = build_code_graph()
            print_timestamp(f"Extracted {len(self.code_nodes)} code nodes")
        else:
            print_timestamp(f"Loaded {len(self.code_nodes)} code nodes from checkpoint")
    
    def analyze_requirement(self, req_text, req_id="REQ-01", top_n=10):
        """
        Analyze a single requirement and find the most relevant Ada code files.
        
        Args:
            req_text: The requirement text to analyze
            req_id: An identifier for the requirement
            top_n: Number of most relevant files to return
            
        Returns:
            List of top_n most relevant files with similarity scores
        """
        print_timestamp(f"Analyzing requirement: '{req_text[:100]}...' (ID: {req_id})")
        
        # Create requirement node
        req_node = RequirementNode(req_id, req_text, "manual_input.txt")
        
        # Generate embedding for the requirement
        start_time = time.time()
        req_embedding = self.model.encode(req_text)
        print_timestamp(f"Generated requirement embedding in {time.time() - start_time:.2f} seconds")
        
        # Calculate file-level relevance
        # First, group nodes by file
        files = {}
        for node in self.code_nodes:
            if node.file_path not in files:
                files[node.file_path] = []
            files[node.file_path].append(node)
        
        print_timestamp(f"Calculating relevance across {len(files)} unique files")
        
        # Calculate relevance for each file
        file_relevance = []
        for file_path, nodes in tqdm(files.items(), desc="Processing files"):
            file_basename = os.path.basename(file_path)
            
            # Extract text from all nodes in this file
            file_text = ""
            for node in nodes:
                if node.type == "PACKAGE":
                    file_text += f"Package {node.name}: {node.body}\n"
                elif hasattr(node, 'summary') and node.summary:
                    file_text += f"{node.type} {node.name}: {node.summary}\n"
                elif node.body:
                    file_text += f"{node.type} {node.name}: {node.body}\n"
                else:
                    file_text += f"{node.type} {node.name}\n"
            
            # Limit text length to avoid memory issues
            if len(file_text) > 10000:
                file_text = file_text[:10000]
            
            # Generate embedding for file text
            try:
                file_embedding = self.model.encode(file_text)
                
                # Calculate vector similarity
                vector_score = cosine_similarity(req_embedding, file_embedding)
                
                # Calculate keyword-based relevance (bag of words)
                req_words = set(req_text.lower().split())
                file_words = set(file_text.lower().split())
                common_words = req_words.intersection(file_words)
                keyword_score = len(common_words) / len(req_words) if req_words else 0
                
                # Follow the Requirements2Code approach: combine scores with weights
                # Vector similarity represents semantic similarity (weight: 0.7)
                # Keyword overlap represents lexical similarity (weight: 0.3)
                combined_score = 0.7 * vector_score + 0.3 * keyword_score
                
                file_relevance.append({
                    "file": file_basename,
                    "file_path": file_path,
                    "score": float(combined_score),  # Convert to standard Python float
                    "vector_score": float(vector_score),  # Convert to standard Python float
                    "keyword_score": float(keyword_score),  # Convert to standard Python float
                    "node_count": len(nodes)
                })
            except Exception as e:
                print_timestamp(f"Error processing {file_path}: {str(e)}")
        
        # Sort by combined score
        file_relevance.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top_n results
        top_results = file_relevance[:top_n]
        print_timestamp(f"Analysis complete. Found {len(top_results)} relevant files")
        
        return top_results
    
    def analyze_and_save(self, req_text, req_id="REQ-01", top_n=10, output_file=None):
        """Analyze a requirement and save results to a file."""
        results = self.analyze_requirement(req_text, req_id, top_n)
        
        # Save results to file
        if output_file is None:
            output_file = os.path.join(OUTPUT_DIR, f"req_{req_id}_analysis.json")
        
        with open(output_file, 'w') as f:
            json.dump({
                "requirement_id": req_id,
                "requirement_text": req_text,
                "top_files": results,
                "analysis_timestamp": datetime.datetime.now().isoformat()
            }, f, indent=2)
        
        print_timestamp(f"Results saved to {output_file}")
        return results

def format_results_table(results):
    """Format results as a table for display."""
    headers = ["Rank", "File", "Score", "Vector Score", "Keyword Score", "Node Count"]
    rows = []
    
    for i, res in enumerate(results):
        rows.append([
            i+1,
            res["file"],
            f"{res['score']:.4f}",
            f"{res['vector_score']:.4f}",
            f"{res['keyword_score']:.4f}",
            res["node_count"]
        ])
    
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]
    
    # Create separator
    separator = "+-" + "-+-".join("-" * width for width in col_widths) + "-+"
    
    # Format the table
    table = []
    table.append(separator)
    table.append("| " + " | ".join(f"{headers[i]:{col_widths[i]}}" for i in range(len(headers))) + " |")
    table.append(separator)
    
    for row in rows:
        table.append("| " + " | ".join(f"{str(row[i]):{col_widths[i]}}" for i in range(len(headers))) + " |")
    
    table.append(separator)
    return "\n".join(table)

def main():
    """Main function to run the analysis for a single requirement."""
    parser = argparse.ArgumentParser(description="Analyze a single requirement against Ada code")
    
    parser.add_argument("--req-id", type=str, default="REQ-01",
                        help="ID for the requirement")
    
    parser.add_argument("--req-text", type=str, 
                        help="Text of the requirement to analyze")
    
    parser.add_argument("--req-file", type=str,
                        help="File containing the requirement text")
    
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top results to show")
    
    parser.add_argument("--output", type=str,
                        help="Output file for the results")
    
    args = parser.parse_args()
    
    # Check that we have either req_text or req_file
    if not args.req_text and not args.req_file:
        print("Error: Either --req-text or --req-file must be specified")
        return 1
    
    # Get requirement text
    req_text = args.req_text
    if args.req_file:
        try:
            with open(args.req_file, 'r') as f:
                req_text = f.read().strip()
        except Exception as e:
            print(f"Error reading requirement file: {str(e)}")
            return 1
    
    # Initialize analyzer and process requirement
    analyzer = Req2CodeAnalyzer()
    results = analyzer.analyze_and_save(req_text, args.req_id, args.top_n, args.output)
    
    # Print results table
    print("\nTop", args.top_n, "files matching the requirement:\n")
    print(format_results_table(results))
    
    # Print requirement summary
    req_words = len(req_text.split())
    print(f"\nRequirement: {args.req_id}")
    print(f"Text: {req_text[:100]}..." if len(req_text) > 100 else f"Text: {req_text}")
    print(f"Word count: {req_words}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_timestamp("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print_timestamp(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 