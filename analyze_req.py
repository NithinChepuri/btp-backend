#!/usr/bin/env python3
"""
Simplified script to analyze a single requirement and find matching Ada files.
This script directly analyzes the sample requirement without command-line arguments.
"""
import os
import json
import time
import datetime
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from constants import ADA_CODE_DIR, OUTPUT_DIR
from code2graph import AdaNode, load_checkpoint, build_code_graph

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Print with timestamp
def print_timestamp(message):
    """Print a message with a timestamp."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Sample requirement
REQUIREMENT_TEXT = """
The system must provide a secure authentication mechanism for users to log in with their credentials. 
The authentication process should validate user credentials against stored data and enforce password 
policies including minimum length and complexity requirements. Failed login attempts should be recorded 
and after a configurable number of failed attempts, the account should be temporarily locked.
"""

REQUIREMENT_ID = "AUTH-01"

def main():
    """Main analysis function."""
    print_timestamp("Starting analysis of sample requirement")
    print(f"\nRequirement {REQUIREMENT_ID}: {REQUIREMENT_TEXT[:100]}...\n")
    
    # Load embedding model
    print_timestamp("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print_timestamp("Model loaded successfully")
    
    # Load or extract code nodes
    code_nodes = load_checkpoint("all_nodes")
    if not code_nodes:
        print_timestamp("No cached code nodes found. Extracting from Ada files...")
        code_nodes = build_code_graph()
        print_timestamp(f"Extracted {len(code_nodes)} code nodes")
    else:
        print_timestamp(f"Loaded {len(code_nodes)} code nodes from checkpoint")
    
    # Generate embedding for the requirement
    start_time = time.time()
    req_embedding = model.encode(REQUIREMENT_TEXT)
    print_timestamp(f"Generated requirement embedding in {time.time() - start_time:.2f} seconds")
    
    # Group nodes by file
    files = {}
    for node in code_nodes:
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
            file_embedding = model.encode(file_text)
            
            # Calculate vector similarity
            vector_score = cosine_similarity(req_embedding, file_embedding)
            
            # Calculate keyword-based relevance (bag of words)
            req_words = set(REQUIREMENT_TEXT.lower().split())
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
                "score": combined_score,
                "vector_score": vector_score,
                "keyword_score": keyword_score,
                "node_count": len(nodes)
            })
        except Exception as e:
            print_timestamp(f"Error processing {file_path}: {str(e)}")
    
    # Sort by combined score
    file_relevance.sort(key=lambda x: x["score"], reverse=True)
    
    # Get top 10 results
    top_results = file_relevance[:10]
    print_timestamp(f"Analysis complete. Found {len(top_results)} relevant files")
    
    # Save results to file
    output_file = os.path.join(OUTPUT_DIR, f"req_{REQUIREMENT_ID}_analysis.json")
    with open(output_file, 'w') as f:
        json.dump({
            "requirement_id": REQUIREMENT_ID,
            "requirement_text": REQUIREMENT_TEXT,
            "top_files": top_results,
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }, f, indent=2)
    
    print_timestamp(f"Results saved to {output_file}")
    
    # Print results
    print("\nTOP 10 FILES MATCHING THE AUTHENTICATION REQUIREMENT:\n")
    print("-" * 80)
    print(f"{'Rank':<5} {'File':<30} {'Score':<10} {'Vector Score':<15} {'Keyword Score':<15}")
    print("-" * 80)
    
    for i, res in enumerate(top_results):
        print(f"{i+1:<5} {res['file']:<30} {res['score']:.4f}    {res['vector_score']:.4f}         {res['keyword_score']:.4f}")
    
    print("-" * 80)
    
    # Print the file contents for the top match if it exists
    if top_results:
        top_file = top_results[0]['file_path']
        print(f"\nExcerpt from top matching file: {top_file}\n")
        try:
            with open(top_file, 'r', encoding='utf-8') as f:
                content = f.read(500)  # Read first 500 chars
            print(content + "..." if len(content) >= 500 else content)
        except Exception as e:
            print(f"Could not read file: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_timestamp("\nOperation cancelled by user")
    except Exception as e:
        print_timestamp(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc() 