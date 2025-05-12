#!/usr/bin/env python3
"""
Simplified script to analyze a single requirement against Ada code files
without needing Neo4j. This script directly outputs the top matching files.
"""
import os
import sys
import json
import argparse
from pathlib import Path
import time

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

from code2graph import print_timestamp
from req2nodes import extract_keywords

# Define weights for scoring
SEMANTIC_WEIGHT = 0.7  # Weight for semantic similarity
KEYWORD_WEIGHT = 0.3   # Weight for keyword matching

# Hardcode the path to Ada code directory
ADA_CODE_DIR = os.path.join("datasets", "ada-awa")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_file_content(file_path):
    """Get file content as text if file exists."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return ""

def analyze_requirement(requirement_text, top_n=10):
    """
    Analyze a requirement against Ada code files.
    
    Args:
        requirement_text: The requirement text to analyze
        top_n: Number of top results to return
        
    Returns:
        List of dictionaries with file_path and score
    """
    print_timestamp(f"Analyzing requirement: '{requirement_text[:50]}...'")
    
    # Load BERT model for embeddings
    print_timestamp("Loading embedding model: all-MiniLM-L6-v2")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print_timestamp("Model loaded successfully")
    
    # 1. Generate embedding for the requirement
    start_time = time.time()
    requirement_embedding = model.encode(requirement_text)
    print_timestamp(f"Generated requirement embedding in {time.time() - start_time:.2f} seconds")
    
    # 2. Extract keywords from the requirement
    keywords = extract_keywords(requirement_text)
    print(f"Extracted keywords: {keywords}")
    
    # 3. Gather unique Ada files
    unique_files = set()
    for root, _, files in os.walk(ADA_CODE_DIR):
        for file in files:
            if file.endswith('.ads') or file.endswith('.adb'):
                file_path = os.path.join(root, file)
                unique_files.add(file_path)
    
    print_timestamp(f"Calculating relevance across {len(unique_files)} unique files")
    
    # 4. Calculate relevance for each file
    results = []
    
    for file_path in tqdm(unique_files, desc="Processing files"):
        file_content = get_file_content(file_path)
        if not file_content.strip():
            continue
        
        # Calculate semantic similarity
        file_embedding = model.encode(file_content)
        semantic_score = cosine_similarity(requirement_embedding, file_embedding)
        
        # Calculate keyword matches
        keyword_score = 0
        if keywords:
            found_keywords = sum(1 for kw in keywords if kw.lower() in file_content.lower())
            keyword_score = found_keywords / len(keywords) if found_keywords else 0
        
        # Combined score
        combined_score = (SEMANTIC_WEIGHT * semantic_score) + (KEYWORD_WEIGHT * keyword_score)
        
        results.append({
            "file_path": file_path,
            "score": float(combined_score),  # Convert to float for JSON serialization
            "semantic_score": float(semantic_score),
            "keyword_score": float(keyword_score)
        })
    
    # 5. Sort results by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # 6. Keep top N results
    top_results = results[:top_n]
    
    print_timestamp(f"Analysis complete. Found {len(top_results)} relevant files")
    
    return top_results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze a single requirement against Ada code files")
    
    parser.add_argument("--req", type=str, required=True,
                        help="The requirement text to analyze")
    
    parser.add_argument("--file", type=str,
                        help="File containing the requirement text (overrides --req)")
    
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top results to show")
    
    parser.add_argument("--output", type=str,
                        help="Output file to save the results as JSON")
    
    args = parser.parse_args()
    
    # Get requirement text
    requirement_text = args.req
    if args.file and os.path.exists(args.file):
        with open(args.file, 'r') as f:
            requirement_text = f.read().strip()
    
    # Analyze requirement
    results = analyze_requirement(requirement_text, args.top_n)
    
    # Print results
    print("\nTop files implementing the requirement:")
    for i, result in enumerate(results):
        print(f"{i+1}. {os.path.basename(result['file_path'])} ({result['file_path']})")
        print(f"   - Score: {result['score']:.4f}")
        print(f"   - Semantic: {result['semantic_score']:.4f}, Keyword: {result['keyword_score']:.4f}")
    
    # Save results if output file specified
    if args.output:
        output_data = {
            "requirement": requirement_text,
            "top_files": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 