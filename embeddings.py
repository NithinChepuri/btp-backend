"""
Module for generating and using embeddings to link nodes in the graph.
Uses sentence-transformers (BERT-based models) locally instead of OpenAI API.
"""
import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Union
from tqdm import tqdm
import time

# Import sentence-transformers for local embedding generation
from sentence_transformers import SentenceTransformer

from constants import MAX_TOKENS, SUMMARIES_DIR
from code2graph import AdaNode
from req2nodes import RequirementNode
from github_integration import GitHubIssue, GitCommit

Node = Union[AdaNode, RequirementNode, GitHubIssue, GitCommit]

# Initialize the BERT model globally for reuse
print("📋 Loading BERT model for embeddings...")
try:
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ BERT model loaded successfully")
except Exception as e:
    print(f"❌ Error loading BERT model: {e}")
    print("   Will attempt to load when needed...")
    EMBEDDING_MODEL = None

def get_bert_embedding(text: str) -> List[float]:
    """Get an embedding from a local BERT model."""
    global EMBEDDING_MODEL
    
    try:
        # Initialize model if not already done
        if EMBEDDING_MODEL is None:
            print("🔄 Initializing BERT model...")
            EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ BERT model initialized")
        
        # Truncate text if necessary to avoid memory issues
        if len(text) > MAX_TOKENS:
            print(f"⚠️ Truncating text from {len(text)} chars to {MAX_TOKENS} chars")
            text = text[:MAX_TOKENS]
        
        # Get embedding from BERT
        start_time = time.time()
        embedding = EMBEDDING_MODEL.encode(text, convert_to_numpy=True)
        embedding_time = time.time() - start_time
        print(f"📊 Generated embedding in {embedding_time:.2f} seconds (dim: {len(embedding)})")
        
        # Convert to list for serialization
        return embedding.tolist()
    
    except Exception as e:
        print(f"❌ Error getting BERT embedding: {e}")
        # Return a zero vector as fallback
        embedding_dim = 384  # Default for all-MiniLM-L6-v2
        print(f"⚠️ Returning zero vector with dimension {embedding_dim}")
        return [0.0] * embedding_dim

def get_node_text(node: Node) -> str:
    """Get the text representation of a node for generating embeddings."""
    if isinstance(node, AdaNode):
        if node.summary:
            return f"{node.type} {node.name}: {node.summary}"
        else:
            return f"{node.type} {node.name}: {node.body}"
    
    elif isinstance(node, RequirementNode):
        return f"Requirement {node.req_id}: {node.text}"
    
    elif isinstance(node, GitHubIssue):
        return f"Issue #{node.issue_id}: {node.title} - {node.body}"
    
    elif isinstance(node, GitCommit):
        files_text = ", ".join(node.changed_files[:5])
        if len(node.changed_files) > 5:
            files_text += f" and {len(node.changed_files) - 5} more"
        
        return f"Commit {node.commit_hash[:8]}: {node.message} - Changed files: {files_text}"
    
    else:
        return str(node)

def generate_embeddings(nodes: List[Node]) -> Dict[str, List[float]]:
    """Generate embeddings for a list of nodes."""
    embeddings = {}
    
    print(f"\n🔤 Generating embeddings for {len(nodes)} nodes")
    print("📝 This may take some time depending on the size of your dataset")
    
    for i, node in enumerate(tqdm(nodes, desc="Generating embeddings")):
        node_id = node.id
        node_type = node.__class__.__name__
        
        # Get text representation
        node_text = get_node_text(node)
        
        # Print verbose info every 10 nodes or for the first node
        if i == 0 or i % 10 == 0:
            print(f"\n🔤 Processing node {i+1}/{len(nodes)}: {node_id} ({node_type})")
            if len(node_text) > 100:
                print(f"   Text preview: {node_text[:100]}...")
            else:
                print(f"   Text: {node_text}")
        
        # Generate embedding
        embedding = get_bert_embedding(node_text)
        
        # Store embedding
        embeddings[node.id] = embedding
        
        # Also store in the node object
        if hasattr(node, 'embedding'):
            node.embedding = embedding
        
        # Print progress every 10 nodes or for the last node
        if i == 0 or i % 10 == 0 or i == len(nodes) - 1:
            print(f"✅ Completed {i+1}/{len(nodes)} embeddings")
    
    print(f"\n✅ Generated {len(embeddings)} embeddings in total")
    return embeddings

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_array = np.array(a)
    b_array = np.array(b)
    
    return np.dot(a_array, b_array) / (np.linalg.norm(a_array) * np.linalg.norm(b_array))

def link_nodes_by_similarity(
    source_nodes: List[Node], 
    target_nodes: List[Node], 
    threshold: float = 0.7,
    max_links: int = 5
) -> Dict[str, List[Tuple[str, float]]]:
    """Link nodes based on embedding similarity."""
    links = {}
    
    print(f"\n🔗 Linking {len(source_nodes)} source nodes to {len(target_nodes)} target nodes")
    print(f"📊 Using similarity threshold: {threshold}, max links per node: {max_links}")
    
    start_time = time.time()
    total_comparisons = len(source_nodes) * len(target_nodes)
    print(f"🧮 Will perform up to {total_comparisons} similarity comparisons")
    
    for source_index, source_node in enumerate(tqdm(source_nodes, desc="Linking nodes")):
        source_id = source_node.id
        
        # Skip if no embedding
        if not hasattr(source_node, 'embedding') or source_node.embedding is None:
            print(f"⚠️ Skipping node {source_id} - no embedding available")
            continue
        
        # Progress update
        if source_index == 0 or source_index % 10 == 0 or source_index == len(source_nodes) - 1:
            print(f"\n🔄 Processing source node {source_index+1}/{len(source_nodes)}: {source_id}")
        
        source_embedding = source_node.embedding
        similarities = []
        
        for target_node in target_nodes:
            # Skip if same node or no embedding
            if source_node.id == target_node.id:
                continue
            
            if not hasattr(target_node, 'embedding') or target_node.embedding is None:
                continue
            
            # Calculate similarity
            similarity = cosine_similarity(source_embedding, target_node.embedding)
            
            # Add to list if above threshold
            if similarity >= threshold:
                similarities.append((target_node.id, similarity))
        
        # Progress update with link count
        if source_index == 0 or source_index % 10 == 0 or source_index == len(source_nodes) - 1:
            print(f"   Found {len(similarities)} potential links above threshold {threshold}")
        
        # Sort by similarity (descending) and take top max_links
        similarities.sort(key=lambda x: x[1], reverse=True)
        if max_links > 0:
            similarities = similarities[:max_links]
        
        links[source_node.id] = similarities
        
        # Also store in the node object
        if hasattr(source_node, 'related_nodes'):
            source_node.related_nodes = [item[0] for item in similarities]
    
    end_time = time.time()
    total_links = sum(len(links_list) for links_list in links.values())
    
    print(f"\n✅ Created {total_links} links between nodes")
    print(f"⏱️ Linking completed in {end_time - start_time:.2f} seconds")
    
    return links

def load_summaries() -> Dict[str, str]:
    """Load pre-generated summaries."""
    print("\n📄 Loading pre-generated summaries...")
    summaries = {}
    
    summary_file = os.path.join(SUMMARIES_DIR, "ada_summaries.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summaries = json.load(f)
            print(f"✅ Loaded {len(summaries)} summaries from {summary_file}")
        except Exception as e:
            print(f"❌ Error loading summaries: {e}")
    else:
        print(f"⚠️ Summaries file not found: {summary_file}")
    
    return summaries

def apply_summaries_to_nodes(nodes: List[AdaNode], summaries: Dict[str, str]) -> None:
    """Apply pre-generated summaries to code nodes."""
    print(f"\n📝 Applying summaries to {len(nodes)} code nodes...")
    applied_count = 0
    
    for node in nodes:
        if node.name in summaries:
            node.summary = summaries[node.name]
            applied_count += 1
    
    print(f"✅ Applied {applied_count} summaries to code nodes")

if __name__ == "__main__":
    from code2graph import build_code_graph
    from req2nodes import parse_requirements
    
    # Load code nodes and requirements
    print("\n🔍 Loading code nodes and requirements...")
    code_nodes = build_code_graph()
    req_nodes = parse_requirements()
    print(f"✅ Loaded {len(code_nodes)} code nodes and {len(req_nodes)} requirements")
    
    # Apply summaries to code nodes
    summaries = load_summaries()
    apply_summaries_to_nodes(code_nodes, summaries)
    
    # Generate embeddings
    print("\n🔤 Generating embeddings...")
    all_nodes = code_nodes + req_nodes
    embeddings = generate_embeddings(all_nodes)
    
    # Link requirements to code
    print("\n🔗 Linking requirements to code...")
    links = link_nodes_by_similarity(req_nodes, code_nodes, threshold=0.6)
    
    # Print some links for debugging
    print("\n🔍 Sample links between requirements and code:")
    for req_id, related_nodes in list(links.items())[:5]:
        req = next((n for n in req_nodes if n.id == req_id), None)
        if req and related_nodes:
            print(f"🔹 Requirement {req.req_id}: {req.text[:50]}...")
            for node_id, similarity in related_nodes[:3]:
                code = next((n for n in code_nodes if n.id == node_id), None)
                if code:
                    print(f"   → {code.type} {code.name} (similarity: {similarity:.3f})") 