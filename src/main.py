import os
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import networkx as nx
from collections import defaultdict
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from .data_processing.ada_processor import AdaDataProcessor

class TraceabilitySystem:
    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.processor = AdaDataProcessor(dataset_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_index = {}
        self.keyword_index = {}
        self.graph_index = nx.DiGraph()
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, 'summaries'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'indexes'), exist_ok=True)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def process_dataset(self):
        """Process the Ada dataset and generate summaries."""
        print("Loading code files...")
        self.processor.load_code_files()
        
        print("Extracting requirements...")
        self.processor.extract_requirements()
        
        print("Generating ground truth...")
        self.processor.generate_ground_truth()
        
        print("Saving summaries...")
        self.processor.save_summaries(os.path.join(self.output_dir, 'summaries'))
        
        print("Building indexes...")
        self._build_indexes()
        
        print("Evaluating traceability...")
        metrics = self.evaluate_traceability()
        
        print("\nEvaluation Metrics:")
        print(f"Average Precision: {metrics['avg_precision']:.3f}")
        print(f"Average Recall: {metrics['avg_recall']:.3f}")
        print(f"Average F1 Score: {metrics['avg_f1']:.3f}")
        
        # Save metrics
        with open(os.path.join(self.output_dir, 'metrics', 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _build_indexes(self):
        """Build vector, keyword, and graph indexes."""
        # Load summaries
        with open(os.path.join(self.output_dir, 'summaries', 'code_summaries.json'), 'r') as f:
            code_summaries = json.load(f)
        
        # Build vector index
        print("Building vector index...")
        for file_path, summary in code_summaries.items():
            # Combine all text fields for embedding
            text = f"{summary['package_name']} {' '.join(summary['procedures'])} {' '.join(summary['functions'])} {' '.join(summary['types'])} {' '.join(summary['constants'])}"
            embedding = self.model.encode(text)
            self.vector_index[file_path] = embedding
        
        # Build keyword index
        print("Building keyword index...")
        stop_words = set(stopwords.words('english'))
        for file_path, summary in code_summaries.items():
            # Combine all text fields
            text = f"{summary['package_name']} {' '.join(summary['procedures'])} {' '.join(summary['functions'])} {' '.join(summary['types'])} {' '.join(summary['constants'])}"
            # Tokenize and remove stopwords
            tokens = [word.lower() for word in word_tokenize(text) if word.lower() not in stop_words]
            self.keyword_index[file_path] = set(tokens)
        
        # Build graph index
        print("Building graph index...")
        for file_path, summary in code_summaries.items():
            self.graph_index.add_node(file_path, **summary)
            for dep in summary['dependencies']:
                # Find the file that contains this dependency
                for other_file, other_summary in code_summaries.items():
                    if other_summary['package_name'] == dep:
                        self.graph_index.add_edge(file_path, other_file)
        
        # Save indexes
        print("Saving indexes...")
        with open(os.path.join(self.output_dir, 'indexes', 'vector_index.pkl'), 'wb') as f:
            pickle.dump(self.vector_index, f)
        with open(os.path.join(self.output_dir, 'indexes', 'keyword_index.pkl'), 'wb') as f:
            pickle.dump(self.keyword_index, f)
        with open(os.path.join(self.output_dir, 'indexes', 'graph_index.pkl'), 'wb') as f:
            pickle.dump(self.graph_index, f)
    
    def trace_requirement(self, requirement_text: str, top_k: int = 11) -> List[Tuple[str, float]]:
        """Trace a requirement to relevant code files."""
        # Get vector similarity scores
        req_embedding = self.model.encode(requirement_text)
        vector_scores = {}
        for doc_id, doc_embedding in self.vector_index.items():
            similarity = cosine_similarity([req_embedding], [doc_embedding])[0][0]
            vector_scores[doc_id] = similarity
        
        # Get keyword match scores
        req_tokens = set(word.lower() for word in word_tokenize(requirement_text) if word.lower() not in stopwords.words('english'))
        keyword_scores = {}
        for doc_id, doc_tokens in self.keyword_index.items():
            matches = len(req_tokens.intersection(doc_tokens))
            keyword_scores[doc_id] = matches / max(len(req_tokens), 1)
        
        # Combine scores
        combined_scores = []
        for doc_id in set(vector_scores.keys()) | set(keyword_scores.keys()):
            vector_score = vector_scores.get(doc_id, 0)
            keyword_score = keyword_scores.get(doc_id, 0)
            
            # Skip documents with very low scores in both indexes
            if keyword_score < 0.1 and vector_score < 0.1:
                continue
            
            # Weight vector scores more heavily (65%) as they capture semantic similarity better
            base_score = (0.35 * keyword_score + 0.65 * vector_score)
            
            # Apply additional scoring factors
            final_score = base_score
            
            # Strong boost for files that appear in both indexes with high scores
            if doc_id in keyword_scores and doc_id in vector_scores:
                if keyword_score > 0.4 and vector_score > 0.4:
                    final_score *= 1.4
                else:
                    final_score *= 1.2
            
            # Extract key terms from requirement
            req_terms = set(word.lower() for word in requirement_text.split())
            doc_name = doc_id.lower()
            
            # Strong boost for exact name matches
            if any(term in doc_name for term in req_terms):
                final_score *= 1.3
            
            # Penalize test files more aggressively
            if 'test' in doc_name.lower():
                final_score *= 0.8
            
            # Penalize utility/helper classes unless they have very high scores
            if any(term in doc_name for term in ['util', 'helper', 'common']):
                final_score *= 0.9
            
            # Boost domain-specific classes
            domain_terms = {'game', 'player', 'level', 'object', 'engine', 'graphics', 'sound', 'input'}
            if any(term in doc_name for term in domain_terms):
                final_score *= 1.2
            
            combined_scores.append((doc_id, final_score))
        
        # Sort by score in descending order
        sorted_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        
        # Apply adaptive thresholding
        if sorted_scores:
            max_score = sorted_scores[0][1]
            threshold = max_score * 0.4
            filtered_scores = [(doc_id, score) for doc_id, score in sorted_scores if score >= threshold]
            
            # If we have enough high-quality results, return them
            if len(filtered_scores) >= top_k:
                return filtered_scores[:top_k]
            
            # If we don't have enough high-quality results, try a lower threshold
            threshold = max_score * 0.2
            filtered_scores = [(doc_id, score) for doc_id, score in sorted_scores if score >= threshold]
            return filtered_scores[:top_k]
        
        return sorted_scores[:top_k]
    
    def evaluate_traceability(self) -> Dict:
        """Evaluate the traceability system's performance."""
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'avg_precision': 0,
            'avg_recall': 0,
            'avg_f1': 0
        }
        
        for req_id, req_text in self.processor.requirements.items():
            # Get predicted links
            predicted_links = set(doc_id for doc_id, _ in self.trace_requirement(req_text))
            
            # Get ground truth links
            ground_truth = self.processor.solution_links.get(req_id, set())
            
            # Calculate metrics
            if predicted_links and ground_truth:
                true_positives = len(predicted_links.intersection(ground_truth))
                precision = true_positives / len(predicted_links)
                recall = true_positives / len(ground_truth)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1'].append(f1)
        
        # Calculate averages
        if metrics['precision']:
            metrics['avg_precision'] = sum(metrics['precision']) / len(metrics['precision'])
            metrics['avg_recall'] = sum(metrics['recall']) / len(metrics['recall'])
            metrics['avg_f1'] = sum(metrics['f1']) / len(metrics['f1'])
        
        return metrics

def main():
    # Set paths
    dataset_path = "datasets/AdaDoom3"
    output_dir = "outputs"
    
    # Create and run traceability system
    system = TraceabilitySystem(dataset_path, output_dir)
    system.process_dataset()

if __name__ == "__main__":
    main() 