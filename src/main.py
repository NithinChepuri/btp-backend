"""
Main script for implementing bidirectional traceability between
requirements and code using the iTrust dataset.
"""
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import json

from src.data_processor import iTrustDataProcessor
from src.code_summarizer import CodeSummarizer
from src.indexer import KeywordIndex, VectorIndex, KnowledgeGraphIndex
from src.rag import RAGSystem
from src.output_manager import OutputManager

class TraceabilitySystem:
    def __init__(self, dataset_path: str):
        """
        Initialize the traceability system components
        
        Args:
            dataset_path: Path to iTrust dataset
        """
        print("\nInitializing Traceability System...")
        
        # Load environment variables
        print("Loading environment variables...")
        load_dotenv()
        
        # Initialize components
        print("\nInitializing components...")
        self.data_processor = iTrustDataProcessor(dataset_path)
        self.code_summarizer = CodeSummarizer()
        
        print("\nInitializing indexes...")
        # Initialize indexes
        self.keyword_index = KeywordIndex()
        self.vector_index = VectorIndex()
        
        # Initialize Neo4j connection
        print("Connecting to Neo4j...")
        try:
            self.knowledge_graph_index = KnowledgeGraphIndex(
                uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                user=os.getenv('NEO4J_USER', 'bst'),
                password=os.getenv('NEO4J_PASSWORD', 'nithinrajeev')
            )
            self.neo4j_available = True
        except ConnectionError as e:
            print(f"\nWarning: Neo4j connection failed: {str(e)}")
            print("System will continue without knowledge graph functionality.")
            self.knowledge_graph_index = None
            self.neo4j_available = False
        
        # Initialize RAG system
        print("Initializing RAG system...")
        self.rag_system = RAGSystem(
            keyword_index=self.keyword_index,
            vector_index=self.vector_index,
            knowledge_graph_index=self.knowledge_graph_index if self.neo4j_available else None,
            llm_model_name=os.getenv('LLAMA_MODEL')
        )
        
        # Initialize output manager
        print("Setting up output directories...")
        self.output_manager = OutputManager()
        
        # Load configuration
        print("Loading configuration...")
        self.min_score = float(os.getenv('MIN_SIMILARITY_SCORE', '0.3'))
        self.max_results = int(os.getenv('MAX_RESULTS_PER_INDEX', '5'))
        self.graph_depth = int(os.getenv('GRAPH_TRAVERSAL_DEPTH', '2'))
        
        # Store generated summaries
        self.code_summaries = {}
        self.req_summaries = {}
        print("System initialization complete!\n")
        
    def initialize_system(self):
        """Initialize the system by loading and processing data"""
        print("\n=== Loading Dataset ===")
        self.data_processor.load_all()
        
        print("\n=== Loading/Generating Summaries ===")
        
        # Try to load cached summaries first
        summaries_dir = Path('outputs/summaries')
        code_summaries_path = summaries_dir / 'summaries.json'
        req_summaries_path = summaries_dir / 'requirement_summaries.json'
        
        summaries_loaded = False
        if code_summaries_path.exists() and req_summaries_path.exists():
            try:
                print("Loading cached summaries...")
                with open(code_summaries_path, 'r', encoding='utf-8') as f:
                    self.code_summaries = json.load(f)
                with open(req_summaries_path, 'r', encoding='utf-8') as f:
                    self.req_summaries = json.load(f)
                print(f"Loaded {len(self.code_summaries)} code summaries and {len(self.req_summaries)} requirement summaries")
                summaries_loaded = True
            except Exception as e:
                print(f"Failed to load cached summaries: {str(e)}")
                summaries_loaded = False
        
        if not summaries_loaded:
            print("Generating fresh summaries...")
            # Generate code summaries
            self.code_summaries = {}
            total_code_files = len(self.data_processor.code_files)
            print(f"Processing {total_code_files} code files...")
            
            for idx, (code_id, code_content) in enumerate(self.data_processor.code_files.items(), 1):
                print(f"\nProcessing file {idx}/{total_code_files}: {code_id}")
                try:
                    print(f"- Generating summary for {code_id}...")
                    summary = self.code_summarizer.summarize_code(code_content)
                    print(f"- Summary generated successfully ({len(summary)} chars)")
                    self.code_summaries[code_id] = summary
                except Exception as e:
                    print(f"- Error generating summary for {code_id}: {str(e)}")
            
            # Generate requirement summaries
            print("\nGenerating requirement summaries...")
            self.req_summaries = {}
            total_reqs = len(self.data_processor.requirements)
            print(f"Processing {total_reqs} requirements...")
            
            for idx, (req_id, req_text) in enumerate(self.data_processor.requirements.items(), 1):
                print(f"\nProcessing requirement {idx}/{total_reqs}: {req_id}")
                try:
                    print(f"- Generating summary for {req_id}...")
                    summary = self.code_summarizer.summarize_text(req_text)
                    print(f"- Summary generated successfully ({len(summary)} chars)")
                    self.req_summaries[req_id] = summary
                except Exception as e:
                    print(f"- Error generating summary for {req_id}: {str(e)}")
            
            # Save generated summaries
            print("\nSaving summaries...")
            summaries_dir.mkdir(exist_ok=True)
            with open(code_summaries_path, 'w', encoding='utf-8') as f:
                json.dump(self.code_summaries, f, indent=2)
            with open(req_summaries_path, 'w', encoding='utf-8') as f:
                json.dump(self.req_summaries, f, indent=2)
        
        print("\n=== Building/Loading Indexes ===")
        indexes_dir = Path('outputs/indexes')
        indexes_dir.mkdir(exist_ok=True)
        
        # Try to load cached indexes
        keyword_index_path = indexes_dir / 'keyword_index.pkl'
        vector_index_path = indexes_dir / 'vector_index.pkl'
        graph_index_path = indexes_dir / 'graph_index.pkl'
        
        indexes_loaded = False
        if all(p.exists() for p in [keyword_index_path, vector_index_path]):
            try:
                print("Loading cached indexes...")
                self.keyword_index.load(str(keyword_index_path))
                self.vector_index.load(str(vector_index_path))
                if self.neo4j_available and graph_index_path.exists():
                    self.knowledge_graph_index.load(str(graph_index_path))
                print("Successfully loaded cached indexes")
                indexes_loaded = True
            except Exception as e:
                print(f"Failed to load cached indexes: {str(e)}")
                indexes_loaded = False
        
        if not indexes_loaded:
            print("Building fresh indexes...")
            # Create fresh instances of indexes
            self.keyword_index = KeywordIndex()
            self.vector_index = VectorIndex()
            if self.neo4j_available:
                self.knowledge_graph_index = KnowledgeGraphIndex(
                    uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                    user=os.getenv('NEO4J_USER', 'bst'),
                    password=os.getenv('NEO4J_PASSWORD', 'nithinrajeev')
                )
            
            # Collect all documents for fitting
            all_documents = []
            all_doc_ids = []
            
            # Add combined code content and summaries
            print("Preparing documents for indexing...")
            for code_id, code_content in self.data_processor.code_files.items():
                if code_id in self.code_summaries:
                    # Combine code content with its summary
                    combined_content = f"""
SUMMARY:
{self.code_summaries[code_id]}

CODE CONTENT:
{code_content}
"""
                    # Add to keyword index
                    self.keyword_index.add_document(code_id, combined_content, is_code=True)
                    
                    # Add to vector index
                    self.vector_index.add_document(code_id, combined_content)
                    
                    # Add to knowledge graph if available
                    if self.neo4j_available:
                        # Get relationships from call graph
                        relationships = []
                        try:
                            # Only get relationships if code_id exists in call graph
                            if code_id in self.data_processor.call_graph:
                                # Get outgoing calls (methods this class calls)
                                for target in self.data_processor.call_graph.successors(code_id):
                                    relationships.append({
                                        'source': code_id,
                                        'target': target,
                                        'type': 'CALLS'
                                    })
                                # Get incoming calls (methods that call this class)
                                for source in self.data_processor.call_graph.predecessors(code_id):
                                    relationships.append({
                                        'source': source,
                                        'target': code_id,
                                        'type': 'CALLS'
                                    })
                            
                            # Add requirement-code relationships
                            for req_id, code_file_id in self.data_processor.solution_links:
                                if code_file_id == code_id:
                                    relationships.append({
                                        'source': req_id,
                                        'target': code_id,
                                        'type': 'IMPLEMENTS'
                                    })
                        except Exception as e:
                            print(f"Warning: Error getting relationships for {code_id}: {str(e)}")
                        
                        # Add to knowledge graph
                        try:
                            self.knowledge_graph_index.add_document(
                                doc_id=code_id,
                                content=combined_content,  # Use combined content
                                relationships=relationships
                            )
                        except Exception as e:
                            print(f"Warning: Error adding {code_id} to knowledge graph: {str(e)}")
            
            # Fit keyword index
            print("Fitting keyword index...")
            self.keyword_index.fit()
            
            print("Building vector index...")
            for idx, (doc_id, content) in enumerate(zip(all_doc_ids, all_documents)):
                self.vector_index.add_document(doc_id, content)
                
                # Add to knowledge graph if available
                if self.neo4j_available:
                    # Get relationships from call graph
                    relationships = []
                    try:
                        # Only get relationships if code_id exists in call graph
                        if doc_id in self.data_processor.call_graph:
                            # Get outgoing calls (methods this class calls)
                            for target in self.data_processor.call_graph.successors(doc_id):
                                relationships.append({
                                    'source': doc_id,
                                    'target': target,
                                    'type': 'CALLS'
                                })
                            # Get incoming calls (methods that call this class)
                            for source in self.data_processor.call_graph.predecessors(doc_id):
                                relationships.append({
                                    'source': source,
                                    'target': doc_id,
                                    'type': 'CALLS'
                                })
                        
                        # Add requirement-code relationships
                        for req_id, code_file_id in self.data_processor.solution_links:
                            if code_file_id == doc_id:
                                relationships.append({
                                    'source': req_id,
                                    'target': doc_id,
                                    'type': 'IMPLEMENTS'
                                })
                    except Exception as e:
                        print(f"Warning: Error getting relationships for {doc_id}: {str(e)}")
                    
                    # Add to knowledge graph
                    try:
                        self.knowledge_graph_index.add_document(
                            doc_id=doc_id,
                            content=content,  # Use combined content
                            relationships=relationships
                        )
                    except Exception as e:
                        print(f"Warning: Failed to add {doc_id} to knowledge graph: {str(e)}")
            
            print("\nSaving indexes...")
            self.keyword_index.save(str(keyword_index_path))
            self.vector_index.save(str(vector_index_path))
            if self.neo4j_available:
                self.knowledge_graph_index.save(str(graph_index_path))
        
        print("\nSystem initialization complete!")
        
    def trace_requirement(self, requirement: str, top_k: int = 11) -> List[Tuple[str, float]]:
        """
        Trace a requirement to related code files with enhanced scoring and filtering
        
        Args:
            requirement: Requirement ID or text
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        # If requirement is an ID, get the text
        if requirement in self.data_processor.requirements:
            requirement_text = self.data_processor.requirements[requirement]
        else:
            requirement_text = requirement
            
        # Get more candidates from each index to ensure diversity
        keyword_results = self.keyword_index.search(requirement_text, top_k=top_k*3)
        vector_results = self.vector_index.search(requirement_text, top_k=top_k*3)
        
        # Convert results to score dictionaries
        keyword_scores = {doc_id: score for doc_id, score in keyword_results}
        vector_scores = {doc_id: score for doc_id, score in vector_results}
        
        # Get all unique document IDs
        all_docs = set(keyword_scores.keys()) | set(vector_scores.keys())
        
        # Enhanced scoring with weighted combination and additional factors
        combined_scores = []
        for doc_id in all_docs:
            # Get base scores
            keyword_score = keyword_scores.get(doc_id, 0.0)
            vector_score = vector_scores.get(doc_id, 0.0)
            
            # Skip documents with very low scores in either index
            if keyword_score < 0.4 or vector_score < 0.4:
                continue
            
            # Weight vector scores more heavily (70%) as they capture semantic similarity better
            base_score = (0.3 * keyword_score + 0.7 * vector_score)
            
            # Apply additional scoring factors
            final_score = base_score
            
            # Strong boost for files that appear in both indexes with high scores
            if doc_id in keyword_scores and doc_id in vector_scores:
                if keyword_score > 0.6 and vector_score > 0.6:
                    final_score *= 1.5
                else:
                    final_score *= 1.2
            
            # Extract key terms from requirement
            req_terms = set(word.lower() for word in requirement_text.split())
            doc_name = doc_id.lower()
            
            # Strong boost for exact name matches
            if any(term in doc_name for term in req_terms):
                final_score *= 1.4
            
            # Penalize test files more aggressively
            if 'test' in doc_name.lower():
                final_score *= 0.6
            
            # Penalize utility/helper classes unless they have very high scores
            if any(term in doc_name for term in ['util', 'helper', 'common']):
                final_score *= 0.7
            
            # Boost domain-specific classes
            domain_terms = {'patient', 'doctor', 'visit', 'record', 'health', 'medical', 'prescription'}
            if any(term in doc_name for term in domain_terms):
                final_score *= 1.3
            
            combined_scores.append((doc_id, final_score))
        
        # Sort by score in descending order
        sorted_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
        
        # Apply stricter adaptive thresholding
        if sorted_scores:
            max_score = sorted_scores[0][1]
            threshold = max_score * 0.7  # Keep only scores within 70% of max score
            filtered_scores = [(doc_id, score) for doc_id, score in sorted_scores if score >= threshold]
            
            # If we have enough high-quality results, return them
            if len(filtered_scores) >= top_k:
                return filtered_scores[:top_k]
            
            # If we don't have enough high-quality results, try a lower threshold
            threshold = max_score * 0.5
            filtered_scores = [(doc_id, score) for doc_id, score in sorted_scores if score >= threshold]
            return filtered_scores[:top_k]
        
        return sorted_scores[:top_k]
    
    def evaluate_traceability(self) -> Dict[str, float]:
        """
        Evaluate the traceability system using ground truth links
        
        Returns:
            Dictionary containing precision, recall, and F1 scores
        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        print("\n=== Evaluating Traceability ===")
        
        # Group solution links by requirement
        solution_links_by_req = {}
        for req_id, code_id in self.data_processor.solution_links:
            if req_id not in solution_links_by_req:
                solution_links_by_req[req_id] = set()
            solution_links_by_req[req_id].add(code_id)
        
        # Store detailed results for each requirement
        detailed_results = []
        
        for req_id, req_text in tqdm(self.data_processor.requirements.items(), desc="Evaluating requirements"):
            # Get system predictions (top 11)
            results = self.trace_requirement(req_id, top_k=11)
            predicted_classes = set(doc_id for doc_id, _ in results)
            
            # Get ground truth - all code files linked to this requirement
            true_classes = solution_links_by_req.get(req_id, set())
            
            # Calculate metrics for this requirement
            req_true_positives = len(predicted_classes & true_classes)
            req_false_positives = len(predicted_classes - true_classes)
            req_false_negatives = len(true_classes - predicted_classes)
            
            # Store detailed results
            detailed_results.append({
                'requirement_id': req_id,
                'true_classes': sorted(list(true_classes)),
                'predicted_classes': [(doc_id, score) for doc_id, score in results],
                'metrics': {
                    'true_positives': req_true_positives,
                    'false_positives': req_false_positives,
                    'false_negatives': req_false_negatives
                }
            })
            
            # Update overall metrics
            true_positives += req_true_positives
            false_positives += req_false_positives
            false_negatives += req_false_negatives
        
        # Calculate final metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Print detailed results
        print("\n=== Detailed Results ===")
        for result in detailed_results:
            req_id = result['requirement_id']
            print(f"\nRequirement: {req_id}")
            print("Ground Truth Classes:")
            for class_name in result['true_classes']:
                print(f"  - {class_name}")
            
            print("\nPredicted Classes (Top 11):")
            for doc_id, score in result['predicted_classes']:
                match = "✓" if doc_id in result['true_classes'] else "✗"
                print(f"  {match} {doc_id} (score: {score:.4f})")
            
            metrics = result['metrics']
            print(f"\nMetrics for this requirement:")
            print(f"  True Positives: {metrics['true_positives']}")
            print(f"  False Positives: {metrics['false_positives']}")
            print(f"  False Negatives: {metrics['false_negatives']}")
            print("-" * 80)
        
        # Print final metrics
        print("\nFinal Evaluation Metrics:")
        print(f"True Positives: {true_positives}")
        print(f"False Positives: {false_positives}")
        print(f"False Negatives: {false_negatives}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'detailed_results': detailed_results
        }
        
        # Save metrics with detailed results
        self.output_manager.save_evaluation_metrics(metrics)
        
        return metrics

    def _combine_search_results(self, keyword_results: List[Tuple[str, float]], vector_results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """
        Combine and normalize results from different indexes
        
        Args:
            keyword_results: List of (doc_id, score) tuples from keyword index
            vector_results: List of (doc_id, score) tuples from vector index
            
        Returns:
            Combined and normalized list of (doc_id, score) tuples
        """
        # Create score dictionaries
        keyword_scores = {doc_id: score for doc_id, score in keyword_results}
        vector_scores = {doc_id: score for doc_id, score in vector_results}
        
        # Get all unique document IDs
        all_docs = set(keyword_scores.keys()) | set(vector_scores.keys())
        
        # Combine scores with equal weights
        combined_scores = []
        for doc_id in all_docs:
            keyword_score = keyword_scores.get(doc_id, 0.0)
            vector_score = vector_scores.get(doc_id, 0.0)
            combined_score = (keyword_score + vector_score) / 2
            combined_scores.append((doc_id, combined_score))
        
        # Sort by score in descending order
        return sorted(combined_scores, key=lambda x: x[1], reverse=True)

    def close(self):
        """Clean up resources"""
        if self.neo4j_available and self.knowledge_graph_index:
            try:
                self.knowledge_graph_index.close()
            except Exception as e:
                print(f"\nWarning: Error while closing Neo4j connection: {str(e)}")

def main():
    """Main entry point"""
    # Initialize system
    dataset_path = Path('datasets/iTrust')
    try:
        system = TraceabilitySystem(str(dataset_path))
        
        # Initialize and build indexes
        system.initialize_system()
        
        # Example: Trace a requirement
        print("\n=== Example Requirement Tracing ===")
        requirement = """
        The system shall allow patients to view their records including
        their health history, diagnoses, and prescriptions.
        """
        results = system.trace_requirement(requirement)
        
        print("\nTop Related Classes:")
        print("-" * 50)
        for doc_id, score in results:
            print(f"Class: {doc_id}")
            print(f"Similarity Score: {score:.4f}")
            print("-" * 50)
        
        # Evaluate system
        print("\n=== System Evaluation ===")
        metrics = system.evaluate_traceability()
        print("\nEvaluation Metrics:")
        print("-" * 50)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("-" * 50)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
    finally:
        # Clean up
        if 'system' in locals():
            system.close()

if __name__ == "__main__":
    main() 