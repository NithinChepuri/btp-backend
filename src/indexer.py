"""
Indexer Module for creating and managing different types of indexes
for code retrieval (Keyword, Vector, and Knowledge Graph).
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from neo4j import GraphDatabase
import networkx as nx
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from scipy.sparse import vstack
from sklearn.metrics.pairwise import cosine_similarity
import re
import torch
from sentence_transformers import util
import pickle

# Download required NLTK data
print("Downloading required NLTK data...")
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt_tab')
    nltk.download('punkt_models')
except Exception as e:
    print(f"Warning: Error downloading NLTK data: {str(e)}")
    print("Falling back to basic tokenization...")

class BaseIndex:
    """Base class for all index types"""
    def __init__(self):
        self.index = {}
        
    def add_document(self, doc_id: str, content: str):
        """Add a document to the index"""
        raise NotImplementedError
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search the index"""
        raise NotImplementedError
        
    def save(self, filepath: str):
        """Save index to file"""
        raise NotImplementedError
        
    def load(self, filepath: str):
        """Load index from file"""
        raise NotImplementedError

class KeywordIndex(BaseIndex):
    """Keyword-based index using TF-IDF with enhanced preprocessing"""
    def __init__(self):
        super().__init__()
        self.tfidf_matrix = None
        self.vectorizer = None
        self.doc_ids = []
        self.doc_contents = {}
        self.stemmer = PorterStemmer()
        
        # Java-specific stop words to remove
        self.java_stop_words = {
            'public', 'private', 'protected', 'static', 'final', 'void', 'class', 'interface',
            'extends', 'implements', 'return', 'new', 'package', 'import', 'throws', 'try',
            'catch', 'finally', 'throw', 'abstract', 'this', 'super', 'null', 'true', 'false'
        }
        
        # Important terms to keep
        self.important_terms = {
            'dao', 'action', 'bean', 'validator', 'controller', 'service', 'exception',
            'create', 'update', 'delete', 'get', 'set', 'view', 'edit', 'add', 'remove',
            'process', 'validate', 'check', 'verify', 'authenticate', 'authorize'
        }
        
    def preprocess_text(self, text: str, is_code: bool = True) -> str:
        """Enhanced text preprocessing for Java code and requirements"""
        try:
            # Extract important parts based on content type
            if is_code:
                # Extract class/interface names and method signatures
                class_pattern = r'(public|private|protected)?\s*(class|interface)\s+(\w+)'
                method_pattern = r'(public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)'
                
                important_lines = []
                
                # Extract class/interface definitions
                class_matches = re.finditer(class_pattern, text)
                for match in class_matches:
                    important_lines.append(match.group())
                    
                # Extract method signatures
                method_matches = re.finditer(method_pattern, text)
                for match in method_matches:
                    important_lines.append(match.group())
                    
                # Extract JavaDoc comments
                javadoc_pattern = r'/\*\*[\s\S]*?\*/'
                javadoc_matches = re.finditer(javadoc_pattern, text)
                for match in javadoc_matches:
                    comment = match.group()
                    # Clean JavaDoc syntax
                    comment = re.sub(r'[/*\s@]*', ' ', comment)
                    important_lines.append(comment)
                    
                # Join important lines with content
                text = '\n'.join(important_lines) + '\n' + text
            else:
                # For requirements, extract sentences with key indicators
                try:
                    sentences = nltk.sent_tokenize(text)
                except:
                    # Fallback to basic sentence splitting
                    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
                
                key_phrases = []
                for sentence in sentences:
                    if any(indicator in sentence.lower() for indicator in 
                          ['shall', 'must', 'should', 'will', 'requires', 'needs to']):
                        key_phrases.append(sentence)
                        
                # Join key phrases with content
                text = '\n'.join(key_phrases) + '\n' + text
            
            # Split camelCase and PascalCase
            words = []
            for word in text.split():
                words.extend(re.findall('[A-Z][a-z]*|[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', word))
            
            # Convert to lowercase and tokenize
            processed = ' '.join(words).lower()
            try:
                tokens = word_tokenize(processed)
            except:
                # Fallback to basic tokenization
                tokens = processed.split()
            
            # Remove stop words but keep important terms
            try:
                stop_words = set(stopwords.words('english')).union(self.java_stop_words)
            except:
                stop_words = self.java_stop_words
                
            tokens = [
                token for token in tokens 
                if (token not in stop_words or 
                    token in self.important_terms or 
                    any(term in token for term in self.important_terms))
            ]
            
            # Stem tokens
            try:
                tokens = [self.stemmer.stem(token) for token in tokens]
            except:
                # Skip stemming if it fails
                pass
            
            return ' '.join(tokens)
            
        except Exception as e:
            print(f"Warning: Error in text preprocessing: {str(e)}")
            # Return basic tokenization as fallback
            return ' '.join(text.lower().split())
            
    def add_document(self, doc_id: str, content: str, is_code: bool = True):
        """Add a document to the index"""
        processed = self.preprocess_text(content, is_code)
        self.doc_contents[doc_id] = processed
        self.doc_ids.append(doc_id)
        
    def fit(self):
        """Fit the TF-IDF vectorizer on all documents"""
        if not self.doc_contents:
            return
            
        # Configure vectorizer with better parameters
        self.vectorizer = TfidfVectorizer(
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.9,  # Ignore terms that appear in more than 90% of documents
            ngram_range=(1, 2),  # Use unigrams and bigrams
            sublinear_tf=True  # Apply sublinear scaling to term frequencies
        )
        
        # Fit vectorizer and transform documents
        docs = [self.doc_contents[doc_id] for doc_id in self.doc_ids]
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for documents similar to the query"""
        if self.tfidf_matrix is None or self.vectorizer is None:
            return []
            
        try:
            # Preprocess query as requirement
            processed_query = self.preprocess_text(query, is_code=False)
            
            # Transform query
            query_vector = self.vectorizer.transform([processed_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
            
            # Create results with scores
            results = list(zip(self.doc_ids, similarities))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Apply adaptive thresholding
            if results:
                max_score = results[0][1]
                min_score = results[-1][1]
                score_range = max_score - min_score
                
                if score_range > 0:
                    threshold = max_score - (score_range * 0.3)  # Keep top 30% of score range
                    results = [(doc_id, score) for doc_id, score in results if score >= threshold]
            
            # Apply diversity ranking
            final_results = []
            seen_prefixes = set()
            
            for doc_id, score in results:
                # Get file prefix (e.g., 'Auth' from 'AuthDAO.java')
                prefix = re.sub(r'([A-Z])[a-z]*', r'\1', doc_id.split('.')[0])
                
                # Add if prefix not seen or score is very high
                if prefix not in seen_prefixes or score > 0.8:
                    final_results.append((doc_id, float(score)))
                    seen_prefixes.add(prefix)
                    
                if len(final_results) >= top_k:
                    break
            
            return final_results
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []
            
    def save(self, file_path: str):
        """Save the index to a file"""
        data = {
            'tfidf_matrix': self.tfidf_matrix,
            'vectorizer': self.vectorizer,
            'doc_ids': self.doc_ids,
            'doc_contents': self.doc_contents
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, file_path: str):
        """Load the index from a file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.tfidf_matrix = data['tfidf_matrix']
            self.vectorizer = data['vectorizer']
            self.doc_ids = data['doc_ids']
            self.doc_contents = data['doc_contents']

class VectorIndex(BaseIndex):
    """Vector-based index using sentence transformers"""
    def __init__(self):
        super().__init__()
        self.vectors = []
        self.doc_ids = []
        self.doc_contents = {}
        self.model = SentenceTransformer('microsoft/graphcodebert-base')
        
    def preprocess_code(self, content: str) -> str:
        """Enhanced preprocessing for code files"""
        # Extract class/interface names and method signatures
        class_pattern = r'(public|private|protected)?\s*(class|interface)\s+(\w+)'
        method_pattern = r'(public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)'
        
        important_lines = []
        
        # Extract class/interface definitions with higher weight
        class_matches = re.finditer(class_pattern, content)
        for match in class_matches:
            # Repeat class definition to give it more weight
            important_lines.extend([match.group()] * 3)
            
        # Extract method signatures with context
        method_matches = re.finditer(method_pattern, content)
        for match in method_matches:
            # Get surrounding context (previous and next line)
            start = max(0, match.start() - 100)
            end = min(len(content), match.end() + 100)
            context = content[start:end]
            important_lines.append(context)
            
        # Extract JavaDoc comments with parameter info
        javadoc_pattern = r'/\*\*[\s\S]*?\*/'
        javadoc_matches = re.finditer(javadoc_pattern, content)
        for match in javadoc_matches:
            comment = match.group()
            # Extract parameter descriptions
            param_pattern = r'@param\s+(\w+)\s+([^\n]*)'
            param_matches = re.finditer(param_pattern, comment)
            for param_match in param_matches:
                important_lines.append(param_match.group())
            # Clean JavaDoc syntax
            comment = re.sub(r'[/*\s@]*', ' ', comment)
            important_lines.append(comment)
            
        # Join important lines with content
        processed = '\n'.join(important_lines) + '\n' + content
        
        # Clean and normalize
        processed = re.sub(r'[^\w\s]', ' ', processed)  # Remove punctuation
        processed = re.sub(r'\s+', ' ', processed).strip()  # Normalize whitespace
        
        # Split camelCase and PascalCase with preservation
        words = []
        for word in processed.split():
            # Keep original word
            words.append(word)
            # Add split version
            words.extend(re.findall('[A-Z][a-z]*|[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', word))
            
        return ' '.join(words)

    def preprocess_requirement(self, content: str) -> str:
        """Enhanced preprocessing for requirements"""
        # Extract key phrases with context
        key_phrases = []
        sentences = nltk.sent_tokenize(content)
        
        for i, sentence in enumerate(sentences):
            # Look for important indicators
            if any(indicator in sentence.lower() for indicator in 
                  ['shall', 'must', 'should', 'will', 'requires', 'needs to']):
                # Get surrounding context
                start = max(0, i - 1)
                end = min(len(sentences), i + 2)
                context = ' '.join(sentences[start:end])
                key_phrases.append(context)
                # Add the key sentence with more weight
                key_phrases.extend([sentence] * 2)
                
        # Extract domain-specific terms
        domain_terms = re.findall(r'\b[A-Z][a-zA-Z]*(?:DAO|Action|Controller|Service|Bean|Validator)\b', content)
        if domain_terms:
            key_phrases.extend(domain_terms * 2)  # Add domain terms with more weight
            
        # Join key phrases with content
        processed = '\n'.join(key_phrases) + '\n' + content
        
        # Clean special characters but keep some punctuation
        processed = re.sub(r'[^a-zA-Z0-9\s\.,]', ' ', processed)
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed

    def add_document(self, doc_id: str, content: str, is_code: bool = True):
        """Add a document to the index"""
        # Preprocess based on content type
        processed = self.preprocess_code(content) if is_code else self.preprocess_requirement(content)
        self.doc_contents[doc_id] = processed
        
        # Generate embedding
        vector = self.model.encode(processed, convert_to_tensor=True)
        self.vectors.append(vector)
        self.doc_ids.append(doc_id)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents with improved scoring"""
        # Preprocess query
        processed_query = self.preprocess_requirement(query)
        query_vector = self.model.encode(processed_query, convert_to_tensor=True)
        
        # Calculate similarities with improved scoring
        similarities = []
        for i, doc_vector in enumerate(self.vectors):
            # Cosine similarity
            sim = util.pytorch_cos_sim(query_vector, doc_vector).item()
            
            # Get document info
            doc_id = self.doc_ids[i]
            doc_content = self.doc_contents[doc_id]
            
            # Calculate term overlap with better weighting
            query_terms = set(processed_query.lower().split())
            doc_terms = set(doc_content.lower().split())
            
            # Weight different types of matches
            exact_matches = len(query_terms & doc_terms)
            partial_matches = sum(1 for qt in query_terms 
                                for dt in doc_terms 
                                if qt in dt or dt in qt)
            
            # Calculate weighted overlap score
            overlap_score = (exact_matches * 0.7 + partial_matches * 0.3) / len(query_terms)
            
            # Boost score based on document type and name relevance
            boost = 1.0
            if doc_id.endswith('.java'):
                name = doc_id.split('.')[0]
                
                # Boost based on name relevance to query
                name_terms = set(re.findall('[A-Z][a-z]*', name))
                name_relevance = len(query_terms.intersection(map(str.lower, name_terms))) / len(query_terms)
                boost += name_relevance * 0.5
                
                # Boost based on file type
                if any(suffix in name for suffix in ['DAO', 'Action', 'Controller', 'Service']):
                    boost *= 1.3
                elif 'Exception' in name:
                    boost *= 0.6
                    
            # Combine scores with adjusted weights
            final_score = (sim * 0.6 + overlap_score * 0.4) * boost
            similarities.append((doc_id, final_score))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply adaptive thresholding
        if similarities:
            max_score = similarities[0][1]
            threshold = max_score * 0.6  # Keep only scores within 60% of max
            similarities = [(doc_id, score) for doc_id, score in similarities if score >= threshold]
        
        # Apply diversity ranking
        final_results = []
        seen_prefixes = set()
        
        for doc_id, score in similarities:
            if doc_id.endswith('.java'):
                # Extract base name without type suffix
                base_name = re.sub(r'(DAO|Action|Controller|Service|Bean|Validator)$', '', 
                                 doc_id.split('.')[0])
                
                # Skip if similar prefix seen unless score is very high
                if base_name in seen_prefixes and score < 0.7:
                    continue
                seen_prefixes.add(base_name)
            
            final_results.append((doc_id, score))
            if len(final_results) >= top_k:
                break
        
        return final_results

    def save(self, file_path: str):
        """Save the index to a file"""
        data = {
            'vectors': [v.tolist() for v in self.vectors],
            'doc_ids': self.doc_ids,
            'doc_contents': self.doc_contents
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, file_path: str):
        """Load the index from a file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.vectors = [torch.tensor(v) for v in data['vectors']]
            self.doc_ids = data['doc_ids']
            self.doc_contents = data['doc_contents']

class KnowledgeGraphIndex(BaseIndex):
    """Knowledge Graph-based index using Neo4j with advanced traceability features"""
    def __init__(self, uri: str, user: str, password: str):
        super().__init__()
        print("\nInitializing Neo4j connection...")
        try:
            # Create the driver instance
            self.driver = GraphDatabase.driver(
                uri, 
                auth=(user, password)
            )
            
            # Verify connection and print database info
            with self.driver.session() as session:
                # Check connection
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                record = result.single()
                print(f"Connected to Neo4j {record['name']} {record['edition']} version {record['versions'][0]}")
                
                # Count nodes and relationships
                counts = session.run("""
                    MATCH (n) 
                    OPTIONAL MATCH (n)-[r]->() 
                    RETURN count(DISTINCT n) as nodes, count(DISTINCT r) as rels
                """)
                count_record = counts.single()
                print(f"Database contains {count_record['nodes']} nodes and {count_record['rels']} relationships")
                
                # Sample some node labels
                labels = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels")
                label_list = labels.single()['labels']
                if label_list:
                    print(f"Node types: {', '.join(label_list[:5])}")
                
            print("Successfully connected to Neo4j!")
            
        except Exception as e:
            print(f"Neo4j connection error: {str(e)}")
            print("Connection details used:")
            print(f"URI: {uri}")
            print(f"User: {user}")
            raise ConnectionError(f"Failed to connect to Neo4j: {str(e)}")
        
        self.graph = nx.DiGraph()
        # Relationship type weights based on importance
        self.relationship_weights = {
            'calls': 1.0,  # Direct function calls
            'imports': 0.8,  # Import relationships
            'inherits': 0.9,  # Inheritance relationships
            'implements': 0.9,  # Interface implementation
            'references': 0.7,  # General references
            'contains': 0.6,  # Package/module containment
            'similar_to': 0.5,  # Semantic similarity
            'requirement': 1.0  # Direct requirement links
        }
        
    def add_document(self, doc_id: str, content: str, relationships: List[Dict[str, str]]):
        """
        Add document to knowledge graph index with weighted relationships
        
        Args:
            doc_id: Document ID
            content: Document content
            relationships: List of relationships with types and optional weights
        """
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
                
            # Add node with content and metadata
            self.graph.add_node(doc_id, 
                              content=content,
                              type='code' if doc_id.endswith(('.py', '.java', '.cpp', '.cs')) else 'requirement')
            
            # Add weighted relationships
            for rel in relationships:
                rel_type = rel.get('type', 'references')
                weight = rel.get('weight', self.relationship_weights.get(rel_type, 0.5))
                
                self.graph.add_edge(
                    rel['source'],
                    rel['target'],
                    type=rel_type,
                    weight=weight
                )
                
                # Add reverse relationship for bidirectional analysis
                if rel_type in ['calls', 'references', 'similar_to']:
                    reverse_type = f"reverse_{rel_type}"
                    self.graph.add_edge(
                        rel['target'],
                        rel['source'],
                        type=reverse_type,
                        weight=weight * 0.8  # Slightly lower weight for reverse relationships
                    )
            
            self._sync_to_neo4j()
        except Exception as e:
            raise ConnectionError(f"Failed to add document to Neo4j: {str(e)}")
        
    def _sync_to_neo4j(self):
        """Synchronize NetworkX graph with Neo4j including relationship weights"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
                
                print("\nSynchronizing with Neo4j...")
                session.run("MATCH (n) DETACH DELETE n")
                
                # Add nodes with type information
                print(f"Adding {len(self.graph.nodes)} nodes to Neo4j...")
                for node, attrs in self.graph.nodes(data=True):
                    session.run(
                        """
                        CREATE (n:Document {
                            id: $id,
                            content: $content,
                            type: $type
                        })
                        """,
                        id=node,
                        content=attrs.get('content', ''),
                        type=attrs.get('type', 'unknown')
                    )
                
                # Add weighted relationships
                print(f"Adding {len(self.graph.edges)} relationships to Neo4j...")
                for source, target, attrs in self.graph.edges(data=True):
                    session.run(
                        """
                        MATCH (s:Document {id: $source})
                        MATCH (t:Document {id: $target})
                        CREATE (s)-[:RELATES {
                            type: $type,
                            weight: $weight
                        }]->(t)
                        """,
                        source=source,
                        target=target,
                        type=attrs.get('type', 'unknown'),
                        weight=attrs.get('weight', 0.5)
                    )
                print("Neo4j graph synchronization complete!")
        except Exception as e:
            raise ConnectionError(f"Failed to sync with Neo4j: {str(e)}")
        
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Enhanced search using multi-step graph traversal and weighted relationships
        
        1. Initial node ranking using personalized PageRank
        2. Path-based analysis for indirect relationships
        3. Relationship type weighting
        4. Bidirectional relationship analysis
        """
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            # Step 1: Calculate personalized PageRank
            # Give higher weight to nodes matching query keywords
            personalization = {}
            query_tokens = set(query.lower().split())
            for node, attrs in self.graph.nodes(data=True):
                content = attrs.get('content', '').lower()
                match_score = sum(1 for token in query_tokens if token in content)
                personalization[node] = 1.0 + (match_score * 0.2)  # Boost matching nodes
            
            pagerank_scores = nx.pagerank(self.graph, 
                                        personalization=personalization,
                                        weight='weight')
            
            # Step 2: Path-based analysis
            path_scores = {}
            for node in self.graph.nodes():
                # Find all paths up to length 3 (configurable)
                paths = []
                for target in self.graph.nodes():
                    if target != node:
                        try:
                            paths.extend(nx.all_simple_paths(self.graph, node, target, cutoff=3))
                        except:
                            continue
                
                # Score paths based on relationship types and weights
                path_score = 0
                for path in paths:
                    path_weight = 1.0
                    for i in range(len(path) - 1):
                        edge_data = self.graph[path[i]][path[i + 1]]
                        rel_type = edge_data.get('type', 'unknown')
                        weight = edge_data.get('weight', 0.5)
                        path_weight *= weight * self.relationship_weights.get(rel_type, 0.5)
                    path_score += path_weight
                
                path_scores[node] = path_score
            
            # Step 3: Combine scores
            final_scores = {}
            for node in self.graph.nodes():
                # Weighted combination of PageRank and path scores
                final_scores[node] = (0.7 * pagerank_scores[node] + 
                                    0.3 * (path_scores[node] / max(path_scores.values())))
            
            # Get results with combined scoring
            results = []
            for node, score in sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
                results.append({
                    'doc_id': node,
                    'score': float(score),
                    'content': self.graph.nodes[node].get('content', ''),
                    'type': self.graph.nodes[node].get('type', 'unknown')
                })
            return results
        except Exception as e:
            raise ConnectionError(f"Failed to search Neo4j graph: {str(e)}")
    
    def close(self):
        """Close Neo4j connection"""
        try:
            if self.driver:
                self.driver.close()
                print("\nClosed Neo4j connection")
        except Exception as e:
            print(f"\nWarning: Error while closing Neo4j connection: {str(e)}")

    def save(self, filepath: str):
        """Save graph to file"""
        import pickle
        data = {
            'nodes': dict(self.graph.nodes(data=True)),
            'edges': list(self.graph.edges(data=True))
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, filepath: str):
        """Load graph from file and sync with Neo4j"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        # Recreate graph
        self.graph = nx.DiGraph()
        for node, attrs in data['nodes'].items():
            self.graph.add_node(node, **attrs)
        for source, target, attrs in data['edges']:
            self.graph.add_edge(source, target, **attrs)
            
        # Sync with Neo4j
        self._sync_to_neo4j()

class CombinedIndex(BaseIndex):
    def __init__(self, keyword_weight: float = 0.45, vector_weight: float = 0.55):  # Rebalanced for recall
        super().__init__()
        self.keyword_index = KeywordIndex()
        self.vector_index = VectorIndex()
        self.keyword_weight = keyword_weight
        self.vector_weight = vector_weight
        
    def add_document(self, doc_id: str, content: str):
        """Add document to both indexes"""
        self.keyword_index.add_document(doc_id, content)
        self.vector_index.add_document(doc_id, content)
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search with improved result combination optimized for recall"""
        # Get significantly more candidates for better recall
        keyword_results = self.keyword_index.search(query, top_k=top_k*8)  # Much larger pool
        vector_results = self.vector_index.search(query, top_k=top_k*8)    # Much larger pool
        
        # Create score maps with aggressive confidence boosting
        keyword_scores = {}
        vector_scores = {}
        
        # Process keyword results with more aggressive boosting
        max_keyword_score = max((score for _, score in keyword_results), default=0)
        for doc_id, score in keyword_results:
            # More aggressive boost for high confidence matches
            if score > 0.6 * max_keyword_score:  # Lower threshold
                score = score * 1.25  # Higher boost
            keyword_scores[doc_id] = score
            
        # Process vector results with more aggressive boosting    
        max_vector_score = max((score for _, score in vector_results), default=0)
        for doc_id, score in vector_results:
            # More aggressive boost for high confidence matches
            if score > 0.6 * max_vector_score:  # Lower threshold
                score = score * 1.25  # Higher boost
            vector_scores[doc_id] = score
        
        # Get all unique document IDs
        all_docs = set(keyword_scores.keys()).union(set(vector_scores.keys()))
        
        # Calculate combined scores with improved weighting for recall
        combined_scores = []
        for doc_id in all_docs:
            keyword_score = keyword_scores.get(doc_id, 0.0)
            vector_score = vector_scores.get(doc_id, 0.0)
            
            # Very permissive initial filtering
            if keyword_score < 0.15 and vector_score < 0.15:  # Much lower threshold
                continue
                
            # Less aggressive exponential penalty
            keyword_exp = 1.1 if keyword_score > 0.5 else 1.2  # Reduced exponents
            vector_exp = 1.2 if vector_score > 0.5 else 1.3    # Reduced exponents
            
            keyword_contrib = self.keyword_weight * (keyword_score ** keyword_exp)
            vector_contrib = self.vector_weight * (vector_score ** vector_exp)
            
            # Enhanced boost for documents in both indexes
            both_indexes_boost = 0.0
            if doc_id in keyword_scores and doc_id in vector_scores:
                min_score = min(keyword_score, vector_score)
                max_score = max(keyword_score, vector_score)
                if min_score > 0.2:  # More permissive threshold
                    both_indexes_boost = 0.45 * (min_score + max_score) / 2  # Higher boost
                    
                    # Extra boost if either score is good
                    if max_score > 0.4:  # Lower threshold
                        both_indexes_boost *= 1.3
            
            base_score = keyword_contrib + vector_contrib + both_indexes_boost
            
            # Enhanced content-based boost with broader matching
            if doc_id.endswith('.java'):
                parts = doc_id.split('.')
                if len(parts) > 1:
                    name = parts[0].lower()
                    
                    # Extract file type
                    file_type = next((suffix for suffix in ['DAO', 'Action', 'Controller', 'Service', 'Bean', 'Validator']
                                    if suffix in parts[0]), None)
                    
                    # Broader query relevance scoring
                    query_terms = set(query.lower().split())
                    name_terms = set(re.findall('[A-Z][a-z]*', parts[0]))
                    
                    # Direct matches with more weight
                    direct_matches = len(query_terms.intersection(map(str.lower, name_terms)))
                    
                    # More permissive partial matches
                    partial_matches = sum(2 for qt in query_terms 
                                       for nt in name_terms 
                                       if (qt in nt.lower() and len(qt) > 2) or  # Shorter minimum length
                                          (nt.lower() in qt and len(nt) > 2))    # Shorter minimum length
                                          
                    # Expanded context matches
                    context_pairs = {
                        'edit': {'update', 'modify', 'change', 'save', 'set'},
                        'view': {'display', 'show', 'get', 'fetch', 'read', 'load'},
                        'delete': {'remove', 'clear', 'erase', 'destroy'},
                        'create': {'add', 'new', 'insert', 'make', 'generate'},
                        'authenticate': {'login', 'security', 'auth', 'access'},
                        'validate': {'check', 'verify', 'ensure', 'confirm'},
                        'process': {'handle', 'execute', 'perform', 'run'},
                        'search': {'find', 'query', 'lookup', 'filter'},
                        'manage': {'admin', 'control', 'maintain'}
                    }
                    
                    context_matches = sum(1 for qt in query_terms
                                        for term, synonyms in context_pairs.items()
                                        if (qt in synonyms and term in name.lower()) or
                                           (term in qt and any(s in name.lower() for s in synonyms)))
                    
                    # Calculate name relevance with adjusted weights for recall
                    name_relevance = (
                        (direct_matches * 1.2) +  # Increased weight
                        (partial_matches * 0.4) + # Increased weight
                        (context_matches * 0.5)   # Increased weight
                    ) / len(query_terms)
                    
                    final_score = base_score * (1 + name_relevance * 1.4)  # Higher impact
                    
                    if file_type:
                        # More aggressive type boosts
                        type_boosts = {
                            'DAO': 1.6,       # Higher boost
                            'Service': 1.5,    # Higher boost
                            'Action': 1.45,    # Higher boost
                            'Controller': 1.4,  # Higher boost
                            'Bean': 1.3,       # Higher boost
                            'Validator': 1.35   # Higher boost
                        }
                        final_score *= type_boosts.get(file_type, 1.0)
                        
                        # Extra boost for exact type matches
                        if file_type.lower() in query.lower():
                            final_score *= 1.2
                            
                    elif 'Exception' in parts[0]:
                        if 'error' in query.lower() or 'exception' in query.lower():
                            final_score *= 0.9  # Less penalty
                        else:
                            final_score *= 0.5  # Less aggressive penalty
                    
                    # More permissive test file handling
                    if 'test' in name.lower():
                        if 'test' in query.lower():
                            final_score *= 1.0  # No penalty when searching for tests
                        else:
                            final_score *= 0.6  # Less aggressive penalty
                else:
                    final_score = base_score
            else:
                final_score = base_score
                
            combined_scores.append((doc_id, final_score))
            
        # Sort by score
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        # More permissive thresholding for better recall
        if combined_scores:
            max_score = combined_scores[0][1]
            scores = [score for _, score in combined_scores]
            mean_score = sum(scores) / len(scores)
            std_dev = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
            
            # More permissive threshold
            threshold = min(
                max_score * 0.4,  # Much more permissive max threshold
                mean_score + 0.3 * std_dev  # More permissive statistical threshold
            )
            combined_scores = [(doc_id, score) for doc_id, score in combined_scores if score >= threshold]
        
        # More permissive diversity ranking
        diverse_results = []
        seen_prefixes = set()
        seen_types = {'DAO': 0, 'Action': 0, 'Controller': 0, 'Service': 0, 'Bean': 0, 'Validator': 0}
        
        for doc_id, score in combined_scores:
            if doc_id.endswith('.java'):
                parts = doc_id.split('.')
                if len(parts) > 1:
                    file_type = next((suffix for suffix in seen_types.keys() 
                                    if suffix in parts[0]), 'Other')
                    
                    # More permissive type diversity
                    if file_type != 'Other':
                        type_scores = [s for d, s in diverse_results if file_type in d.split('.')[0]]
                        if type_scores:
                            max_type_score = max(type_scores)
                            # Allow more results if scores are reasonably close
                            if seen_types[file_type] >= 3 and score < max_type_score * 0.75:  # More permissive
                                continue
                        seen_types[file_type] += 1
                    
                    # More permissive prefix diversity
                    prefix = re.sub(r'(DAO|Action|Controller|Service|Bean|Validator)$', '', parts[0])
                    if prefix in seen_prefixes:
                        prefix_scores = [s for d, s in diverse_results if prefix in d.split('.')[0]]
                        if prefix_scores:
                            max_prefix_score = max(prefix_scores)
                            # Allow more duplicates
                            if score < max_prefix_score * 0.8:  # More permissive
                                continue
                    seen_prefixes.add(prefix)
                
            diverse_results.append((doc_id, score))
            if len(diverse_results) >= top_k:
                break
                
        return diverse_results[:top_k]
        
    def save(self, filepath: str):
        """Save both indexes"""
        import os
        base_path = os.path.splitext(filepath)[0]
        self.keyword_index.save(f"{base_path}_keyword.pkl")
        self.vector_index.save(f"{base_path}_vector.pkl")
        
    def load(self, filepath: str):
        """Load both indexes"""
        import os
        base_path = os.path.splitext(filepath)[0]
        self.keyword_index.load(f"{base_path}_keyword.pkl")
        self.vector_index.load(f"{base_path}_vector.pkl") 