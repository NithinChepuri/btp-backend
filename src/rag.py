"""
Retrieval Augmented Generation (RAG) Module for combining different indexes
and handling the retrieval augmented generation process.
"""
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .indexer import KeywordIndex, VectorIndex, KnowledgeGraphIndex

@dataclass
class RetrievalResult:
    """Data class for storing retrieval results"""
    doc_id: str
    score: float
    content: str
    source: str  # Which index this result came from

class RAGSystem:
    """RAG system combining multiple indexes for enhanced retrieval"""
    
    def __init__(
        self,
        keyword_index: Optional[KeywordIndex] = None,
        vector_index: Optional[VectorIndex] = None,
        knowledge_graph_index: Optional[KnowledgeGraphIndex] = None,
        llm_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    ):
        """
        Initialize RAG system with different indexes
        
        Args:
            keyword_index: Keyword-based index
            vector_index: Vector-based index
            knowledge_graph_index: Knowledge graph-based index
            llm_model_name: Name of the LLM model to use
        """
        self.keyword_index = keyword_index
        self.vector_index = vector_index
        self.knowledge_graph_index = knowledge_graph_index
        
        # Initialize LLM
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(self.device)
        
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents from all indexes
        
        Args:
            query: Search query
            top_k: Number of top results to return from each index
            min_score: Minimum relevance score threshold
            
        Returns:
            List of retrieval results from all indexes
        """
        results = []
        
        # Get results from keyword index
        if self.keyword_index:
            keyword_results = self.keyword_index.search(query, top_k)
            for result in keyword_results:
                if result['score'] >= min_score:
                    results.append(RetrievalResult(
                        doc_id=result['doc_id'],
                        score=result['score'],
                        content=result['content'],
                        source='keyword'
                    ))
                    
        # Get results from vector index
        if self.vector_index:
            vector_results = self.vector_index.search(query, top_k)
            for result in vector_results:
                if result['score'] >= min_score:
                    results.append(RetrievalResult(
                        doc_id=result['doc_id'],
                        score=result['score'],
                        content=result['content'],
                        source='vector'
                    ))
                    
        # Get results from knowledge graph index
        if self.knowledge_graph_index:
            graph_results = self.knowledge_graph_index.search(query, top_k)
            for result in graph_results:
                if result['score'] >= min_score:
                    results.append(RetrievalResult(
                        doc_id=result['doc_id'],
                        score=result['score'],
                        content=result['content'],
                        source='graph'
                    ))
                    
        # Sort by score and remove duplicates
        results.sort(key=lambda x: x.score, reverse=True)
        unique_results = []
        seen_docs = set()
        for result in results:
            if result.doc_id not in seen_docs:
                unique_results.append(result)
                seen_docs.add(result.doc_id)
                
        return unique_results[:top_k]
    
    def generate_prompt(self, query: str, retrieved_docs: List[RetrievalResult]) -> str:
        """
        Generate a prompt for the LLM using the query and retrieved documents
        
        Args:
            query: Original query
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted prompt string
        """
        context = "\n\n".join([
            f"Document {i+1} (Source: {doc.source}, Score: {doc.score:.2f}):\n{doc.content}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        prompt = f"""Given the following context documents and query, identify the most relevant code classes and explain their relationship to the query.

Context:
{context}

Query: {query}

Please provide the names of the relevant classes in a JSON list format and explain why each class is relevant."""
        
        return prompt
    
    def generate_response(self, prompt: str, max_length: int = 1000) -> str:
        """
        Generate a response using the LLM
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of the generated response
            
        Returns:
            Generated response string
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=4,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            early_stopping=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def process_query(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline
        
        Args:
            query: Input query
            top_k: Number of top results to retrieve
            min_score: Minimum relevance score threshold
            
        Returns:
            Dictionary containing retrieved documents and generated response
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k, min_score)
        
        # Generate prompt
        prompt = self.generate_prompt(query, retrieved_docs)
        
        # Generate response
        response = self.generate_response(prompt)
        
        return {
            'retrieved_documents': [
                {
                    'doc_id': doc.doc_id,
                    'score': doc.score,
                    'source': doc.source
                }
                for doc in retrieved_docs
            ],
            'response': response
        } 