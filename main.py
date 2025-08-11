import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import redis
from pathlib import Path

# Document processing
import pdfplumber
import PyMuPDF as fitz
from unstructured.partition.auto import partition
import spacy

# Text processing and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Vector storage and search
import faiss
from elasticsearch import Elasticsearch
import weaviate

# LLM integration
from openai import OpenAI
import mistralai

# Evaluation
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    vector_dim: int = 384
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    redis_host: str = "localhost"
    redis_port: int = 6379
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200

class DocumentProcessor:
    """Handles document loading and initial processing with Mistral OCR integration"""
    
    def __init__(self, mistral_api_key: str):
        self.mistral_client = mistralai.Mistral(api_key=mistral_api_key)
        self.nlp = spacy.load("en_core_web_sm")
    
    def extract_with_mistral_ocr(self, file_path: str) -> str:
        """Extract text using Mistral OCR capabilities"""
        try:
            # For demonstration - in practice, you'd use Mistral's vision/OCR API
            with open(file_path, 'rb') as file:
                # This would be replaced with actual Mistral OCR API call
                # For now, using traditional PDF extraction as fallback
                return self._extract_pdf_text(file_path)
        except Exception as e:
            logger.error(f"Mistral OCR extraction failed: {e}")
            return self._extract_pdf_text(file_path)
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Fallback PDF text extraction"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
        return text
    
    def extract_document_structure(self, file_path: str) -> Dict[str, Any]:
        """Extract hierarchical structure from PDF"""
        structure = {"sections": [], "metadata": {}}
        
        try:
            doc = fitz.open(file_path)
            toc = doc.get_toc()  # Table of contents
            
            for level, title, page in toc:
                structure["sections"].append({
                    "level": level,
                    "title": title,
                    "page": page,
                    "content": ""
                })
            
            structure["metadata"] = {
                "total_pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", "")
            }
            
            doc.close()
        except Exception as e:
            logger.error(f"Structure extraction failed: {e}")
        
        return structure

class IntelligentChunker:
    """Implements hierarchical and semantic chunking strategies"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "]
        )
        self.nlp = spacy.load("en_core_web_sm")
    
    def hierarchical_chunk(self, text: str, structure: Dict[str, Any]) -> List[Document]:
        """Create hierarchical chunks based on document structure"""
        chunks = []
        
        if structure["sections"]:
            # Chunk by sections first
            for section in structure["sections"]:
                section_text = section.get("content", "")
                if section_text:
                    section_chunks = self.text_splitter.split_text(section_text)
                    
                    for i, chunk in enumerate(section_chunks):
                        metadata = {
                            "section_title": section["title"],
                            "section_level": section["level"],
                            "chunk_index": i,
                            "page": section.get("page", 0)
                        }
                        chunks.append(Document(page_content=chunk, metadata=metadata))
        else:
            # Fallback to simple chunking
            simple_chunks = self.text_splitter.split_text(text)
            for i, chunk in enumerate(simple_chunks):
                metadata = {"chunk_index": i}
                chunks.append(Document(page_content=chunk, metadata=metadata))
        
        return chunks
    
    def semantic_chunk(self, text: str) -> List[Document]:
        """Create chunks based on semantic boundaries"""
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        # Group sentences into semantic chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.config.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={"type": "semantic", "sentence_count": len(current_chunk)}
                ))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Document(
                page_content=chunk_text,
                metadata={"type": "semantic", "sentence_count": len(current_chunk)}
            ))
        
        return chunks

class MultiResolutionIndexer:
    """Handles hierarchical summaries and metadata enrichment"""
    
    def __init__(self, config: RAGConfig, openai_api_key: str):
        self.config = config
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = SentenceTransformer(config.embedding_model)
    
    def generate_hierarchical_summaries(self, chunks: List[Document]) -> Dict[str, str]:
        """Generate summaries at different levels"""
        summaries = {}
        
        # Group chunks by section
        sections = {}
        for chunk in chunks:
            section_title = chunk.metadata.get("section_title", "default")
            if section_title not in sections:
                sections[section_title] = []
            sections[section_title].append(chunk.page_content)
        
        # Generate section summaries
        for section_title, section_chunks in sections.items():
            combined_text = "\n".join(section_chunks)
            
            if len(combined_text) > 100:  # Only summarize substantial sections
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=[
                            {"role": "system", "content": "Create a concise summary of the following text section."},
                            {"role": "user", "content": combined_text[:8000]}  # Limit context
                        ],
                        max_tokens=200
                    )
                    summaries[section_title] = response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Summary generation failed for {section_title}: {e}")
                    summaries[section_title] = combined_text[:300] + "..."
        
        return summaries
    
    def create_embeddings(self, chunks: List[Document]) -> np.ndarray:
        """Create embeddings for chunks"""
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings

class AdvancedRetriever:
    """Implements hybrid search, query expansion, and reranking"""
    
    def __init__(self, config: RAGConfig, openai_api_key: str):
        self.config = config
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.reranker = CrossEncoder(config.reranker_model)
        
        # Initialize vector store
        self.index = faiss.IndexFlatIP(config.vector_dim)
        
        # Initialize Redis for caching
        try:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                decode_responses=True
            )
        except:
            self.redis_client = None
            logger.warning("Redis connection failed, caching disabled")
        
        # Initialize Elasticsearch for keyword search
        try:
            self.es_client = Elasticsearch([
                f"http://{config.elasticsearch_host}:{config.elasticsearch_port}"
            ])
        except:
            self.es_client = None
            logger.warning("Elasticsearch connection failed, keyword search disabled")
    
    def build_index(self, chunks: List[Document], embeddings: np.ndarray):
        """Build vector and keyword indices"""
        # Add to vector index
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        self.chunks = chunks
        
        # Add to Elasticsearch for keyword search
        if self.es_client:
            try:
                # Create index if not exists
                if not self.es_client.indices.exists(index="documents"):
                    self.es_client.indices.create(
                        index="documents",
                        body={
                            "mappings": {
                                "properties": {
                                    "content": {"type": "text"},
                                    "metadata": {"type": "object"}
                                }
                            }
                        }
                    )
                
                # Index documents
                for i, chunk in enumerate(chunks):
                    self.es_client.index(
                        index="documents",
                        id=i,
                        body={
                            "content": chunk.page_content,
                            "metadata": chunk.metadata
                        }
                    )
            except Exception as e:
                logger.error(f"Elasticsearch indexing failed: {e}")
    
    def expand_query(self, query: str) -> List[str]:
        """Generate expanded queries using LLM"""
        cache_key = f"query_expansion:{hashlib.md5(query.encode()).hexdigest()}"
        
        # Check cache
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate 3 related queries that capture different aspects of the original query. Return as a JSON list."},
                    {"role": "user", "content": f"Original query: {query}"}
                ],
                max_tokens=150
            )
            
            expanded_queries = json.loads(response.choices[0].message.content)
            
            # Cache result
            if self.redis_client:
                self.redis_client.setex(cache_key, 3600, json.dumps(expanded_queries))
            
            return expanded_queries
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
    
    def vector_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Perform vector similarity search"""
        k = k or self.config.top_k_retrieval
        
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def keyword_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Perform keyword search using Elasticsearch"""
        if not self.es_client:
            return []
        
        k = k or self.config.top_k_retrieval
        
        try:
            response = self.es_client.search(
                index="documents",
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["content"],
                            "type": "best_fields"
                        }
                    },
                    "size": k
                }
            )
            
            results = []
            for hit in response['hits']['hits']:
                doc_id = int(hit['_id'])
                score = hit['_score']
                if doc_id < len(self.chunks):
                    results.append((self.chunks[doc_id], score))
            
            return results
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def hybrid_search(self, query: str) -> List[Tuple[Document, float]]:
        """Combine vector and keyword search results"""
        vector_results = self.vector_search(query)
        keyword_results = self.keyword_search(query)
        
        # Normalize and combine scores
        combined_results = {}
        
        # Add vector results
        for doc, score in vector_results:
            doc_key = doc.page_content[:100]  # Use content snippet as key
            combined_results[doc_key] = {
                'doc': doc,
                'vector_score': score,
                'keyword_score': 0.0
            }
        
        # Add keyword results
        for doc, score in keyword_results:
            doc_key = doc.page_content[:100]
            if doc_key in combined_results:
                combined_results[doc_key]['keyword_score'] = score
            else:
                combined_results[doc_key] = {
                    'doc': doc,
                    'vector_score': 0.0,
                    'keyword_score': score
                }
        
        # Combine scores (weighted average)
        final_results = []
        for doc_data in combined_results.values():
            combined_score = (
                0.7 * doc_data['vector_score'] + 
                0.3 * doc_data['keyword_score']
            )
            final_results.append((doc_data['doc'], combined_score))
        
        # Sort by combined score
        final_results.sort(key=lambda x: x[1], reverse=True)
        return final_results[:self.config.top_k_retrieval]
    
    def rerank_results(self, query: str, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Rerank results using cross-encoder"""
        if not results:
            return results
        
        try:
            # Prepare pairs for reranking
            pairs = [(query, doc.page_content) for doc, _ in results]
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Combine with original results
            reranked_results = []
            for (doc, original_score), rerank_score in zip(results, rerank_scores):
                reranked_results.append((doc, float(rerank_score)))
            
            # Sort by rerank score and return top k
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            return reranked_results[:self.config.top_k_rerank]
        
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:self.config.top_k_rerank]

class ContextOptimizer:
    """Handles context compression and iterative retrieval"""
    
    def __init__(self, config: RAGConfig, openai_api_key: str):
        self.config = config
        self.openai_client = OpenAI(api_key=openai_api_key)
    
    def compress_context(self, query: str, contexts: List[str]) -> str:
        """Extract only relevant snippets from retrieved contexts"""
        try:
            combined_context = "\n".join(contexts)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract only the most relevant information from the following contexts that directly answers the query. Keep the essential details but remove redundant information."
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nContexts:\n{combined_context}"
                    }
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Context compression failed: {e}")
            return "\n".join(contexts)
    
    def iterative_retrieval(self, query: str, retriever: AdvancedRetriever, max_iterations: int = 2) -> str:
        """Perform iterative retrieval for complex queries"""
        current_query = query
        all_contexts = []
        
        for iteration in range(max_iterations):
            # Retrieve with current query
            results = retriever.hybrid_search(current_query)
            reranked_results = retriever.rerank_results(current_query, results)
            
            contexts = [doc.page_content for doc, _ in reranked_results]
            all_contexts.extend(contexts)
            
            if iteration < max_iterations - 1:
                # Generate refined query based on partial answer
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "Based on the retrieved information, generate a refined query that would help find additional relevant information."
                            },
                            {
                                "role": "user",
                                "content": f"Original query: {query}\nRetrieved info: {contexts[0] if contexts else ''}"
                            }
                        ],
                        max_tokens=100
                    )
                    current_query = response.choices[0].message.content
                except Exception as e:
                    logger.error(f"Query refinement failed: {e}")
                    break
        
        # Compress final context
        return self.compress_context(query, all_contexts)

class RAGPipeline:
    """Main RAG pipeline orchestrating all components"""
    
    def __init__(self, config: RAGConfig, mistral_api_key: str, openai_api_key: str):
        self.config = config
        self.document_processor = DocumentProcessor(mistral_api_key)
        self.chunker = IntelligentChunker(config)
        self.indexer = MultiResolutionIndexer(config, openai_api_key)
        self.retriever = AdvancedRetriever(config, openai_api_key)
        self.context_optimizer = ContextOptimizer(config, openai_api_key)
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        self.chunks = []
        self.summaries = {}
        self.is_indexed = False
    
    def process_document(self, file_path: str, use_semantic_chunking: bool = False) -> None:
        """Process a document and build the index"""
        logger.info(f"Processing document: {file_path}")
        
        # Extract text and structure
        text = self.document_processor.extract_with_mistral_ocr(file_path)
        structure = self.document_processor.extract_document_structure(file_path)
        
        # Create chunks
        if use_semantic_chunking:
            self.chunks = self.chunker.semantic_chunk(text)
        else:
            self.chunks = self.chunker.hierarchical_chunk(text, structure)
        
        logger.info(f"Created {len(self.chunks)} chunks")
        
        # Generate hierarchical summaries
        self.summaries = self.indexer.generate_hierarchical_summaries(self.chunks)
        
        # Create embeddings and build index
        embeddings = self.indexer.create_embeddings(self.chunks)
        self.retriever.build_index(self.chunks, embeddings)
        
        self.is_indexed = True
        logger.info("Document processing completed")
    
    def query(self, question: str, use_iterative_retrieval: bool = False) -> Dict[str, Any]:
        """Process a query and generate response"""
        if not self.is_indexed:
            raise ValueError("No document has been processed. Call process_document() first.")
        
        logger.info(f"Processing query: {question}")
        
        # Expand query
        expanded_queries = self.retriever.expand_query(question)
        
        if use_iterative_retrieval:
            # Use iterative retrieval
            context = self.context_optimizer.iterative_retrieval(question, self.retriever)
        else:
            # Standard retrieval
            results = self.retriever.hybrid_search(question)
            reranked_results = self.retriever.rerank_results(question, results)
            contexts = [doc.page_content for doc, _ in reranked_results]
            context = self.context_optimizer.compress_context(question, contexts)
        
        # Generate final answer
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context. Use only the information from the context to answer questions. If the context doesn't contain enough information, say so."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                    }
                ],
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = "I apologize, but I encountered an error generating the response."
        
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "expanded_queries": expanded_queries,
            "num_chunks_used": len(context.split("\n")),
        }
    
    def evaluate_performance(self, test_queries: List[Dict[str, str]]) -> Dict[str, float]:
        """Evaluate RAG performance using RAGAS metrics"""
        try:
            # Prepare evaluation data
            questions = [q["question"] for q in test_queries]
            ground_truths = [q["answer"] for q in test_queries]
            
            # Generate answers
            answers = []
            contexts = []
            
            for query_data in test_queries:
                result = self.query(query_data["question"])
                answers.append(result["answer"])
                contexts.append([result["context"]])  # RAGAS expects list of contexts
            
            # Create dataset for RAGAS
            dataset = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truths": ground_truths
            }
            
            # Evaluate
            result = evaluate(
                dataset,
                metrics=[answer_relevancy, context_recall, faithfulness]
            )
            
            return {
                "answer_relevancy": result["answer_relevancy"],
                "context_recall": result["context_recall"],
                "faithfulness": result["faithfulness"]
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Configuration
    config = RAGConfig(
        chunk_size=600,
        chunk_overlap=60,
        top_k_retrieval=15,
        top_k_rerank=5
    )
    
    # Initialize pipeline
    rag = RAGPipeline(
        config=config,
        mistral_api_key="your_mistral_api_key",
        openai_api_key="your_openai_api_key"
    )
    
    # Process document
    document_path = "path/to/your/large_document.pdf"
    rag.process_document(document_path, use_semantic_chunking=True)
    
    # Query the system
    result = rag.query(
        "What are the main findings discussed in chapter 3?",
        use_iterative_retrieval=True
    )
    
    print("Question:", result["question"])
    print("Answer:", result["answer"])
    print("Context used:", len(result["context"]), "characters")
    print("Expanded queries:", result["expanded_queries"])
    
    # Evaluate performance (optional)
    test_queries = [
        {
            "question": "What are the main findings?",
            "answer": "Expected answer from document..."
        }
    ]
    
    performance = rag.evaluate_performance(test_queries)
    print("Performance metrics:", performance)