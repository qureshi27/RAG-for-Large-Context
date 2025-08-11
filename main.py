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
import shutil

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel, Field

# Document processing
import sys
print(sys.executable)
import pdfplumber
sys.path.append("C:/Users/jason/Desktop/RAG/RAG-System-with-Mistral-OCR-and-ChromaDB/")
# import PyMuPDF as fitz
from unstructured.partition.auto import partition
import spacy

# Text processing and embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Vector storage and search
import chromadb
from chromadb.config import Settings

# LLM integration
from openai import OpenAI
import mistralai

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Knowledgebase folder if it doesn't exist
KNOWLEDGEBASE_DIR = Path("Knowledgebase")
KNOWLEDGEBASE_DIR.mkdir(exist_ok=True)
PROCESSED_FILES_LOG = KNOWLEDGEBASE_DIR / "processed_files.log"

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
    chroma_db_path: str = "./chroma_db"
    chroma_collection_name: str = "document_chunks"

# NOTE: The following classes (DocumentProcessor, IntelligentChunker, MultiResolutionIndexer,
# AdvancedRetriever, ContextOptimizer) are identical to your provided code.
# They are collapsed here for brevity but are included in the full script below.
# ... (Paste all your original classes here without changes) ...
#<editor-fold desc="Original RAG Classes">
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
    """Implements hybrid search, query expansion, and reranking with ChromaDB"""
    
    def __init__(self, config: RAGConfig, openai_api_key: str):
        self.config = config
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.reranker = CrossEncoder(config.reranker_model)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_db_path,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.chroma_collection_name
        )
        logger.info(f"Initialized ChromaDB collection: {config.chroma_collection_name}")
        
        # Initialize Redis for caching
        try:
            self.redis_client = redis.Redis(
                host=config.redis_host,
                port=config.redis_port,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection successful.")
        except Exception as e:
            self.redis_client = None
            logger.warning(f"Redis connection failed, caching disabled: {e}")
            
    def build_index(self, chunks: List[Document], embeddings: np.ndarray, file_hashes: List[str]):
        """Build ChromaDB vector index, avoiding duplicates."""
        logger.info(f"Building index for {len(chunks)} new chunks")
        
        # Prepare data for ChromaDB
        documents, metadatas, ids, embeddings_list = [], [], [], []
        
        for i, (chunk, embedding, file_hash) in enumerate(zip(chunks, embeddings, file_hashes)):
            chunk_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
            chunk_id = f"chunk_{file_hash}_{i}_{chunk_hash}"
            
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            
            metadata = {
                "file_hash": file_hash,
                "chunk_index": str(i),
                **{k: str(v) for k, v in chunk.metadata.items()}
            }
            metadatas.append(metadata)
            embeddings_list.append(embedding.tolist())
        
        if not ids:
            logger.info("No new chunks to add to the index.")
            return

        try:
            self.collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
        except Exception as e:
            logger.error(f"ChromaDB indexing failed: {e}")
            raise
    
    def expand_query(self, query: str) -> List[str]:
        """Generate expanded queries using LLM"""
        cache_key = f"query_expansion:{hashlib.md5(query.encode()).hexdigest()}"
        
        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate 3 related queries that capture different aspects of the original query. Return as a JSON list of strings."},
                    {"role": "user", "content": f"Original query: {query}"}
                ],
                max_tokens=150
            )
            expanded_queries = json.loads(response.choices[0].message.content)
            
            if self.redis_client:
                self.redis_client.setex(cache_key, 3600, json.dumps(expanded_queries))
            
            return expanded_queries
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
    
    def vector_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Perform vector similarity search using ChromaDB"""
        k = k or self.config.top_k_retrieval
        
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            if results["documents"]:
                for doc_text, metadata, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                    doc = Document(page_content=doc_text, metadata=metadata)
                    similarity_score = 1.0 / (1.0 + distance)
                    search_results.append((doc, similarity_score))
            
            return search_results
        except Exception as e:
            logger.error(f"ChromaDB vector search failed: {e}")
            return []

    def rerank_results(self, query: str, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Rerank results using cross-encoder"""
        if not results:
            return []
        
        try:
            pairs = [(query, doc.page_content) for doc, _ in results]
            rerank_scores = self.reranker.predict(pairs)
            
            reranked_results = [(doc, float(rerank_score)) for (doc, _), rerank_score in zip(results, rerank_scores)]
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            return reranked_results[:self.config.top_k_rerank]
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results[:self.config.top_k_rerank]

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the ChromaDB collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.config.chroma_collection_name,
                "embedding_model": self.config.embedding_model
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def reset_collection(self):
        """Delete and recreate the ChromaDB collection"""
        try:
            self.chroma_client.delete_collection(name=self.config.chroma_collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.config.chroma_collection_name,
                metadata={"description": "Document chunks for RAG system"}
            )
            logger.info(f"Successfully reset ChromaDB collection: {self.config.chroma_collection_name}")
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")

class ContextOptimizer:
    """Handles context compression and iterative retrieval"""
    
    def __init__(self, config: RAGConfig, openai_api_key: str):
        self.config = config
        self.openai_client = OpenAI(api_key=openai_api_key)
    
    def compress_context(self, query: str, contexts: List[str]) -> str:
        """Extract only relevant snippets from retrieved contexts"""
        if not contexts:
            return ""
        try:
            combined_context = "\n---\n".join(contexts)
            
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
#</editor-fold>

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
        
        self.processed_hashes = self._load_processed_hashes()
        self.is_indexed = self.retriever.get_collection_stats().get("total_chunks", 0) > 0

    def _load_processed_hashes(self) -> set:
        """Load the set of hashes of already processed files."""
        if not PROCESSED_FILES_LOG.exists():
            return set()
        with open(PROCESSED_FILES_LOG, "r") as f:
            return set(line.strip() for line in f)

    def _mark_file_as_processed(self, file_hash: str):
        """Add a file hash to the processed log."""
        self.processed_hashes.add(file_hash)
        with open(PROCESSED_FILES_LOG, "a") as f:
            f.write(f"{file_hash}\n")

    def process_documents(self, file_paths: List[str], use_semantic_chunking: bool = False) -> int:
        """Process a list of documents and update the index."""
        all_chunks = []
        all_file_hashes = []
        new_files_processed = 0

        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}. Skipping.")
                continue

            with open(path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash in self.processed_hashes:
                logger.info(f"Skipping already processed file: {path.name}")
                continue

            logger.info(f"Processing new document: {path.name}")
            
            text = self.document_processor.extract_with_mistral_ocr(file_path)
            structure = self.document_processor.extract_document_structure(file_path)
            
            if use_semantic_chunking:
                chunks = self.chunker.semantic_chunk(text)
            else:
                chunks = self.chunker.hierarchical_chunk(text, structure)
            
            for chunk in chunks:
                chunk.metadata['source'] = path.name
            
            all_chunks.extend(chunks)
            all_file_hashes.extend([file_hash] * len(chunks))
            self._mark_file_as_processed(file_hash)
            new_files_processed += 1

        if not all_chunks:
            logger.info("No new documents to process.")
            return 0
            
        logger.info(f"Created {len(all_chunks)} chunks from {new_files_processed} new documents.")
        
        embeddings = self.indexer.create_embeddings(all_chunks)
        self.retriever.build_index(all_chunks, embeddings, all_file_hashes)
        
        self.is_indexed = True
        logger.info("Document processing and indexing complete.")
        return new_files_processed

    def query(self, question: str, use_iterative_retrieval: bool = False) -> Dict[str, Any]:
        """Process a query and generate response"""
        if not self.is_indexed:
            raise ValueError("No documents have been indexed yet.")
        
        logger.info(f"Processing query: {question}")
        
        expanded_queries = self.retriever.expand_query(question)
        
        # Combine search results from original and expanded queries
        all_results = set()
        for q in [question] + expanded_queries:
            results = self.retriever.vector_search(q)
            for doc, score in results:
                # Use page content as a unique identifier for the document
                all_results.add((doc.page_content, doc.metadata.get('source', 'unknown')))

        # Convert back to list of Document objects for reranking
        unique_docs = [Document(page_content=content, metadata={'source': source}) for content, source in all_results]
        
        reranked_results = self.retriever.rerank_results(question, [(doc, 0.0) for doc in unique_docs]) # Pass dummy scores
        
        contexts = [doc.page_content for doc, _ in reranked_results]
        sources = [doc.metadata.get('source', 'unknown') for doc, _ in reranked_results]
        
        compressed_context = self.context_optimizer.compress_context(question, contexts)
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the question based ONLY on the provided context. Cite the source document for each piece of information used."},
                    {"role": "user", "content": f"Context:\n{compressed_context}\n\nQuestion: {question}\n\nAnswer:"}
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
            "context": compressed_context,
            "sources": list(set(sources)),
            "expanded_queries": expanded_queries,
        }

# --- FASTAPI APPLICATION ---

# API Keys - Replace with your actual keys, preferably from environment variables
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "your_mistral_api_key")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")

# Initialize configuration and the main RAG pipeline
config = RAGConfig(
    chunk_size=600,
    chunk_overlap=60,
    top_k_retrieval=15,
    top_k_rerank=5,
    chroma_db_path="./my_rag_db",
)
rag_pipeline = RAGPipeline(config, MISTRAL_API_KEY, OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI(
    title="Advanced RAG Pipeline API",
    description="An API for processing documents and answering questions using a sophisticated RAG pipeline.",
    version="1.0.0"
)

# Pydantic Models for API requests and responses
class QueryRequest(BaseModel):
    question: str = Field(..., example="What are the main findings discussed in chapter 3?")
    use_iterative_retrieval: bool = Field(False, description="Use iterative retrieval for complex queries.")

class QueryResponse(BaseModel):
    question: str
    answer: str
    context: str
    sources: List[str]
    expanded_queries: List[str]

class StatusResponse(BaseModel):
    is_indexed: bool
    collection_stats: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """On startup, process any new documents in the Knowledgebase folder."""
    logger.info("Application startup: Checking for new documents in Knowledgebase...")
    knowledge_base_files = [str(f) for f in KNOWLEDGEBASE_DIR.glob("*.pdf")]
    if knowledge_base_files:
        await asyncio.to_thread(rag_pipeline.process_documents, knowledge_base_files)
    logger.info("Initial document processing complete.")


@app.post("/process-documents/", summary="Upload and Process Documents")
async def process_documents_endpoint(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDF documents. The system will also scan the 'Knowledgebase'
    folder for any new documents and process them all.
    """
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    uploaded_file_paths = []
    for file in files:
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        uploaded_file_paths.append(str(file_path))

    # Also check the knowledgebase directory for new files
    knowledge_base_files = [str(f) for f in KNOWLEDGEBASE_DIR.glob("*.pdf")]
    all_files_to_process = list(set(uploaded_file_paths + knowledge_base_files))
    
    try:
        processed_count = await asyncio.to_thread(rag_pipeline.process_documents, all_files_to_process)
        return {
            "message": f"Successfully processed {processed_count} new documents.",
            "total_chunks_in_db": rag_pipeline.retriever.get_collection_stats().get("total_chunks")
        }
    except Exception as e:
        logger.error(f"Error during document processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {e}")
    finally:
        # Clean up temporary uploaded files
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/query/", response_model=QueryResponse, summary="Query the RAG System")
async def query_endpoint(request: QueryRequest):
    """
    Ask a question to the RAG system. The system will retrieve relevant context
    from the indexed documents and generate a comprehensive answer.
    """
    if not rag_pipeline.is_indexed:
        raise HTTPException(status_code=400, detail="No documents have been processed. Please upload documents first.")
    
    try:
        result = await asyncio.to_thread(
            rag_pipeline.query,
            request.question,
            request.use_iterative_retrieval
        )
        return result
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during the query: {e}")

@app.get("/status/", response_model=StatusResponse, summary="Get System Status")
async def get_status():
    """
    Retrieve the current status of the RAG system, including whether it has
    been indexed and statistics about the document collection.
    """
    stats = await asyncio.to_thread(rag_pipeline.retriever.get_collection_stats)
    return {
        "is_indexed": rag_pipeline.is_indexed,
        "collection_stats": stats
    }

@app.post("/reset-index/", summary="Reset the Document Index")
async def reset_index():
    """
    Delete all data from the vector store. This will require reprocessing
    all documents. Also clears the processed files log.
    """
    try:
        await asyncio.to_thread(rag_pipeline.retriever.reset_collection)
        # Clear the log of processed files
        if PROCESSED_FILES_LOG.exists():
            os.remove(PROCESSED_FILES_LOG)
        rag_pipeline.processed_hashes.clear()
        rag_pipeline.is_indexed = False
        return {"message": "Successfully reset the document index and cleared the processed files log."}
    except Exception as e:
        logger.error(f"Error during index reset: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while resetting the index: {e}")

if __name__ == "__main__":
    import uvicorn
    # To run the server, execute `uvicorn main:app --reload` in your terminal
    # where 'main' is the name of your Python file.
    uvicorn.run(app, host="0.0.0.0", port=8000)