"""
Vector database service using Qdrant with hybrid retrieval (BM25 + vectors).
"""
import structlog
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
import numpy as np

from src.core.config import Settings

logger = structlog.get_logger()


class VectorService:
    """
    Vector database service with hybrid retrieval capabilities.
    """
    
    def __init__(self, qdrant_url: str, collection_name: str = "pdf_documents"):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None
        self.embedding_model: Optional[TextEmbedding] = None
        self.bm25_corpus: List[str] = []
        self.bm25_model: Optional[BM25Okapi] = None
        self.document_chunks: List[Dict[str, Any]] = []
        
    async def initialize(self):
        """Initialize the vector service."""
        logger.info("Initializing vector service", qdrant_url=self.qdrant_url)
        
        try:
            # Initialize Qdrant client
            self.client = QdrantClient(url=self.qdrant_url)
            
            # Initialize embedding model
            self.embedding_model = TextEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                max_length=512
            )
            
            # Create collection if it doesn't exist
            await self._create_collection_if_not_exists()
            
            # Load existing documents for BM25
            await self._load_existing_documents()
            
            logger.info("Vector service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize vector service", error=str(e))
            raise
    
    async def _create_collection_if_not_exists(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info("Creating new collection", collection=self.collection_name)
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # BAAI/bge-small-en-v1.5 dimension
                        distance=Distance.COSINE
                    )
                )
                
                logger.info("Collection created successfully")
            else:
                logger.info("Collection already exists")
                
        except Exception as e:
            logger.error("Failed to create collection", error=str(e))
            raise
    
    async def _load_existing_documents(self):
        """Load existing documents to rebuild BM25 index."""
        try:
            # Get all points from collection
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True
            )
            
            points = scroll_result[0]
            self.document_chunks = []
            self.bm25_corpus = []
            
            for point in points:
                chunk_data = {
                    "id": str(point.id),
                    "content": point.payload.get("content", ""),
                    "metadata": point.payload.get("metadata", {})
                }
                self.document_chunks.append(chunk_data)
                self.bm25_corpus.append(chunk_data["content"])
            
            # Rebuild BM25 index
            if self.bm25_corpus:
                tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
                self.bm25_model = BM25Okapi(tokenized_corpus)
                
                logger.info(
                    "Loaded existing documents",
                    num_chunks=len(self.document_chunks)
                )
            else:
                logger.info("No existing documents found")
                
        except Exception as e:
            logger.error("Failed to load existing documents", error=str(e))
            # Continue without existing documents
            self.document_chunks = []
            self.bm25_corpus = []
    
    async def add_documents(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Add document chunks to the vector database.
        
        Args:
            document_id: Unique document identifier
            chunks: List of document chunks with content and metadata
            
        Returns:
            bool: Success status
        """
        logger.info(
            "Adding document chunks",
            document_id=document_id,
            num_chunks=len(chunks)
        )
        
        try:
            # Generate embeddings
            texts = [chunk["content"] for chunk in chunks]
            embeddings = list(self.embedding_model.embed(texts))
            
            # Prepare points for Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = str(uuid.uuid4())
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "content": chunk["content"],
                        "document_id": document_id,
                        "chunk_index": i,
                        "metadata": {
                            **chunk.get("metadata", {}),
                            "created_at": datetime.utcnow().isoformat()
                        }
                    }
                )
                points.append(point)
                
                # Update local storage for BM25
                chunk_data = {
                    "id": point_id,
                    "content": chunk["content"],
                    "metadata": {
                        "document_id": document_id,
                        "chunk_index": i,
                        **chunk.get("metadata", {})
                    }
                }
                self.document_chunks.append(chunk_data)
                self.bm25_corpus.append(chunk["content"])
            
            # Upload to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            # Rebuild BM25 index
            tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
            self.bm25_model = BM25Okapi(tokenized_corpus)
            
            logger.info(
                "Document chunks added successfully",
                document_id=document_id,
                total_chunks=len(self.document_chunks)
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to add document chunks",
                document_id=document_id,
                error=str(e)
            )
            return False
    
    async def search_similar(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.7  # Weight for vector search (1-alpha for BM25)
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity and BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for vector search (0-1)
            
        Returns:
            List of similar documents with scores
        """
        logger.info("Performing hybrid search", query=query, top_k=top_k)
        
        try:
            if not self.document_chunks:
                logger.warning("No documents available for search")
                return []
            
            # Vector search
            vector_results = await self._vector_search(query, top_k * 2)
            
            # BM25 search
            bm25_results = await self._bm25_search(query, top_k * 2)
            
            # Combine results using RRF (Reciprocal Rank Fusion)
            combined_results = self._combine_results(
                vector_results,
                bm25_results,
                alpha,
                top_k
            )
            
            logger.info(
                "Hybrid search completed",
                num_results=len(combined_results)
            )
            
            return combined_results
            
        except Exception as e:
            logger.error("Hybrid search failed", error=str(e))
            return []
    
    async def _vector_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Perform vector similarity search."""
        try:
            # Generate query embedding
            query_embedding = list(self.embedding_model.embed([query]))[0]
            
            # Search in Qdrant
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True
            )
            
            # Return (document_id, score) tuples
            results = []
            for result in search_result:
                doc_id = str(result.id)
                score = result.score
                results.append((doc_id, score))
            
            return results
            
        except Exception as e:
            logger.error("Vector search failed", error=str(e))
            return []
    
    async def _bm25_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Perform BM25 search."""
        try:
            if self.bm25_model is None:
                return []
            
            # Tokenize query
            tokenized_query = query.split()
            
            # Get BM25 scores
            scores = self.bm25_model.get_scores(tokenized_query)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.document_chunks):
                    doc_id = self.document_chunks[idx]["id"]
                    score = float(scores[idx])
                    results.append((doc_id, score))
            
            return results
            
        except Exception as e:
            logger.error("BM25 search failed", error=str(e))
            return []
    
    def _combine_results(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        alpha: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Combine vector and BM25 results using weighted scoring."""
        try:
            # Normalize scores
            vector_scores = self._normalize_scores([score for _, score in vector_results])
            bm25_scores = self._normalize_scores([score for _, score in bm25_results])
            
            # Create score dictionaries
            vector_dict = {doc_id: score for (doc_id, _), score in 
                          zip(vector_results, vector_scores)}
            bm25_dict = {doc_id: score for (doc_id, _), score in 
                        zip(bm25_results, bm25_scores)}
            
            # Combine scores
            all_doc_ids = set(vector_dict.keys()) | set(bm25_dict.keys())
            combined_scores = {}
            
            for doc_id in all_doc_ids:
                vector_score = vector_dict.get(doc_id, 0.0)
                bm25_score = bm25_dict.get(doc_id, 0.0)
                combined_score = alpha * vector_score + (1 - alpha) * bm25_score
                combined_scores[doc_id] = combined_score
            
            # Sort by combined score
            sorted_results = sorted(
                combined_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]
            
            # Build final results
            final_results = []
            for doc_id, score in sorted_results:
                # Find document chunk
                chunk = next(
                    (chunk for chunk in self.document_chunks if chunk["id"] == doc_id),
                    None
                )
                
                if chunk:
                    final_results.append({
                        "id": doc_id,
                        "content": chunk["content"],
                        "score": score,
                        "metadata": chunk["metadata"]
                    })
            
            return final_results
            
        except Exception as e:
            logger.error("Failed to combine results", error=str(e))
            return []
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores or max(scores) == min(scores):
            return [0.0] * len(scores)
        
        min_score = min(scores)
        max_score = max(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            # Find points to delete
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=1000,
                with_payload=False
            )
            
            point_ids = [point.id for point in scroll_result[0]]
            
            if point_ids:
                # Delete from Qdrant
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=point_ids)
                )
                
                # Remove from local storage
                self.document_chunks = [
                    chunk for chunk in self.document_chunks
                    if chunk["metadata"].get("document_id") != document_id
                ]
                self.bm25_corpus = [chunk["content"] for chunk in self.document_chunks]
                
                # Rebuild BM25 index
                if self.bm25_corpus:
                    tokenized_corpus = [doc.split() for doc in self.bm25_corpus]
                    self.bm25_model = BM25Okapi(tokenized_corpus)
                else:
                    self.bm25_model = None
                
                logger.info(
                    "Document deleted successfully",
                    document_id=document_id,
                    deleted_chunks=len(point_ids)
                )
                
                return True
            else:
                logger.warning("No chunks found for document", document_id=document_id)
                return False
                
        except Exception as e:
            logger.error("Failed to delete document", document_id=document_id, error=str(e))
            return False
    
    async def get_document_count(self) -> int:
        """Get total number of document chunks."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logger.error("Failed to get document count", error=str(e))
            return 0
    
    async def close(self):
        """Close the vector service."""
        logger.info("Closing vector service")
        if self.client:
            self.client.close()