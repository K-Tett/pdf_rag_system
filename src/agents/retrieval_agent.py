"""
Document retrieval agent with reranking capabilities.
"""
import structlog
from typing import Dict, Any, List
from langchain.schema import BaseMessage
from sentence_transformers import CrossEncoder
import asyncio

from src.services.vector_service import VectorService
from src.core.config import Settings

logger = structlog.get_logger()


class RetrievalAgent:
    """
    Agent responsible for retrieving and reranking relevant documents.
    """
    
    def __init__(
        self,
        vector_service: VectorService,
        primary_llm,
        fallback_llm=None,
        settings: Settings = None
    ):
        self.vector_service = vector_service
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.settings = settings or Settings()
        
        # Initialize reranker
        self.reranker = None
        self._initialize_reranker()
    
    def _initialize_reranker(self):
        """Initialize the reranking model."""
        try:
            self.reranker = CrossEncoder(
                self.settings.RERANK_MODEL,
                max_length=512
            )
            logger.info("Reranker initialized", model=self.settings.RERANK_MODEL)
        except Exception as e:
            logger.warning("Failed to initialize reranker", error=str(e))
            self.reranker = None
    
    async def process(
        self,
        question: str,
        conversation_history: List[BaseMessage] = None
    ) -> Dict[str, Any]:
        """
        Retrieve and rerank relevant documents for a question.
        
        Args:
            question: The user question
            conversation_history: Previous conversation messages for context
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        logger.info("Starting document retrieval", question=question)
        
        try:
            # Enhance query with conversation context if available
            enhanced_query = self._enhance_query_with_context(question, conversation_history)
            
            # Retrieve documents using hybrid search
            retrieved_docs = await self.vector_service.search_similar(
                query=enhanced_query,
                top_k=self.settings.RETRIEVAL_TOP_K,
                alpha=0.7  # Weight for vector search
            )
            
            logger.info(
                "Initial retrieval complete",
                num_docs=len(retrieved_docs),
                query=enhanced_query
            )
            
            if not retrieved_docs:
                return {
                    "documents": [],
                    "query_used": enhanced_query,
                    "retrieval_metadata": {
                        "initial_count": 0,
                        "reranked_count": 0,
                        "reranking_enabled": False
                    }
                }
            
            # Rerank documents if reranker is available
            if self.reranker and len(retrieved_docs) > 1:
                reranked_docs = await self._rerank_documents(question, retrieved_docs)
                final_docs = reranked_docs[:self.settings.RERANK_TOP_K]
                
                logger.info(
                    "Document reranking complete",
                    original_count=len(retrieved_docs),
                    reranked_count=len(final_docs)
                )
            else:
                final_docs = retrieved_docs[:self.settings.RERANK_TOP_K]
                logger.info("Skipping reranking", reason="reranker not available" if not self.reranker else "insufficient documents")
            
            # Add retrieval scores and format for synthesis
            formatted_docs = self._format_documents(final_docs)
            
            return {
                "documents": formatted_docs,
                "query_used": enhanced_query,
                "retrieval_metadata": {
                    "initial_count": len(retrieved_docs),
                    "reranked_count": len(final_docs),
                    "reranking_enabled": self.reranker is not None,
                    "search_type": "hybrid"
                }
            }
            
        except Exception as e:
            logger.error("Document retrieval failed", error=str(e))
            return {
                "documents": [],
                "query_used": question,
                "retrieval_metadata": {
                    "error": str(e),
                    "initial_count": 0,
                    "reranked_count": 0,
                    "reranking_enabled": False
                }
            }
    
    def _enhance_query_with_context(
        self,
        question: str,
        conversation_history: List[BaseMessage] = None
    ) -> str:
        """
        Enhance the query with conversation context.
        
        Args:
            question: Original question
            conversation_history: Previous messages
            
        Returns:
            Enhanced query string
        """
        if not conversation_history or len(conversation_history) < 2:
            return question
        
        try:
            # Get the last few exchanges for context
            recent_messages = conversation_history[-4:]  # Last 2 exchanges
            
            # Extract key terms from recent conversation
            context_terms = []
            for message in recent_messages:
                content = message.content.lower()
                
                # Extract potential key terms (simple approach)
                words = content.split()
                key_words = [
                    word for word in words 
                    if len(word) > 4 and word.isalpha()
                    and word not in {'question', 'answer', 'please', 'could', 'would', 'should'}
                ][:3]  # Top 3 key words per message
                
                context_terms.extend(key_words)
            
            # Add unique context terms to the query
            unique_terms = list(set(context_terms))[:5]  # Limit to 5 terms
            
            if unique_terms:
                enhanced_query = f"{question} {' '.join(unique_terms)}"
                logger.debug("Enhanced query with context", 
                           original=question, 
                           enhanced=enhanced_query,
                           context_terms=unique_terms)
                return enhanced_query
            
        except Exception as e:
            logger.warning("Failed to enhance query with context", error=str(e))
        
        return question
    
    async def _rerank_documents(
        self,
        question: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder model.
        
        Args:
            question: User question
            documents: List of retrieved documents
            
        Returns:
            Reranked list of documents
        """
        try:
            # Prepare query-document pairs
            pairs = []
            for doc in documents:
                content = doc.get("content", "")
                # Truncate content if too long
                if len(content) > 1000:
                    content = content[:1000] + "..."
                pairs.append([question, content])
            
            # Get reranking scores
            scores = self.reranker.predict(pairs)
            
            # Combine documents with scores and sort
            scored_docs = []
            for doc, score in zip(documents, scores):
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = float(score)
                scored_docs.append(doc_copy)
            
            # Sort by rerank score (descending)
            reranked_docs = sorted(
                scored_docs,
                key=lambda x: x["rerank_score"],
                reverse=True
            )
            
            logger.debug(
                "Documents reranked",
                top_scores=[doc["rerank_score"] for doc in reranked_docs[:3]]
            )
            
            return reranked_docs
            
        except Exception as e:
            logger.error("Reranking failed", error=str(e))
            # Return original documents if reranking fails
            return documents
    
    def _format_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format documents for synthesis agent.
        
        Args:
            documents: Raw retrieved documents
            
        Returns:
            Formatted documents with standardized structure
        """
        formatted = []
        
        for i, doc in enumerate(documents):
            formatted_doc = {
                "id": doc.get("id", f"doc_{i}"),
                "content": doc.get("content", ""),
                "title": self._extract_title(doc),
                "source": "pdf_document",
                "relevance_score": doc.get("score", 0.0),
                "rerank_score": doc.get("rerank_score"),
                "metadata": {
                    "document_id": doc.get("metadata", {}).get("document_id"),
                    "chunk_index": doc.get("metadata", {}).get("chunk_index"),
                    "filename": doc.get("metadata", {}).get("filename"),
                    "author": doc.get("metadata", {}).get("author"),
                    **doc.get("metadata", {})
                }
            }
            
            formatted.append(formatted_doc)
        
        return formatted
    
    def _extract_title(self, document: Dict[str, Any]) -> str:
        """
        Extract or generate a title for the document.
        
        Args:
            document: Document dictionary
            
        Returns:
            Document title
        """
        metadata = document.get("metadata", {})
        
        # Try to get title from metadata
        if metadata.get("title"):
            return metadata["title"]
        
        # Try to get filename
        if metadata.get("filename"):
            filename = metadata["filename"]
            # Remove extension and clean up
            title = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
            return title.title()
        
        # Try to extract from content (first line that looks like a title)
        content = document.get("content", "")
        lines = content.split("\n")[:5]  # Check first 5 lines
        
        for line in lines:
            line = line.strip()
            if 10 < len(line) < 100 and not line.startswith(("Page", "Figure", "Table")):
                return line
        
        # Fallback
        return "Document Extract"
    
    async def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        try:
            doc_count = await self.vector_service.get_document_count()
            
            return {
                "total_documents": doc_count,
                "reranker_available": self.reranker is not None,
                "reranker_model": self.settings.RERANK_MODEL if self.reranker else None,
                "retrieval_top_k": self.settings.RETRIEVAL_TOP_K,
                "rerank_top_k": self.settings.RERANK_TOP_K
            }
        except Exception as e:
            logger.error("Failed to get retrieval stats", error=str(e))
            return {"error": str(e)}