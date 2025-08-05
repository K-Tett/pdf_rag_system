import structlog
from typing import Dict, Any, List, Optional, AsyncGenerator
import asyncio
from datetime import datetime

from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_ollama.llms import OllamaLLM

from src.core.config import Settings
from src.core.models import ChatResponse, StreamingChunk, Source
from src.services.vector_service import VectorService
from src.services.memory_service import ConversationMemory

logger = structlog.get_logger()


class AgentOrchestrator:
    def __init__(self, settings: Settings, vector_service: VectorService):
        self.settings = settings
        self.vector_service = vector_service
        self.memory = ConversationMemory(settings.SESSION_TIMEOUT)
        
        # Initialize LLMs
        self._init_llms()
    
    def _init_llms(self):
        """Initialize LLM instances."""
        # Primary LLM (Ollama)
        self.primary_llm = OllamaLLM(
            base_url=self.settings.OLLAMA_URL,
            model=self.settings.OLLAMA_MODEL,
            temperature=0.1
        )
        
        # Fallback LLM (OpenAI)
        self.fallback_llm = None
        if self.settings.has_openai_key:
            self.fallback_llm = ChatOpenAI(
                api_key=self.settings.OPENAI_API_KEY,
                model=self.settings.OPENAI_MODEL,
                temperature=0.1
            )
    
    async def process_question(
        self,
        question: str,
        session_id: str,
        stream: bool = False
    ) -> ChatResponse:
        logger.info("Processing question", question=question, session_id=session_id)
        
        try:
            # Get conversation history
            conversation_history = self.memory.get_conversation_history(session_id)
            
            # Step 1: Simple routing decision
            route = self._simple_route(question)
            logger.info("Route decision", route=route)
            
            # Step 2: Retrieve information
            sources = []
            if route in ["pdf", "both"]:
                pdf_sources = await self._retrieve_from_pdfs(question)
                sources.extend(pdf_sources)
            
            if route in ["web", "both"]:
                web_sources = await self._search_web(question)
                sources.extend(web_sources)
            
            # Step 3: Generate answer
            answer = await self._generate_answer(question, sources, conversation_history)
            
            # Step 4: Update conversation memory
            self.memory.add_message(session_id, HumanMessage(content=question))
            self.memory.add_message(session_id, AIMessage(content=answer))
            
            # Create response
            response = ChatResponse(
                answer=answer,
                sources=[self._dict_to_source(src) for src in sources[:5]],  # Limit to top 5
                confidence_score=0.8,  # Simple confidence
                metadata={
                    "session_id": session_id,
                    "route_decision": route,
                    "num_sources": len(sources),
                    "processing_time": 1.0  # Placeholder
                }
            )
            
            logger.info("Question processing complete", session_id=session_id)
            return response
            
        except Exception as e:
            logger.error("Error processing question", error=str(e), session_id=session_id)
            
            # Return error response
            return ChatResponse(
                answer="I apologize, but I encountered an error while processing your question. Please try again.",
                sources=[],
                confidence_score=0.0,
                metadata={
                    "session_id": session_id,
                    "error": str(e)
                }
            )
    
    def _simple_route(self, question: str) -> str:
        """Simple heuristic-based routing."""
        question_lower = question.lower()
        
        # Web search indicators
        web_keywords = [
            "recent", "latest", "current", "today", "this month", "this year",
            "breaking", "news", "announced", "released", "just", "now",
            "what did", "openai", "google", "microsoft"
        ]
        
        # PDF search indicators
        pdf_keywords = [
            "paper", "study", "research", "author", "dataset", "experiment",
            "methodology", "results", "according to", "in the study",
            "accuracy", "performance", "model", "algorithm"
        ]
        
        web_score = sum(1 for keyword in web_keywords if keyword in question_lower)
        pdf_score = sum(1 for keyword in pdf_keywords if keyword in question_lower)
        
        if web_score > pdf_score and web_score > 0:
            return "web"
        elif pdf_score > 0:
            return "pdf"
        else:
            return "pdf"  # Default to PDF
    
    async def _retrieve_from_pdfs(self, question: str) -> List[Dict[str, Any]]:
        try:
            results = await self.vector_service.search_similar(
                query=question,
                top_k=5,
                alpha=0.7
            )
            
            # Convert to standardized format
            sources = []
            for result in results:
                sources.append({
                    "title": result.get("metadata", {}).get("filename", "Document"),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "metadata": {
                        "source_type": "pdf",
                        **result.get("metadata", {})
                    }
                })
            
            logger.info("PDF retrieval complete", num_results=len(sources))
            return sources
            
        except Exception as e:
            logger.error("PDF retrieval failed", error=str(e))
            return []
    
    async def _search_web(self, question: str) -> List[Dict[str, Any]]:
        """Search the web for current information."""
        try:
            # Simple web search using DuckDuckGo
            from duckduckgo_search import DDGS
            
            ddgs = DDGS()
            results = ddgs.text(keywords=question, max_results=3)
            
            sources = []
            for result in results:
                sources.append({
                    "title": result.get("title", "Web Result"),
                    "content": result.get("body", ""),
                    "score": 1.0,
                    "metadata": {
                        "source_type": "web",
                        "url": result.get("href", "")
                    }
                })
            
            logger.info("Web search complete", num_results=len(sources))
            return sources
            
        except Exception as e:
            logger.error("Web search failed", error=str(e))
            return []
    
    async def _generate_answer(
        self,
        question: str,
        sources: List[Dict[str, Any]],
        conversation_history: List[BaseMessage]
    ) -> str:
        """Generate an answer using the available sources."""
        try:
            # Prepare context from sources
            context_parts = []
            for i, source in enumerate(sources[:3], 1):  # Use top 3 sources
                content = source.get("content", "")[:500]  # Limit content length
                title = source.get("title", f"Source {i}")
                context_parts.append(f"Source {i} ({title}): {content}")
            
            context = "\n\n".join(context_parts)
            
            # Simple prompt for answer generation
            prompt = f"""Based on the following sources, answer the question: "{question}"

Sources:
{context}

Instructions:
- Provide a clear, informative answer based on the sources
- If the sources don't contain relevant information, say so
- Keep the answer concise but helpful
- Mention which sources support your answer

Answer:"""

            # Try to generate answer
            try:
                if hasattr(self.primary_llm, 'ainvoke'):
                    response = await self.primary_llm.ainvoke([HumanMessage(content=prompt)])
                else:
                    response = self.primary_llm.invoke([HumanMessage(content=prompt)])
                
                answer = response.content if hasattr(response, 'content') else str(response)
                
            except Exception as e:
                logger.warning("Primary LLM failed for generation", error=str(e))
                
                if self.fallback_llm:
                    logger.info("Trying fallback LLM for generation")
                    if hasattr(self.fallback_llm, 'ainvoke'):
                        response = await self.fallback_llm.ainvoke([HumanMessage(content=prompt)])
                    else:
                        response = self.fallback_llm.invoke([HumanMessage(content=prompt)])
                    
                    answer = response.content if hasattr(response, 'content') else str(response)
                else:
                    raise e
            
            # Clean up the answer
            answer = answer.strip()
            if not answer:
                answer = "I couldn't generate a proper answer based on the available sources."
            
            return answer
            
        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            return "I apologize, but I encountered an error while generating the answer."
    
    def _dict_to_source(self, source_dict: Dict[str, Any]) -> Source:
        """Convert dictionary to Source object."""
        return Source(
            title=source_dict.get("title", ""),
            content=source_dict.get("content", ""),
            score=source_dict.get("score", 0.0),
            metadata=source_dict.get("metadata", {})
        )
    
    async def clear_session_memory(self, session_id: str):
        """Clear conversation memory for a session."""
        self.memory.clear_session(session_id)
        logger.info("Session memory cleared", session_id=session_id)
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get session status and conversation history."""
        return self.memory.get_session_stats(session_id)
    
    async def process_question_stream(
        self,
        question: str,
        session_id: str
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Process a question with streaming response (simplified).
        """
        try:
            # For simplicity, just return the complete answer as a single chunk
            response = await self.process_question(question, session_id, stream=False)
            
            # Yield the complete answer
            yield StreamingChunk(
                type="complete",
                content=response.answer,
                metadata={
                    "confidence": response.confidence_score,
                    "sources": [src.__dict__ for src in response.sources],
                    **response.metadata
                }
            )
            
        except Exception as e:
            logger.error("Streaming failed", error=str(e))
            yield StreamingChunk(
                type="error",
                content=f"Error: {str(e)}",
                metadata={"error": str(e)}
            )