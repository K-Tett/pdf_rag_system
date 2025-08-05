"""
Synthesis agent for generating comprehensive answers from multiple sources.
"""
import structlog
from typing import Dict, Any, List, AsyncGenerator
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import asyncio
import json
import re

from src.core.config import Settings
from src.core.models import StreamingChunk, Source

logger = structlog.get_logger()


class SynthesisAgent:
    """
    Agent responsible for synthesizing information from multiple sources into coherent answers.
    """
    
    def __init__(
        self,
        primary_llm,
        fallback_llm=None,
        settings: Settings = None
    ):
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.settings = settings or Settings()
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates for synthesis."""
        
        self.synthesis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert research assistant that synthesizes information from multiple sources to provide comprehensive, accurate answers.

Your task is to:
1. Analyze the provided sources (PDF documents and web search results)
2. Extract relevant information that answers the user's question
3. Synthesize a coherent, well-structured response
4. Properly cite sources using [Source X] notation
5. Indicate confidence level and any limitations

Guidelines:
- Prioritize authoritative sources (academic papers, official documentation)
- If sources conflict, acknowledge the conflict and explain different perspectives
- Be clear about what information comes from which source
- If the answer is not found in the sources, say so explicitly
- Provide specific details and examples when available
- Structure your response logically with clear sections if needed

Source Priority:
1. Academic papers and research documents (highest credibility)
2. Official documentation and authoritative websites
3. Recent news and announcements (for current information)
4. General web sources (lowest priority, use with caution)

Citation Format:
- Use [Source 1], [Source 2], etc. to reference sources
- Match the source numbers to the provided source list
- Cite specific claims, not just general statements

Confidence Assessment:
- High (0.8-1.0): Multiple authoritative sources agree
- Medium (0.5-0.79): Some authoritative sources, or single reliable source
- Low (0.2-0.49): Limited or potentially unreliable sources
- Very Low (0.0-0.19): No relevant sources or conflicting information"""),
            
            MessagesPlaceholder(variable_name="conversation_history", optional=True),
            
            HumanMessage(content="""Question: {question}

PDF Document Sources:
{pdf_sources}

Web Search Sources:
{web_sources}

Please provide a comprehensive answer to the question using the available sources. Include proper citations and assess your confidence level.""")
        ])
    
    async def process(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]] = None,
        web_results: List[Dict[str, Any]] = None,
        conversation_history: List[BaseMessage] = None
    ) -> Dict[str, Any]:
        """
        Synthesize information from multiple sources to answer a question.
        
        Args:
            question: The user question
            retrieved_docs: Documents from PDF retrieval
            web_results: Results from web search
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with synthesized answer and metadata
        """
        logger.info("Starting answer synthesis", question=question)
        
        try:
            # Prepare sources
            pdf_sources_text = self._format_pdf_sources(retrieved_docs or [])
            web_sources_text = self._format_web_sources(web_results or [])
            
            # Create the prompt
            messages = self.synthesis_prompt.format_messages(
                question=question,
                pdf_sources=pdf_sources_text,
                web_sources=web_sources_text,
                conversation_history=conversation_history or []
            )
            
            # Generate response
            try:
                response = await self.primary_llm.ainvoke(messages)
                answer = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.warning("Primary LLM failed for synthesis", error=str(e))
                
                if self.fallback_llm:
                    logger.info("Trying fallback LLM for synthesis")
                    response = await self.fallback_llm.ainvoke(messages)
                    answer = response.content
                else:
                    raise e
            
            # Parse the response and extract metadata
            parsed_result = self._parse_synthesis_response(
                answer,
                retrieved_docs or [],
                web_results or []
            )
            
            logger.info(
                "Answer synthesis complete",
                confidence=parsed_result["confidence"],
                num_sources=len(parsed_result["sources"])
            )
            
            return parsed_result
            
        except Exception as e:
            logger.error("Answer synthesis failed", error=str(e))
            
            return {
                "answer": "I apologize, but I encountered an error while generating the answer. Please try rephrasing your question.",
                "confidence": 0.0,
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "synthesis_failed": True
                }
            }
    
    async def process_stream(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]] = None,
        web_results: List[Dict[str, Any]] = None,
        conversation_history: List[BaseMessage] = None
    ) -> AsyncGenerator[StreamingChunk, None]:
        """
        Generate streaming synthesis response.
        
        Args:
            question: The user question
            retrieved_docs: Documents from PDF retrieval
            web_results: Results from web search
            conversation_history: Previous conversation messages
            
        Yields:
            StreamingChunk objects with partial responses
        """
        logger.info("Starting streaming synthesis", question=question)
        
        try:
            # Prepare sources
            pdf_sources_text = self._format_pdf_sources(retrieved_docs or [])
            web_sources_text = self._format_web_sources(web_results or [])
            
            # Create the prompt
            messages = self.synthesis_prompt.format_messages(
                question=question,
                pdf_sources=pdf_sources_text,
                web_sources=web_sources_text,
                conversation_history=conversation_history or []
            )
            
            # Stream response
            full_response = ""
            
            try:
                # Note: This is a simplified streaming implementation
                # In a real implementation, you'd use the LLM's streaming capability
                response = await self.primary_llm.ainvoke(messages)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                # Simulate streaming by chunking the response
                words = answer.split()
                current_chunk = ""
                
                for i, word in enumerate(words):
                    current_chunk += word + " "
                    
                    # Send chunk every few words
                    if (i + 1) % 3 == 0 or i == len(words) - 1:
                        yield StreamingChunk(
                            type="token",
                            content=current_chunk,
                            metadata=None
                        )
                        full_response += current_chunk
                        current_chunk = ""
                        
                        # Small delay for realistic streaming
                        await asyncio.sleep(0.05)
                
            except Exception as e:
                logger.warning("Primary LLM failed for streaming synthesis", error=str(e))
                
                if self.fallback_llm:
                    response = await self.fallback_llm.ainvoke(messages)
                    answer = response.content
                    full_response = answer
                    
                    yield StreamingChunk(
                        type="token",
                        content=answer,
                        metadata=None
                    )
                else:
                    raise e
            
            # Parse final response
            parsed_result = self._parse_synthesis_response(
                full_response,
                retrieved_docs or [],
                web_results or []
            )
            
            # Send completion chunk
            yield StreamingChunk(
                type="complete",
                content=parsed_result["answer"],
                metadata={
                    "confidence": parsed_result["confidence"],
                    "sources": [source.__dict__ for source in parsed_result["sources"]],
                    **parsed_result["metadata"]
                }
            )
            
        except Exception as e:
            logger.error("Streaming synthesis failed", error=str(e))
            
            yield StreamingChunk(
                type="error",
                content=f"Error during synthesis: {str(e)}",
                metadata={"error": str(e)}
            )
    
    def _format_pdf_sources(self, documents: List[Dict[str, Any]]) -> str:
        """Format PDF documents as source text."""
        if not documents:
            return "No PDF documents available."
        
        formatted_sources = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            title = doc.get("title", f"Document {i}")
            metadata = doc.get("metadata", {})
            
            source_text = f"""Source {i}: {title}
Content: {content}
Metadata: Document ID: {metadata.get('document_id', 'N/A')}, Filename: {metadata.get('filename', 'N/A')}
Relevance Score: {doc.get('relevance_score', 0.0):.2f}"""
            
            if doc.get('rerank_score') is not None:
                source_text += f", Rerank Score: {doc.get('rerank_score', 0.0):.2f}"
            
            formatted_sources.append(source_text)
        
        return "\n\n".join(formatted_sources)
    
    def _format_web_sources(self, results: List[Dict[str, Any]]) -> str:
        """Format web search results as source text."""
        if not results:
            return "No web search results available."
        
        formatted_sources = []
        source_num = 1
        
        for result in results:
            content = result.get("content", result.get("snippet", ""))
            title = result.get("title", f"Web Result {source_num}")
            url = result.get("url", result.get("metadata", {}).get("url", ""))
            
            source_text = f"""Source {source_num}: {title}
Content: {content}
URL: {url}
Relevance Score: {result.get('relevance_score', 0.0):.2f}"""
            
            formatted_sources.append(source_text)
            source_num += 1
        
        return "\n\n".join(formatted_sources)
    
    def _parse_synthesis_response(
        self,
        response: str,
        pdf_docs: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse the synthesis response and extract metadata."""
        
        # Extract confidence from response if mentioned
        confidence = self._extract_confidence(response)
        
        # Extract cited sources
        sources = self._extract_sources(response, pdf_docs, web_results)
        
        # Clean up the response
        cleaned_response = self._clean_response(response)
        
        return {
            "answer": cleaned_response,
            "confidence": confidence,
            "sources": sources,
            "metadata": {
                "response_length": len(cleaned_response),
                "num_pdf_sources": len(pdf_docs),
                "num_web_sources": len(web_results),
                "citations_found": len(re.findall(r'\[Source \d+\]', response)),
                "synthesis_method": "llm_generated"
            }
        }
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response text."""
        # Look for confidence indicators in the response
        confidence_patterns = [
            r'confidence[:\s]+(\d+(?:\.\d+)?)',
            r'confident[:\s]+(\d+(?:\.\d+)?)',
            r'certainty[:\s]+(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*confidence',
        ]
        
        for pattern in confidence_patterns:
            matches = re.findall(pattern, response.lower())
            if matches:
                try:
                    confidence = float(matches[0])
                    # Normalize to 0-1 if it's in percentage
                    if confidence > 1:
                        confidence = confidence / 100
                    return max(0.0, min(1.0, confidence))
                except ValueError:
                    continue
        
        # Fallback: assess confidence based on source quality and response characteristics
        return self._assess_implicit_confidence(response)
    
    def _assess_implicit_confidence(self, response: str) -> float:
        """Assess confidence based on response characteristics."""
        base_confidence = 0.5
        
        # Check for uncertainty indicators
        uncertainty_indicators = [
            "might", "could", "possibly", "perhaps", "unclear", "uncertain",
            "not sure", "may be", "appears to", "seems to", "likely"
        ]
        
        certainty_indicators = [
            "definitely", "certainly", "clearly", "confirmed", "established",
            "proven", "demonstrated", "according to", "research shows"
        ]
        
        response_lower = response.lower()
        
        # Count indicators
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in response_lower)
        certainty_count = sum(1 for indicator in certainty_indicators if indicator in response_lower)
        
        # Adjust confidence
        confidence_adjustment = (certainty_count - uncertainty_count) * 0.1
        
        # Check for citations
        citations = len(re.findall(r'\[Source \d+\]', response))
        if citations > 0:
            base_confidence += 0.2
        
        # Check response length and detail
        if len(response) > 500:
            base_confidence += 0.1
        
        return max(0.1, min(0.9, base_confidence + confidence_adjustment))
    
    def _extract_sources(
        self,
        response: str,
        pdf_docs: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]]
    ) -> List[Source]:
        """Extract and format sources mentioned in the response."""
        sources = []
        
        # Find all source citations in the response
        citations = re.findall(r'\[Source (\d+)\]', response)
        cited_source_nums = set(int(num) for num in citations)
        
        # Map source numbers to actual sources
        all_sources = pdf_docs + web_results
        
        for source_num in sorted(cited_source_nums):
            if 1 <= source_num <= len(all_sources):
                source_data = all_sources[source_num - 1]
                
                # Determine source type
                if source_num <= len(pdf_docs):
                    source_type = "pdf_document"
                    title = source_data.get("title", "PDF Document")
                    metadata = source_data.get("metadata", {})
                else:
                    source_type = "web_search"
                    title = source_data.get("title", "Web Result")
                    metadata = source_data.get("metadata", {})
                
                source = Source(
                    title=title,
                    content=source_data.get("content", "")[:500],  # Limit content length
                    score=source_data.get("relevance_score", 0.0),
                    metadata={
                        "source_type": source_type,
                        "source_number": source_num,
                        **metadata
                    }
                )
                
                sources.append(source)
        
        return sources
    
    def _clean_response(self, response: str) -> str:
        """Clean up the response text."""
        # Remove any system-level instructions that might have leaked through
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and system instructions
            if not line or line.startswith(('System:', 'Assistant:', 'Human:')):
                continue
            
            cleaned_lines.append(line)
        
        return '\n\n'.join(cleaned_lines) if cleaned_lines else response