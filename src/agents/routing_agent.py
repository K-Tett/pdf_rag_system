"""
Routing agent to determine whether to use PDF retrieval, web search, or both.
"""
import structlog
from typing import Dict, Any, List
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class RoutingResult(BaseModel):
    """Routing decision result model."""
    route: str = Field(description="Routing decision: 'pdf', 'web', or 'both'")
    reasoning: str = Field(description="Explanation for the routing decision")
    confidence: float = Field(description="Confidence in the routing decision (0-1)")
    search_strategy: str = Field(description="Recommended search strategy")


class RoutingAgent:
    """
    Agent responsible for routing queries to appropriate information sources.
    """
    
    def __init__(self, primary_llm, fallback_llm=None):
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.output_parser = JsonOutputParser(pydantic_object=RoutingResult)
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates."""
        self.routing_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a routing agent that determines the best information source for answering questions.

You have access to two information sources:
1. PDF documents: Academic papers and research documents about generative AI
2. Web search: Current information from the internet

Routing Rules:

Route to "pdf" when:
- Question asks about specific research papers, authors, or methodologies
- Question references academic concepts, experiments, or results from papers
- Question asks about technical details that would be in research literature
- Question uses phrases like "in the paper", "according to the study", "the authors report"
- Question asks about specific datasets, benchmarks, or evaluation metrics mentioned in papers

Route to "web" when:
- Question asks about recent news, announcements, or current events
- Question asks "what did [company] release this month/recently"
- Question asks about current stock prices, market conditions, or breaking news
- Question asks about real-time information or very recent developments
- Question clearly indicates need for current/up-to-date information

Route to "both" when:
- Question might benefit from both academic background and current information
- Question asks for comparisons that might need both historical and current data
- Uncertain about the best single source
- Question is complex and multifaceted

Search Strategy Guidelines:
- "comprehensive": Use when question requires thorough analysis
- "specific": Use when looking for particular facts or figures
- "comparative": Use when comparing different approaches or results
- "current": Use when timeliness is important

Respond with a JSON object:
{
    "route": "pdf|web|both",
    "reasoning": "detailed explanation of routing decision",
    "confidence": float_between_0_and_1,
    "search_strategy": "comprehensive|specific|comparative|current"
}"""),
            MessagesPlaceholder(variable_name="conversation_history", optional=True),
            HumanMessage(content="Question to route: {question}")
        ])
    
    async def process(
        self,
        question: str,
        conversation_history: List[BaseMessage] = None
    ) -> Dict[str, Any]:
        """
        Determine the appropriate routing for a question.
        
        Args:
            question: The user question to route
            conversation_history: Previous conversation messages for context
            
        Returns:
            Dictionary with routing decision and metadata
        """
        logger.info("Determining routing for question", question=question)
        
        try:
            # Simplified prompt for Ollama
            simple_prompt = f"""Analyze this question: "{question}"

Should this question be answered using:
- PDF documents (academic papers, research)
- Web search (current news, recent events)
- Both sources

Respond with ONLY a JSON object:
{{
    "route": "pdf",
    "reasoning": "explanation here",
    "confidence": 0.8,
    "search_strategy": "comprehensive"
}}

Route options: "pdf", "web", or "both"
Strategy options: "comprehensive", "specific", "comparative", "current"""

            # Try primary LLM first
            try:
                if hasattr(self.primary_llm, 'ainvoke'):
                    response = await self.primary_llm.ainvoke([HumanMessage(content=simple_prompt)])
                else:
                    response = self.primary_llm.invoke([HumanMessage(content=simple_prompt)])
                
                response_text = response.content if hasattr(response, 'content') else str(response)
                result = self._extract_json_from_response(response_text)
                
            except Exception as e:
                logger.warning("Primary LLM failed for routing", error=str(e))
                
                if self.fallback_llm:
                    logger.info("Trying fallback LLM for routing")
                    if hasattr(self.fallback_llm, 'ainvoke'):
                        response = await self.fallback_llm.ainvoke([HumanMessage(content=simple_prompt)])
                    else:
                        response = self.fallback_llm.invoke([HumanMessage(content=simple_prompt)])
                    
                    response_text = response.content if hasattr(response, 'content') else str(response)
                    result = self._extract_json_from_response(response_text)
                else:
                    raise e
            
            # Validate and clean result
            validated_result = self._validate_result(result, question)
            
            logger.info(
                "Routing decision complete",
                route=validated_result["route"],
                confidence=validated_result["confidence"]
            )
            
            return validated_result
            
        except Exception as e:
            logger.error("Routing failed", error=str(e))
            
            # Return fallback routing (default to PDF search)
            return self._get_fallback_routing(question)
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response that might contain extra text."""
        import json
        import re
        
        try:
            # First try to parse the entire response as JSON
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to find JSON within the response
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            # If no valid JSON found, create a default response
            logger.warning("Could not extract valid JSON from routing response", response=response_text[:200])
            return {
                "route": "pdf",
                "reasoning": "Default routing due to parsing failure",
                "confidence": 0.6,
                "search_strategy": "comprehensive"
            }
    
    def _validate_result(self, result: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Validate and clean the routing result."""
        
        # Ensure valid route
        route = result.get("route", "pdf").lower()
        if route not in ["pdf", "web", "both"]:
            route = "pdf"
        
        # Ensure valid search strategy
        strategy = result.get("search_strategy", "comprehensive").lower()
        if strategy not in ["comprehensive", "specific", "comparative", "current"]:
            strategy = "comprehensive"
        
        validated = {
            "route": route,
            "reasoning": result.get("reasoning", f"Routing question to {route} based on content analysis"),
            "confidence": max(0.0, min(1.0, result.get("confidence", 0.7))),
            "search_strategy": strategy
        }
        
        return validated
    
    def _get_fallback_routing(self, question: str) -> Dict[str, Any]:
        """Get fallback routing when LLM fails."""
        
        # Simple heuristic-based routing
        question_lower = question.lower()
        
        # Web search indicators
        web_keywords = [
            "recent", "latest", "current", "today", "this month", "this year",
            "breaking", "news", "announced", "released", "just", "now"
        ]
        
        # PDF search indicators
        pdf_keywords = [
            "paper", "study", "research", "author", "dataset", "experiment",
            "methodology", "results", "according to", "in the study"
        ]
        
        web_score = sum(1 for keyword in web_keywords if keyword in question_lower)
        pdf_score = sum(1 for keyword in pdf_keywords if keyword in question_lower)
        
        if web_score > pdf_score and web_score > 0:
            route = "web"
            strategy = "current"
        elif pdf_score > 0:
            route = "pdf"
            strategy = "comprehensive"
        else:
            route = "pdf"  # Default to PDF
            strategy = "comprehensive"
        
        return {
            "route": route,
            "reasoning": f"Fallback routing based on keyword analysis (web_score: {web_score}, pdf_score: {pdf_score})",
            "confidence": 0.6,
            "search_strategy": strategy
        }