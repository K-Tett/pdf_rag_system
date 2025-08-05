"""
Clarification agent to detect and handle ambiguous queries.
"""
import structlog
from typing import Dict, Any, List, Optional
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = structlog.get_logger()


class ClarificationResult(BaseModel):
    """Clarification result model."""
    is_ambiguous: bool = Field(description="Whether the question is ambiguous")
    ambiguity_reasons: List[str] = Field(description="Reasons why the question is ambiguous")
    clarified_question: str = Field(description="Clarified version of the question")
    suggested_clarifications: List[str] = Field(description="Suggested clarification questions")
    confidence: float = Field(description="Confidence in the clarification (0-1)")


class ClarificationAgent:
    """
    Agent responsible for detecting ambiguous queries and requesting clarification.
    """
    
    def __init__(self, primary_llm, fallback_llm=None):
        self.primary_llm = primary_llm
        self.fallback_llm = fallback_llm
        self.output_parser = JsonOutputParser(pydantic_object=ClarificationResult)
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates."""
        self.clarification_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a clarification agent tasked with analyzing user questions to detect ambiguity and improve clarity.

Your job is to:
1. Identify if a question is ambiguous or underspecified
2. Determine what additional information might be needed
3. Provide a clarified version of the question when possible
4. Suggest specific clarification questions when needed

Common types of ambiguity to look for:
- Vague terms ("good accuracy", "enough examples", "better performance")
- Missing context (which dataset, model, paper, method)
- Unclear scope (time period, domain, specific aspect)
- Ambiguous pronouns or references
- Multiple possible interpretations

If the question is clear and specific, return it unchanged with is_ambiguous=false.

Respond with a JSON object matching this format:
{
    "is_ambiguous": boolean,
    "ambiguity_reasons": ["reason1", "reason2"],
    "clarified_question": "improved question text",
    "suggested_clarifications": ["clarification question 1", "clarification question 2"],
    "confidence": float_between_0_and_1
}"""),
            MessagesPlaceholder(variable_name="conversation_history", optional=True),
            HumanMessage(content="Question to analyze: {question}")
        ])
    
    async def process(
        self,
        question: str,
        conversation_history: List[BaseMessage] = None
    ) -> Dict[str, Any]:
        """
        Process a question to detect ambiguity and provide clarification.
        
        Args:
            question: The user question to analyze
            conversation_history: Previous conversation messages for context
            
        Returns:
            Dictionary with clarification results
        """
        logger.info("Analyzing question for ambiguity", question=question)
        
        try:
            # Simplified prompt that works better with Ollama
            simple_prompt = f"""Analyze this question for ambiguity: "{question}"

Is this question clear and specific, or does it need clarification?

Respond with ONLY a JSON object in this exact format:
{{
    "is_ambiguous": true/false,
    "ambiguity_reasons": ["reason1", "reason2"],
    "clarified_question": "improved question text",
    "suggested_clarifications": ["question1", "question2"],
    "confidence": 0.8
}}"""

            # Try primary LLM first
            try:
                if hasattr(self.primary_llm, 'ainvoke'):
                    response = await self.primary_llm.ainvoke([HumanMessage(content=simple_prompt)])
                else:
                    response = self.primary_llm.invoke([HumanMessage(content=simple_prompt)])
                
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Clean the response to extract JSON
                result = self._extract_json_from_response(response_text)
                
            except Exception as e:
                logger.warning("Primary LLM failed for clarification", error=str(e))
                
                if self.fallback_llm:
                    logger.info("Trying fallback LLM for clarification")
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
                "Clarification analysis complete",
                is_ambiguous=validated_result["is_ambiguous"],
                confidence=validated_result["confidence"]
            )
            
            return validated_result
            
        except Exception as e:
            logger.error("Clarification analysis failed", error=str(e))
            
            # Return fallback result
            return {
                "is_ambiguous": False,
                "ambiguity_reasons": [],
                "clarified_question": question,
                "suggested_clarifications": [],
                "confidence": 0.5
            }
    
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
            logger.warning("Could not extract valid JSON from response", response=response_text[:200])
            return {
                "is_ambiguous": False,
                "ambiguity_reasons": [],
                "clarified_question": "",
                "suggested_clarifications": [],
                "confidence": 0.5
            }
    
    def _validate_result(self, result: Dict[str, Any], original_question: str) -> Dict[str, Any]:
        """Validate and clean the clarification result."""
        
        # Ensure all required fields are present
        validated = {
            "is_ambiguous": result.get("is_ambiguous", False),
            "ambiguity_reasons": result.get("ambiguity_reasons", []),
            "clarified_question": result.get("clarified_question", original_question),
            "suggested_clarifications": result.get("suggested_clarifications", []),
            "confidence": max(0.0, min(1.0, result.get("confidence", 0.5)))
        }
        
        # Clean up lists
        validated["ambiguity_reasons"] = [
            reason for reason in validated["ambiguity_reasons"] 
            if isinstance(reason, str) and reason.strip()
        ]
        
        validated["suggested_clarifications"] = [
            clarification for clarification in validated["suggested_clarifications"]
            if isinstance(clarification, str) and clarification.strip()
        ]
        
        # Ensure clarified question is not empty
        if not validated["clarified_question"].strip():
            validated["clarified_question"] = original_question
        
        # Consistency check
        if validated["is_ambiguous"] and not validated["ambiguity_reasons"]:
            validated["ambiguity_reasons"] = ["Question lacks specificity"]
        
        if not validated["is_ambiguous"]:
            validated["ambiguity_reasons"] = []
            validated["suggested_clarifications"] = []
        
        return validated
    
    async def suggest_follow_up_questions(
        self,
        question: str,
        answer: str,
        confidence: float
    ) -> List[str]:
        """
        Suggest follow-up questions based on the answer quality.
        
        Args:
            question: Original question
            answer: Generated answer
            confidence: Confidence score of the answer
            
        Returns:
            List of suggested follow-up questions
        """
        if confidence > 0.8:
            # High confidence - suggest related questions
            follow_up_prompt = f"""Based on this Q&A pair, suggest 2-3 related follow-up questions that might be interesting:

Question: {question}
Answer: {answer}

Suggest questions that:
- Explore related aspects of the topic
- Ask for more specific details
- Connect to broader implications

Return only the questions, one per line."""
            
        else:
            # Lower confidence - suggest clarifying questions
            follow_up_prompt = f"""The answer to this question had low confidence. Suggest 2-3 clarifying questions that might help get a better answer:

Question: {question}
Answer: {answer}

Suggest questions that:
- Ask for more specific information
- Clarify ambiguous terms
- Provide additional context

Return only the questions, one per line."""
        
        try:
            response = await self.primary_llm.ainvoke([HumanMessage(content=follow_up_prompt)])
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse follow-up questions
            questions = [
                q.strip() for q in content.split('\n')
                if q.strip() and not q.strip().startswith(('-', '*', '•'))
            ]
            
            return questions[:3]  # Limit to 3 questions
            
        except Exception as e:
            logger.error("Failed to generate follow-up questions", error=str(e))
            return []