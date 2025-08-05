"""
Web search agent using DuckDuckGo for current information retrieval.
"""
import structlog
from typing import Dict, Any, List
from langchain.schema import BaseMessage
from ddgs import DDGS
import asyncio
import aiohttp
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime

from src.core.config import Settings

logger = structlog.get_logger()


class WebSearchAgent:
    """
    Agent responsible for web search and content extraction.
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
        
        # Initialize search client
        self.ddgs = DDGS()
    
    async def process(
        self,
        question: str,
        conversation_history: List[BaseMessage] = None
    ) -> Dict[str, Any]:
        """
        Perform web search and extract relevant information.
        
        Args:
            question: The user question
            conversation_history: Previous conversation messages for context
            
        Returns:
            Dictionary with search results and metadata
        """
        logger.info("Starting web search", question=question)
        
        try:
            # Generate search queries
            search_queries = self._generate_search_queries(question, conversation_history)
            
            # Perform searches
            all_results = []
            for query in search_queries:
                try:
                    results = await self._search_web(query)
                    all_results.extend(results)
                    
                    # Add small delay between searches
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.warning("Search failed for query", query=query, error=str(e))
                    continue
            
            # Deduplicate and rank results
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_results(question, unique_results)
            
            # Limit to top results
            final_results = ranked_results[:self.settings.WEB_SEARCH_TOP_K]
            
            # Extract content from top results
            enriched_results = await self._enrich_results_with_content(final_results)
            
            logger.info(
                "Web search completed",
                num_queries=len(search_queries),
                total_results=len(all_results),
                final_results=len(enriched_results)
            )
            
            return {
                "results": enriched_results,
                "search_metadata": {
                    "queries_used": search_queries,
                    "total_found": len(all_results),
                    "unique_results": len(unique_results),
                    "final_count": len(enriched_results),
                    "search_timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error("Web search failed", error=str(e))
            return {
                "results": [],
                "search_metadata": {
                    "error": str(e),
                    "queries_used": [],
                    "total_found": 0,
                    "final_count": 0
                }
            }
    
    def _generate_search_queries(
        self,
        question: str,
        conversation_history: List[BaseMessage] = None
    ) -> List[str]:
        """
        Generate multiple search queries for better coverage.
        
        Args:
            question: Original question
            conversation_history: Previous conversation messages
            
        Returns:
            List of search queries
        """
        queries = [question]  # Always include the original question
        
        try:
            # Extract key terms for alternative queries
            question_lower = question.lower()
            
            # Generate variations based on question type
            if "latest" in question_lower or "recent" in question_lower:
                # Add time-specific variations
                queries.append(f"{question} 2024 2025")
                queries.append(f"{question.replace('latest', 'new').replace('recent', 'current')}")
            
            elif "what did" in question_lower and ("release" in question_lower or "announce" in question_lower):
                # Company/product announcement queries
                match = re.search(r'what did (\w+)', question_lower)
                if match:
                    company = match.group(1)
                    queries.append(f"{company} announcement 2024 2025")
                    queries.append(f"{company} new product release")
            
            elif "how to" in question_lower or "tutorial" in question_lower:
                # How-to queries
                queries.append(f"{question} tutorial guide")
                queries.append(f"{question} best practices")
            
            # Remove duplicates and limit
            unique_queries = []
            for query in queries:
                if query not in unique_queries:
                    unique_queries.append(query)
            
            return unique_queries[:3]  # Limit to 3 queries
            
        except Exception as e:
            logger.warning("Failed to generate search queries", error=str(e))
            return [question]
    
    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform web search using DuckDuckGo.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        try:
            logger.debug("Performing web search", query=query)
            
            # Use DuckDuckGo search
            results = self.ddgs.text(
                keywords=query,
                max_results=10,
                region='wt-wt',  # World-wide
                safesearch='moderate'
            )
            
            formatted_results = []
            for result in results:
                formatted_result = {
                    "title": result.get("title", ""),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", ""),
                    "source": self._extract_domain(result.get("href", "")),
                    "search_query": query,
                    "relevance_score": 1.0  # Will be updated in ranking
                }
                formatted_results.append(formatted_result)
            
            logger.debug("Web search completed", query=query, results_count=len(formatted_results))
            return formatted_results
            
        except Exception as e:
            logger.error("DuckDuckGo search failed", query=query, error=str(e))
            return []
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except Exception:
            return "unknown"
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate results based on URL and title similarity.
        
        Args:
            results: List of search results
            
        Returns:
            Deduplicated results
        """
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            title = result.get("title", "").lower()
            
            # Skip if URL already seen
            if url in seen_urls:
                continue
            
            # Skip if very similar title already seen
            title_words = set(title.split())
            is_similar = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if len(title_words & seen_words) / max(len(title_words), len(seen_words), 1) > 0.8:
                    is_similar = True
                    break
            
            if not is_similar:
                seen_urls.add(url)
                seen_titles.add(title)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, question: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank search results by relevance to the question.
        
        Args:
            question: Original question
            results: Search results to rank
            
        Returns:
            Ranked results
        """
        question_words = set(question.lower().split())
        
        for result in results:
            score = 0.0
            
            # Title relevance (weighted higher)
            title_words = set(result.get("title", "").lower().split())
            title_overlap = len(question_words & title_words) / max(len(question_words), 1)
            score += title_overlap * 3.0
            
            # Snippet relevance
            snippet_words = set(result.get("snippet", "").lower().split())
            snippet_overlap = len(question_words & snippet_words) / max(len(question_words), 1)
            score += snippet_overlap * 2.0
            
            # Source quality (simple heuristic)
            source = result.get("source", "").lower()
            if any(domain in source for domain in ["github.com", "stackoverflow.com", "arxiv.org"]):
                score += 0.5
            elif any(domain in source for domain in ["wikipedia.org", ".edu", ".org"]):
                score += 0.3
            elif any(domain in source for domain in [".gov", "reuters.com", "bbc.com"]):
                score += 0.4
            
            # Recency indicators in title/snippet
            text_content = (result.get("title", "") + " " + result.get("snippet", "")).lower()
            if any(term in text_content for term in ["2024", "2025", "latest", "recent", "new"]):
                score += 0.2
            
            result["relevance_score"] = score
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    
    async def _enrich_results_with_content(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich results by extracting content from web pages.
        
        Args:
            results: Search results to enrich
            
        Returns:
            Enriched results with extracted content
        """
        enriched = []
        
        for result in results:
            try:
                # Try to extract more content from the page
                content = await self._extract_page_content(result.get("url", ""))
                
                enriched_result = {
                    **result,
                    "content": content or result.get("snippet", ""),
                    "source": "web_search",
                    "metadata": {
                        "url": result.get("url", ""),
                        "domain": result.get("source", ""),
                        "search_query": result.get("search_query", ""),
                        "extracted_content": bool(content),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                
                enriched.append(enriched_result)
                
            except Exception as e:
                logger.warning("Failed to enrich result", url=result.get("url"), error=str(e))
                
                # Use snippet as fallback
                enriched_result = {
                    **result,
                    "content": result.get("snippet", ""),
                    "source": "web_search",
                    "metadata": {
                        "url": result.get("url", ""),
                        "domain": result.get("source", ""),
                        "search_query": result.get("search_query", ""),
                        "extracted_content": False,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                enriched.append(enriched_result)
        
        return enriched
    
    async def _extract_page_content(self, url: str, max_length: int = 2000) -> str:
        """
        Extract text content from a web page.
        
        Args:
            url: URL to extract content from
            max_length: Maximum content length
            
        Returns:
            Extracted text content
        """
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; RAGBot/1.0)'
                }
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Simple HTML content extraction
                        # Remove script and style tags
                        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
                        
                        # Remove HTML tags
                        text = re.sub(r'<[^>]+>', ' ', html)
                        
                        # Clean up whitespace
                        text = re.sub(r'\s+', ' ', text).strip()
                        
                        # Truncate if too long
                        if len(text) > max_length:
                            text = text[:max_length] + "..."
                        
                        return text
                    
        except Exception as e:
            logger.debug("Failed to extract page content", url=url, error=str(e))
        
        return ""