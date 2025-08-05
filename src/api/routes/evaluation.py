"""
Evaluation API routes for testing system performance.
"""
import structlog
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from typing import List, Dict, Any
import json
import tempfile
import os

from src.services.evaluation_service import EvaluationHarness
from src.agents.orchestrator import AgentOrchestrator
from src.core.models import EvaluationRequest, EvaluationResult

logger = structlog.get_logger()
router = APIRouter()


@router.post("/single", response_model=EvaluationResult)
async def evaluate_single_question(
    request: EvaluationRequest,
    orchestrator: AgentOrchestrator = Depends()
):
    """
    Evaluate a single question-answer pair.
    """
    logger.info("Evaluating single question", question=request.question)
    
    try:
        harness = EvaluationHarness(orchestrator)
        
        result = await harness.evaluate_single_question(
            question=request.question,
            expected_answer=request.expected_answer,
            context=request.context
        )
        
        logger.info(
            "Single question evaluation complete",
            pass_threshold=result.pass_threshold,
            primary_score=result.metadata.get('primary_score', 0.0)
        )
        
        return result
        
    except Exception as e:
        logger.error("Failed to evaluate single question", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@router.post("/dataset")
async def evaluate_dataset(
    evaluation_pairs: List[EvaluationRequest],
    orchestrator: AgentOrchestrator = Depends()
):
    """
    Evaluate a dataset of question-answer pairs.
    """
    logger.info("Evaluating dataset", num_pairs=len(evaluation_pairs))
    
    try:
        harness = EvaluationHarness(orchestrator)
        
        # Convert requests to the format expected by the harness
        pairs = []
        for req in evaluation_pairs:
            pair = {
                'question': req.question,
                'expected_answer': req.expected_answer
            }
            if req.context:
                pair['context'] = req.context
            pairs.append(pair)
        
        results = await harness.evaluate_dataset(pairs)
        
        logger.info(
            "Dataset evaluation complete",
            total_questions=results.get('total_questions', 0),
            pass_rate=results.get('aggregate_metrics', {}).get('pass_rate', 0.0)
        )
        
        return results
        
    except Exception as e:
        logger.error("Failed to evaluate dataset", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset evaluation failed: {str(e)}"
        )


@router.post("/dataset/upload")
async def evaluate_uploaded_dataset(
    file: UploadFile = File(...),
    orchestrator: AgentOrchestrator = Depends()
):
    """
    Evaluate a dataset uploaded as JSON file.
    """
    logger.info("Evaluating uploaded dataset", filename=file.filename)
    
    # Validate file type
    if not file.filename.endswith('.json'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only JSON files are supported"
        )
    
    try:
        # Read and parse the uploaded file
        content = await file.read()
        data = json.loads(content.decode('utf-8'))
        
        harness = EvaluationHarness(orchestrator)
        
        # Load evaluation pairs from the uploaded data
        if isinstance(data, list):
            evaluation_pairs = data
        elif isinstance(data, dict) and 'evaluation_pairs' in data:
            evaluation_pairs = data['evaluation_pairs']
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid JSON format. Expected list of evaluation pairs or object with 'evaluation_pairs' key"
            )
        
        # Validate format
        for i, pair in enumerate(evaluation_pairs):
            if 'question' not in pair or 'expected_answer' not in pair:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Evaluation pair {i} missing required fields 'question' or 'expected_answer'"
                )
        
        # Run evaluation
        results = await harness.evaluate_dataset(evaluation_pairs)
        
        logger.info(
            "Uploaded dataset evaluation complete",
            filename=file.filename,
            total_questions=results.get('total_questions', 0),
            pass_rate=results.get('aggregate_metrics', {}).get('pass_rate', 0.0)
        )
        
        return results
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON file"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to evaluate uploaded dataset", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Uploaded dataset evaluation failed: {str(e)}"
        )


@router.get("/report/{evaluation_id}")
async def get_evaluation_report(
    evaluation_id: str,
    orchestrator: AgentOrchestrator = Depends()
):
    """
    Get a formatted evaluation report.
    Note: This is a simplified implementation. In production, you'd store evaluation results.
    """
    # This is a placeholder - in practice, you'd retrieve stored evaluation results
    # and generate the report from those
    
    return {
        "message": "Evaluation report endpoint placeholder",
        "evaluation_id": evaluation_id,
        "note": "In production, this would retrieve stored evaluation results and generate a formatted report"
    }


@router.get("/history")
async def get_evaluation_history(
    limit: int = 10,
    orchestrator: AgentOrchestrator = Depends()
):
    """
    Get evaluation history.
    Note: This is a simplified implementation using in-memory storage.
    """
    try:
        # Create temporary harness to get history
        # In production, this would be stored in a database
        harness = EvaluationHarness(orchestrator)
        history = harness.get_evaluation_history()
        
        # Limit results
        limited_history = history[-limit:] if len(history) > limit else history
        
        return {
            "total_evaluations": len(history),
            "returned_count": len(limited_history),
            "evaluations": limited_history
        }
        
    except Exception as e:
        logger.error("Failed to get evaluation history", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get evaluation history: {str(e)}"
        )


@router.post("/benchmark")
async def run_standard_benchmark(
    benchmark_name: str = "basic",
    orchestrator: AgentOrchestrator = Depends()
):
    """
    Run a standard benchmark evaluation.
    """
    logger.info("Running standard benchmark", benchmark=benchmark_name)
    
    try:
        # Define standard benchmark questions
        benchmarks = {
            "basic": [
                {
                    "question": "What is retrieval-augmented generation?",
                    "expected_answer": "Retrieval-augmented generation (RAG) is a technique that combines information retrieval with text generation to improve the factual accuracy and relevance of generated responses."
                },
                {
                    "question": "How does hybrid search work?",
                    "expected_answer": "Hybrid search combines multiple retrieval methods, typically dense vector search and sparse keyword search (like BM25), to improve retrieval accuracy by leveraging both semantic similarity and exact keyword matching."
                },
                {
                    "question": "What are the benefits of using embeddings for document retrieval?",
                    "expected_answer": "Embeddings enable semantic search by representing documents and queries in a high-dimensional space where semantically similar content has similar vector representations, allowing for more nuanced and context-aware retrieval."
                }
            ],
            "advanced": [
                {
                    "question": "Compare different chunking strategies for RAG systems.",
                    "expected_answer": "Common chunking strategies include fixed-size chunking (simple but may break context), sentence-based chunking (preserves semantic units), and semantic chunking (groups related content). Each has trade-offs between context preservation and retrieval granularity."
                },
                {
                    "question": "What role does reranking play in retrieval systems?",
                    "expected_answer": "Reranking improves retrieval quality by using more sophisticated models to reorder initially retrieved documents based on their relevance to the query, typically using cross-encoder models that can better understand query-document relationships."
                }
            ]
        }
        
        if benchmark_name not in benchmarks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown benchmark: {benchmark_name}. Available: {list(benchmarks.keys())}"
            )
        
        harness = EvaluationHarness(orchestrator)
        results = await harness.evaluate_dataset(benchmarks[benchmark_name])
        
        # Add benchmark metadata
        results['benchmark_name'] = benchmark_name
        results['benchmark_type'] = 'standard'
        
        logger.info(
            "Standard benchmark complete",
            benchmark=benchmark_name,
            pass_rate=results.get('aggregate_metrics', {}).get('pass_rate', 0.0)
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to run standard benchmark", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Benchmark evaluation failed: {str(e)}"
        )


@router.get("/metrics/available")
async def get_available_metrics():
    """
    Get list of available evaluation metrics.
    """
    return {
        "metrics": [
            {
                "name": "rouge1_f",
                "description": "ROUGE-1 F1 score (unigram overlap)",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            {
                "name": "rouge2_f", 
                "description": "ROUGE-2 F1 score (bigram overlap)",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            {
                "name": "rougeL_f",
                "description": "ROUGE-L F1 score (longest common subsequence)",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            {
                "name": "bert_f1",
                "description": "BERTScore F1 (semantic similarity using BERT)",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            {
                "name": "semantic_similarity",
                "description": "Cosine similarity between sentence embeddings",
                "range": [-1.0, 1.0],
                "higher_is_better": True
            },
            {
                "name": "has_specific_details",
                "description": "Whether answer contains specific details (numbers, dates, names)",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            {
                "name": "has_citations",
                "description": "Whether answer contains citations or source references",
                "range": [0.0, 1.0],
                "higher_is_better": True
            },
            {
                "name": "is_coherent",
                "description": "Basic coherence assessment of the generated text",
                "range": [0.0, 1.0],
                "higher_is_better": True
            }
        ]
    }


@router.delete("/history")
async def clear_evaluation_history(
    orchestrator: AgentOrchestrator = Depends()
):
    """
    Clear evaluation history.
    """
    try:
        # Create temporary harness to clear history
        harness = EvaluationHarness(orchestrator)
        harness.clear_evaluation_history()
        
        logger.info("Evaluation history cleared")
        
        return {"message": "Evaluation history cleared successfully"}
        
    except Exception as e:
        logger.error("Failed to clear evaluation history", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear evaluation history: {str(e)}"
        )