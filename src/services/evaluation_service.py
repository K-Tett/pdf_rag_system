"""
Evaluation service for assessing RAG system performance.
"""
import structlog
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Evaluation metrics
from rouge_score import rouge_scorer
try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None

from sentence_transformers import SentenceTransformer
import numpy as np

from src.agents.orchestrator import AgentOrchestrator
from src.core.models import EvaluationRequest, EvaluationResult

logger = structlog.get_logger()


class EvaluationMetrics:
    """
    Collection of evaluation metrics for RAG systems.
    """
    
    def __init__(self):
        """Initialize evaluation metrics."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.warning("Failed to load sentence transformer", error=str(e))
            self.sentence_model = None
    
    def calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores between generated and reference text.
        
        Args:
            generated: Generated answer
            reference: Reference/expected answer
            
        Returns:
            Dictionary with ROUGE scores
        """
        try:
            scores = self.rouge_scorer.score(reference, generated)
            
            return {
                'rouge1_f': scores['rouge1'].fmeasure,
                'rouge1_p': scores['rouge1'].precision,
                'rouge1_r': scores['rouge1'].recall,
                'rouge2_f': scores['rouge2'].fmeasure,
                'rouge2_p': scores['rouge2'].precision,
                'rouge2_r': scores['rouge2'].recall,
                'rougeL_f': scores['rougeL'].fmeasure,
                'rougeL_p': scores['rougeL'].precision,
                'rougeL_r': scores['rougeL'].recall,
            }
        except Exception as e:
            logger.error("Failed to calculate ROUGE scores", error=str(e))
            return {metric: 0.0 for metric in ['rouge1_f', 'rouge2_f', 'rougeL_f']}
    
    def calculate_bert_score(self, generated: str, reference: str) -> Dict[str, float]:
        """
        Calculate BERTScore between generated and reference text.
        
        Args:
            generated: Generated answer
            reference: Reference/expected answer
            
        Returns:
            Dictionary with BERTScore metrics
        """
        if bert_score is None:
            logger.warning("BERTScore not available")
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
        
        try:
            P, R, F1 = bert_score([generated], [reference], lang="en", verbose=False)
            
            return {
                'bert_precision': float(P[0]),
                'bert_recall': float(R[0]),
                'bert_f1': float(F1[0])
            }
        except Exception as e:
            logger.error("Failed to calculate BERTScore", error=str(e))
            return {'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0}
    
    def calculate_semantic_similarity(self, generated: str, reference: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Args:
            generated: Generated answer
            reference: Reference/expected answer
            
        Returns:
            Semantic similarity score (0-1)
        """
        if self.sentence_model is None:
            return 0.0
        
        try:
            # Generate embeddings
            embeddings = self.sentence_model.encode([generated, reference])
            
            # Calculate cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error("Failed to calculate semantic similarity", error=str(e))
            return 0.0
    
    def calculate_length_ratio(self, generated: str, reference: str) -> float:
        """
        Calculate length ratio between generated and reference text.
        
        Args:
            generated: Generated answer
            reference: Reference/expected answer
            
        Returns:
            Length ratio (generated/reference)
        """
        gen_len = len(generated.split())
        ref_len = len(reference.split())
        
        if ref_len == 0:
            return 0.0
        
        return gen_len / ref_len
    
    def assess_answer_quality(self, generated: str, reference: str, question: str = "") -> Dict[str, Any]:
        """
        Comprehensive answer quality assessment.
        
        Args:
            generated: Generated answer
            reference: Reference/expected answer
            question: Original question (optional)
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # ROUGE scores
        rouge_scores = self.calculate_rouge_scores(generated, reference)
        metrics.update(rouge_scores)
        
        # BERTScore
        bert_scores = self.calculate_bert_score(generated, reference)
        metrics.update(bert_scores)
        
        # Semantic similarity
        metrics['semantic_similarity'] = self.calculate_semantic_similarity(generated, reference)
        
        # Length metrics
        metrics['length_ratio'] = self.calculate_length_ratio(generated, reference)
        metrics['generated_length'] = len(generated.split())
        metrics['reference_length'] = len(reference.split())
        
        # Quality indicators
        metrics['has_specific_details'] = self._has_specific_details(generated)
        metrics['has_citations'] = self._has_citations(generated)
        metrics['is_coherent'] = self._assess_coherence(generated)
        
        return metrics
    
    def _has_specific_details(self, text: str) -> bool:
        """Check if text contains specific details (numbers, dates, names)."""
        import re
        
        # Look for numbers, percentages, dates, proper nouns
        patterns = [
            r'\d+\.?\d*%',  # Percentages
            r'\d{4}',       # Years
            r'\d+\.?\d*',   # Numbers
            r'[A-Z][a-z]+ [A-Z][a-z]+',  # Proper nouns (names)
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _has_citations(self, text: str) -> bool:
        """Check if text contains citations or source references."""
        import re
        
        citation_patterns = [
            r'\[Source \d+\]',
            r'\(.*\d{4}.*\)',
            r'according to',
            r'as stated in',
            r'from the paper',
        ]
        
        text_lower = text.lower()
        for pattern in citation_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _assess_coherence(self, text: str) -> bool:
        """Simple coherence assessment based on text structure."""
        sentences = text.split('. ')
        
        # Basic checks
        if len(sentences) < 2:
            return True  # Single sentence is coherent by default
        
        # Check for reasonable sentence length variation
        lengths = [len(s.split()) for s in sentences]
        avg_length = np.mean(lengths)
        
        # Very short or very long sentences might indicate incoherence
        if avg_length < 3 or avg_length > 50:
            return False
        
        return True


class EvaluationHarness:
    """
    Evaluation harness for testing RAG system performance.
    """
    
    def __init__(self, orchestrator: AgentOrchestrator):
        """
        Initialize evaluation harness.
        
        Args:
            orchestrator: Agent orchestrator to evaluate
        """
        self.orchestrator = orchestrator
        self.metrics = EvaluationMetrics()
        self.evaluation_history: List[Dict[str, Any]] = []
    
    async def evaluate_single_question(
        self,
        question: str,
        expected_answer: str,
        context: Optional[str] = None,
        session_id: str = "eval_session"
    ) -> EvaluationResult:
        """
        Evaluate a single question-answer pair.
        
        Args:
            question: Question to ask
            expected_answer: Expected/reference answer
            context: Optional context information
            session_id: Session ID for evaluation
            
        Returns:
            EvaluationResult with scores and metadata
        """
        logger.info("Evaluating single question", question=question)
        
        start_time = datetime.utcnow()
        
        try:
            # Generate answer using the orchestrator
            response = await self.orchestrator.process_question(
                question=question,
                session_id=session_id,
                stream=False
            )
            
            generated_answer = response.answer
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Calculate metrics
            scores = self.metrics.assess_answer_quality(
                generated=generated_answer,
                reference=expected_answer,
                question=question
            )
            
            # Determine if answer passes threshold
            # Using ROUGE-L F1 score as primary metric
            primary_score = scores.get('rougeL_f', 0.0)
            pass_threshold = primary_score >= 0.3  # Configurable threshold
            
            # Create evaluation result
            result = EvaluationResult(
                question=question,
                generated_answer=generated_answer,
                expected_answer=expected_answer,
                scores=scores,
                pass_threshold=pass_threshold,
                metadata={
                    'processing_time': processing_time,
                    'confidence_score': response.confidence_score,
                    'sources_used': len(response.sources),
                    'context_provided': context is not None,
                    'primary_score': primary_score,
                    'evaluation_timestamp': start_time.isoformat()
                }
            )
            
            # Store in history
            self.evaluation_history.append({
                'timestamp': start_time.isoformat(),
                'question': question,
                'result': result.dict()
            })
            
            logger.info(
                "Question evaluation complete",
                primary_score=primary_score,
                pass_threshold=pass_threshold,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to evaluate question", error=str(e))
            
            # Return error result
            return EvaluationResult(
                question=question,
                generated_answer=f"Error during evaluation: {str(e)}",
                expected_answer=expected_answer,
                scores={'error': 1.0},
                pass_threshold=False,
                metadata={
                    'error': str(e),
                    'processing_time': (datetime.utcnow() - start_time).total_seconds(),
                    'evaluation_timestamp': start_time.isoformat()
                }
            )
    
    async def evaluate_dataset(
        self,
        evaluation_pairs: List[Dict[str, str]],
        session_id: str = "eval_dataset"
    ) -> Dict[str, Any]:
        """
        Evaluate a dataset of question-answer pairs.
        
        Args:
            evaluation_pairs: List of {'question': str, 'expected_answer': str, 'context': str?} dicts
            session_id: Session ID for evaluation
            
        Returns:
            Dictionary with aggregate evaluation results
        """
        logger.info("Evaluating dataset", num_pairs=len(evaluation_pairs))
        
        start_time = datetime.utcnow()
        results = []
        
        # Process each evaluation pair
        for i, pair in enumerate(evaluation_pairs):
            try:
                logger.info(f"Evaluating question {i+1}/{len(evaluation_pairs)}")
                
                result = await self.evaluate_single_question(
                    question=pair['question'],
                    expected_answer=pair['expected_answer'],
                    context=pair.get('context'),
                    session_id=f"{session_id}_{i}"
                )
                
                results.append(result)
                
                # Small delay between evaluations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to evaluate pair {i}", error=str(e))
                continue
        
        # Calculate aggregate metrics
        aggregate_metrics = self._calculate_aggregate_metrics(results)
        
        evaluation_summary = {
            'dataset_name': 'custom_dataset',
            'total_questions': len(evaluation_pairs),
            'successful_evaluations': len(results),
            'evaluation_time': (datetime.utcnow() - start_time).total_seconds(),
            'aggregate_metrics': aggregate_metrics,
            'individual_results': [result.dict() for result in results],
            'timestamp': start_time.isoformat()
        }
        
        logger.info(
            "Dataset evaluation complete",
            total_questions=len(evaluation_pairs),
            successful=len(results),
            pass_rate=aggregate_metrics.get('pass_rate', 0.0)
        )
        
        return evaluation_summary
    
    def _calculate_aggregate_metrics(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Calculate aggregate metrics from individual results."""
        if not results:
            return {}
        
        # Extract scores from all results
        all_scores = {}
        pass_count = 0
        
        for result in results:
            if result.pass_threshold:
                pass_count += 1
            
            for metric, value in result.scores.items():
                if metric not in all_scores:
                    all_scores[metric] = []
                all_scores[metric].append(value)
        
        # Calculate aggregate statistics
        aggregate = {
            'pass_rate': pass_count / len(results),
            'total_evaluated': len(results)
        }
        
        for metric, values in all_scores.items():
            if metric == 'error':
                continue
            
            aggregate[f'{metric}_mean'] = np.mean(values)
            aggregate[f'{metric}_median'] = np.median(values)
            aggregate[f'{metric}_std'] = np.std(values)
            aggregate[f'{metric}_min'] = np.min(values)
            aggregate[f'{metric}_max'] = np.max(values)
        
        return aggregate
    
    def load_evaluation_dataset(self, dataset_path: str) -> List[Dict[str, str]]:
        """
        Load evaluation dataset from JSON file.
        
        Args:
            dataset_path: Path to JSON file with evaluation data
            
        Returns:
            List of evaluation pairs
        """
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            # Validate format
            if isinstance(data, list):
                evaluation_pairs = data
            elif isinstance(data, dict) and 'evaluation_pairs' in data:
                evaluation_pairs = data['evaluation_pairs']
            else:
                raise ValueError("Invalid dataset format")
            
            # Validate required fields
            for pair in evaluation_pairs:
                if 'question' not in pair or 'expected_answer' not in pair:
                    raise ValueError("Each evaluation pair must have 'question' and 'expected_answer'")
            
            logger.info("Loaded evaluation dataset", num_pairs=len(evaluation_pairs))
            return evaluation_pairs
            
        except Exception as e:
            logger.error("Failed to load evaluation dataset", path=dataset_path, error=str(e))
            return []
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info("Saved evaluation results", output_path=output_path)
            
        except Exception as e:
            logger.error("Failed to save evaluation results", path=output_path, error=str(e))
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted report string
        """
        try:
            aggregate = results.get('aggregate_metrics', {})
            
            report = f"""
# RAG System Evaluation Report

**Dataset:** {results.get('dataset_name', 'Unknown')}
**Timestamp:** {results.get('timestamp', 'Unknown')}
**Total Questions:** {results.get('total_questions', 0)}
**Successful Evaluations:** {results.get('successful_evaluations', 0)}
**Evaluation Time:** {results.get('evaluation_time', 0):.2f} seconds

## Summary Metrics

- **Pass Rate:** {aggregate.get('pass_rate', 0.0):.1%}
- **ROUGE-L F1 (mean):** {aggregate.get('rougeL_f_mean', 0.0):.3f}
- **Semantic Similarity (mean):** {aggregate.get('semantic_similarity_mean', 0.0):.3f}
- **BERTScore F1 (mean):** {aggregate.get('bert_f1_mean', 0.0):.3f}

## Detailed Metrics

| Metric | Mean | Median | Std | Min | Max |
|--------|------|--------|-----|-----|-----|
"""
            
            # Add detailed metrics table
            metrics_to_show = ['rougeL_f', 'semantic_similarity', 'bert_f1', 'rouge1_f', 'rouge2_f']
            
            for metric in metrics_to_show:
                if f'{metric}_mean' in aggregate:
                    mean_val = aggregate.get(f'{metric}_mean', 0.0)
                    median_val = aggregate.get(f'{metric}_median', 0.0)
                    std_val = aggregate.get(f'{metric}_std', 0.0)
                    min_val = aggregate.get(f'{metric}_min', 0.0)
                    max_val = aggregate.get(f'{metric}_max', 0.0)
                    
                    report += f"| {metric} | {mean_val:.3f} | {median_val:.3f} | {std_val:.3f} | {min_val:.3f} | {max_val:.3f} |\n"
            
            # Add quality indicators
            quality_metrics = ['has_specific_details', 'has_citations', 'is_coherent']
            if any(f'{m}_mean' in aggregate for m in quality_metrics):
                report += "\n## Quality Indicators\n\n"
                for metric in quality_metrics:
                    if f'{metric}_mean' in aggregate:
                        score = aggregate.get(f'{metric}_mean', 0.0)
                        report += f"- **{metric.replace('_', ' ').title()}:** {score:.1%}\n"
            
            return report
            
        except Exception as e:
            logger.error("Failed to generate evaluation report", error=str(e))
            return f"Error generating report: {str(e)}"
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        return self.evaluation_history.copy()
    
    def clear_evaluation_history(self):
        """Clear evaluation history."""
        self.evaluation_history.clear()
        logger.info("Evaluation history cleared")