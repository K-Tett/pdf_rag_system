#!/usr/bin/env python3
"""
PDF Ingestion Script for Bulk Document Processing

This script helps you ingest multiple PDF documents into the RAG system.
It can process individual files, directories, or download sample papers.
"""

import os
import sys
import argparse
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import List, Dict, Any
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.config import Settings


class PDFIngester:
    """Handles bulk PDF ingestion into the RAG system."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000", api_key: str = None):
        """
        Initialize the ingester.
        
        Args:
            api_base_url: Base URL of the RAG system API
            api_key: API key for authentication
        """
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key or self._get_api_key()
        self.session = None
        
        if not self.api_key:
            raise ValueError("API key is required. Set API_KEY environment variable or pass it directly.")
    
    def _get_api_key(self) -> str:
        """Get API key from environment or .env file."""
        # Try environment variable first
        api_key = os.getenv('API_KEY')
        if api_key:
            return api_key
        
        # Try .env file
        env_file = Path('.env')
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('API_KEY='):
                        return line.split('=', 1)[1].strip()
        
        return None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.api_key}'},
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def check_system_health(self) -> bool:
        """Check if the RAG system is healthy and accessible."""
        try:
            async with self.session.get(f'{self.api_base_url}/health') as response:
                if response.status == 200:
                    data = await response.model_dump_json()
                    return data.get('status') == 'healthy'
                return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    async def upload_pdf(self, file_path: Path, process_immediately: bool = True) -> Dict[str, Any]:
        """
        Upload a single PDF file.
        
        Args:
            file_path: Path to the PDF file
            process_immediately: Whether to process the document immediately
            
        Returns:
            Response data from the API
        """
        try:
            with open(file_path, 'rb') as file:
                data = aiohttp.FormData()
                data.add_field('file', file, filename=file_path.name, content_type='application/pdf')
                data.add_field('process_immediately', str(process_immediately).lower())
                
                async with self.session.post(
                    f'{self.api_base_url}/documents/upload',
                    data=data
                ) as response:
                    if response.status == 200:
                        result = await response.model_dump_json()
                        print(f"✅ Uploaded: {file_path.name}")
                        return result
                    else:
                        error_text = await response.text()
                        print(f"❌ Upload failed for {file_path.name}: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            print(f"❌ Error uploading {file_path.name}: {e}")
            return None
    
    async def ingest_directory(self, directory: Path, recursive: bool = False) -> List[Dict[str, Any]]:
        """
        Ingest all PDF files from a directory.
        
        Args:
            directory: Directory containing PDF files
            recursive: Whether to search subdirectories
            
        Returns:
            List of upload results
        """
        pdf_files = []
        
        if recursive:
            pdf_files = list(directory.rglob("*.pdf"))
        else:
            pdf_files = list(directory.glob("*.pdf"))
        
        if not pdf_files:
            print(f"⚠️ No PDF files found in {directory}")
            return []
        
        print(f"📚 Found {len(pdf_files)} PDF files")
        
        results = []
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"📄 Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            result = await self.upload_pdf(pdf_file)
            if result:
                results.append(result)
            
            # Small delay between uploads to avoid overwhelming the system
            await asyncio.sleep(0.5)
        
        return results
    
    async def download_sample_papers(self, output_dir: Path) -> List[Path]:
        """
        Download sample research papers for testing.
        
        Args:
            output_dir: Directory to save downloaded papers
            
        Returns:
            List of downloaded file paths
        """
        # Note: In a real implementation, you might download from arXiv or other sources
        # For this demo, we'll create sample content
        
        output_dir.mkdir(exist_ok=True)
        sample_papers = []
        
        # Create sample PDF content (placeholder)
        sample_content = {
            "rag_survey.txt": """
# Retrieval-Augmented Generation: A Comprehensive Survey

## Abstract
Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing large language models with external knowledge sources. This survey provides a comprehensive overview of RAG techniques, applications, and future directions.

## Introduction
Large language models (LLMs) have shown remarkable capabilities in natural language understanding and generation. However, they face limitations in accessing up-to-date information and domain-specific knowledge.

## Methodology
RAG systems typically consist of three main components:
1. Retrieval system for finding relevant documents
2. Knowledge base containing external information  
3. Generation model that incorporates retrieved information

## Results
Our analysis shows that RAG systems achieve 15-25% improvement in factual accuracy compared to standalone LLMs across various benchmarks.

## Conclusion
RAG represents a promising direction for building more reliable and knowledgeable AI systems.
""",
            "hybrid_search.txt": """
# Hybrid Search for Information Retrieval

## Abstract
This paper presents a novel approach to information retrieval that combines dense vector search with sparse keyword matching for improved retrieval performance.

## Introduction
Traditional keyword-based search and modern dense retrieval methods each have distinct advantages and limitations.

## Methodology
Our hybrid approach uses:
- Dense embeddings for semantic similarity
- BM25 for exact keyword matching
- Fusion techniques for combining results

## Experiments
We evaluated on MS MARCO and Natural Questions datasets, achieving:
- 12% improvement in MRR@10
- 8% improvement in Recall@100

## Results
The hybrid approach consistently outperforms both dense and sparse methods individually across multiple domains.
""",
            "prompt_engineering.txt": """
# Prompt Engineering for Large Language Models

## Abstract
Effective prompt design is crucial for maximizing the performance of large language models. This study analyzes various prompting strategies and their impact on model outputs.

## Background
Prompt engineering has become an essential skill for working with LLMs, requiring understanding of model behavior and task requirements.

## Techniques
We evaluate several prompting strategies:
1. Zero-shot prompting
2. Few-shot learning
3. Chain-of-thought reasoning
4. Instruction tuning

## Results
Chain-of-thought prompting shows 20-30% improvement on reasoning tasks, while few-shot examples improve performance by 15% on average.

## Applications
These techniques are particularly effective for:
- Mathematical reasoning
- Code generation
- Question answering
- Text summarization
"""
        }
        
        print("📥 Creating sample papers...")
        
        for filename, content in sample_content.items():
            # Convert to PDF-like format (simplified)
            pdf_filename = filename.replace('.txt', '.pdf')
            file_path = output_dir / pdf_filename
            
            # For this demo, we'll save as text files that can be processed
            # In a real implementation, you'd create actual PDFs
            with open(file_path.with_suffix('.txt'), 'w') as f:
                f.write(content)
            
            print(f"📝 Created: {pdf_filename} (as .txt for demo)")
            sample_papers.append(file_path.with_suffix('.txt'))
        
        return sample_papers
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system status including document count."""
        try:
            async with self.session.get(f'{self.api_base_url}/documents/') as response:
                if response.status == 200:
                    data = await response.model_dump_json()
                    return {
                        'total_documents': data.get('total_count', 0),
                        'documents': data.get('documents', [])
                    }
                return {}
        except Exception as e:
            print(f"❌ Error getting system status: {e}")
            return {}


async def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description='Ingest PDF documents into the RAG system')
    parser.add_argument('--file', type=str, help='Single PDF file to ingest')
    parser.add_argument('--directory', type=str, help='Directory containing PDF files')
    parser.add_argument('--recursive', action='store_true', help='Search subdirectories recursively')
    parser.add_argument('--sample', action='store_true', help='Download and ingest sample papers')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000', help='API base URL')
    parser.add_argument('--api-key', type=str, help='API key (or set API_KEY environment variable)')
    parser.add_argument('--output-dir', type=str, default='./sample_papers', help='Output directory for sample papers')
    
    args = parser.parse_args()
    
    if not any([args.file, args.directory, args.sample]):
        parser.print_help()
        print("\n❌ Please specify --file, --directory, or --sample")
        return
    
    try:
        async with PDFIngester(args.api_url, args.api_key) as ingester:
            # Check system health
            print("🩺 Checking system health...")
            if not await ingester.check_system_health():
                print("❌ RAG system is not healthy. Please check if services are running.")
                return
            
            print("✅ RAG system is healthy")
            
            # Get initial status
            initial_status = await ingester.get_system_status()
            initial_count = initial_status.get('total_documents', 0)
            print(f"📊 Current documents in system: {initial_count}")
            
            results = []
            
            # Handle sample papers
            if args.sample:
                print("📥 Downloading sample papers...")
                output_dir = Path(args.output_dir)
                sample_files = await ingester.download_sample_papers(output_dir)
                
                print(f"📚 Ingesting {len(sample_files)} sample papers...")
                for file_path in sample_files:
                    # Note: For demo purposes, these are .txt files
                    # In production, you'd have actual PDFs
                    print(f"📄 Would ingest: {file_path.name} (sample content created)")
                
                print("✅ Sample papers ready for ingestion")
                print(f"💡 You can now upload the files in {output_dir} manually via the UI")
            
            # Handle single file
            elif args.file:
                file_path = Path(args.file)
                if not file_path.exists():
                    print(f"❌ File not found: {file_path}")
                    return
                
                if not file_path.suffix.lower() == '.pdf':
                    print(f"❌ File is not a PDF: {file_path}")
                    return
                
                print(f"📄 Ingesting single file: {file_path.name}")
                result = await ingester.upload_pdf(file_path)
                if result:
                    results.append(result)
            
            # Handle directory
            elif args.directory:
                directory = Path(args.directory)
                if not directory.exists():
                    print(f"❌ Directory not found: {directory}")
                    return
                
                print(f"📁 Ingesting from directory: {directory}")
                results = await ingester.ingest_directory(directory, args.recursive)
            
            # Show final status
            if results:
                final_status = await ingester.get_system_status()
                final_count = final_status.get('total_documents', 0)
                
                print(f"\n📊 Ingestion Summary:")
                print(f"  • Successfully uploaded: {len(results)} documents")
                print(f"  • Documents before: {initial_count}")
                print(f"  • Documents after: {final_count}")
                print(f"  • Net increase: {final_count - initial_count}")
                
                # Show uploaded documents
                print(f"\n📚 Uploaded Documents:")
                for result in results[:5]:  # Show first 5
                    status = result.get('processing_status', 'unknown')
                    filename = result.get('filename', 'unknown')
                    print(f"  • {filename}: {status}")
                
                if len(results) > 5:
                    print(f"  • ... and {len(results) - 5} more")
                
                print(f"\n🚀 Ready to ask questions! Visit http://localhost:8501")
            
    except KeyboardInterrupt:
        print("\n⏹️ Ingestion cancelled by user")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)