#!/usr/bin/env python3
"""
System Verification Script for PDF RAG System

This script performs comprehensive health checks and verification
of all system components to ensure proper deployment.
"""

import asyncio
import aiohttp
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse


class SystemVerifier:
    """Comprehensive system verification for PDF RAG deployment."""
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        """
        Initialize the verifier.
        
        Args:
            base_url: Base URL of the RAG system
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or self._get_api_key()
        self.session = None
        self.checks_passed = 0
        self.checks_total = 0
        self.failures = []
    
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
        
        return "pdf-rag-secret-key"  # Default fallback
    
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def print_check(self, name: str, passed: bool, details: str = ""):
        """Print check result with formatting."""
        self.checks_total += 1
        status = "✅" if passed else "❌"
        
        if passed:
            self.checks_passed += 1
            print(f"{status} {name}")
            if details:
                print(f"   {details}")
        else:
            print(f"{status} {name}")
            if details:
                print(f"   ❌ {details}")
            self.failures.append(f"{name}: {details}")
    
    async def check_basic_connectivity(self) -> bool:
        """Check basic HTTP connectivity to the system."""
        try:
            async with self.session.get(f"{self.base_url}/") as response:
                if response.status == 200:
                    data = await response.model_dump_json()
                    version = data.get('version', 'unknown')
                    self.print_check("Basic Connectivity", True, f"API version: {version}")
                    return True
                else:
                    self.print_check("Basic Connectivity", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.print_check("Basic Connectivity", False, str(e))
            return False
    
    async def check_health_endpoints(self) -> bool:
        """Check all health endpoints."""
        endpoints = [
            ("/health/", "Basic Health"),
            ("/health/detailed", "Detailed Health"),
            ("/health/qdrant", "Qdrant Health"),
            ("/health/ollama", "Ollama Health"),
        ]
        
        all_healthy = True
        
        for endpoint, name in endpoints:
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    if response.status == 200:
                        data = await response.model_dump_json()
                        status = data.get('status', 'unknown')
                        self.print_check(name, status == 'healthy', f"Status: {status}")
                        if status != 'healthy':
                            all_healthy = False
                    else:
                        self.print_check(name, False, f"HTTP {response.status}")
                        all_healthy = False
            except Exception as e:
                self.print_check(name, False, str(e))
                all_healthy = False
        
        return all_healthy
    
    async def check_authentication(self) -> bool:
        """Check API authentication."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            # Test authenticated endpoint
            async with self.session.get(f"{self.base_url}/documents/", headers=headers) as response:
                if response.status == 200:
                    self.print_check("API Authentication", True, "Valid API key")
                    return True
                elif response.status == 401:
                    self.print_check("API Authentication", False, "Invalid API key")
                    return False
                else:
                    self.print_check("API Authentication", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.print_check("API Authentication", False, str(e))
            return False
    
    async def check_document_operations(self) -> bool:
        """Check document management operations."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            # List documents
            async with self.session.get(f"{self.base_url}/documents/", headers=headers) as response:
                if response.status == 200:
                    data = await response.model_dump_json()
                    count = data.get('total_count', 0)
                    self.print_check("Document Listing", True, f"{count} documents available")
                    
                    # Check document stats
                    async with self.session.get(f"{self.base_url}/documents/stats/overview", headers=headers) as stats_response:
                        if stats_response.status == 200:
                            stats = await stats_response.model_dump_json()
                            completed = stats.get('completed_documents', 0)
                            self.print_check("Document Statistics", True, f"{completed} completed documents")
                            return True
                        else:
                            self.print_check("Document Statistics", False, f"HTTP {stats_response.status}")
                            return False
                else:
                    self.print_check("Document Listing", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.print_check("Document Operations", False, str(e))
            return False
    
    async def check_chat_functionality(self) -> bool:
        """Check chat/question-answering functionality."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        test_question = {
            "question": "What is artificial intelligence?",
            "session_id": "verification_test"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/ask",
                headers=headers,
                json=test_question
            ) as response:
                if response.status == 200:
                    data = await response.model_dump_json()
                    answer = data.get('answer', '')
                    confidence = data.get('confidence_score', 0)
                    
                    if len(answer) > 10:  # Basic sanity check
                        self.print_check("Chat Functionality", True, f"Generated response (confidence: {confidence:.2f})")
                        
                        # Test memory clearing
                        clear_request = {"session_id": "verification_test"}
                        async with self.session.post(
                            f"{self.base_url}/chat/clear-memory",
                            headers=headers,
                            json=clear_request
                        ) as clear_response:
                            if clear_response.status == 200:
                                self.print_check("Session Memory", True, "Memory cleared successfully")
                                return True
                            else:
                                self.print_check("Session Memory", False, f"HTTP {clear_response.status}")
                                return False
                    else:
                        self.print_check("Chat Functionality", False, "Response too short or empty")
                        return False
                else:
                    self.print_check("Chat Functionality", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.print_check("Chat Functionality", False, str(e))
            return False
    
    async def check_evaluation_system(self) -> bool:
        """Check evaluation system functionality."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        test_evaluation = {
            "question": "What is machine learning?",
            "expected_answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed."
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/evaluation/single",
                headers=headers,
                json=test_evaluation
            ) as response:
                if response.status == 200:
                    data = await response.model_dump_json()
                    scores = data.get('scores', {})
                    primary_score = scores.get('rougeL_f', 0)
                    self.print_check("Evaluation System", True, f"ROUGE-L: {primary_score:.3f}")
                    
                    # Check available metrics
                    async with self.session.get(f"{self.base_url}/evaluation/metrics/available") as metrics_response:
                        if metrics_response.status == 200:
                            metrics_data = await metrics_response.model_dump_json()
                            metric_count = len(metrics_data.get('metrics', []))
                            self.print_check("Evaluation Metrics", True, f"{metric_count} metrics available")
                            return True
                        else:
                            self.print_check("Evaluation Metrics", False, f"HTTP {metrics_response.status}")
                            return False
                else:
                    self.print_check("Evaluation System", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.print_check("Evaluation System", False, str(e))
            return False
    
    async def check_streaming_functionality(self) -> bool:
        """Check streaming response functionality."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        test_question = {
            "question": "What is deep learning?",
            "session_id": "streaming_test"
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/chat/ask/stream",
                headers=headers,
                json=test_question
            ) as response:
                if response.status == 200:
                    # Read first few chunks to verify streaming works
                    chunks_received = 0
                    async for line in response.content:
                        if line.startswith(b'data: '):
                            chunks_received += 1
                            if chunks_received >= 3:  # Got some chunks
                                break
                    
                    if chunks_received > 0:
                        self.print_check("Streaming Responses", True, f"Received {chunks_received} chunks")
                        return True
                    else:
                        self.print_check("Streaming Responses", False, "No streaming chunks received")
                        return False
                else:
                    self.print_check("Streaming Responses", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.print_check("Streaming Responses", False, str(e))
            return False
    
    async def check_frontend_accessibility(self) -> bool:
        """Check if Streamlit frontend is accessible."""
        frontend_url = "http://localhost:8501"
        
        try:
            async with self.session.get(frontend_url) as response:
                if response.status == 200:
                    self.print_check("Frontend Accessibility", True, "Streamlit UI accessible")
                    return True
                else:
                    self.print_check("Frontend Accessibility", False, f"HTTP {response.status}")
                    return False
        except Exception as e:
            self.print_check("Frontend Accessibility", False, str(e))
            return False
    
    async def check_external_services(self) -> bool:
        """Check external service availability."""
        services = [
            ("http://localhost:6333/health", "Qdrant Vector DB"),
            ("http://localhost:11434/api/tags", "Ollama LLM Server"),
        ]
        
        all_available = True
        
        for url, name in services:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        self.print_check(f"External Service: {name}", True, "Service responding")
                    else:
                        self.print_check(f"External Service: {name}", False, f"HTTP {response.status}")
                        all_available = False
            except Exception as e:
                self.print_check(f"External Service: {name}", False, str(e))
                all_available = False
        
        return all_available
    
    def check_file_structure(self) -> bool:
        """Check required files and directories exist."""
        required_paths = [
            ("docker-compose.yml", "file", "Docker Compose Configuration"),
            ("requirements.txt", "file", "Python Dependencies"),
            (".env", "file", "Environment Configuration"),
            ("src/", "dir", "Source Code Directory"),
            ("data/", "dir", "Data Directory"),
            ("logs/", "dir", "Logs Directory"),
            ("frontend/", "dir", "Frontend Directory"),
            ("scripts/", "dir", "Scripts Directory"),
        ]
        
        all_exist = True
        
        for path, path_type, description in required_paths:
            path_obj = Path(path)
            
            if path_type == "file":
                exists = path_obj.is_file()
            else:
                exists = path_obj.is_dir()
            
            if exists:
                self.print_check(f"File Structure: {description}", True, f"{path} exists")
            else:
                self.print_check(f"File Structure: {description}", False, f"{path} missing")
                all_exist = False
        
        return all_exist
    
    async def run_comprehensive_verification(self) -> bool:
        """Run all verification checks."""
        print("🔍 PDF RAG System Verification")
        print("=" * 50)
        
        # File structure checks (synchronous)
        print("\n📁 File Structure Checks:")
        file_structure_ok = self.check_file_structure()
        
        # Network connectivity checks
        print("\n🌐 Connectivity Checks:")
        basic_connectivity = await self.check_basic_connectivity()
        
        if not basic_connectivity:
            print("\n❌ Basic connectivity failed. Cannot proceed with further checks.")
            return False
        
        # Health checks
        print("\n🩺 Health Checks:")
        health_ok = await self.check_health_endpoints()
        
        # Authentication checks
        print("\n🔐 Authentication Checks:")
        auth_ok = await self.check_authentication()
        
        # Functional checks
        print("\n⚙️ Functional Checks:")
        doc_ops_ok = await self.check_document_operations()
        chat_ok = await self.check_chat_functionality()
        eval_ok = await self.check_evaluation_system()
        streaming_ok = await self.check_streaming_functionality()
        
        # External services
        print("\n🔌 External Services:")
        external_ok = await self.check_external_services()
        
        # Frontend
        print("\n🖥️ Frontend Checks:")
        frontend_ok = await self.check_frontend_accessibility()
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 50)
        
        success_rate = (self.checks_passed / self.checks_total) * 100 if self.checks_total > 0 else 0
        
        print(f"✅ Checks Passed: {self.checks_passed}/{self.checks_total} ({success_rate:.1f}%)")
        
        if self.failures:
            print(f"\n❌ Failed Checks ({len(self.failures)}):")
            for failure in self.failures:
                print(f"   • {failure}")
        
        overall_success = success_rate >= 80  # 80% success rate threshold
        
        if overall_success:
            print(f"\n🎉 VERIFICATION SUCCESSFUL!")
            print("   Your PDF RAG system is ready for use!")
            print("\n🚀 Quick Start:")
            print("   • Frontend: http://localhost:8501")
            print("   • API Docs: http://localhost:8000/docs")
            print(f"   • API Key: {self.api_key}")
        else:
            print(f"\n⚠️ VERIFICATION INCOMPLETE")
            print("   Some components are not working properly.")
            print("   Please check the failed items above.")
        
        return overall_success


async def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description='Verify PDF RAG system deployment')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000', help='API base URL')
    parser.add_argument('--api-key', type=str, help='API key (or set API_KEY environment variable)')
    parser.add_argument('--quick', action='store_true', help='Run only basic checks')
    
    args = parser.parse_args()
    
    try:
        async with SystemVerifier(args.api_url, args.api_key) as verifier:
            if args.quick:
                # Quick verification - basic checks only
                print("🔍 Quick System Verification")
                print("=" * 30)
                
                basic_ok = await verifier.check_basic_connectivity()
                if basic_ok:
                    health_ok = await verifier.check_health_endpoints()
                    auth_ok = await verifier.check_authentication()
                    
                    if health_ok and auth_ok:
                        print("\n✅ Quick verification passed!")
                        return 0
                    else:
                        print("\n❌ Quick verification failed!")
                        return 1
                else:
                    print("\n❌ System not accessible!")
                    return 1
            else:
                # Comprehensive verification
                success = await verifier.run_comprehensive_verification()
                return 0 if success else 1
                
    except KeyboardInterrupt:
        print("\n⏹️ Verification cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)