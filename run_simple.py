#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent
src_path = project_root / "src"

# Add paths to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

# Set environment variables
os.environ.setdefault("PYTHONPATH", f"{project_root}:{src_path}")

print(f"🐍 Python Path: {sys.path[:3]}...")
print(f"📁 Working Directory: {os.getcwd()}")
print(f"🔧 Project Root: {project_root}")

try:
    # Test imports
    print("🧪 Testing imports...")
    from src.core.config import Settings
    print("✅ Config import successful")
    
    from src.services.vector_service import VectorService
    print("✅ Vector service import successful")
    
    from src.agents.orchestrator import AgentOrchestrator
    print("✅ Simple orchestrator import successful")
    
    from src.api.main_simple import app
    print("✅ Simple FastAPI app import successful")
    
    # Run the app
    print("🚀 Starting FastAPI server...")
    import uvicorn
    uvicorn.run(
        "src.api.main_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n🔍 Debugging info:")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Python executable: {sys.executable}")
    print(f"   Python version: {sys.version}")
    print(f"   Python path: {sys.path}")
    
    print("\n💡 Try running from the project root directory:")
    print("   cd /path/to/pdf-rag-system")
    print("   python run_simple.py")
    
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)