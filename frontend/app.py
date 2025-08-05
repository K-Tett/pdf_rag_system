"""
Streamlit frontend for the PDF RAG system.
"""
import streamlit as st
import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
import uuid

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "pdf-rag-secret-key")

# Helper functions
def get_headers():
    """Get headers with API key."""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
st.set_page_config(
    page_title="PDF RAG System - Chat with Your Documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
    }
    .status-healthy { color: #28a745; }
    .status-unhealthy { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_headers():
    """Get headers with API key."""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

def get_session_id():
    """Get or create session ID."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None):
    """Make API request with error handling."""
    url = f"{BACKEND_URL}{endpoint}"
    headers = get_headers()
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=30)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=30)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None

def upload_file_request(file_data, filename: str):
    """Upload file with proper handling."""
    url = f"{BACKEND_URL}/documents/upload"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    files = {"file": (filename, file_data, "application/pdf")}
    data = {"process_immediately": True}
    
    try:
        response = requests.post(url, headers=headers, files=files, data=data, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {str(e)}")
        return None

def check_backend_health():
    """Check if backend is healthy."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def format_confidence(confidence: float) -> str:
    """Format confidence score with color."""
    if confidence >= 0.8:
        return f'<span class="confidence-high">High ({confidence:.1%})</span>'
    elif confidence >= 0.5:
        return f'<span class="confidence-medium">Medium ({confidence:.1%})</span>'
    else:
        return f'<span class="confidence-low">Low ({confidence:.1%})</span>'

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

# Main app
def main():
    st.markdown('<h1 class="main-header">📚 Chat with Your PDF Documents</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload PDFs and ask questions - powered by AI agents and hybrid search</p>', unsafe_allow_html=True)
    
    # Check backend health
    if not check_backend_health():
        st.error("⚠️ Backend service is not available. Please run `docker-compose up -d` and wait for services to start.")
        st.info("💡 **First time setup?** Run these commands:")
        st.code("""
# Start the system
docker-compose up -d

# Wait for Ollama model (first time only)
docker exec ollama ollama pull mistral

# Check if ready
curl http://localhost:8000/health
        """)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Document Management")
        
        # Upload section with better UI
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Upload Your PDFs")
        st.markdown("*Upload academic papers, research documents, or any PDF files you want to chat with.*")
        
        uploaded_file = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            help="Select one or more PDF documents to add to your knowledge base",
            accept_multiple_files=False
        )
        
        col1, col2 = st.columns(2)
        with col1:
            upload_button = st.button("📤 Upload & Process", use_container_width=True)
        with col2:
            if st.button("🔄 Refresh List", use_container_width=True):
                st.session_state.documents = []
                st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file and upload_button:
            with st.spinner("🔄 Uploading and processing document... This may take a minute."):
                result = upload_file_request(uploaded_file.getvalue(), uploaded_file.name)
                if result:
                    st.success(f"✅ '{uploaded_file.name}' uploaded successfully!")
                    st.balloons()
                    st.session_state.documents = []  # Force refresh
                    st.experimental_rerun()
        
        st.divider()
        
        # Document list with better formatting
        st.subheader("📚 Your Document Library")
        
        if not st.session_state.documents:
            documents_data = make_api_request("/documents/")
            if documents_data:
                st.session_state.documents = documents_data.get("documents", [])
        
        if st.session_state.documents:
            st.success(f"📊 {len(st.session_state.documents)} documents loaded")
            
            for doc in st.session_state.documents:
                status_emoji = "✅" if doc['processing_status'] == 'completed' else "⏳"
                
                with st.expander(f"{status_emoji} {doc['filename']}", expanded=False):
                    st.markdown(f"**📖 Title:** {doc.get('title', 'N/A')}")
                    st.markdown(f"**👤 Author:** {doc.get('author', 'N/A')}")
                    st.markdown(f"**📊 Status:** {doc['processing_status']}")
                    st.markdown(f"**🧩 Chunks:** {doc['num_chunks']}")
                    st.markdown(f"**📅 Uploaded:** {doc['upload_date'][:10]}")
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{doc['id']}", use_container_width=True):
                        if make_api_request(f"/documents/{doc['id']}", method="DELETE"):
                            st.success("✅ Document deleted!")
                            st.session_state.documents = []
                            st.experimental_rerun()
        else:
            st.info("📁 No documents uploaded yet")
            st.markdown("**👆 Upload your first PDF above to get started!**")
        
        st.divider()
        
        # Session management
        st.subheader("💬 Chat Session")
        session_id = get_session_id()
        st.caption(f"Session: {session_id[:8]}...")
        
        if st.button("🧹 Clear Conversation", use_container_width=True):
            data = {"session_id": session_id}
            if make_api_request("/chat/clear-memory", method="POST", data=data):
                st.session_state.messages = []
                st.success("✅ Conversation cleared!")
                st.experimental_rerun()
        
        # System status
        st.divider()
        st.subheader("🔧 System Status")
        
        status_container = st.empty()
        if st.button("🔍 Check Status", use_container_width=True):
            with st.spinner("Checking system status..."):
                health_data = make_api_request("/health/")
                if health_data:
                    status = health_data['status']
                    status_class = "status-healthy" if status == "healthy" else "status-unhealthy"
                    status_container.markdown(f'**Overall Status:** <span class="{status_class}">{status.upper()}</span>', unsafe_allow_html=True)
                    
                    deps = health_data.get('dependencies', {})
                    for service, service_status in deps.items():
                        emoji = "✅" if service_status == "healthy" else "❌"
                        st.caption(f"{emoji} {service.title()}: {service_status}")
                else:
                    status_container.error("❌ Unable to get system status")

    # Main chat interface
    st.header("💬 Ask Questions About Your Documents")
    
    # Instructions for new users
    if not st.session_state.messages and not st.session_state.documents:
        st.info("👋 **Welcome!** To get started:")
        st.markdown("""
        1. **📤 Upload a PDF** using the sidebar (academic papers work great!)
        2. **⏳ Wait for processing** (usually takes 30-60 seconds)
        3. **💬 Ask questions** about your documents in the chat below
        4. **🔍 Get answers** with sources and confidence scores
        """)
        
        st.markdown("**🌐 Web Search:** Ask current questions like *'What did OpenAI release recently?'* and the system will search the web!")
    
    elif not st.session_state.documents:
        st.warning("📁 **No documents uploaded yet.** Upload a PDF in the sidebar to start asking questions!")
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><strong>🙋 You:</strong> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message"><strong>🤖 Assistant:</strong> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
                
                # Show sources and confidence if available
                if "metadata" in message:
                    metadata = message["metadata"]
                    
                    # Show confidence and sources in columns
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        # Confidence score
                        if "confidence" in metadata:
                            confidence_html = format_confidence(metadata["confidence"])
                            st.markdown(f"**🎯 Confidence:** {confidence_html}", unsafe_allow_html=True)
                    
                    with col2:
                        # Show source count
                        if "sources" in metadata and metadata["sources"]:
                            st.markdown(f"**📚 Sources:** {len(metadata['sources'])} documents used")
                    
                    # Sources detail
                    if "sources" in metadata and metadata["sources"]:
                        with st.expander(f"📖 View {len(metadata['sources'])} Sources", expanded=False):
                            for i, source in enumerate(metadata["sources"], 1):
                                st.markdown(f"**Source {i}: {source.get('title', 'Unknown')}**")
                                st.markdown(f"*Score: {source.get('score', 0):.2f}*")
                                st.markdown(source.get('content', 'No content available')[:300] + "...")
                                if source.get('metadata', {}).get('url'):
                                    st.markdown(f"🔗 [View Original]({source['metadata']['url']})")
                                st.divider()
    
    # Chat input with better styling
    st.markdown("### 💭 Ask a Question")
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your question here...",
                placeholder="e.g., What is the main contribution of the paper? How does this method work?",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("🚀 Ask", use_container_width=True)
        
        # Quick action buttons
        if st.session_state.documents:
            st.markdown("**💡 Quick Questions:**")
            quick_questions = [
                "What is the main contribution of this paper?",
                "What methodology was used in this research?", 
                "What were the key results and findings?",
                "How does this work compare to previous approaches?"
            ]
            
            cols = st.columns(2)
            for i, question in enumerate(quick_questions):
                col = cols[i % 2]
                with col:
                    if st.form_submit_button(f"💬 {question}", use_container_width=True):
                        user_input = question
                        submit_button = True
    
    # Handle chat submission
    if submit_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show thinking message
        with st.spinner("Thinking..."):
            # Make API request
            session_id = get_session_id()
            data = {
                "question": user_input,
                "session_id": session_id
            }
            
            response = make_api_request("/chat/ask", method="POST", data=data)
            
            if response:
                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "metadata": {
                        "confidence": response.get("confidence_score", 0.0),
                        "sources": response.get("sources", [])
                    }
                }
                st.session_state.messages.append(assistant_message)
            else:
                error_message = {
                    "role": "assistant",
                    "content": "I'm sorry, I encountered an error while processing your question. Please try again."
                }
                st.session_state.messages.append(error_message)
        
        st.experimental_rerun()

    # Example questions for new users
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("### 🌟 Try These Example Questions")
        st.markdown("*Click any button below to try different types of questions:*")
        
        # Organize examples by type
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📄 Document Questions:**")
            doc_questions = [
                "What is the main contribution of this paper?",
                "What methodology was used in this research?",
                "What were the experimental results?"
            ]
            for question in doc_questions:
                if st.button(question, key=f"doc_{hash(question)}", use_container_width=True):
                    st.session_state.example_question = question
                    st.experimental_rerun()
        
        with col2:
            st.markdown("**🌐 Web Search Questions:**")
            web_questions = [
                "What did OpenAI release this month?",
                "Latest developments in AI research",
                "Recent advances in machine learning"
            ]
            for question in web_questions:
                if st.button(question, key=f"web_{hash(question)}", use_container_width=True):
                    st.session_state.example_question = question
                    st.experimental_rerun()
        
        # Tips section
        st.markdown("---")
        st.markdown("### 💡 Tips for Better Results")
        st.markdown("""
        - **📊 Be specific**: Ask about particular results, methods, or findings
        - **🔍 Ask follow-ups**: Build on previous answers for deeper understanding  
        - **🌐 Current info**: Ask about recent events and the system will search the web
        - **❓ Clarification**: If a question is ambiguous, the system will ask for clarification
        - **📚 Multiple docs**: The system can synthesize information across all your documents
        """)
        
        # Handle example question selection
        if hasattr(st.session_state, 'example_question'):
            user_input = st.session_state.example_question
            del st.session_state.example_question
            
            # Process the example question
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("🤔 Thinking... This may take a moment for the first question."):
                session_id = get_session_id()
                data = {
                    "question": user_input,
                    "session_id": session_id
                }
                
                response = make_api_request("/chat/ask", method="POST", data=data)
                
                if response:
                    assistant_message = {
                        "role": "assistant",
                        "content": response["answer"],
                        "metadata": {
                            "confidence": response.get("confidence_score", 0.0),
                            "sources": response.get("sources", [])
                        }
                    }
                    st.session_state.messages.append(assistant_message)
                
            st.experimental_rerun()

# Footer with helpful information
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <strong>🚀 PDF RAG System</strong> | Built with LangGraph Multi-Agent Architecture<br>
        <small>💡 Upload PDFs → Ask Questions → Get AI-Powered Answers with Sources</small><br>
        <small>🔧 System Status: <a href="http://localhost:8000/health" target="_blank">Health Check</a> | 
        📖 API Docs: <a href="http://localhost:8000/docs" target="_blank">Documentation</a></small>
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()