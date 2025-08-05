#!/bin/bash

# PDF RAG System Setup Script
# This script helps you set up the PDF RAG system quickly

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Docker
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check Python (for local development)
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_status "Found Python $PYTHON_VERSION"
        
        if [[ "$(printf '%s\n' "3.11" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.11" ]]; then
            print_warning "Python 3.11+ is recommended. Current version: $PYTHON_VERSION"
        fi
    else
        print_warning "Python 3 not found. Required for local development."
    fi
    
    # Check available memory
    if command_exists free; then
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
        if [ "$TOTAL_MEM" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Performance may be limited."
        fi
    fi
    
    print_success "System requirements check completed"
}

# Function to setup environment file
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env file from template"
        else
            print_error ".env.example not found"
            exit 1
        fi
    else
        print_warning ".env file already exists. Skipping creation."
    fi
    
    # Generate random API key if using default
    if grep -q "API_KEY=pdf-rag-secret-key" .env; then
        NEW_API_KEY=$(openssl rand -hex 32)
        sed -i.bak "s/API_KEY=pdf-rag-secret-key/API_KEY=$NEW_API_KEY/" .env
        print_success "Generated secure API key"
        print_warning "Your API key: $NEW_API_KEY"
        print_warning "Save this key - you'll need it for API access!"
    fi
    
    print_status "Environment file setup completed"
}

# Function to setup directories
setup_directories() {
    print_status "Creating required directories..."
    
    mkdir -p data
    mkdir -p logs
    mkdir -p config
    
    print_success "Directories created"
}

# Function to pull and setup Ollama models
setup_ollama() {
    print_status "Setting up Ollama models..."
    
    # Start Ollama container
    docker-compose up -d ollama
    
    # Wait for Ollama to be ready
    print_status "Waiting for Ollama to start..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker exec ollama ollama list >/dev/null 2>&1; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Ollama failed to start within 60 seconds"
        return 1
    fi
    
    # Pull the default model
    print_status "Pulling Ollama model (this may take a while)..."
    docker exec ollama ollama pull mistral
    
    print_success "Ollama setup completed"
}

# Function to start all services
start_services() {
    print_status "Starting all services..."
    
    # Start infrastructure services first
    docker-compose up -d qdrant
    
    # Wait for Qdrant to be ready
    print_status "Waiting for Qdrant to start..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:6333/health >/dev/null 2>&1; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    # Start Ollama if not already running
    if ! docker-compose ps ollama | grep -q "Up"; then
        setup_ollama
    fi
    
    # Start application services
    docker-compose up -d backend frontend
    
    # Wait for backend to be ready
    print_status "Waiting for backend to start..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Backend failed to start within 60 seconds"
        return 1
    fi
    
    print_success "All services started successfully"
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check service health
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        print_success "Backend health check passed"
    else
        print_error "Backend health check failed"
        return 1
    fi
    
    # Check Streamlit
    if curl -s http://localhost:8501 >/dev/null 2>&1; then
        print_success "Frontend is accessible"
    else
        print_error "Frontend is not accessible"
        return 1
    fi
    
    # Check Qdrant
    if curl -s http://localhost:6333/health >/dev/null 2>&1; then
        print_success "Qdrant is running"
    else
        print_error "Qdrant is not accessible"
        return 1
    fi
    
    print_success "Installation verification completed"
}

# Function to show next steps
show_next_steps() {
    print_success "Setup completed successfully!"
    echo
    echo "🚀 Your PDF RAG System is ready!"
    echo
    echo "🌐 Access the Chat Interface:"
    echo "     👉 http://localhost:8501"
    echo
    echo "📋 Additional Access Points:"
    echo "  🔧 API Documentation:   http://localhost:8000/docs"
    echo "  ❤️  Health Check:        http://localhost:8000/health"
    echo
    echo "📚 How to Use:"
    echo "  1. Open http://localhost:8501 in your browser"
    echo "  2. Upload PDF documents using the sidebar upload button"
    echo "  3. Wait for processing (you'll see a progress indicator)"
    echo "  4. Start asking questions in the chat interface!"
    echo
    echo "💡 Example Questions to Try:"
    echo "  • 'What is the main contribution of this paper?'"
    echo "  • 'How does the proposed method work?'"
    echo "  • 'What were the experimental results?'"
    echo "  • 'What did OpenAI release recently?' (web search)"
    echo
    echo "🔧 System Management:"
    echo "  📊 View logs:     docker-compose logs -f"
    echo "  🔄 Restart:       docker-compose restart"
    echo "  🛑 Stop:          docker-compose down"
    echo "  🧹 Reset data:    docker-compose down -v"
    echo
    echo "🎯 No coding required - just use the web interface!"
    echo
}

# Function to setup for development
setup_development() {
    print_status "Setting up development environment..."
    
    if ! command_exists python3; then
        print_error "Python 3 is required for development setup"
        return 1
    fi
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created virtual environment"
    fi
    
    # Activate virtual environment and install dependencies
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -r requirements-frontend.txt
    pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy
    
    print_success "Development environment setup completed"
    print_status "To activate: source venv/bin/activate"
}

# Main function
main() {
    echo "🚀 PDF RAG System Setup"
    echo "======================="
    echo
    
    # Parse command line arguments
    SETUP_TYPE="production"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev|--development)
                SETUP_TYPE="development"
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--dev|--development] [--help]"
                echo
                echo "Options:"
                echo "  --dev, --development    Setup for development (includes Python venv)"
                echo "  --help, -h             Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Run setup steps
    check_requirements
    setup_environment
    setup_directories
    
    if [ "$SETUP_TYPE" = "development" ]; then
        setup_development
    fi
    
    start_services
    verify_installation
    show_next_steps
}

# Error handling
trap 'print_error "Setup failed! Check the logs above for details."' ERR

# Run main function
main "$@"