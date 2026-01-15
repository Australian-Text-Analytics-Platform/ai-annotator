#!/bin/bash
# Start FastAPI backend and Streamlit frontend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_DIR/.pids"
LOG_DIR="$PROJECT_DIR/.logs"

# Create directories
mkdir -p "$PID_DIR" "$LOG_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting AI Annotator services...${NC}"

# Check if services are already running
if [ -f "$PID_DIR/fastapi.pid" ]; then
    PID=$(cat "$PID_DIR/fastapi.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${YELLOW}FastAPI is already running (PID: $PID)${NC}"
        FASTAPI_RUNNING=true
    else
        rm "$PID_DIR/fastapi.pid"
        FASTAPI_RUNNING=false
    fi
else
    FASTAPI_RUNNING=false
fi

if [ -f "$PID_DIR/streamlit.pid" ]; then
    PID=$(cat "$PID_DIR/streamlit.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${YELLOW}Streamlit is already running (PID: $PID)${NC}"
        STREAMLIT_RUNNING=true
    else
        rm "$PID_DIR/streamlit.pid"
        STREAMLIT_RUNNING=false
    fi
else
    STREAMLIT_RUNNING=false
fi

# Activate conda environment
echo "Activating conda environment: atap_classifier"
source /Users/seb/mambaforge/bin/activate atap_classifier

cd "$PROJECT_DIR"

# Start FastAPI backend
if [ "$FASTAPI_RUNNING" = false ]; then
    echo "Starting FastAPI backend..."
    nohup python run_api.py > "$LOG_DIR/fastapi.log" 2>&1 &
    FASTAPI_PID=$!
    echo $FASTAPI_PID > "$PID_DIR/fastapi.pid"
    echo -e "${GREEN}FastAPI started (PID: $FASTAPI_PID)${NC}"
    sleep 2  # Give FastAPI time to start
fi

# Start Streamlit frontend
if [ "$STREAMLIT_RUNNING" = false ]; then
    echo "Starting Streamlit frontend..."
    nohup streamlit run streamlit_app/app.py > "$LOG_DIR/streamlit.log" 2>&1 &
    STREAMLIT_PID=$!
    echo $STREAMLIT_PID > "$PID_DIR/streamlit.pid"
    echo -e "${GREEN}Streamlit started (PID: $STREAMLIT_PID)${NC}"
fi

echo ""
echo -e "${GREEN}Services started successfully!${NC}"
echo ""
echo "FastAPI backend:    http://localhost:8002"
echo "  - API docs:       http://localhost:8002/docs"
echo "Streamlit frontend: http://localhost:8501"
echo ""
echo "Logs:"
echo "  - FastAPI:   $LOG_DIR/fastapi.log"
echo "  - Streamlit: $LOG_DIR/streamlit.log"
echo ""
echo -e "Run ${YELLOW}./scripts/status.sh${NC} to check service status"
echo -e "Run ${YELLOW}./scripts/stop.sh${NC} to stop all services"
