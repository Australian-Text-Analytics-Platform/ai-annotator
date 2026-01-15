#!/bin/bash
# Check status of FastAPI backend and Streamlit frontend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_DIR/.pids"
LOG_DIR="$PROJECT_DIR/.logs"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "AI Annotator Service Status"
echo "============================"
echo ""

# Check FastAPI
echo -n "FastAPI (port 8002):   "
FASTAPI_PORT_PID=$(lsof -ti:8002 2>/dev/null || true)
if [ -n "$FASTAPI_PORT_PID" ]; then
    echo -e "${GREEN}RUNNING${NC} (PID: $FASTAPI_PORT_PID)"
    FASTAPI_STATUS="running"
else
    echo -e "${RED}STOPPED${NC}"
    FASTAPI_STATUS="stopped"
fi

# Check Streamlit
echo -n "Streamlit (port 8501): "
STREAMLIT_PORT_PID=$(lsof -ti:8501 2>/dev/null || true)
if [ -n "$STREAMLIT_PORT_PID" ]; then
    echo -e "${GREEN}RUNNING${NC} (PID: $STREAMLIT_PORT_PID)"
    STREAMLIT_STATUS="running"
else
    echo -e "${RED}STOPPED${NC}"
    STREAMLIT_STATUS="stopped"
fi

echo ""
echo "URLs:"
if [ "$FASTAPI_STATUS" = "running" ]; then
    echo -e "  FastAPI backend:    ${GREEN}http://localhost:8002${NC}"
    echo -e "    - API docs:       ${GREEN}http://localhost:8002/docs${NC}"
else
    echo -e "  FastAPI backend:    ${RED}not available${NC}"
fi

if [ "$STREAMLIT_STATUS" = "running" ]; then
    echo -e "  Streamlit frontend: ${GREEN}http://localhost:8501${NC}"
else
    echo -e "  Streamlit frontend: ${RED}not available${NC}"
fi

# Show recent log entries if available
echo ""
echo "Recent logs:"
if [ -f "$LOG_DIR/fastapi.log" ]; then
    echo -e "${YELLOW}FastAPI (last 3 lines):${NC}"
    tail -3 "$LOG_DIR/fastapi.log" 2>/dev/null | sed 's/^/  /'
else
    echo "  FastAPI: no log file"
fi

echo ""
if [ -f "$LOG_DIR/streamlit.log" ]; then
    echo -e "${YELLOW}Streamlit (last 3 lines):${NC}"
    tail -3 "$LOG_DIR/streamlit.log" 2>/dev/null | sed 's/^/  /'
else
    echo "  Streamlit: no log file"
fi

echo ""
echo -e "Run ${YELLOW}./scripts/start.sh${NC} to start services"
echo -e "Run ${YELLOW}./scripts/stop.sh${NC} to stop services"
