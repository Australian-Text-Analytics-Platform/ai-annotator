#!/bin/bash
# Stop FastAPI backend and Streamlit frontend

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PID_DIR="$PROJECT_DIR/.pids"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping AI Annotator services...${NC}"

STOPPED_ANY=false

# Stop FastAPI
if [ -f "$PID_DIR/fastapi.pid" ]; then
    PID=$(cat "$PID_DIR/fastapi.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        echo -e "${GREEN}FastAPI stopped (PID: $PID)${NC}"
        STOPPED_ANY=true
    else
        echo -e "${YELLOW}FastAPI was not running${NC}"
    fi
    rm "$PID_DIR/fastapi.pid"
else
    echo -e "${YELLOW}FastAPI PID file not found${NC}"
fi

# Stop Streamlit
if [ -f "$PID_DIR/streamlit.pid" ]; then
    PID=$(cat "$PID_DIR/streamlit.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null
        echo -e "${GREEN}Streamlit stopped (PID: $PID)${NC}"
        STOPPED_ANY=true
    else
        echo -e "${YELLOW}Streamlit was not running${NC}"
    fi
    rm "$PID_DIR/streamlit.pid"
else
    echo -e "${YELLOW}Streamlit PID file not found${NC}"
fi

# Also kill any orphaned processes on the ports
echo ""
echo "Checking for orphaned processes..."

FASTAPI_PORT_PID=$(lsof -ti:8002 2>/dev/null || true)
if [ -n "$FASTAPI_PORT_PID" ]; then
    kill $FASTAPI_PORT_PID 2>/dev/null || true
    echo -e "${GREEN}Killed process on port 8002 (PID: $FASTAPI_PORT_PID)${NC}"
    STOPPED_ANY=true
fi

STREAMLIT_PORT_PID=$(lsof -ti:8501 2>/dev/null || true)
if [ -n "$STREAMLIT_PORT_PID" ]; then
    kill $STREAMLIT_PORT_PID 2>/dev/null || true
    echo -e "${GREEN}Killed process on port 8501 (PID: $STREAMLIT_PORT_PID)${NC}"
    STOPPED_ANY=true
fi

echo ""
if [ "$STOPPED_ANY" = true ]; then
    echo -e "${GREEN}All services stopped.${NC}"
else
    echo -e "${YELLOW}No services were running.${NC}"
fi
