# Service Scripts

Simple bash scripts to manage the AI Annotator services (FastAPI backend and Streamlit frontend).

## Prerequisites

- Conda environment `atap_classifier` must be set up
- `.env` file configured with required API keys (see main README)

## Usage

**Start all services:**
```bash
./scripts/start.sh
```
This starts the FastAPI backend (port 8002) and Streamlit frontend (port 8501) in the background.

**Check status:**
```bash
./scripts/status.sh
```
Shows whether services are running and displays recent log entries.

**Stop all services:**
```bash
./scripts/stop.sh
```
Stops both services and cleans up any orphaned processes.

## Service URLs

Once running:
- **Streamlit UI:** http://localhost:8501
- **FastAPI:** http://localhost:8002
- **API Docs:** http://localhost:8002/docs

## Logs

Logs are stored in `.logs/` at the project root:
- `fastapi.log` - Backend server logs
- `streamlit.log` - Frontend server logs

View logs in real-time:
```bash
tail -f .logs/fastapi.log
tail -f .logs/streamlit.log
```
