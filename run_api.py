#!/usr/bin/env python
"""
Development server for FastAPI classifier service
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "classifier_fastapi.api.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
