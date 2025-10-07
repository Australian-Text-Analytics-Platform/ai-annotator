"""API route modules"""

from classifier_fastapi.api.routes import health, classify, jobs, models

__all__ = ["health", "classify", "jobs", "models"]
