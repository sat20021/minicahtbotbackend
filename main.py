"""
Root entry point for ASGI servers (Render, Docker, Uvicorn, etc.)
Re-exports the FastAPI app instance from app.main.
"""
from app.main import app

__all__ = ["app"]
