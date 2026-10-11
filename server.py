"""
Universal AI-Powered Data Cleaning Platform — Server Entry Point

Run with: python server.py
"""

import uvicorn

from backend.app import app  # noqa: F401

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=7860, reload=True)
