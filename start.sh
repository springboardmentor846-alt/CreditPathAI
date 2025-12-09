#!/usr/bin/env bash
# start.sh — start the FastAPI server (Render-friendly)
export PYTHONUNBUFFERED=1
# Use PORT provided by host (Render) or default 8000
uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2