import sys
import os
import json
from http.server import BaseHTTPRequestHandler

# Add the backend directory to the Python path before importing
backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Root endpoint with API documentation"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        response_data = {
            "message": "Video Chatbot API",
            "version": "1.0.0",
            "endpoints": {
                "upload": "/api/upload - POST - Upload video files (mp4/webm)",
                "chat": "/api/chat/stream - POST - Stream chat responses",
                "health": "/api/health - GET - Health check",
            },
            "status": "ready",
        }
        self.wfile.write(json.dumps(response_data).encode())

    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# Fallback for FastAPI compatibility (if needed for local development)
try:
    from backend.main import app as fastapi_app

    app = fastapi_app
except ImportError:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def read_root():
        return {
                "message": "Video Chatbot API (FastAPI Fallback)",
            "version": "1.0.0",
            "note": "Using FastAPI fallback mode",
                "endpoints": {
                "upload": "/api/upload - POST - Upload video files (mp4/webm)",
                "chat": "/api/chat/stream - POST - Stream chat responses",
                "health": "/api/health - GET - Health check",
            },
            "status": "fallback",
        }
