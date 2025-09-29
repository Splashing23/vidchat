# VidChat

This application features a Next.js frontend that lets you upload a short video, extracts audio and frames on the backend, and provides an LLM-powered chat interface over the video transcript and visual features.

This repository contains a Next.js (React) app in the repo root and a FastAPI backend in `backend/` that runs a small video processing + LLM orchestration service.

## Features
- Upload a video (.mp4, .webm)
- Extract visual and audio information behind-the-scenes
- Stateless streaming chat endpoint (/chat/stream) that cites relevant segments/frames and streams the generated answer

## Prerequisites
- Node.js (recommended v18+)
- npm (bundled with Node)
- Python 3.8+ (3.11 recommended)
- Git (optional)

On Windows, PowerShell is used by the provided scripts. The repository includes cross-platform helper scripts to start the frontend and backend together.

## Quickstart (recommended for development)

1. Install Node dependencies at the repository root:

   npm install

2. Provide an OpenAI API key. The launcher scripts look for `OPENAI_API_KEY` either in the environment or in a repository file in the root folder named `.env.local` with a line like:

   OPENAI_API_KEY=sk-...

   - On Windows PowerShell you can set it for the session like:
     $env:OPENAI_API_KEY = "sk-..."

3. Start the app in development (runs Next dev + backend concurrently):

   npm run dev

This uses `scripts/dev-start.js` which will check for `OPENAI_API_KEY` and then run the backend and Next dev using `concurrently`.

Here is a sample video you may download and test on: [WeAreGoingOnBullrun.mp4](http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4)

If you prefer to run the backend individually, see the manual instructions below.

## Running the backend manually

The backend is a FastAPI service located in `backend/`. There are helper scripts included for convenience.

POSIX (macOS / Linux):

- From the repository root you can run the launcher that prefers POSIX helpers:

  npm run backend

Or directly in the `backend/` directory:

  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  venv/bin/python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

Windows (PowerShell):

The repository provides `backend/start.ps1` which will create a venv (if missing), install requirements, copy an env example, and start `uvicorn`.

From the repo root (preferred):

  npm run backend:win

Or run the node helper which picks the correct launcher for your platform:

  npm run backend

Manual PowerShell flow (if you want to perform steps yourself):

  py -3 -m venv backend\venv
  backend\venv\Scripts\Activate.ps1
  pip install -r backend\requirements.txt
  backend\venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

Notes:
- The backend script will try to find `OPENAI_API_KEY` from the environment, then `.env.local` at the repo root, then `backend/.env`.
- If no API key is provided, LLM features will be disabled but the server will still run for non-LLM endpoints.

## Environment variables
- OPENAI_API_KEY (required for LLM features) — set in your shell or in `.env.local` or `backend/.env`

The backend will also read `.env` inside `backend/` if present. The launcher scripts will copy `backend/env_example.txt` to `backend/.env` if you prefer the example as a starting point.

## API endpoints (backend)
- POST /upload — accepts a multipart upload (`file` or `files`) plus a `description` form field. Processes a single video and returns transcript/pages metadata.
- POST /chat/stream — accepts a JSON chat request and returns a streaming text response (Server-Sent Events / text/event-stream) with structured steps, partial content, costs, and image citation events.
- GET /health — simple health check
- POST /delete_document — remove files associated with a single uploaded document
- POST /delete_all_uploads — remove everything in the `uploads/` directory

FastAPI also exposes interactive docs at `/docs` and OpenAPI JSON at `/openapi.json` when the server is running.

## Frontend

The frontend is a Next.js app (app router) in the repo root. The dev server is started by `npm run dev` which uses `next dev` combined with the backend via `concurrently` (see `scripts/dev-start.js`).

Key files:
- `app/` — Next.js application code
- `app/components/StatelessChatSection.tsx` — chat UI component
- `utils/chunkedUpload.ts` — helper for uploading large files in chunks (frontend)

## File storage
- Uploaded videos, sampled frames, and derived artifacts are stored under `backend/uploads/` during development. The backend serves `uploads/` statically at `/uploads`.

## Troubleshooting
- Missing OPENAI_API_KEY: launcher scripts will refuse to start LLM features and will print an error. Set the key in your environment or add `.env.local`.
- Python not found: `backend/start.ps1` attempts to use `py -3`, then `python`, then `python3`. Make sure one of these is on PATH.
- Virtual env issues on Windows: ensure PowerShell execution policy allows running local scripts or run the launcher from an elevated prompt.

## Acknowledgements
Adapted from [Vectorless](https://github.com/roe-ai/vectorless) by Roe AI
