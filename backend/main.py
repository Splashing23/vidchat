from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any
import json
import tempfile
import os
import asyncio

from models import (
    ChatRequest,
    UploadResponse,
    DocumentData,
    DocumentPage,
)
from video_processor import VideoProcessor
import logging
try:
    # FastAPI provides StaticFiles wrapper; fall back to starlette if needed
    from fastapi.staticfiles import StaticFiles
except Exception:
    from starlette.staticfiles import StaticFiles
from llm_service import LLMService

app = FastAPI(title="Video Chatbot API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://localhost:3004",
        "http://localhost:3005",
    ],  # Next.js dev server on various ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
llm_service = LLMService()
video_processor = VideoProcessor()
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# Serve uploaded files (videos) from /uploads
uploads_dir = os.path.join(os.getcwd(), "uploads")
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    file: UploadFile | None = File(None), files: List[UploadFile] | None = File(None), description: str = Form(...)
):
    """Process a single video file and return the extracted full transcript to the client.

    This endpoint accepts either a single 'file' field or a 'files' field (list). We
    normalize to a single file and process it. This makes the API tolerant to clients
    that use different multipart field names.
    """

    # Normalize incoming files: prefer 'files' list if present, otherwise single 'file'
    incoming_files: List[UploadFile] = []
    if files:
        incoming_files = files
    elif file:
        incoming_files = [file]

    if not incoming_files:
        raise HTTPException(status_code=400, detail="No file uploaded. Please provide a video file under form field 'file' or 'files'.")

    # Only support a single file for now
    if len(incoming_files) > 1:
        raise HTTPException(status_code=400, detail="Please upload only one video file at a time.")

    file_obj = incoming_files[0]
    filename = file_obj.filename or "upload_1"
    lower = filename.lower()

    if not (lower.endswith(".mp4") or lower.endswith(".webm")):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}. Only .mp4 and .webm are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        temp_file.write(await file_obj.read())
        temp_video_path = temp_file.name

    try:
        video_info = video_processor.process_video(temp_video_path, filename)

        # Map single full-transcript segment to DocumentPage
        pages = []
        for seg in video_info["segments"]:
            pages.append(
                DocumentPage(
                    page_number=seg["page_number"],
                    text=seg["text"],
                    char_count=seg.get("char_count"),
                    start_time=seg.get("start_time"),
                    end_time=seg.get("end_time"),
                )
            )

        document = DocumentData(
            id=1,
            filename=filename,
            pages=pages,
            total_pages=len(pages),
            stored_filename=video_info.get("stored_filename"),
            is_video=True,
            processing_cost=float(video_info.get("processing_cost", 0.0) or 0.0),
        )

        return UploadResponse(documents=[document], message="Successfully processed 1 file")

    except Exception as e:
        logger.exception("Video processing error for %s", filename)
        raise HTTPException(status_code=500, detail=f"Error processing {filename}: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Handle chat requests with streaming response - stateless"""
    import time

    logger.info("Streaming chat request started")
    logger.info("Question: %s", request.question)
    logger.info("Received %s documents", len(request.documents))

    async def stream_response():
        start_time = time.time()
        try:
            total_cost = 0.0

            # Convert DocumentData to the format expected by LLMService
            documents_dict = []
            # Include preprocessing cost from uploaded documents
            preprocessing_cost = 0.0
            for doc in request.documents:
                pages_dict = []
                for page in doc.pages:
                    pages_dict.append(
                        {"page_number": page.page_number, "text": page.text}
                    )

                documents_dict.append(
                    {
                        "id": doc.id,
                        "filename": doc.filename,
                        "pages": pages_dict,
                        "total_pages": doc.total_pages,
                    }
                )
                try:
                    preprocessing_cost += float(getattr(doc, "processing_cost", 0.0) or 0.0)
                except Exception:
                    pass

            # Add preprocessing cost to running total
            total_cost += preprocessing_cost

            # Step 1: Select relevant documents
            step1_start = time.time()
            doc_selection_status = {
                "type": "status",
                "step": "document_selection",
                "message": "Searching relevant information...",
                "step_number": 1,
                "total_steps": 2,
            }
            yield f"data: {json.dumps(doc_selection_status)}\n\n"

            logger.info("Step 1: Starting document selection...")
            # Normalize chat history to list of dicts for LLMService
            chat_history = []
            if request.chat_history:
                for msg in request.chat_history:
                    # Support both pydantic model instances and dicts
                    if isinstance(msg, dict):
                        chat_history.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
                    else:
                        # pydantic model
                        try:
                            chat_history.append({"role": getattr(msg, "role", "user"), "content": getattr(msg, "content", "")})
                        except Exception:
                            # Fallback generic
                            chat_history.append({"role": "user", "content": str(msg)})

            selected_docs, step1_cost = await llm_service.select_documents(
                request.description,
                documents_dict,
                request.question,
                chat_history,
            )
            total_cost += step1_cost
            step1_time = time.time() - step1_start
            logger.info("Step 1: Document selection completed in %.2fs", step1_time)

            # Send completion status for document selection
            doc_selection_complete = {
                "type": "step_complete",
                "step": "document_selection",
                "selected_documents": [
                    {"id": doc["id"], "filename": doc["filename"]}
                    for doc in selected_docs
                ],
                "cost": step1_cost,
                "time_taken": step1_time,
            }
            yield f"data: {json.dumps(doc_selection_complete)}\n\n"

            # Step 2: Find relevant pages
            # step2_start = time.time()
            # page_selection_status = {
            #     "type": "status",
            #     "step": "page_selection",
            #     "message": "Finding relevant pages in selected documents...",
            #     "step_number": 2,
            #     "total_steps": 2,
            # }
            # yield f"data: {json.dumps(page_selection_status)}\n\n"

            logger.info("Step 2: Starting page selection...")
            # Build a documents_map for resolving page references later: { filename: {page_number: {start_time, end_time}} }
            documents_map: Dict[str, Dict[int, Dict[str, float]]] = {}
            relevant_pages = []
            step2_cost = 0.0
            if selected_docs:
                # Build documents_map and a flattened relevant_pages list containing page metadata
                for doc in selected_docs:
                    filename = str(doc.get("filename") or "unknown")
                    pages = doc.get("pages", [])
                    documents_map.setdefault(filename, {})
                    for p in pages:
                        page_num = int(p.get("page_number"))
                        start_time = float(p.get("start_time") or 0.0)
                        end_time = float(p.get("end_time") or start_time)
                        documents_map[filename][page_num] = {"start_time": start_time, "end_time": end_time}
                        page_with_source = {
                            "page_number": page_num,
                            "text": p.get("text", ""),
                            "start_time": start_time,
                            "end_time": end_time,
                            "source_document": filename,
                        }
                        relevant_pages.append(page_with_source)
            total_cost += step2_cost
            step2_time = 0 # time.time() - step2_start
            logger.info("Step 2: Page selection (full-transcript) completed in %.2fs", step2_time)

            # Send completion status for page selection
            page_selection_complete = {
                "type": "step_complete",
                "step": "page_selection",
                "relevant_pages_count": len(relevant_pages),
                "cost": step2_cost,
                "time_taken": step2_time,
            }
            yield f"data: {json.dumps(page_selection_complete)}\n\n"

            # Step 3: Generate answer
            step3_start = time.time()
            answer_generation_status = {
                "type": "status",
                "step": "answer_generation",
                "message": "Generating comprehensive answer...",
                "step_number": 2,
                "total_steps": 2,
            }
            yield f"data: {json.dumps(answer_generation_status)}\n\n"

            logger.info("Step 3: Starting answer generation...")

            # Stream the answer generation
            # Ensure model is a valid string
            model = request.model or "gpt-5-mini"
            async for chunk in llm_service.generate_answer_stream(
                relevant_pages, request.question, chat_history, model, documents_map=documents_map
            ):
                if chunk.get("type") == "content":
                    content_data = {
                        "type": "content",
                        "content": chunk["content"],
                    }
                    yield f"data: {json.dumps(content_data)}\n\n"
                elif chunk.get("type") == "cost":
                    try:
                        total_cost += float(chunk.get("cost", 0.0))
                    except Exception:
                        pass

            step3_time = time.time() - step3_start
            logger.info("Step 3: Answer generation completed in %.2fs", step3_time)

            # Send final completion
            total_time = time.time() - start_time
            completion_data = {
                "type": "complete",
                "timing_breakdown": {
                    "document_selection": step1_time,
                    "page_detection": step2_time,
                    "answer_generation": step3_time,
                    "total_time": total_time,
                },
                "cost_breakdown": {
                    "document_selection": step1_cost,
                    "page_detection": step2_cost,
                    "answer_generation": total_cost - step1_cost - step2_cost,
                    "total_cost": total_cost,
                },
            }
            yield f"data: {json.dumps(completion_data)}\n\n"

            logger.info("Request completed in %.2fs, total cost: $%.4f", total_time, total_cost)

        except Exception:
            logger.exception("Error in stream_response")
            error_data = {"type": "error", "error": "Internal server error"}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "mode": "stateless"}


@app.post("/delete_document")
async def delete_document(payload: Dict[str, Any]):
    """Delete stored video file, sampled frames, and frames sidecar for a document.

    Expects JSON body with one of: stored_filename, storedFilename, filename.
    Returns a list of removed files and any errors encountered.
    """
    try:
        key = payload.get("stored_filename") or payload.get("storedFilename") or payload.get("filename")
        if not key:
            raise HTTPException(status_code=400, detail="No filename provided for deletion")

        # Normalize to basename inside uploads_dir and prevent path traversal
        target_name = os.path.basename(str(key))
        removed = []
        errors = []

        # Candidate files to remove
        video_path = os.path.join(uploads_dir, target_name)
        if os.path.exists(video_path):
            try:
                os.unlink(video_path)
                removed.append(os.path.relpath(video_path, uploads_dir))
            except Exception as e:
                logger.exception("Failed to remove video: %s", video_path)
                errors.append(str(e))

        # Remove frame images with pattern <basename>_frame_*.jpg
        basename = os.path.splitext(target_name)[0]
        try:
            for fname in os.listdir(uploads_dir):
                if fname.startswith(f"{basename}_frame_") or fname == f"{basename}_frames.json":
                    fpath = os.path.join(uploads_dir, fname)
                    try:
                        os.unlink(fpath)
                        removed.append(os.path.relpath(fpath, uploads_dir))
                    except Exception as e:
                        logger.exception("Failed to remove frame/sidecar: %s", fpath)
                        errors.append(str(e))
        except Exception:
            logger.exception("Error scanning uploads directory for cleanup")

        # Attempt to remove common audio artifact created during processing (if present)
        audio_candidate = os.path.join(uploads_dir, "audio.wav")
        if os.path.exists(audio_candidate):
            try:
                os.unlink(audio_candidate)
                removed.append(os.path.relpath(audio_candidate, uploads_dir))
            except Exception:
                logger.exception("Failed to remove audio artifact: %s", audio_candidate)

        return {"removed": removed, "errors": errors}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error in delete_document")
        raise HTTPException(status_code=500, detail="Internal error during deletion")


@app.post("/delete_all_uploads")
async def delete_all_uploads():
    """Delete all files and folders under the uploads directory.

    This is intended as a session-level cleanup invoked when a user explicitly
    starts a new session or deletes their uploaded video from the UI.
    """
    removed = []
    errors = []
    try:
        for fname in os.listdir(uploads_dir):
            fpath = os.path.join(uploads_dir, fname)
            try:
                # Remove files, symlinks, and directories
                if os.path.islink(fpath) or os.path.isfile(fpath):
                    os.unlink(fpath)
                    removed.append(os.path.relpath(fpath, uploads_dir))
                elif os.path.isdir(fpath):
                    import shutil

                    shutil.rmtree(fpath)
                    removed.append(os.path.relpath(fpath, uploads_dir))
            except Exception as e:
                logger.exception("Failed to remove upload entry: %s", fpath)
                errors.append(str(e))
    except Exception:
        logger.exception("Error scanning uploads directory for delete_all_uploads")
        raise HTTPException(status_code=500, detail="Failed to enumerate uploads for deletion")

    return {"removed": removed, "errors": errors}
