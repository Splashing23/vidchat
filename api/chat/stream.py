import sys
import os
import json
import time
import asyncio
from http.server import BaseHTTPRequestHandler

# Add the backend directory to the Python path before importing
backend_path = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from backend.models import ChatRequest
from backend.llm_service import LLMService


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Set CORS headers for streaming
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # Read request body
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode("utf-8"))

            # Debug: print a preview of incoming documents payload
            try:
                docs_preview = request_data.get("documents")[:2] if isinstance(request_data.get("documents"), list) else request_data.get("documents")
                print(f"Incoming request preview - documents count: {len(request_data.get('documents') or [])}, sample: {docs_preview}")
            except Exception:
                print("Incoming request preview unavailable")

            # Parse request using ChatRequest model
            request = ChatRequest(**request_data)

            # Initialize LLM service
            llm_service = LLMService()

            # Process the chat request and stream response
            asyncio.run(self._process_chat_request(request, llm_service))

        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode())
            print(f"❌ Error in chat handler: {str(e)}")

    async def _process_chat_request(self, request, llm_service):
        """Process chat request with streaming response"""
        start_time = time.time()
        print(f"🌊 Streaming chat request started")
        print(f"📝 Question: {request.question}")
        print(f"📊 Received {len(request.documents)} documents")

        try:
            total_cost = 0.0

            # Convert DocumentData to the format expected by LLMService
            documents_dict = []
            # Build a documents_map used to resolve page/frame references to timestamps and stored filenames
            documents_map = {}
            for doc in request.documents:
                try:
                    print(f"Inspecting incoming document: filename={doc.filename} type(frames)={type(getattr(doc, 'frames', None))}")
                    # Print a small sample of frames if present
                    if getattr(doc, 'frames', None):
                        try:
                            sample = list(getattr(doc, 'frames'))[:3]
                            print(f"  sample frames: {sample}")
                        except Exception:
                            print("  could not sample frames")
                except Exception:
                    pass
                pages_dict = []
                for page in doc.pages:
                    pages_dict.append(
                        {"page_number": page.page_number, "text": page.text}
                    )

                # Build per-document map: page_number -> {start_time, end_time}; frames -> {frame_number: {time, stored_filename}}
                dm_pages = {}
                dm_frames = {}
                try:
                    for p in doc.pages:
                        if p.page_number is not None:
                            dm_pages[int(p.page_number)] = {"start_time": p.start_time, "end_time": p.end_time}
                except Exception:
                    pass
                try:
                    for f in (doc.frames or []):
                        # f may be a pydantic model or dict
                        try:
                            # Prefer attribute access for pydantic models, but only fall back to dict access
                            fn_val = getattr(f, "frame_number", None)
                            if fn_val is None and isinstance(f, dict):
                                fn_val = f.get("frame_number")
                            if fn_val is None:
                                # skip if we can't determine frame number
                                continue
                            fn = int(fn_val)

                            ft = getattr(f, "time", None)
                            if ft is None and isinstance(f, dict):
                                ft = f.get("time")

                            sf = getattr(f, "stored_filename", None)
                            if (sf is None or sf == "") and isinstance(f, dict):
                                sf = f.get("stored_filename") or f.get("filename")
                            if (sf is None or sf == ""):
                                # final attempt: attribute filename
                                sf = getattr(f, "filename", None) if hasattr(f, "filename") else None
                        except Exception:
                            continue
                        if fn is not None:
                            try:
                                # Use string keys for frames to avoid surprises when serializing/inspecting
                                dm_frames[str(int(fn))] = {"time": float(ft) if ft is not None else None, "stored_filename": sf}
                            except Exception:
                                dm_frames[str(int(fn))] = {"time": None, "stored_filename": sf}
                except Exception:
                    pass

                # If no frames were present in the DocumentData, attempt to fallback by scanning uploads/ for frame files
                try:
                    if not dm_frames:
                        uploads_dir = os.path.join(os.getcwd(), "uploads")
                        basename = os.path.splitext(doc.filename)[0]
                        print(f"No frames found in document metadata for {doc.filename}; scanning uploads for pattern {basename}_frame_*.jpg in {uploads_dir}")
                        if os.path.isdir(uploads_dir):
                            for fname in os.listdir(uploads_dir):
                                if fname.startswith(f"{basename}_frame_") and fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    # parse frame number
                                    try:
                                        parts = fname.rsplit('_frame_', 1)
                                        if len(parts) == 2:
                                            num_part = parts[1].split('.')[0]
                                            fn = int(num_part)
                                            dm_frames[str(int(fn))] = {"time": float(fn), "stored_filename": fname}
                                    except Exception:
                                        continue
                        print(f"Fallback frames discovered for {doc.filename}: {len(dm_frames)}")
                except Exception:
                    pass

                # Debug: print how many frames we collected for this document and a small preview
                try:
                    print(f"Document {doc.filename} frames collected: {len(dm_frames)}")
                    try:
                        preview = json.dumps({"pages": list(dm_pages.keys()), "frames_sample": list(dm_frames.items())[:5]}, default=str)
                        print(f"  preview: {preview}")
                    except Exception:
                        pass
                except Exception:
                    pass

                documents_map[doc.filename] = {"pages": dm_pages, "frames": dm_frames}

                # Append a lightweight representation for selection steps
                documents_dict.append(
                    {
                        "id": doc.id,
                        "filename": doc.filename,
                        "pages": pages_dict,
                        "total_pages": doc.total_pages,
                    }
                )

            # Debug: dump the full documents_map preview before calling LLMService
            try:
                dm_str = json.dumps(documents_map, default=str)
                print(f"Full documents_map (truncated): {dm_str[:1500]}{('...' if len(dm_str) > 1500 else '')}")
            except Exception:
                pass

            # Step 1: Select relevant segments
            step1_start = time.time()
            doc_selection_status = {
                "type": "status",
                "step": "document_selection",
                "message": "Searching for relevant information...",
                "step_number": 1,
                "total_steps": 2,
            }
            data = f"data: {json.dumps(doc_selection_status)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            print("⏱️ Step 1: Starting document selection...")
            selected_docs, step1_cost = await llm_service.select_documents(
                request.description,
                documents_dict,
                request.question,
                request.chat_history,
            )
            total_cost += step1_cost
            step1_time = time.time() - step1_start
            msg = f"✅ Step 1: Document selection completed in {step1_time:.2f}s"
            print(msg)

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
            data = f"data: {json.dumps(doc_selection_complete)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            # Step 2: Find relevant parts?
            # step2_start = time.time()
            # page_selection_status = {
            #     "type": "status",
            #     "step": "page_selection",
            #     "message": "Finding relevant parts in selected segments...",
            #     "step_number": 2,
            #     "total_steps": 3,
            # }
            # data = f"data: {json.dumps(page_selection_status)}\n\n"
            # self.wfile.write(data.encode())
            # self.wfile.flush()

            print("⏱️ Step 2: Starting page selection...")

            async def process_document(doc):
                return await llm_service.find_relevant_pages(
                    doc["pages"],
                    request.question,
                    doc["filename"],
                    request.chat_history,
                )

            # Create tasks for all documents
            doc_tasks = [process_document(doc) for doc in selected_docs]

            # Wait for all documents to complete
            doc_results = await asyncio.gather(*doc_tasks)

            # Combine results
            all_relevant_pages = []
            step2_cost = 0.0
            for doc_relevant_pages, doc_cost in doc_results:
                all_relevant_pages.extend(doc_relevant_pages)
                step2_cost += doc_cost

            relevant_pages = all_relevant_pages
            total_cost += step2_cost
            step2_time = 0 # time.time() - step2_start
            msg = f"✅ Step 2: Page selection completed in {step2_time:.2f}s"
            print(msg)

            # Send completion status for page selection
            page_selection_complete = {
                "type": "step_complete",
                "step": "page_selection",
                "relevant_pages_count": len(relevant_pages),
                "cost": step2_cost,
                "time_taken": step2_time,
            }
            data = f"data: {json.dumps(page_selection_complete)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            # Step 3: Generate answer
            step3_start = time.time()
            answer_generation_status = {
                "type": "status",
                "step": "answer_generation",
                "message": "Generating comprehensive answer...",
                "step_number": 2,
                "total_steps": 2,
            }
            data = f"data: {json.dumps(answer_generation_status)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            print("⏱️ Step 3: Starting answer generation...")

            # Stream the answer generation
            async for chunk in llm_service.generate_answer_stream(
                relevant_pages, request.question, request.chat_history, request.model, documents_map
            ):
                if chunk.get("type") == "content":
                    content_data = {
                        "type": "content",
                        "content": chunk["content"],
                    }
                    data = f"data: {json.dumps(content_data)}\n\n"
                    self.wfile.write(data.encode())
                    self.wfile.flush()
                elif chunk.get("type") == "cost":
                    total_cost += chunk["cost"]
                elif chunk.get("type") == "image_citation":
                    # Forward image citation metadata to client so UI can render thumbnails / jump links
                    img_data = {"type": "image_citation", "citation": chunk.get("citation")}
                    data = f"data: {json.dumps(img_data)}\n\n"
                    self.wfile.write(data.encode())
                    self.wfile.flush()

            step3_time = time.time() - step3_start
            msg = f"✅ Step 3: Answer generation completed in {step3_time:.2f}s"
            print(msg)

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
            data = f"data: {json.dumps(completion_data)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            cost_msg = f"🎉 Request completed in {total_time:.2f}s, total cost: ${total_cost:.4f}"
            print(cost_msg)

        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            data = f"data: {json.dumps(error_data)}\n\n"
            self.wfile.write(data.encode())
            print(f"❌ Error in stream_response: {str(e)}")

    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        # Add GET method for testing
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        response_data = {
            "message": "Chat stream endpoint",
            "method": "POST",
            "description": "Use POST method to send chat requests",
        }

        self.wfile.write(json.dumps(response_data).encode())
