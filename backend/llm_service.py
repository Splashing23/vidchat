import asyncio
import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import logging

load_dotenv()

# Module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


class LLMService:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            logger.warning("OpenAI API key not set. LLM features will be disabled.")
            self.client = None
        else:
            # instantiate async client when key is present
            try:
                self.client = AsyncOpenAI(api_key=api_key)
            except Exception as e:
                logger.exception("Failed to instantiate AsyncOpenAI client")
                self.client = None

        # Default model
        self.model = "gpt-5-nano"

        # Pricing tiers (per 1M tokens, approximate)
        self.pricing = {
            "gpt-5": {"input": 1.25, "output": 10.0},
            "gpt-5-mini": {"input": 0.25, "output": 2.0},
            # gpt-5-nano: ultra-low-cost, lower-capacity option for quick, cheap responses
            "gpt-5-nano": {"input": 0.05, "output": 0.4},
        }

    def calculate_cost(self, usage_data, model="gpt-5-nano"):
        # logger.info("calculate_cost usage_data: %s", usage_data)
        """Calculate cost based on token usage"""
        if not usage_data or model not in self.pricing:
            return 0.0

        input_tokens = usage_data.prompt_tokens
        output_tokens = usage_data.completion_tokens

        input_cost = (input_tokens / 1_000_000 * 1.0) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1_000_000 * 1.0) * self.pricing[model]["output"]

        return input_cost + output_cost

    async def select_documents(
        self,
        description: str,
        documents: List[Dict[str, Any]],
        question: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Select relevant documents based on description, question, and chat history
        """

        doc_summaries = []
        for doc in documents:
            doc_summaries.append(
                {
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "total_pages": doc["total_pages"],
                    "first_page_preview": (doc["pages"][0]["text"][:500] + "..."),
                }
            )

        # Format chat history
        history_context = ""
        if chat_history:
            history_context = "\n\nChat History:\n"
            for msg in chat_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                else:
                    role = getattr(msg, "role", "unknown")
                    content = getattr(msg, "content", "")
                history_context += f"{role.capitalize()}: {content}\n"

        prompt = f"""
            Based on the following document collection description, chat history, 
            and current question, select which documents are most likely to 
            contain the answer.

            <Document Collection Description>
            {description}
            <Document Collection Description>

            <Available Documents>
            {json.dumps(doc_summaries, indent=2)}
            <Available Documents>

            <Chat History>
            {history_context}
            <Chat History>

            <Current Question>
            {question}
            <Current Question>

            Return a JSON array of document IDs (numbers) that are most relevant to 
            the current question and conversation context.
            Only return the JSON array, no other text.
            Example: [1, 3, 5]
            """

        try:
            if not self.client:
                logger.warning("AsyncOpenAI client not available, skipping remote document selection")
                # fallback: return all documents
                return documents, 0.0

            response = await self.client.chat.completions.create(
                model="gpt-5",
                messages=[{"role": "user", "content": prompt}],
            )

            content_str = None
            try:
                content_str = response.choices[0].message.content
            except Exception:
                content_str = None
            if not content_str:
                content_str = "[]"
            selected_ids = json.loads(content_str)
            cost = self.calculate_cost(response.usage, self.model)

            # Return full document objects for selected IDs
            selected_docs = []
            for doc in documents:
                if doc["id"] in selected_ids:
                    selected_docs.append(doc)

            return selected_docs, cost

        except Exception:
            logger.exception("Error in document selection")
            # Fallback: return all documents
            return documents, 0.0

    async def find_relevant_pages(
        self,
        pages: List[Dict[str, Any]],
        question: str,
        filename: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[Dict[str, Any]], float]:
        # logger.info("find_relevant_pages filename=%s", filename)
        """Find relevant pages by processing 20 pages at a time in parallel"""

        # Create chunks of 20 pages
        chunks = []
        for i in range(0, len(pages), 20):
            chunk = pages[i : i + 20]
            chunks.append(chunk)

        # Process all chunks in parallel
        chunk_tasks = []
        for chunk_index, chunk in enumerate(chunks):
            task = self._process_page_chunk(
                chunk, question, filename, chunk_index, chat_history
            )
            chunk_tasks.append(task)

        # Wait for all chunks to complete
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        # Combine results from all chunks
        relevant_pages = []
        total_cost = 0.0
        for result in chunk_results:
            if isinstance(result, Exception):
                logger.exception("Error in chunk processing: %s", result)
                continue
            if isinstance(result, tuple) and len(result) == 2:
                pages, cost = result
                relevant_pages.extend(pages)
                total_cost += cost
            elif isinstance(result, list):
                # Fallback for old format
                relevant_pages.extend(result)

        return relevant_pages, total_cost

    async def _process_page_chunk(
        self,
        chunk: List[Dict[str, Any]],
        question: str,
        filename: str,
        chunk_index: int,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[Dict[str, Any]], float]:
        """Process a single chunk of pages"""
        import time

        chunk_start = time.time()
        print(f"    🔍 Processing chunk {chunk_index + 1} with {len(chunk)} pages...")

        # Prepare content for LLM
        pages_content = []
        for page in chunk:
            # Defensive check for required fields
            if "page_number" not in page:
                print(f"Warning: page missing 'page_number': {page.keys()}")
                continue
            if "text" not in page:
                print(f"Warning: page missing 'text': {page.keys()}")
                continue

            pages_content.append(
                {
                    "page_number": page["page_number"],
                    "page_content": (page["text"]),
                }
            )

        # Format chat history for context
        history_context = ""
        if chat_history:
            history_context = "\n\nRecent Chat History:\n"
            for msg in chat_history:
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                else:
                    role = getattr(msg, "role", "unknown")
                    content = getattr(msg, "content", "")
                history_context += f"{role.capitalize()}: {content}...\n"

        prompt = f"""
            Analyze the following pages from document "{filename}" and determine 
            which pages are relevant to the current question, considering the conversation context. 
            Return empty array if no pages are relevant.
            
            <Chat History>
            {history_context}
            <Chat History>
            
            <Current Question>
            {question}
            <Current Question>

            <Document Page Content>
            {json.dumps(pages_content, indent=2)}
            <Document Page Content>

            Return a JSON array of page numbers relevant to the current question
            Only return the JSON array, no other text.
            Example: [1, 3, 5]
            """

        try:
            if not self.client:
                logger.warning("AsyncOpenAI client not available, using fallback page selection")
                # fallback: include first page of chunk
                if chunk:
                    first_page = chunk[0].copy()
                    first_page["source_document"] = filename
                    return [first_page], 0.0
                return [], 0.0

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )

            content_str = None
            try:
                content_str = response.choices[0].message.content
            except Exception:
                content_str = None
            if not content_str:
                content_str = "[]"

            # Debug log the raw response so we can inspect model output formats
            # logger.info("Raw page selection response: %s", content_str)

            # Normalize model output into a list of integers (page numbers).
            # The model may return: [1,2], ["1","2"], "1,2", "1-3", or other variants.
            def _normalize_page_list(raw: str):
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = raw

                numbers = set()

                # If parsed is a list, handle each item
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, int):
                            numbers.add(item)
                        elif isinstance(item, str):
                            item = item.strip()
                            # comma separated inside a string
                            if "," in item:
                                for part in item.split(","):
                                    part = part.strip()
                                    if part.isdigit():
                                        numbers.add(int(part))
                                    elif "-" in part:
                                        start_end = part.split("-")
                                        try:
                                            start = int(start_end[0])
                                            end = int(start_end[1])
                                            for n in range(start, end + 1):
                                                numbers.add(n)
                                        except Exception:
                                            continue
                            elif item.isdigit():
                                numbers.add(int(item))
                            elif "-" in item:
                                # range expressed as string
                                try:
                                    start, end = item.split("-", 1)
                                    start = int(start.strip())
                                    end = int(end.strip())
                                    for n in range(start, end + 1):
                                        numbers.add(n)
                                except Exception:
                                    continue
                elif isinstance(parsed, str):
                    s = parsed.strip()
                    # Try comma-separated
                    if "," in s:
                        for part in s.split(","):
                            part = part.strip()
                            if part.isdigit():
                                numbers.add(int(part))
                            elif "-" in part:
                                try:
                                    a, b = part.split("-", 1)
                                    a = int(a.strip())
                                    b = int(b.strip())
                                    for n in range(a, b + 1):
                                        numbers.add(n)
                                except Exception:
                                    continue
                    elif s.isdigit():
                        numbers.add(int(s))
                    elif "-" in s:
                        try:
                            a, b = s.split("-", 1)
                            a = int(a.strip())
                            b = int(b.strip())
                            for n in range(a, b + 1):
                                numbers.add(n)
                        except Exception:
                            pass

                return sorted(numbers)

            try:
                relevant_page_numbers = _normalize_page_list(content_str)
            except Exception:
                logger.exception("Failed to normalize page list from model output")
                relevant_page_numbers = []

            cost = self.calculate_cost(response.usage, model=self.model)

            # Add full page data for relevant pages
            relevant_pages = []
            for page in chunk:
                if "page_number" not in page:
                    continue
                try:
                    page_num = int(page["page_number"])
                except Exception:
                    continue
                if page_num in relevant_page_numbers:
                    page_with_source = page.copy()
                    page_with_source["source_document"] = filename
                    relevant_pages.append(page_with_source)

            chunk_time = time.time() - chunk_start
            # logger.info("Chunk %s completed in %.2fs, found %s relevant pages", chunk_index + 1, chunk_time, len(relevant_pages))
            return relevant_pages, cost

        except Exception as e:
            chunk_time = time.time() - chunk_start
            logger.exception("Chunk %s failed in %.2fs", chunk_index + 1, chunk_time)
            # Fallback: include first page of chunk
            if chunk:
                first_page = chunk[0].copy()
                first_page["source_document"] = filename
                return [first_page], 0.0
            return [], 0.0

    async def generate_answer_stream(
        self,
        relevant_pages: List[Dict[str, Any]],
        question: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        model: str = "gpt-5-nano",
        documents_map: Optional[Dict[str, Any]] = None,
    ):
        """Generate final answer using all relevant pages with streaming"""
        # logger.info("documents_map: %s", documents_map)
        if not relevant_pages:
            yield {
                "type": "content",
                "content": "I couldn't find any relevant information to answer your question.",
            }
            yield {"type": "cost", "cost": 0.0}
            return

        # Format chat history for conversational context
        history_context = ""
        if chat_history:
            history_context = "\n\nConversation History:\n"
            for msg in chat_history:  # Include last 4 messages for context
                if isinstance(msg, dict):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                else:
                    role = getattr(msg, "role", "unknown")
                    content = getattr(msg, "content", "")
                history_context += f"{role.capitalize()}: {content}\n"
        # logger.info("relevant_pages: %s", relevant_pages)
        # Debug: show documents_map preview
        try:
            dm_preview = json.dumps(documents_map or {}, default=str)
            # logger.info("documents_map preview: %s", (dm_preview[:200] + '...') if len(dm_preview) > 200 else dm_preview)
        except Exception:
            logger.exception("Failed to stringify documents_map for debug preview")
        # Helper to safely convert values to int
        def safe_int_val(v) -> Optional[int]:
            """Safely convert a value to int or return None if not convertible."""
            if v is None:
                return None
            try:
                return int(v)
            except Exception:
                try:
                    return int(float(v))
                except Exception:
                    return None
        # Build an image-context block that lists sampled frame URLs and timestamps so the model can use visual context.
        # We provide URLs relative to the server origin (/uploads/<stored_filename>). If you host uploads elsewhere, adapt accordingly.
        image_context_lines: List[str] = []
        total_images = 0
        captions_found = False
        try:
            if documents_map:
                # Debug: log top-level document keys we received
                # logger.info("Documents received for image-context building: %s", list((documents_map or {}).keys()))
                for fname, fmap in (documents_map or {}).items():
                    try:
                        frames_map = fmap.get("frames") or {}
                        # logger.info("Frames map for %s: type=%s keys_preview=%s", fname, type(frames_map), list(frames_map.keys())[:10] if isinstance(frames_map, dict) else (len(frames_map) if isinstance(frames_map, list) else 'n/a'))
                    except Exception:
                        logger.exception("Error inspecting frames map for %s", fname)
                        frames_map = {}

                    # frames_map may be dict keyed by frame_number or list
                    if isinstance(frames_map, dict):
                        # If the frames_map is empty, attempt a filesystem fallback to find sampled frames
                        if not frames_map:
                            try:
                                uploads_dir = os.path.join(os.getcwd(), "uploads")
                                basename = os.path.splitext(fname)[0]
                                # logger.info("No frames in documents_map for %s; scanning uploads_dir=%s for %s_frame_*.jpg or sidecar JSON", fname, uploads_dir, basename)

                                # First, try to load a sidecar JSON produced at upload time
                                sidecar_path = os.path.join(uploads_dir, f"{basename}_frames.json")
                                loaded_sidecar = False
                                if os.path.exists(sidecar_path):
                                    try:
                                        with open(sidecar_path, "r", encoding="utf-8") as sf:
                                            side = json.load(sf)
                                        if isinstance(side, dict) and isinstance(side.get("frames"), list):
                                            frames_map = {}
                                            for fm in side.get("frames", []):
                                                try:
                                                    k = str(int(fm.get("frame_number") if fm.get("frame_number") is not None else int(fm.get("time", 0))))
                                                except Exception:
                                                    k = str(fm.get("time") or fm.get("stored_filename") or "0")
                                                frames_map[k] = fm
                                            loaded_sidecar = True
                                            # logger.info("Loaded frames sidecar with captions for %s: %s", fname, sidecar_path)
                                    except Exception:
                                        logger.exception("Failed to load frames sidecar for %s", fname)

                                # If no sidecar, fall back to scanning for image files
                                if not loaded_sidecar and os.path.isdir(uploads_dir):
                                    found = 0
                                    for file_name in os.listdir(uploads_dir):
                                        if file_name.startswith(f"{basename}_frame_") and file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                                            try:
                                                parts = file_name.rsplit('_frame_', 1)
                                                if len(parts) == 2:
                                                    num_part = parts[1].split('.')[0]
                                                    fn_key = str(int(num_part))
                                                    # avoid overwriting existing entries
                                                    if fn_key not in frames_map:
                                                        frames_map[fn_key] = {"time": float(int(num_part)), "stored_filename": file_name}
                                                        found += 1
                                            except Exception:
                                                logger.exception("Error parsing frame filename during fallback: %s", file_name)
                                                continue
                                    # logger.info("Fallback scan for %s found %d frames", fname, found)

                                # Persist the discovered frames back into the documents_map so later
                                # stages (Responses API attachment) can see them. `fmap` may be a
                                # separate dict object, so write into the top-level documents_map as well
                                try:
                                    if isinstance(documents_map, dict) and fname in documents_map:
                                        try:
                                            documents_map[fname]["frames"] = frames_map
                                        except Exception:
                                            fmap["frames"] = frames_map
                                    else:
                                        fmap["frames"] = frames_map
                                    # logger.info("Persisted %d fallback frames into documents_map for %s", len(frames_map), fname)
                                except Exception:
                                    logger.exception("Failed to persist fallback frames for %s", fname)
                            except Exception:
                                logger.exception("Error during filesystem fallback scan for frames for %s", fname)
                        for fn, fmeta in frames_map.items():
                            fn_i = safe_int_val(fn)
                            if fn_i is None:
                                fn_i = safe_int_val(getattr(fmeta, "frame_number", None) or (fmeta or {}).get("frame_number"))
                            if fn_i is None:
                                continue
                            time_v = None
                            stored = None
                            if isinstance(fmeta, dict):
                                time_v = fmeta.get("time")
                                stored = fmeta.get("stored_filename") or fmeta.get("filename")
                            else:
                                time_v = getattr(fmeta, "time", None)
                                stored = getattr(fmeta, "stored_filename", None) or getattr(fmeta, "filename", None)
                            # logger.info("Frame %s meta for %s -> stored=%s time=%s raw=%s", fn_i, fname, stored, time_v, (str(fmeta)[:200] if fmeta is not None else 'None'))
                            # Prefer providing a short dense caption if available so the model
                            # can reason over text rather than raw pixels. Keep captions single-line.
                            caption = None
                            if isinstance(fmeta, dict):
                                caption = fmeta.get("caption")
                            else:
                                caption = getattr(fmeta, "caption", None)
                            if caption is not None:
                                try:
                                    caption = str(caption).replace('\n', ' ').replace('\r', ' ')[:400]
                                except Exception:
                                    caption = None
                            if caption:
                                captions_found = True
                            if stored:
                                if caption:
                                    image_context_lines.append(f"IMAGE {fname} FRAME {fn_i} URL /uploads/{stored} TIME {time_v} CAPTION {caption}")
                                else:
                                    image_context_lines.append(f"IMAGE {fname} FRAME {fn_i} URL /uploads/{stored} TIME {time_v}")
                                total_images += 1
                    elif isinstance(frames_map, list):
                        for fmeta in frames_map:
                            fn_i = safe_int_val(getattr(fmeta, "frame_number", None) or (fmeta or {}).get("frame_number"))
                            if fn_i is None:
                                continue
                            time_v = getattr(fmeta, "time", None) if not isinstance(fmeta, dict) else fmeta.get("time")
                            stored = getattr(fmeta, "stored_filename", None) or (fmeta or {}).get("stored_filename") or getattr(fmeta, "filename", None) or (fmeta or {}).get("filename")
                            # logger.info("Frame %s meta for %s (list-entry) -> stored=%s time=%s raw=%s", fn_i, fname, stored, time_v, (str(fmeta)[:200] if fmeta is not None else 'None'))
                            caption = None
                            if isinstance(fmeta, dict):
                                caption = fmeta.get("caption")
                            else:
                                caption = getattr(fmeta, "caption", None)
                            if caption is not None:
                                try:
                                    caption = str(caption).replace('\n', ' ').replace('\r', ' ')[:400]
                                except Exception:
                                    caption = None
                            if stored:
                                if caption:
                                    image_context_lines.append(f"IMAGE {fname} FRAME {fn_i} URL /uploads/{stored} TIME {time_v} CAPTION {caption}")
                                else:
                                    image_context_lines.append(f"IMAGE {fname} FRAME {fn_i} URL /uploads/{stored} TIME {time_v}")
                                total_images += 1
        except Exception:
            # Be resilient if documents_map is malformed
            logger.exception("Error building image context for LLM request")

        prompt = f"""
            Based on the following chat history context, video document context, and current question, answer the question. 
            Provide answer and cite which documents and pages you're referencing.

            IMPORTANT: When referencing specific segments, use this special format:
            $PAGE_START{{filename}}:{{page_numbers_or_ranges}}$PAGE_END

            Examples:
            - For single segment: $PAGE_STARTvideo.mp4:5$PAGE_END
            - For multiple segments: $PAGE_STARTvideo.mp4:2,7,12$PAGE_END 
            - For segment range: $PAGE_STARTvideo.mp4:15-18$PAGE_END

            <Chat History>
            {history_context}
            <Chat History>
            
            <Current Question>
            {question}
            <Current Question>

            <Document Page Content>
            {json.dumps(relevant_pages)}
            <Document Page Content>

            Please provide answer based on the information in the documents and use the special page reference format when citing specific pages.
            If you use visual frames from the provided image context, cite them using this special format:
            $FRAME_START{{filename}}:{{frame_numbers_or_ranges}}$FRAME_END

            Examples for frames:
            - Single frame: $FRAME_STARTvideo.mp4:12$FRAME_END
            - Multiple frames: $FRAME_STARTvideo.mp4:12,13,17$FRAME_END
            - Frame range: $FRAME_STARTvideo.mp4:10-15$FRAME_END

            No need to mention the chat history in the answer, just focus on the current question.
            """

    # Prepend image context as a system message if images exist so the model receives the visual context
        messages_for_model = []
        if total_images > 0:
            # Provide concise image context: each line includes filename, frame number, URL, and time
            image_block = "\n".join(image_context_lines)
            messages_for_model.append({"role": "system", "content": "Embedded visual context (frame samples):\n" + image_block})
        # The main user prompt follows
        messages_for_model.append({"role": "user", "content": prompt})

        # Estimate a simple per-image input cost and yield it as an upfront cost event so UI can show it immediately.
        # These rates are conservative estimates and can be tuned.
        per_image_cost = {
            "gpt-5": 0.05,
            "gpt-5-mini": 0.01,
            "gpt-5-nano": 0.002,
        }
        image_input_cost = 0.0
        try:
            rate = per_image_cost.get(model, 0.002)
            image_input_cost = float(total_images) * float(rate)
        except Exception:
            image_input_cost = 0.0

        # Helper: resolve $PAGE_START and $FRAME_START markers into human timestamps using documents_map
        import re

        def seconds_to_hhmmss(s: float) -> str:
            s = int(round(s))
            h = s // 3600
            m = (s % 3600) // 60
            sec = s % 60
            if h:
                return f"{h:02d}:{m:02d}:{sec:02d}"
            return f"{m:02d}:{sec:02d}"

        _PAGE_RE = re.compile(r"\$PAGE_START([^:}]+):([^\$]+)\$PAGE_END")
        _FRAME_RE = re.compile(r"\$FRAME_START([^:}]+):([^\$]+)\$FRAME_END")

        def resolve_refs(text: str) -> tuple[str, List[Dict[str, Any]]]:
            """Resolve both page and frame markers. Returns (resolved_text, image_citations)

            image_citations is a list of dicts with keys: filename, frame_numbers, times, stored_filenames, human_readable
            """
            if not documents_map:
                return text, []

            dm = documents_map or {}
            image_citations: List[Dict[str, Any]] = []

            # First resolve page markers (keep behavior similar to before)
            def _page_repl(m):
                filename = m.group(1).strip()
                pages_str = m.group(2).strip()
                parts = [p.strip() for p in pages_str.split(",") if p.strip()]
                ts_parts = []
                for part in parts:
                    if "-" in part:
                        try:
                            a, b = part.split("-", 1)
                            a_i = safe_int_val(a.strip()); b_i = safe_int_val(b.strip())
                        except Exception:
                            continue
                        # Ensure a_i and b_i are valid integers
                        if a_i is None or b_i is None:
                            continue
                        if a_i > b_i:
                            a_i, b_i = b_i, a_i
                        times = []
                        for n in range(a_i, b_i + 1):
                            file_map = dm.get(filename) or {}
                            pages_map = file_map.get("pages") or {}
                            page = None
                            if isinstance(pages_map, dict):
                                try:
                                    page = pages_map.get(n)
                                except Exception:
                                    page = None
                            if isinstance(page, dict):
                                start = page.get("start_time")
                                end = page.get("end_time")
                                if start is not None and end is not None:
                                    times.append((start, end))
                        if times:
                            start = times[0][0]
                            end = times[-1][1]
                            ts_parts.append(f"{seconds_to_hhmmss(start)}–{seconds_to_hhmmss(end)}")
                    else:
                        n = safe_int_val(part)
                        if n is None:
                            continue
                        file_map = dm.get(filename) or {}
                        pages_map = file_map.get("pages") or {}
                        page = None
                        if isinstance(pages_map, dict):
                            try:
                                page = pages_map.get(n)
                            except Exception:
                                page = None
                        if isinstance(page, dict):
                            start = page.get("start_time")
                            end = page.get("end_time")
                            if start is not None and end is not None:
                                ts_parts.append(f"{seconds_to_hhmmss(start)}-{seconds_to_hhmmss(end)}")
                if ts_parts:
                    return f"(see {filename} @ {', '.join(ts_parts)})"
                return m.group(0)

            text_after_pages = _PAGE_RE.sub(_page_repl, text)

            # Now find frame markers and replace them with human readable text, while collecting image citation metadata
            def _frame_repl(m):
                filename = m.group(1).strip()
                frames_str = m.group(2).strip()
                parts = [p.strip() for p in frames_str.split(",") if p.strip()]
                readable_parts = []
                collected_numbers: List[int] = []
                times: List[float] = []
                stored_filenames: List[str] = []
                for part in parts:
                    if "-" in part:
                        try:
                            a, b = part.split("-", 1)
                            a_i = safe_int_val(a.strip()); b_i = safe_int_val(b.strip())
                        except Exception:
                            continue
                        # Ensure a_i and b_i are valid integers
                        if a_i is None or b_i is None:
                            continue
                        if a_i > b_i:
                            a_i, b_i = b_i, a_i
                        range_times = []
                        range_numbers = []
                        range_stored = []
                        def _get_frame_by_number(frames_map, num):
                            # frames_map may be a dict keyed by int or a list of frame dicts
                            if isinstance(frames_map, dict):
                                try:
                                    f = frames_map.get(num)
                                except Exception:
                                    f = None
                                if isinstance(f, dict):
                                    return f
                            elif isinstance(frames_map, list):
                                for item in frames_map:
                                    if isinstance(item, dict) and item.get("frame_number") == num:
                                        return item
                            return None

                        for n in range(a_i, b_i + 1):
                            file_map = dm.get(filename) or {}
                            frames_map = file_map["frames"] if isinstance(file_map, dict) and "frames" in file_map else {}
                            frame = _get_frame_by_number(frames_map, n)
                            if isinstance(frame, dict):
                                raw_time = frame.get("time")
                                if raw_time is None:
                                    continue
                                try:
                                    t_f = float(raw_time)
                                except Exception:
                                    continue
                                range_numbers.append(n)
                                range_times.append(t_f)
                                range_stored.append(str(frame.get("stored_filename") or frame.get("filename") or ""))
                        if range_times:
                            start = range_times[0]; end = range_times[-1]
                            readable_parts.append(f"{seconds_to_hhmmss(start)}–{seconds_to_hhmmss(end)}")
                            collected_numbers.extend(range_numbers)
                            times.extend(range_times)
                            stored_filenames.extend(range_stored)
                    else:
                        n = safe_int_val(part)
                        if n is None:
                            continue
                        file_map = dm.get(filename) or {}
                        frames_map = file_map["frames"] if isinstance(file_map, dict) and "frames" in file_map else {}
                        def _get_frame(frames_map, num):
                            if isinstance(frames_map, dict):
                                try:
                                    f = frames_map.get(num)
                                except Exception:
                                    f = None
                                if isinstance(f, dict):
                                    return f
                            elif isinstance(frames_map, list):
                                for item in frames_map:
                                    if isinstance(item, dict) and item.get("frame_number") == num:
                                        return item
                            return None
                        frame = _get_frame(frames_map, n)
                        if isinstance(frame, dict):
                            raw_time = frame.get("time")
                            if raw_time is None:
                                continue
                            try:
                                t_f = float(raw_time)
                            except Exception:
                                continue
                            collected_numbers.append(n)
                            times.append(t_f)
                            stored_filenames.append(str(frame.get("stored_filename") or frame.get("filename") or ""))
                            readable_parts.append(f"{seconds_to_hhmmss(t_f)}")

                if collected_numbers:
                    # record an image citation entry
                    image_citations.append(
                        {
                            "filename": filename,
                            "frame_numbers": collected_numbers,
                            "times": times,
                            "stored_filenames": stored_filenames,
                            "human_readable": readable_parts,
                        }
                    )
                    return f"(see images from {filename} @ {', '.join(readable_parts)})"
                return m.group(0)

            final_text = _FRAME_RE.sub(_frame_repl, text_after_pages)
            return final_text, image_citations

        try:
            if not self.client:
                # Graceful fallback when LLM client unavailable
                fallback_text = "LLM features are not available because the OpenAI API key is not configured."
                # provide a minimal content answer based on the provided pages
                if relevant_pages:
                    sample = relevant_pages[0].get("text", "")
                    fallback_text += "\n\nContext from document: " + sample[:100]
                yield {"type": "content", "content": fallback_text}
                yield {"type": "cost", "cost": 0.0}
                return

            # logger.info("Total sampled images available: %d", total_images)
            # logger.info("Sample image_context_lines: %s", (image_context_lines[:5]))

            # If we have an image input cost, emit it first so the client can display it immediately
            if image_input_cost and image_input_cost > 0:
                yield {"type": "cost", "cost": image_input_cost}
            # If we have images, prefer the Responses API which supports inline image inputs
            if total_images > 0 and hasattr(self.client, "responses"):
                try:
                    import base64

                    uploads_dir = os.path.join(os.getcwd(), "uploads")

                    # logger.info("Preparing to attach frames to Responses API call; uploads_dir=%s", uploads_dir)

                    # If captions exist for frames, prefer passing caption text only (skip image attachments)
                    if captions_found:
                        # logger.info("Detected captions for frames; using captions-only visual context and skipping image attachments")
                        # Emit a small event so clients can display that captions will be used
                        try:
                            yield {"type": "log", "level": "info", "message": "Using frame captions for visual context (images not attached)."}
                        except Exception:
                            # Some clients may not handle 'log' type; continue silently
                            pass

                        # Prepare a Responses API call which includes the image_block (captions)
                        # so the model receives the caption text as visual context.
                        try:
                            image_block = "\n".join(image_context_lines) if image_context_lines else ""
                        except Exception:
                            image_block = ""

                        content_items = []
                        # Provide the embedded visual context first so the model can reference captions
                        if image_block:
                            content_items.append({"type": "input_text", "text": "Embedded visual context (frame samples):\n" + image_block})
                        # Then the user's prompt
                        content_items.append({"type": "input_text", "text": prompt})

                        resp_input = [
                            {
                                "role": "user",
                                "content": content_items,
                            }
                        ]

                        # Cast to Any for SDK input typing compatibility
                        resp_input_any: Any = resp_input
                        response = await self.client.responses.create(
                            model=model,
                            input=resp_input_any,
                        )
                    else:
                        # Build content array: user prompt followed by input_image entries for each frame
                        content_items = []
                        # The user's prompt is the main input_text
                        content_items.append({"type": "input_text", "text": prompt})

                        # For debugging: count of frames we attempted to attach
                        attempted_attach = 0
                        attached_images = 0

                        # For each document/frame, append images
                        # We'll attach all sampled frames; this may be large for long videos.
                        # logger.info("documents_map keys: %s", list((documents_map or {}).keys()))
                        for fname, fmap in (documents_map or {}).items():
                            frames_map = fmap.get("frames") or {}
                            # logger.info("Document %s frames_map type=%s", fname, type(frames_map))
                            frames_list = []
                            if isinstance(frames_map, dict):
                                # gather dict values
                                for k, v in frames_map.items():
                                    if isinstance(v, dict):
                                        frames_list.append(v)
                            elif isinstance(frames_map, list):
                                frames_list = list(frames_map)

                            # logger.info("Found %d frame entries for document %s", len(frames_list), fname)

                            for fmeta in frames_list:
                                attempted_attach += 1
                                try:
                                    if not isinstance(fmeta, dict):
                                        # logger.info("Skipping non-dict frame metadata: %s", type(fmeta))
                                        continue
                                    stored = fmeta.get("stored_filename") or fmeta.get("filename")
                                    frame_num = fmeta.get("frame_number")
                                    # logger.info("Frame metadata: stored=%s frame_number=%s", stored, frame_num)
                                    if not stored:
                                        # logger.info("Skipping frame with no stored filename: %s", fmeta)
                                        continue
                                    img_path = os.path.join(uploads_dir, stored)
                                    if not os.path.exists(img_path):
                                        logger.warning("Frame file not found, skipping: %s", img_path)
                                        continue
                                    # Log file size for diagnostic
                                    try:
                                        size = os.path.getsize(img_path)
                                    except Exception:
                                        size = None
                                    # logger.info("Attaching image file %s (size=%s bytes)", img_path, size)
                                    with open(img_path, "rb") as fh:
                                        b64 = base64.b64encode(fh.read()).decode("utf-8")
                                    data_url = f"data:image/jpeg;base64,{b64}"
                                    content_items.append({"type": "input_image", "image_url": data_url})
                                    attached_images += 1
                                except Exception:
                                    logger.exception("Failed to attach image %s to model input", fmeta)

                        # logger.info("Frame attachment summary: attempted=%d attached=%d", attempted_attach, attached_images)

                        # Construct input for Responses API
                        resp_input = [
                            {
                                "role": "user",
                                "content": content_items,
                            }
                        ]

                        # Make the call (async)
                        resp_input_any: Any = resp_input
                        response = await self.client.responses.create(
                            model=model,
                            input=resp_input_any,
                        )

                    # logger.info("Responses API raw response: %s", response)
                    try:
                        u = getattr(response, "usage", None)
                        # logger.info("Responses API usage: %s", u)
                    except Exception:
                        pass

                    # Extract textual output robustly
                    output_text = None
                    try:
                        output_text = getattr(response, "output_text", None)
                    except Exception:
                        output_text = None
                    if not output_text:
                        try:
                            # SDK may return output as list of content blocks
                            out = getattr(response, "output", None) or []
                            if isinstance(out, list) and len(out) > 0:
                                # content may be nested
                                first = out[0]
                                # try to find text in first
                                if isinstance(first, dict):
                                    # look for 'content' -> list -> element with 'text'
                                    cont = first.get("content") or []
                                    for c in cont:
                                        if isinstance(c, dict) and c.get("type") == "output_text":
                                            output_text = c.get("text")
                                            break
                                else:
                                    output_text = str(first)
                        except Exception:
                            output_text = None

                    if not output_text:
                        # As a last resort, stringify response
                        try:
                            output_text = str(response)
                        except Exception:
                            output_text = ""

                    # logger.info("Responses API output_text length: %d", len(output_text or ""))

                    # Resolve refs and emit
                    resolved_text, image_citations = resolve_refs(output_text or "")
                    yield {"type": "content", "content": resolved_text}
                    for img in image_citations:
                        yield {"type": "image_citation", "citation": img}

                    # Emit usage-based cost if available
                    try:
                        usage = getattr(response, "usage", None)
                        if usage:
                            yield {"type": "cost", "cost": self.calculate_cost(usage, model=model)}
                    except Exception:
                        pass
                except Exception:
                    logger.exception("Error while calling Responses API with images")
                    # Fall back to chat streaming if Responses API fails
                    pass
                return

            # Fallback: no images or Responses API not available -> use streaming chat completions
            if total_images > 0 and not hasattr(self.client, "responses"):
                logger.warning("Responses API not available on client; falling back to streaming chat completions. total_images=%d", total_images)
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages_for_model,
                stream=True,
                stream_options={"include_usage": True},
            )

            buffer = ""
            async for chunk in stream:
                if chunk.usage:
                    yield {
                        "type": "cost",
                        "cost": self.calculate_cost(chunk.usage, model=model),
                    }
                if len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta.content is not None:
                        # Buffer a sliding window to handle markers split across chunks
                        piece = delta.content
                        buffer += piece
                        # Resolve both page and frame markers and collect image citations
                        resolved_text, image_citations = resolve_refs(buffer)
                        # Emit resolved text
                        yield {
                            "type": "content",
                            "content": resolved_text,
                        }
                        # Emit any image citation events so UI can show thumbnails/jump links separately
                        for img in image_citations:
                            yield {
                                "type": "image_citation",
                                "citation": img,
                            }
                        buffer = ""

        except Exception:
            logger.exception("Error generating answer stream")
            yield {"type": "content", "content": "Error generating answer. See server logs for details."}
            yield {"type": "cost", "cost": 0.0}
