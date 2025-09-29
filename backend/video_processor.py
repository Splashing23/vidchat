import os
import tempfile
import math
from typing import List, Dict, Any, Optional
import logging
try:
    import numpy as np
except Exception:
    np = None

try:
    # moviepy's primary entry for loading clips
    from moviepy import VideoFileClip
    _MOVIEPY_AVAILABLE = True
except Exception:
    VideoFileClip = None  # type: ignore
    _MOVIEPY_AVAILABLE = False

try:
    # prefer the new OpenAI client API
    from openai import OpenAI
    _OPENAI_CLIENT_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_CLIENT_AVAILABLE = False


# Configure module logger to ensure messages go to the backend terminal
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class VideoProcessor:
    """Simple video processor that extracts duration, attempts transcription using OpenAI whisper-1
    (if OPENAI_API_KEY is set and openai package available), and produces timestamped text segments.

    Requirements: system ffmpeg/ffprobe must be available on PATH for duration and audio extraction.
    If transcription is not possible the processor will return minimal placeholder segments.
    """

    def __init__(self, uploads_dir: Optional[str] = None, segment_seconds: int = 15, max_duration_seconds: int = 180):
        self.segment_seconds = segment_seconds
        self.max_duration_seconds = max_duration_seconds
        self.uploads_dir = uploads_dir or os.path.join(os.getcwd(), "uploads")
        os.makedirs(self.uploads_dir, exist_ok=True)

    def _get_duration(self, path: str) -> Optional[float]:
        if not _MOVIEPY_AVAILABLE or VideoFileClip is None:
            return None
        try:
            clip = VideoFileClip(path)
            try:
                duration = clip.duration
                return float(duration)
            finally:
                try:
                    clip.close()
                except Exception:
                        logger.exception("Error closing video clip")
        except Exception:
            return None

    def _extract_audio(self, video_path: str, out_audio: str) -> bool:
        if not _MOVIEPY_AVAILABLE or VideoFileClip is None:
            # logger.info("moviepy or videofileclip modules not available")
            return False
        try:
            clip = VideoFileClip(video_path)
            try:
                if not clip.audio:
                    # logger.info("no audio track found in video with path %s", video_path)
                    return False
                # write audio file in wav format with 16 kHz sampling if possible
                # moviepy wraps ffmpeg; providing fps and nbytes helps ensure
                # a PCM-like WAV that transcription tools accept.
                clip.audio.write_audiofile(out_audio, logger=None)
                return True
            finally:
                try:
                    clip.close()
                except Exception:
                        logger.exception("Error closing video clip")
        except Exception as e:
            logger.exception("exception during audio extraction: %s", e)
            return False

    def _transcribe_with_openai(self, audio_path: str) -> Optional[Dict[str, Any]]:
        # Prefer the explicit new OpenAI sync client pattern the app expects.
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key or not _OPENAI_CLIENT_AVAILABLE or OpenAI is None:
                return None

            client = OpenAI(api_key=api_key)
            with open(audio_path, "rb") as fh:
                # Prefer verbose JSON (per-segment tokens) when available
                resp = None
                try:
                    resp = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=fh,
                        response_format="verbose_json",
                        timestamp_granularities=["segment"],
                    )
                except Exception:
                    # Fallback to flexible call shapes
                    audio_obj = getattr(client, "audio", None)
                    if audio_obj is not None:
                        transcriptions = getattr(audio_obj, "transcriptions", None)
                        if transcriptions is not None and hasattr(transcriptions, "create"):
                            try:
                                fh.seek(0)
                                resp = transcriptions.create(
                                    model="whisper-1",
                                    file=fh,
                                    response_format="verbose_json",
                                    timestamp_granularities=["segment"],
                                )
                            except Exception:
                                resp = None

                # The SDK may return either plain dicts or model objects (e.g., TranscriptionVerbose
                # and TranscriptionSegment). Handle both shapes robustly.
                transcript_text = None
                segments = []
                token_count = 0

                if isinstance(resp, dict):
                    transcript_text = resp.get("text") if isinstance(resp.get("text"), str) else None
                    segments = resp.get("segments") or []
                else:
                    # SDK model object: prefer attribute access
                    transcript_text = getattr(resp, "text", None)
                    segments = getattr(resp, "segments", []) or []

                # Sum tokens across segments when available. Support both dict and object segment shapes.
                if isinstance(segments, list):
                    for seg in segments:
                        toks = None
                        if isinstance(seg, dict):
                            toks = seg.get("tokens") or []
                        else:
                            # TranscriptionSegment objects expose tokens as an attribute
                            toks = getattr(seg, "tokens", None) or getattr(seg, "token_ids", None) or []
                        if isinstance(toks, list):
                            token_count += len(toks)

                transcription_cost = 0.0
                if token_count > 0:
                    transcription_cost = (token_count / 1_000_000.0) * 1.25

                return {
                    "text": transcript_text or "",
                    "token_count": token_count,
                    "cost": transcription_cost,
                    "segments": segments,
                    "raw": resp,
                }
        except Exception:
            logger.exception("Error transcribing audio: %s", audio_path)
            return None

    def process_video(self, video_path: str, original_filename: str) -> Dict[str, Any]:
        # Save video into uploads dir so it can be served
        stored_name = os.path.basename(original_filename)
        stored_path = os.path.join(self.uploads_dir, stored_name)
        try:
            os.replace(video_path, stored_path)
        except Exception:
            # fallback to copy
            import shutil

            shutil.copyfile(video_path, stored_path)

        # logger.info("saved video to: %s", stored_path)

        # Get duration
        duration = self._get_duration(stored_path) or 0.0
        # logger.info("duration: %s", duration)
        if duration > self.max_duration_seconds:
            raise Exception(f"Video is too long ({duration:.1f}s). Max allowed is {self.max_duration_seconds}s.")

        # Attempt to extract audio and transcribe
        audio_path = "uploads/audio.wav"

        extracted = self._extract_audio(stored_path, audio_path)
        # logger.info("audio extracted: %s -> %s", extracted, audio_path if extracted else "N/A")
        transcript_info: Optional[Dict[str, Any]] = None
        transcription_error_msg: Optional[str] = None
        if extracted:
            transcript_info = self._transcribe_with_openai(audio_path)
            if transcript_info is None:
                transcription_error_msg = (
                    "Transcription failed or OpenAI client not available. "
                    "Ensure OPENAI_API_KEY is set and the OpenAI Python SDK is installed."
                )
        else:
            transcription_error_msg = (
                "Audio extraction failed. Ensure ffmpeg is installed and available on PATH, "
                "and that the video file contains an audio track."
            )

        # try:
        #     os.unlink(audio_path)
        # except Exception:
        #     pass

        # Fallback transcript text
        transcript_text = ""
        processing_cost = 0.0
        segments_info = []
        if transcript_info:
            transcript_text = transcript_info.get("text", "") or ""
            processing_cost = float(transcript_info.get("cost", 0.0) or 0.0)
            segments_info = transcript_info.get("segments") or []
            try:
                pass # logger.info("transcription token_count: %s cost: %s", transcript_info.get("token_count"), processing_cost)
            except Exception:
                pass
            try:
                pass # logger.info("transcript_info raw (truncated): %s", str(transcript_info.get("raw"))[:1000])
            except Exception:
                pass
        elif transcription_error_msg:
            transcript_text = transcription_error_msg
            logger.warning("transcription error: %s", transcription_error_msg)

        # For simplicity and to support querying the entire transcription at once,
        # return a single segment containing the full transcription text. This avoids
        # chunked/page-based retrieval and lets the chat pipeline query the entire
        # transcript in one pass.
        full_text = transcript_text or ""
        # logger.info("final transcript length: %s chars", len(full_text))
        # Emit the full transcribed text to the backend terminal for debugging.
        # NOTE: This may be large for long transcripts; consider switching to DEBUG in production.
        try:
            pass # logger.info("transcribed_text:\n%s", full_text)
        except Exception:
            pass # logger.exception("Failed to log transcribed text")

        # If raw per-segment data exists from the transcription, return one DocumentPage
        # per transcription segment for reliable timestamping. Otherwise, fall back to
        # a single full-transcript page.
        segments: List[Dict[str, Any]] = []
        if segments_info and isinstance(segments_info, list):
            page_num = 1
            for seg in segments_info:
                # seg may be a dict or an SDK object
                if isinstance(seg, dict):
                    seg_text = seg.get("text", "")
                    start = float(seg.get("start") or seg.get("start_time") or 0.0)
                    end = float(seg.get("end") or seg.get("end_time") or start)
                else:
                    seg_text = getattr(seg, "text", "") or ""
                    start = float(getattr(seg, "start", 0.0) or getattr(seg, "start_time", 0.0) or 0.0)
                    end = float(getattr(seg, "end", start) or getattr(seg, "end_time", start) or start)

                segments.append(
                    {
                        "page_number": page_num,
                        "text": seg_text.strip(),
                        "start_time": round(start, 3),
                        "end_time": round(end, 3),
                        "char_count": len(seg_text or ""),
                    }
                )
                page_num += 1

        # If no per-segment info, return a single page with the full transcript.
        if not segments:
            segments = [
                {
                    "page_number": 1,
                    "text": full_text,
                    "start_time": 0.0,
                    "end_time": round(duration or 0.0, 2),
                    "char_count": len(full_text),
                }
            ]

        # Sample one frame per second (visual context) if moviepy is available
        frames_metadata: List[Dict[str, Any]] = []
        try:
            if _MOVIEPY_AVAILABLE and VideoFileClip is not None:
                clip = VideoFileClip(stored_path)
                try:
                    # sample one frame every two seconds to reduce number of images
                    step_seconds = 2  # one frame per 2 seconds
                    total_seconds = int(math.ceil(duration))
                    for sec in range(0, total_seconds, step_seconds):
                        try:
                            frame = clip.get_frame(sec)
                            # Save frame as JPG
                            from PIL import Image

                            if frame is None:
                                raise ValueError("frame is None")
                            # Ensure frame is a numpy array
                            if np is not None and hasattr(frame, 'astype'):
                                arr = frame.astype('uint8')
                            else:
                                arr = frame
                            frame_img = Image.fromarray(arr)
                            frame_filename = f"{os.path.splitext(stored_name)[0]}_frame_{sec}.jpg"
                            frame_path = os.path.join(self.uploads_dir, frame_filename)
                            frame_img.save(frame_path, format='JPEG', quality=75)
                            frames_metadata.append({
                                "frame_number": sec,
                                "time": float(sec),
                                "filename": frame_filename,
                                "stored_filename": frame_filename,
                            })
                        except Exception:
                            logger.exception("Failed to extract/save frame at %ss", sec)
                finally:
                    try:
                        clip.close()
                    except Exception:
                        logger.exception("Error closing clip after frame extraction")
        except Exception:
            logger.exception("Error while sampling frames for visual context")

        # logger.info("returning %s segment(s); processing_cost=%s frames=%s", len(segments), processing_cost, len(frames_metadata))

        # Attempt to generate short captions for each sampled frame at upload time so
        # the frontend and LLM can use captions instead of attaching raw images.
        captions_cost = 0.0
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and _OPENAI_CLIENT_AVAILABLE and OpenAI is not None and frames_metadata:
                client = OpenAI(api_key=api_key)
                import base64

                # estimate a tiny per-image captioning cost (conservative)
                per_caption_rate = 0.005
                attempted = 0
                succeeded = 0

                # Prepare a worker that will caption a single frame. We'll run workers in a ThreadPool
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _caption_worker(fm_item: Dict[str, Any]) -> bool:
                    fn_local = fm_item.get("stored_filename") or fm_item.get("filename")
                    if not fn_local:
                        return False
                    img_path_local = os.path.join(self.uploads_dir, fn_local)
                    if not os.path.exists(img_path_local):
                        logger.warning("Captioning: frame file missing %s", img_path_local)
                        fm_item["caption_full"] = f"Frame at {fm_item.get('time', '0')}s"
                        fm_item["caption"] = fm_item["caption_full"]
                        return False
                    try:
                        with open(img_path_local, "rb") as fh:
                            b64 = base64.b64encode(fh.read()).decode("utf-8")
                        data_url = f"data:image/jpeg;base64,{b64}"

                        # Build a minimal Responses API input: image + instruction to produce a single-line caption
                        content_items_local = [
                            {"type": "input_image", "image_url": data_url},
                            {"type": "input_text", "text": "Provide a dense caption describing this image. Return only the caption text."},
                        ]

                        resp = None
                        try:
                            resp_input = [{"role": "user", "content": content_items_local}]
                            resp_input_any: Any = resp_input
                            resp = client.responses.create(
                                model="gpt-5-mini",
                                input=resp_input_any,
                            )
                        except Exception:
                            try:
                                audio_obj = getattr(client, "responses", None)
                                if audio_obj is not None and hasattr(audio_obj, "create"):
                                    resp = audio_obj.create(
                                        model="gpt-5-mini",
                                        input=[{"role": "user", "content": content_items_local}],
                                    )
                            except Exception:
                                resp = None

                        caption_text_local = None
                        try:
                            caption_text_local = getattr(resp, "output_text", None)
                        except Exception:
                            caption_text_local = None

                        if not caption_text_local and isinstance(resp, dict):
                            caption_text_local = resp.get("output_text") or resp.get("output")

                        if not caption_text_local and resp is not None:
                            try:
                                out = getattr(resp, "output", None) or (resp.get("output") if isinstance(resp, dict) else None) or []
                                if isinstance(out, list) and len(out) > 0:
                                    first = out[0]
                                    if isinstance(first, dict):
                                        cont = first.get("content") or getattr(first, "content", None) or []
                                        for c in cont:
                                            if isinstance(c, dict) and (c.get("type") == "output_text" or getattr(c, "type", None) == "output_text"):
                                                caption_text_local = c.get("text") or getattr(c, "text", None)
                                                break
                                    else:
                                        caption_text_local = str(first)
                            except Exception:
                                caption_text_local = None

                        if caption_text_local:
                            try:
                                import re

                                raw = str(caption_text_local)
                                caption_full_local = raw.strip()
                                s = ' '.join([ln.strip() for ln in caption_full_local.splitlines() if ln.strip()])
                                s = re.sub(r"\s+", " ", s).strip()
                                max_len = 240
                                if len(s) > max_len:
                                    s = s[: max_len - 1].rstrip() + "…"
                                caption_line_local = s
                            except Exception:
                                caption_full_local = str(caption_text_local).strip()
                                caption_line_local = caption_full_local.splitlines()[0].strip()

                            fm_item["caption_full"] = caption_full_local
                            fm_item["caption"] = caption_line_local
                            return True
                        else:
                            fm_item["caption_full"] = f"Frame at {fm_item.get('time', '0')}s"
                            fm_item["caption"] = fm_item["caption_full"]
                            return False
                    except Exception:
                        logger.exception("Failed to generate caption for frame %s", fn_local)
                        fm_item["caption_full"] = f"Frame at {fm_item.get('time', '0')}s"
                        fm_item["caption"] = fm_item["caption_full"]
                        return False

                # Run workers with a conservative parallelism cap so we don't overload the API
                workers = min(6, max(1, len(frames_metadata)))
                futures = []
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for fm in frames_metadata:
                        fn = fm.get("stored_filename") or fm.get("filename")
                        if not fn:
                            continue
                        # Count attempts for frames that appear to have files
                        img_path_check = os.path.join(self.uploads_dir, fn)
                        if not os.path.exists(img_path_check):
                            logger.warning("Captioning: frame file missing %s", img_path_check)
                            fm["caption_full"] = f"Frame at {fm.get('time', '0')}s"
                            fm["caption"] = fm["caption_full"]
                            continue
                        attempted += 1
                        futures.append(ex.submit(_caption_worker, fm))

                    for fut in as_completed(futures):
                        try:
                            ok = fut.result()
                            if ok:
                                succeeded += 1
                        except Exception:
                            logger.exception("Caption worker raised an exception")

                # Add conservative captioning cost estimate
                captions_cost = float(attempted) * float(per_caption_rate)
                processing_cost = float(processing_cost) + captions_cost
                # logger.info("Captioning attempted=%d succeeded=%d estimated_cost=%s", attempted, succeeded, captions_cost)
                # Persist frames metadata (including captions) to a sidecar JSON so other
                # services can load captions without requiring the upload path to keep state.
                try:
                    import json

                    basename = os.path.splitext(stored_name)[0]
                    sidecar_path = os.path.join(self.uploads_dir, f"{basename}_frames.json")
                    with open(sidecar_path, "w", encoding="utf-8") as sf:
                        json.dump({"frames": frames_metadata}, sf, ensure_ascii=False, indent=2)
                    # logger.info("Wrote frames sidecar: %s", sidecar_path)
                except Exception:
                    logger.exception("Failed to write frames sidecar JSON")
        except Exception:
            logger.exception("Error during frame captioning step")

        return {
            "stored_filename": stored_name,
            "stored_path": stored_path,
            "duration": duration,
            "segments": segments,
            "processing_cost": processing_cost,
            "raw_segments": segments_info,
            "frames": frames_metadata,
            "captions_cost": captions_cost,
        }
