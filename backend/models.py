from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[str] = None


class DocumentPage(BaseModel):
    page_number: int
    text: str
    char_count: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class VideoSegment(BaseModel):
    page_number: int
    text: str
    start_time: float
    end_time: float
    char_count: Optional[int] = None


class ImageFrame(BaseModel):
    frame_number: int
    time: float
    filename: str
    stored_filename: Optional[str] = None


class DocumentData(BaseModel):
    id: int
    filename: str
    pages: List[DocumentPage]
    total_pages: int
    # For videos, stored_filename is the server-side filename under /uploads
    stored_filename: Optional[str] = None
    is_video: Optional[bool] = False
    # Cost incurred during preprocessing (transcription/extraction)
    processing_cost: Optional[float] = 0.0
    # Whether a real transcription was produced or a placeholder
    transcription_available: Optional[bool] = True
    transcription_error: Optional[str] = None
    # Per-second sampled image frames (one frame per second) for visual context
    frames: Optional[List[ImageFrame]] = []


class VideoData(BaseModel):
    id: int
    filename: str
    stored_filename: str
    duration: float
    segments: List[VideoSegment]


class ChatRequest(BaseModel):
    question: str
    documents: List[DocumentData]  # Documents sent from client
    description: str  # Collection description
    chat_history: Optional[List[ChatMessage]] = []
    model: Optional[str] = "gpt-5-nano"


class ChatResponse(BaseModel):
    answer: str
    selected_documents: List[str]
    relevant_pages_count: int
    total_cost: Optional[float] = 0.0


class UploadResponse(BaseModel):
    documents: List[DocumentData]  # Return processed documents to client
    message: str


class UpdateDescriptionRequest(BaseModel):
    description: str


class AddDocumentsResponse(BaseModel):
    documents: List[DocumentData]  # Return all documents including new ones
    message: str
    new_documents_count: int


class SessionData(BaseModel):
    session_id: str
    description: str
    documents: List[Dict[str, Any]]
    created_at: datetime
    total_session_cost: Optional[float] = 0.0
