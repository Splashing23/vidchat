'use client';

import React, { useState, useRef, useEffect } from 'react';
import config from '../../config';
import ReactMarkdown from 'react-markdown';
import { Send, Upload, X, FileText, Eye, EyeOff, Settings, Plus, Trash2, Edit2, Check, RotateCcw } from 'lucide-react';
import ChunkedUploader from '../utils/chunkedUpload';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  progress?: {
    status: string;
    step: number;
    total: number;
    stepCost?: number;
    stepTime?: number;
  };
  metadata?: {
    selectedDocuments?: Array<{id: number, filename: string}>;
    relevantPagesCount?: number;
    timing?: {
      document_selection?: number;
      page_detection?: number;
      answer_generation?: number;
      total_time?: number;
    };
    costs?: {
      document_selection?: number;
      page_detection?: number;
      answer_generation?: number;
      total_cost?: number;
    };
    imageCitations?: Array<any>;
    model?: string;
  };
}

interface DocumentData {
  id: number;
  filename: string;
  pages: Array<{
    page_number: number;
    text: string;
    char_count?: number;
    start_time?: number;
    end_time?: number;
  }>;
  total_pages: number;
  stored_filename?: string;
  is_video?: boolean;
  processing_cost?: number;
  transcription_available?: boolean;
  transcription_error?: string | null;
}

interface StatelessChatSectionProps {
  documents: DocumentData[];
  description: string;
  onReset: () => void;
  onUpdateDocuments: (documents: DocumentData[]) => void;
  onUpdateDescription: (description: string) => void;
}

export default function StatelessChatSection({ 
  documents, 
  description, 
  onReset, 
  onUpdateDocuments, 
  onUpdateDescription 
}: StatelessChatSectionProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  // Hide the uploaded-documents dropdown by default; user can expand it
  const [showDocuments, setShowDocuments] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('gpt-5');
  const [isEditingDescription, setIsEditingDescription] = useState(false);
  const [editedDescription, setEditedDescription] = useState('');
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);
  const [selectedNewFiles, setSelectedNewFiles] = useState<File[]>([]);
  const [showUploadSection, setShowUploadSection] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [uploadProgress, setUploadProgress] = useState<{
    percentComplete: number;
    processedFiles: number;
    totalFiles: number;
    currentChunk: number;
    totalChunks: number;
    isChunked: boolean;
  } | null>(null);
  const [totalSessionCost, setTotalSessionCost] = useState<number>(0);
  const [selectedPageContent, setSelectedPageContent] = useState<{
    content: string;
    pageNumber: number;
    filename: string;
  } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // When parent clears documents (start new session), clear chat messages and session state
  useEffect(() => {
    if (!documents || documents.length === 0) {
      // Clear messages, selected page modal, and session cost when there are no uploaded documents
      setMessages([]);
      setSelectedPageContent(null);
      setTotalSessionCost(0);
      setIsLoading(false);
      setIsUploadingFiles(false);
      setSelectedNewFiles([]);
      setShowUploadSection(false);
      setUploadProgress(null);
    }
  }, [documents]);

  const handleSendMessage = async () => {
    if (!currentQuestion.trim() || isLoading) return;

    const question = currentQuestion;
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentQuestion('');
    setIsLoading(true);

    // Create assistant message placeholder for streaming
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
      progress: {
        status: 'Starting...',
        step: 0,
        total: 3,
      },
      metadata: {
        selectedDocuments: [],
        relevantPagesCount: 0,
      },
    };
    
    setMessages(prev => [...prev, assistantMessage]);

    try {
      const response = await fetch(`${config.apiBaseUrl}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          documents: documents,
          description: description,
          model: selectedModel,
          chat_history: messages.map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp.toISOString()
          }))
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No reader available');
      }

      let accumulatedContent = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'status') {
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId 
                    ? {
                        ...msg,
                        progress: {
                          status: data.message,
                          step: data.step_number,
                          total: data.total_steps,
                        }
                      }
                    : msg
                ));
              } else if (data.type === 'step_complete') {
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId 
                    ? {
                        ...msg,
                        progress: {
                          ...msg.progress!,
                          stepCost: data.cost,
                          stepTime: data.time_taken,
                        },
                        metadata: {
                          ...msg.metadata,
                          ...(data.step === 'document_selection' && {
                            selectedDocuments: data.selected_documents,
                          }),
                          ...(data.step === 'page_selection' && {
                            relevantPagesCount: data.relevant_pages_count,
                          }),
                        }
                      }
                    : msg
                ));
              } else if (data.type === 'content') {
                accumulatedContent += data.content;
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId 
                    ? { ...msg, content: accumulatedContent }
                    : msg
                ));
              } else if (data.type === 'complete') {
                // Complete: update final metadata but do not add total_cost here because incremental "cost" events
                // have already been applied to the session total as they arrived.
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId 
                    ? {
                        ...msg,
                        isStreaming: false,
                        progress: undefined,
                        metadata: {
                          ...msg.metadata,
                          timing: data.timing_breakdown,
                          costs: data.cost_breakdown,
                          model: selectedModel
                        }
                      }
                    : msg
                ));
                
                // Reset loading state when streaming is complete
                setIsLoading(false);
              } else if (data.type === 'error') {
                throw new Error(data.error);
              } else if (data.type === 'image_citation') {
                // Attach image citation to the streaming assistant message so UI can render thumbnails/jump links
                const citation = data.citation;
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId
                    ? {
                        ...msg,
                        metadata: {
                          ...msg.metadata,
                          imageCitations: [...(msg.metadata?.imageCitations || []), citation]
                        }
                      }
                    : msg
                ));
              } else if (data.type === 'cost') {
                // Incrementally add cost chunks to the visible total so users see charges as they accrue
                const c = Number(data.cost || 0);
                if (!Number.isNaN(c) && c !== 0) {
                  setTotalSessionCost(prev => prev + c);
                }
              }
            } catch (parseError) {
              console.warn('Failed to parse SSE data:', line);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessageId 
          ? {
              ...msg,
              content: 'Sorry, I encountered an error while processing your question. Please try again.',
              isStreaming: false,
              progress: undefined
            }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleEditDescription = () => {
    setEditedDescription(description);
    setIsEditingDescription(true);
  };

  const handleSaveDescription = () => {
    if (!editedDescription.trim()) {
      return;
    }
    onUpdateDescription(editedDescription);
    setIsEditingDescription(false);
  };

  const handleCancelEdit = () => {
    setIsEditingDescription(false);
    setEditedDescription('');
  };

  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);

    // Only allow a single file from the picker
    if (files.length > 1) {
      setUploadError('Only one video may be uploaded at a time');
      return;
    }

    // If a video already exists, prevent adding another
    if (documents.length > 0) {
      setUploadError('A video is already uploaded. Please delete it before uploading another.');
      return;
    }

    const file = files[0];
    if (!file) {
      setUploadError('No file selected');
      return;
    }

    const allowedExt = ['.mp4', '.webm'];
    if (!allowedExt.some(ext => file.name.toLowerCase().endsWith(ext))) {
      setUploadError('Only video files (.mp4, .webm) are allowed');
      return;
    }

    // Check duration: load file into a blob URL and use an offscreen video element
    try {
      const url = URL.createObjectURL(file);
      const videoEl = document.createElement('video');
      videoEl.preload = 'metadata';
      videoEl.src = url;

      const duration: number = await new Promise((resolve, reject) => {
        const onLoaded = () => {
          cleanup();
          resolve(videoEl.duration || 0);
        };
        const onError = () => {
          cleanup();
          reject(new Error('Failed to read video metadata'));
        };
        const cleanup = () => {
          videoEl.removeEventListener('loadedmetadata', onLoaded);
          videoEl.removeEventListener('error', onError);
          try { URL.revokeObjectURL(url); } catch (e) {}
        };
        videoEl.addEventListener('loadedmetadata', onLoaded);
        videoEl.addEventListener('error', onError);
      });

      // Reject if duration >= 180 seconds (3 minutes)
      if (!isFinite(duration) || duration >= 180) {
        setUploadError('Video must be shorter than 3 minutes');
        return;
      }

      // Accept the file (store only the one)
      setSelectedNewFiles([file]);
      setUploadError('');
    } catch (err) {
      console.error('Error checking video duration:', err);
      setUploadError('Failed to validate video. Please try a different file.');
    }
  };

  const removeNewFile = (index: number) => {
    setSelectedNewFiles(files => files.filter((_, i) => i !== index));
  };

  const handleUploadNewFiles = async () => {
    if (selectedNewFiles.length === 0) {
      setUploadError('Please select a video file');
      return;
    }

    setIsUploadingFiles(true);
    setUploadError('');
    setUploadProgress(null);

    try {
      // Check if chunking will be needed
      const needsChunking = ChunkedUploader.needsChunking(selectedNewFiles);
      
      // Use the new ChunkedUploader
      const result = await ChunkedUploader.upload({
        endpoint: `${config.apiBaseUrl}/upload`,
        files: selectedNewFiles,
        description: description,
        onProgress: (progress) => {
          setUploadProgress({
            ...progress,
            isChunked: needsChunking
          });
          console.log(`Upload progress: ${progress.percentComplete}% (${progress.processedFiles}/${progress.totalFiles} files, chunk ${progress.currentChunk}/${progress.totalChunks})`);
        },
        onChunkComplete: (chunkIndex, totalChunks) => {
          console.log(`Completed chunk ${chunkIndex + 1} of ${totalChunks}`);
        }
      });

      if (!result.success) {
        throw new Error(result.error || 'Upload failed');
      }

      console.log('Upload response:', result);
      
      // Validate the response format
      if (!result.documents || !Array.isArray(result.documents)) {
        console.error('Invalid response format. Expected documents array, got:', result);
        throw new Error(`Invalid response format from server. Expected documents array, got: ${JSON.stringify(result, null, 2)}`);
      }
      
      // Validate server returned documents array and take the first one as the single video
      const returnedDocs = result.documents || [];
      if (!Array.isArray(returnedDocs) || returnedDocs.length === 0) {
        throw new Error('Server did not return any processed video');
      }

      const returnedDoc = returnedDocs[0] as DocumentData;
      // Assign a safe id (1) or preserve existing id if present
      const newDoc: DocumentData = {
        ...returnedDoc,
        id: 1,
      };

      // Replace existing documents with the single uploaded video
      onUpdateDocuments([newDoc]);
      // Add transcription/processing cost to the visible total
      if (newDoc.processing_cost && typeof newDoc.processing_cost === 'number') {
        setTotalSessionCost(prev => prev + newDoc.processing_cost!);
      }

      setSelectedNewFiles([]);
      setShowUploadSection(false);
      
      // Show success message with chunking info
      const uploadMessage = result.chunked 
        ? `Successfully uploaded video using ${ChunkedUploader.estimateChunks(selectedNewFiles)} chunks`
        : `Successfully uploaded video`;

      console.log(uploadMessage);
      
    } catch (error) {
      console.error('Upload error:', error);
      setUploadError(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      setIsUploadingFiles(false);
      setUploadProgress(null);
    }
  };

  const handleCancelUpload = () => {
    setSelectedNewFiles([]);
    setShowUploadSection(false);
    setUploadError('');
  };

  const handleDeleteDocument = (documentId: number) => {
    // Confirm deletion
    if (confirm('Are you sure you want to delete this video?')) {
      const docToDelete = documents.find(d => d.id === documentId);
      // Ask backend to remove stored files for this document
      if (docToDelete) {
        (async () => {
            try {
            // Use the top-level reset to ensure all uploads and client state are cleared
            await fetch(`${config.apiBaseUrl}/delete_all_uploads`, { method: 'POST' });
            // Also call onReset so parent clears localStorage and UI state
            onReset();
          } catch (err) {
            console.warn('Failed to notify server to delete document files', err);
          }
        })();
      }
      // UI state will be cleared by onReset above
    }
  };

  // Function to process page and frame references in content before markdown
  const processPageReferences = (content: string): string => {
    let result = content;

    // Parse page markers: $PAGE_STARTfilename:pages$PAGE_END
    if (result.includes('$PAGE_START')) {
      const pageRefRegex = /\$PAGE_START([^:]+):([^\$]+)\$PAGE_END/g;
      result = result.replace(pageRefRegex, (match, filename, pageSpec) => {
        const cleanFilename = filename.trim();
        const cleanPageSpec = pageSpec.trim();
        return `[(Segment ${cleanPageSpec})](javascript:void(0) "data-page-ref=${cleanFilename}:${cleanPageSpec}")`;
      });
    }

    // Parse frame markers: $FRAME_STARTfilename:frames$FRAME_END
    if (result.includes('$FRAME_START')) {
      const frameRefRegex = /\$FRAME_START([^:]+):([^\$]+)\$FRAME_END/g;
      result = result.replace(frameRefRegex, (match, filename, frameSpec) => {
        const cleanFilename = filename.trim();
        const cleanFrameSpec = frameSpec.trim();
        return `[(Frame ${cleanFrameSpec})](javascript:void(0) "data-frame-ref=${cleanFilename}:${cleanFrameSpec}")`;
      });
    }

    return result;
  };

  // Helper function to parse page specifications like "5", "2,7,12", "15-18"
  const parsePageSpecification = (pageSpec: string): number[] => {
    const pages: number[] = [];
    
    // Split by commas for multiple pages/ranges
    const parts = pageSpec.split(',');
    
    parts.forEach(part => {
      part = part.trim();
      
      if (part.includes('-')) {
        // Handle range like "15-18"
        const [start, end] = part.split('-').map(p => parseInt(p.trim()));
        if (start && end && start <= end) {
          for (let i = start; i <= end; i++) {
            pages.push(i);
          }
        }
      } else {
        // Handle single page
        const pageNum = parseInt(part);
        if (pageNum) {
          pages.push(pageNum);
        }
      }
    });
    
    return pages;
  };

  // Handle clicking on an inline frame reference inserted from $FRAME_START markers
  const handleFrameReference = (filename: string, frameSpec: string) => {
    // Parse frame specification (supports single, comma-separated, and ranges)
    const frames = parsePageSpecification(frameSpec);
    if (frames.length === 0) {
      console.error('No frame numbers parsed:', frameSpec);
      return;
    }
    const frameNum = frames[0];

    // Find the document and frame metadata
    const document = documents.find(doc => doc.filename === filename);
    if (!document) {
      console.error('Document not found for frame ref:', filename);
      return;
    }

    // document.frames may be present
    const framesMeta: any = (document as any).frames || [];
    let matched: any = null;
    if (Array.isArray(framesMeta)) {
      matched = framesMeta.find((f: any) => Number(f.frame_number) === Number(frameNum));
    } else if (framesMeta && typeof framesMeta === 'object') {
      const key = String(frameNum);
      matched = framesMeta[key] || null;
    }

    if (matched) {
      const storedFilename = matched.stored_filename || matched.filename || (document as any).stored_filename;
      const timeSec = matched.time != null ? Number(matched.time) : Number(frameNum);
      // Prefer full caption if available
      const captionText = matched.caption_full || matched.caption || matched.captionText || matched.description || null;
      setSelectedPageContent({
        content: captionText ? String(captionText) : `Frame ${frameNum} from ${filename}`,
        pageNumber: frameNum,
        filename: filename,
        // @ts-ignore
        storedFilename,
        // @ts-ignore
        startTime: timeSec,
      } as any);
      return;
    }

    // Fallback: open modal with stored filename if present and use frame number as seconds
    const stored = (document as any).stored_filename || (document as any).storedFilename;
    if (stored) {
      setSelectedPageContent({
        content: `Frame ${frameNum} from ${filename}`,
        pageNumber: frameNum,
        filename: filename,
        // @ts-ignore
        storedFilename: stored,
        // @ts-ignore
        startTime: frameNum,
      } as any);
      return;
    }

    console.error('No frame metadata or stored filename available for', filename);
  };

  const handlePageReference = (filename: string, pageNumber: number) => {
    // Find the document and page content
    const document = documents.find(doc => doc.filename === filename);
    if (!document) {
      console.error('Document not found:', filename);
      return;
    }
    const page = document.pages.find(p => p.page_number === pageNumber);
    if (!page) {
      console.error('Page not found:', pageNumber, 'in', filename);
      return;
    }

    // If this document is a video (has stored_filename), open video player modal.
    // Default startTime to 0 when page.start_time is missing so the video still shows.
    const storedFilename = (document as any).stored_filename || (document as any).storedFilename;
    if (storedFilename) {
      const startTimeVal = page.start_time !== undefined && page.start_time !== null ? page.start_time : 0;
      setSelectedPageContent({
        content: page.text,
        pageNumber: pageNumber,
        filename: filename,
        // include extra metadata
        // @ts-ignore
        storedFilename,
        // @ts-ignore
        startTime: startTimeVal,
      } as any);
      return;
    }

    // Otherwise show text content modal
    setSelectedPageContent({
      content: page.text,
      pageNumber: pageNumber,
      filename: filename,
    });
  };

  return (
    <div className="bg-white rounded-lg shadow-lg flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-gray-200 p-4">
        <div className="flex justify-between items-center mb-3">
          <div className="flex-1">
                <h2 className="text-xl font-semibold text-gray-800">Chat with your video</h2>
            <div className="flex items-center space-x-2 mt-1">
              <p className="text-sm text-gray-600">
                {documents.length} video {documents.length > 0 ? '•' : ''}
              </p>
              {isEditingDescription ? (
                <div className="flex items-center space-x-2 flex-1">
                  <input
                    type="text"
                    value={editedDescription}
                    onChange={(e) => setEditedDescription(e.target.value)}
                    className="text-sm text-gray-600 border border-gray-300 rounded px-2 py-1 flex-1"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        handleSaveDescription();
                      } else if (e.key === 'Escape') {
                        handleCancelEdit();
                      }
                    }}
                    autoFocus
                  />
                  <button
                    onClick={handleSaveDescription}
                    className="text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700"
                  >
                    Save
                  </button>
                  <button
                    onClick={handleCancelEdit}
                    className="text-xs bg-gray-500 text-white px-2 py-1 rounded hover:bg-gray-600"
                  >
                    Cancel
                  </button>
                </div>
                              ) : documents.length > 0 ? (
                  <div className="flex items-center space-x-2">
                    <p className="text-sm text-gray-600">{description}</p>
                    {/* <button
                      onClick={handleEditDescription}
                      className="text-xs text-blue-600 hover:text-blue-800"
                      title="Edit description"
                    >
                      Update video usage guide ✏️ 
                    </button> */}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 italic">Upload your video to get started</p>
                )}
            </div>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => documents.length === 0 && setShowUploadSection(!showUploadSection)}
              disabled={documents.length > 0}
              className={documents.length === 0 
                ? "bg-blue-600 text-white hover:bg-blue-700 px-4 py-2 rounded text-sm font-medium"
                : "bg-gray-200 text-gray-500 px-4 py-2 rounded text-sm cursor-not-allowed"
              }
            >
              {documents.length === 0 ? "Add Your Video" : "Video uploaded"}
            </button>
            <button
              onClick={() => { if (isUploadingFiles) return; onReset(); }}
              disabled={isUploadingFiles}
              title={isUploadingFiles ? "Can't start a new session while upload is in progress" : "Start a new session"}
              className={isUploadingFiles
                ? "text-gray-400 px-3 py-1 rounded border border-gray-200 cursor-not-allowed"
                : "text-gray-500 hover:text-gray-700 px-3 py-1 rounded border border-gray-300 hover:border-gray-400"
              }
            >
              Start new session
            </button>
          </div>
        </div>

        {/* Upload Section */}
        {showUploadSection && (
          <div className="bg-gray-50 rounded-lg p-4 mb-3">
                <h3 className="text-lg font-medium text-gray-800 mb-3">Add New Video</h3>
            
            <div className="mb-4">
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-gray-400 transition-colors cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".mp4,.webm"
                  multiple={false}
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <div className="text-gray-600">
                  <svg className="mx-auto h-8 w-8 text-gray-400 mb-2" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <p className="text-sm">Click to select video files (MP4/WEBM)</p>
                </div>
              </div>
            </div>

            {selectedNewFiles.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">
                  Selected File
                </h4>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {selectedNewFiles.slice(0,1).map((file, index) => (
                    <div key={index} className="flex items-center justify-between bg-white p-2 rounded border">
                      <span className="text-sm text-gray-700 truncate">{file.name}</span>
                      {/* <button
                        onClick={() => removeNewFile(index)}
                        className="text-red-500 hover:text-red-700 flex-shrink-0 ml-2"
                      >
                        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button> */}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {uploadError && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
                <p className="text-red-600 text-sm">{uploadError}</p>
              </div>
            )}

            <div className="flex space-x-2">
              <button
                onClick={handleUploadNewFiles}
                disabled={isUploadingFiles || selectedNewFiles.length === 0}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-sm"
              >
                {isUploadingFiles ? 'Uploading...' : `Upload`}
              </button>
              {/* <button
                onClick={handleCancelUpload}
                className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600 text-sm"
              >
                Cancel
              </button> */}
            </div>
          </div>
        )}
        
        {/* Document List */}
        {documents.length > 0 && (
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <button
                onClick={() => setShowDocuments(!showDocuments)}
                className="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-gray-900"
              >
                        <span>Uploaded video</span>
                <svg
                  className={`w-4 h-4 transition-transform ${showDocuments ? 'rotate-90' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
              <div className="flex items-center space-x-3">
                <span className="text-xs text-gray-500">{documents.length > 0 ? '1 video' : '0 videos'}</span>
                {/* <span className="text-xs font-medium text-green-600">
                  Cost: ${totalSessionCost.toFixed(4)}
                </span> */}
              </div>
            </div>
            {showDocuments && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                {documents.map((doc, index) => (
                <div
                  key={index}
                  className="flex items-center space-x-2 bg-white p-2 rounded border border-gray-200"
                >
                  <div className="flex-shrink-0">
                    <svg className="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-gray-700 truncate" title={doc.filename}>
                      {doc.filename}
                    </p>
                    {/* <p className="text-xs text-gray-500">
                      {doc.total_pages} pages
                    </p> */}
                  </div>
                  {/* <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteDocument(doc.id);
                    }}
                    className="flex-shrink-0 ml-2 text-red-500 hover:text-red-700 p-1"
                    title="Delete video"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button> */}
                </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Messages */}
    <div className="flex-[3] min-h-0 overflow-y-auto p-4 space-y-4">
        {documents.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No video uploaded</h3>
              <p className="text-gray-500 mb-4">
                Upload video files to start chatting about their content. Click "Add Files" above to get started.
              </p>
              <button
                onClick={() => setShowUploadSection(true)}
                className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                </svg>
                Add Your Video
              </button>
            </div>
          </div>
            ) : messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-8">
            <p>Ask me anything about your uploaded video!</p>
            <p className="text-sm mt-2">Examples:</p>
            <ul className="text-sm mt-1 space-y-1">
              <li>• "What are the main topics covered?"</li>
              <li>• "Summarize the key findings"</li>
              <li>• "What does it say about [specific topic]?"</li>
            </ul>
          </div>
        ) : null}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-lg ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              {/* Progress indicator for streaming messages */}
              {message.isStreaming && message.progress && (
                <div className="mb-3 p-3 bg-blue-50 rounded-lg border-l-4 border-blue-400">
                  <div className="flex items-center space-x-2 mb-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <span className="text-sm font-medium text-blue-800">
                      {message.progress.status}
                    </span>
                  </div>
                  
                  <div className="w-full bg-blue-200 rounded-full h-2 mb-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500 ease-out"
                      style={{ width: `${(message.progress.step / message.progress.total) * 100}%` }}
                    ></div>
                  </div>
                  
                  <div className="flex justify-between text-xs text-blue-600">
                    <span>Step {message.progress.step} of {message.progress.total}</span>
                    {message.progress.stepTime && message.progress.stepCost && (
                      <span>{message.progress.stepTime.toFixed(2)}s • ${message.progress.stepCost.toFixed(4)}</span>
                    )}
                  </div>
                </div>
              )}

              <div className="prose prose-sm max-w-none">
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                    a: ({ href, title, children }) => {
                      // Handle page reference links
                      if (title && title.startsWith('data-page-ref=')) {
                        const pageRef = title.replace('data-page-ref=', '');
                        const [filename, pageSpec] = pageRef.split(':');
                        const pages = parsePageSpecification(pageSpec);

                        if (pages.length > 0) {
                          const pageNum = pages[0]; // Use first page for click
                          return (
                            <button
                              onClick={() => handlePageReference(filename, pageNum)}
                              className="text-blue-600 hover:text-blue-800 underline font-medium cursor-pointer mx-0.5"
                              title={`View ${filename} - Page ${pageNum} content`}
                            >
                              {children}
                            </button>
                          );
                        }
                      }

                      // Handle frame reference links (vision citations)
                      if (title && title.startsWith('data-frame-ref=')) {
                        const frameRef = title.replace('data-frame-ref=', '');
                        const [filename, frameSpec] = frameRef.split(':');
                        return (
                          <button
                            onClick={() => handleFrameReference(filename, frameSpec)}
                            className="text-blue-600 hover:text-blue-800 underline font-medium cursor-pointer mx-0.5"
                            title={`View ${filename} - Frame ${frameSpec}`}
                          >
                            {children}
                          </button>
                        );
                      }

                      // Regular links
                      return (
                        <a href={href} title={title} className="text-blue-600 hover:text-blue-800 underline">
                          {children}
                        </a>
                      );
                    },
                    h1: ({ children }) => <h1 className="text-lg font-bold mb-2">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-base font-bold mb-2">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-sm font-bold mb-1">{children}</h3>,
                    ul: ({ children }) => <ul className="list-disc ml-4 mb-2">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal ml-4 mb-2">{children}</ol>,
                    li: ({ children }) => <li className="mb-1">{children}</li>,
                    code: ({ children, ...props }) => {
                      const isInline = !props.className?.includes('language-');
                      const bgColor = message.role === 'user' ? 'bg-blue-700 bg-opacity-50' : 'bg-gray-200';
                      return isInline ? (
                        <code className={`${bgColor} px-1 rounded text-xs font-mono`}>{children}</code>
                      ) : (
                        <code className={`block ${bgColor} p-2 rounded text-xs font-mono whitespace-pre-wrap`}>{children}</code>
                      );
                    },
                    strong: ({ children }) => <strong className="font-bold">{children}</strong>,
                    em: ({ children }) => <em className="italic">{children}</em>,
                  }}
                >
                  {processPageReferences(message.content)}
                </ReactMarkdown>
              </div>
              {message.metadata && (
                <div className="mt-2 text-xs opacity-60 space-y-1">
                  <div className="flex flex-wrap gap-x-4 gap-y-1">
                    {message.metadata.selectedDocuments && (
                      <span><span className="font-medium">Video:</span> {message.metadata.selectedDocuments.map(doc => `${doc.filename}`).join(', ')}</span>
                    )}
                    {/* {message.metadata.relevantPagesCount && (
                      <span><span className="font-medium">Pages:</span> {message.metadata.relevantPagesCount}</span>
                    )} */}
                    {message.metadata.model && (
                      <span><span className="font-medium">Model:</span> {message.metadata.model}</span>
                    )}
                    {/* {message.metadata.responseTime && (
                      <span><span className="font-medium">Response Time:</span> {message.metadata.responseTime.toFixed(2)}s</span>
                    )} */}
                    {/* {message.metadata.costs?.total_cost && (
                      <span><span className="font-medium">Cost:</span> ${message.metadata.costs.total_cost.toFixed(4)}</span>
                    )} */}
                  </div>
                </div>
              )}
              {/* Image citations rendered separately so users can view thumbnails and jump to timestamps */}
              {message.metadata?.imageCitations && message.metadata.imageCitations.length > 0 && (
                <div className="mt-2 space-y-2">
                  <div className="text-xs font-medium text-gray-600">Referenced frames:</div>
                  <div className="flex flex-wrap gap-2">
                    {message.metadata.imageCitations.map((cit: any, idx: number) => (
                      <div key={idx} className="w-40 bg-white border rounded p-2">
                        <div className="text-xs font-medium text-gray-700 mb-1">{cit.filename}</div>
                        <div className="grid grid-cols-2 gap-1 mb-2">
                          {(cit.stored_filenames || cit.stored_filenames === undefined) && (cit.stored_filenames || cit.frame_numbers || []).slice(0,4).map((sf: any, i: number) => {
                            // stored_filenames may be an array of filenames or undefined; if undefined try cit.stored_filenames
                            const thumb = Array.isArray(cit.stored_filenames) ? cit.stored_filenames[i] : sf;
                            const url = thumb ? `${config.apiBaseUrl}/uploads/${thumb}` : undefined;
                            const timeSec = Array.isArray(cit.times) ? cit.times[i] : undefined;
                            // Try to surface the dense caption (caption_full) from the uploaded document frames metadata
                            let captionText: string | undefined = undefined;
                            try {
                              const doc = documents.find(d => d.filename === cit.filename) as any | undefined;
                              const frameNum = Array.isArray(cit.frame_numbers) ? cit.frame_numbers[i] : undefined;
                              if (doc) {
                                const framesMeta: any = doc.frames || [];
                                let fm: any = null;
                                if (Array.isArray(framesMeta)) {
                                  fm = framesMeta.find((f: any) => Number(f.frame_number) === Number(frameNum));
                                } else if (framesMeta && typeof framesMeta === 'object') {
                                  fm = framesMeta[String(frameNum)] || null;
                                }
                                if (fm) {
                                  captionText = fm.caption_full || fm.caption || fm.captionText || fm.description || undefined;
                                }
                              }
                            } catch (err) {
                              console.warn('Failed to look up frame caption', err);
                            }

                            return (
                              <div key={i} className="w-full">
                                {url ? (
                                  <img src={url} alt={`frame-${i}`} className="w-full h-16 object-cover rounded" />
                                ) : (
                                  <div className="w-full h-16 bg-gray-100 rounded flex items-center justify-center text-xs text-gray-500">No image</div>
                                )}

                                {/* Caption (full if available) */}
                                {captionText ? (
                                  <div className="text-xs text-gray-600 mt-1 whitespace-normal">{captionText}</div>
                                ) : null}

                                <div className="flex items-center justify-between mt-1 text-xs">
                                  <button
                                    onClick={() => {
                                      // Try to find a player element; if found, seek
                                      const player = document.getElementById('video-player') as HTMLVideoElement | null;
                                      if (player && typeof timeSec === 'number') {
                                        player.currentTime = timeSec;
                                        player.play();
                                      } else if (typeof timeSec === 'number') {
                                        // If no player in modal, try to open page modal for the filename
                                        handlePageReference(cit.filename, cit.frame_numbers ? cit.frame_numbers[0] : 0);
                                      }
                                    }}
                                    className="text-blue-600 hover:underline"
                                  >
                                    {Array.isArray(cit.human_readable) && cit.human_readable[i] ? cit.human_readable[i] : (typeof timeSec === 'number' ? `${Math.floor(timeSec)}s` : 'View')}
                                  </button>
                                  <a href={`${config.apiBaseUrl}/uploads/${Array.isArray(cit.stored_filenames) ? cit.stored_filenames[i] : cit.stored_filenames}`} target="_blank" rel="noreferrer" className="text-gray-400">⤢</a>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              <p className={`text-xs mt-1 ${
                message.role === 'user' ? 'text-blue-200' : 'text-gray-500'
              }`}>
                {message.timestamp.toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 p-4">
        {/* Model Selection */}
        <div className="mb-3">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            AI Model:
          </label>
          <div className="flex space-x-4">
                        <label className="flex items-center space-x-2">
              <input
                type="radio"
                name="model"
                value="gpt-5-nano"
                checked={selectedModel === 'gpt-5-nano'}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={isLoading}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">GPT-5 Nano (Fastest, Lowest Cost)</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                name="model"
                value="gpt-5-mini"
                checked={selectedModel === 'gpt-5-mini'}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={isLoading}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">GPT-5 Mini (Faster, Lower Cost)</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                name="model"
                value="gpt-5"
                checked={selectedModel === 'gpt-5'}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={isLoading}
                className="text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm text-gray-700">GPT-5 (Higher Quality, Higher Cost)</span>
            </label>
          </div>
        </div>
        
        <div className="flex space-x-2">
          <textarea
            value={currentQuestion}
            onChange={(e) => setCurrentQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={documents.length === 0 ? "Upload video first to start chatting..." : "Ask a question about your video..."}
            className="flex-1 px-3 py-2 text-gray-700 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={2}
            disabled={isLoading || documents.length === 0}
          />
          <button
            onClick={handleSendMessage}
            disabled={!currentQuestion.trim() || isLoading || documents.length === 0}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>

      {/* Page Content Modal */}
      {selectedPageContent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-4xl max-h-[80vh] w-full flex flex-col">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <div>
                <h3 className="text-lg font-semibold text-gray-800">
                  {selectedPageContent.filename} - Page {selectedPageContent.pageNumber}
                </h3>
                <p className="text-sm text-gray-600">Extracted Text Content</p>
              </div>
              <button
                onClick={() => setSelectedPageContent(null)}
                className="text-gray-500 hover:text-gray-700 focus:outline-none"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            {/* Modal Content */}
            <div className="flex-1 overflow-auto p-6">
              <div className="prose prose-sm max-w-none">
                {(
                  // If storedFilename and startTime are present (allow startTime === 0), render a video player
                  // @ts-ignore
                  selectedPageContent && (selectedPageContent as any).storedFilename && ((selectedPageContent as any).startTime !== undefined && (selectedPageContent as any).startTime !== null) ? (
                    <div>
                      {/* Video player */}
                      <video
                        id="video-player"
                        controls
                        className="w-full max-h-[60vh] mb-4"
                        src={`${config.apiBaseUrl}/uploads/${(selectedPageContent as any).storedFilename}`}
                      />
                      <div className="mb-4">
                        <button
                          onClick={() => {
                            const el = document.getElementById('video-player') as HTMLVideoElement | null;
                            if (el) {
                              el.currentTime = (selectedPageContent as any).startTime || 0;
                              el.play();
                            }
                          }}
                          className="px-3 py-2 bg-blue-600 text-white rounded"
                        >
                          Play at {Math.floor((selectedPageContent as any).startTime)}s
                        </button>
                      </div>
                      <div className="whitespace-pre-wrap text-gray-800 leading-relaxed">{selectedPageContent.content}</div>
                    </div>
                  ) : (
                    <div className="whitespace-pre-wrap text-gray-800 leading-relaxed">{selectedPageContent.content}</div>
                  )
                )}
              </div>
            </div>
            
            {/* Modal Footer */}
            <div className="flex justify-end p-4 border-t border-gray-200">
              <button
                onClick={() => setSelectedPageContent(null)}
                className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}