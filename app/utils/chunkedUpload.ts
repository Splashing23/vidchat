/**
 * Chunked Upload Utility for handling large file uploads to Vercel
 * Automatically splits large uploads into manageable chunks
 */

interface ChunkedUploadOptions {
  endpoint: string;
  files: FileList | File[];
  description?: string;
  chunkSizeMB?: number;
  onProgress?: (progress: ChunkedUploadProgress) => void;
  onChunkComplete?: (chunkIndex: number, totalChunks: number) => void;
}

interface ChunkedUploadProgress {
  totalFiles: number;
  processedFiles: number;
  currentChunk: number;
  totalChunks: number;
  percentComplete: number;
}

interface ChunkedUploadResult {
  success: boolean;
  documents?: any[];
  error?: string;
  chunked?: boolean;
  totalFiles?: number;
}

export class ChunkedUploader {
  private static readonly MAX_PAYLOAD_SIZE_MB = 4.5;
  private static readonly MAX_FILE_SIZE_MB = 4.5;
  private static readonly DEFAULT_CHUNK_SIZE_MB = 3.5;

  static async upload(options: ChunkedUploadOptions): Promise<ChunkedUploadResult> {
    const {
      endpoint,
      files,
      description = '',
      chunkSizeMB = this.DEFAULT_CHUNK_SIZE_MB,
      onProgress,
      onChunkComplete
    } = options;

    const fileArray = Array.from(files);
    // If the user is uploading exactly one file, always use the singleUpload flow.
    // This avoids chunked multipart semantics which the backend no longer supports.
    if (fileArray.length === 1) {
      return this.singleUpload(endpoint, fileArray, description, onProgress);
    }
    
    // Note: individual file size checks removed to allow larger video files; chunking will be used for large uploads

    // Calculate total size and determine if chunking is needed
    const totalSize = fileArray.reduce((sum, file) => sum + file.size, 0);
    const totalSizeMB = totalSize / (1024 * 1024);

    // If total size is within limits, do a single upload
    if (totalSizeMB <= this.MAX_PAYLOAD_SIZE_MB) {
      return this.singleUpload(endpoint, fileArray, description, onProgress);
    }

    // Otherwise, use chunked upload
    return this.chunkedUpload(endpoint, fileArray, description, chunkSizeMB, onProgress, onChunkComplete);
  }

  private static async singleUpload(
    endpoint: string,
    files: File[],
    description: string,
    onProgress?: (progress: ChunkedUploadProgress) => void
  ): Promise<ChunkedUploadResult> {
    const formData = new FormData();
    // If only one file is being uploaded, the backend expects the field name 'file'.
    if (files.length === 1) {
      formData.append('file', files[0]);
    } else {
      files.forEach(file => {
        formData.append('files', file);
      });
    }
    formData.append('description', description);

    if (onProgress) {
      onProgress({
        totalFiles: files.length,
        processedFiles: 0,
        currentChunk: 1,
        totalChunks: 1,
        percentComplete: 0
      });
    }

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (onProgress) {
        onProgress({
          totalFiles: files.length,
          processedFiles: files.length,
          currentChunk: 1,
          totalChunks: 1,
          percentComplete: 100
        });
      }

      if (!response.ok) {
        return {
          success: false,
          error: result.error || `HTTP ${response.status}: ${response.statusText}`,
          chunked: false
        };
      }

      return {
        success: true,
        documents: result.documents,
        chunked: false,
        totalFiles: files.length
      };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Upload failed',
        chunked: false
      };
    }
  }

  private static async chunkedUpload(
    endpoint: string,
    files: File[],
    description: string,
    chunkSizeMB: number,
    onProgress?: (progress: ChunkedUploadProgress) => void,
    onChunkComplete?: (chunkIndex: number, totalChunks: number) => void
  ): Promise<ChunkedUploadResult> {
    const chunkSizeBytes = chunkSizeMB * 1024 * 1024;
    const chunks = this.createChunks(files, chunkSizeBytes);
    const uploadId = this.generateUploadId();

    let allDocuments: any[] = [];
    let totalProcessedFiles = 0;

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const formData = new FormData();
      // For chunked uploads we append files as 'files' (multiple per request)
      chunk.forEach(file => {
        formData.append('files', file);
      });
      formData.append('description', description);

      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'X-Chunk-Index': i.toString(),
            'X-Total-Chunks': chunks.length.toString(),
            'X-Upload-ID': uploadId,
          },
          body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
          return {
            success: false,
            error: result.error || `Chunk ${i + 1} failed: HTTP ${response.status}`,
            chunked: true
          };
        }

        // Update progress
        totalProcessedFiles += chunk.length;
        if (onProgress) {
          onProgress({
            totalFiles: files.length,
            processedFiles: totalProcessedFiles,
            currentChunk: i + 1,
            totalChunks: chunks.length,
            percentComplete: Math.round((totalProcessedFiles / files.length) * 100)
          });
        }

        if (onChunkComplete) {
          onChunkComplete(i, chunks.length);
        }

        // Each chunk now returns documents directly
        if (result.documents && Array.isArray(result.documents)) {
          allDocuments.push(...result.documents);
        }

      } catch (error) {
        return {
          success: false,
          error: `Chunk ${i + 1} failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
          chunked: true
        };
      }
    }

    return {
      success: true,
      documents: allDocuments,
      chunked: true,
      totalFiles: files.length
    };
  }

  private static createChunks(files: File[], chunkSizeBytes: number): File[][] {
    const chunks: File[][] = [];
    let currentChunk: File[] = [];
    let currentChunkSize = 0;

    for (const file of files) {
      // If adding this file would exceed chunk size, start a new chunk
      if (currentChunkSize + file.size > chunkSizeBytes && currentChunk.length > 0) {
        chunks.push(currentChunk);
        currentChunk = [];
        currentChunkSize = 0;
      }

      currentChunk.push(file);
      currentChunkSize += file.size;
    }

    // Add the last chunk if it has files
    if (currentChunk.length > 0) {
      chunks.push(currentChunk);
    }

    return chunks;
  }

  private static generateUploadId(): string {
    return `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Utility method to check if files need chunking
  static needsChunking(files: FileList | File[]): boolean {
    const fileArray = Array.from(files);
    const totalSize = fileArray.reduce((sum, file) => sum + file.size, 0);
    return totalSize > this.MAX_PAYLOAD_SIZE_MB * 1024 * 1024;
  }

  // Utility method to estimate number of chunks needed
  static estimateChunks(files: FileList | File[], chunkSizeMB: number = this.DEFAULT_CHUNK_SIZE_MB): number {
    const fileArray = Array.from(files);
    const chunks = this.createChunks(fileArray, chunkSizeMB * 1024 * 1024);
    return chunks.length;
  }
}

// Export default instance for convenience
export default ChunkedUploader; 