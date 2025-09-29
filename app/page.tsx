'use client';

import { useState, useEffect } from 'react';
import StatelessChatSection from './components/StatelessChatSection';
import config from '../config';

interface DocumentData {
  id: number;
  filename: string;
  pages: { page_number: number; text: string }[];
  total_pages: number;
}

// GitHub Icon Component
function GitHubIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="currentColor"
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
    </svg>
  );
}

export default function Home() {
  const [documents, setDocuments] = useState<DocumentData[]>([]);
  const [description, setDescription] = useState('');

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const savedDocuments = localStorage.getItem('chatbot_documents');
      const savedDescription = localStorage.getItem('chatbot_description');
      
      if (savedDocuments) {
        const parsedDocuments = JSON.parse(savedDocuments);
        if (Array.isArray(parsedDocuments)) {
          setDocuments(parsedDocuments);
        } else {
          console.warn('Invalid documents format in localStorage, clearing...');
          localStorage.removeItem('chatbot_documents');
          setDocuments([]);
        }
      }
      if (savedDescription) {
        setDescription(savedDescription);
      }
    } catch (error) {
      console.error('Error loading from localStorage:', error);
    }
  }, []);

  // Save to localStorage whenever documents or description changes
  useEffect(() => {
    try {
      if (documents.length > 0) {
        localStorage.setItem('chatbot_documents', JSON.stringify(documents));
      }
      if (description) {
        localStorage.setItem('chatbot_description', description);
      }
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  }, [documents, description]);

  // Ensure uploaded files are cleaned up when the user leaves the page or closes the tab.
  // We use navigator.sendBeacon when available (fire-and-forget) and fall back to fetch with keepalive.
  useEffect(() => {
    const sendDeleteAll = () => {
      try {
        const url = `${config.apiBaseUrl}/delete_all_uploads`;
        // Prefer sendBeacon where available (no body required for this endpoint)
        if (typeof navigator !== 'undefined' && typeof navigator.sendBeacon === 'function') {
          try {
            // send an empty JSON payload
            const blob = new Blob([JSON.stringify({})], { type: 'application/json' });
            navigator.sendBeacon(url, blob);
            return;
          } catch (e) {
            // fall through to fetch
          }
        }

        // Fallback: small POST with keepalive flag
        try {
          fetch(url, { method: 'POST', keepalive: true }).catch(() => {});
        } catch (e) {
          // ignore
        }
      } catch (err) {
        // swallow errors during unload
      }
    };

    window.addEventListener('pagehide', sendDeleteAll, { passive: true });
    window.addEventListener('beforeunload', sendDeleteAll);

    return () => {
      window.removeEventListener('pagehide', sendDeleteAll as EventListener);
      window.removeEventListener('beforeunload', sendDeleteAll as EventListener);
    };
  }, []);

  const handleReset = () => {
    // Tell the backend to delete all uploads for this session and clear client memory
    (async () => {
      try {
  await fetch(`${config.apiBaseUrl}/delete_all_uploads`, { method: 'POST' });
      } catch (err) {
        console.warn('Failed to notify server to delete all uploads', err);
      }
    })();

    setDocuments([]);
    setDescription('');
    try {
      localStorage.removeItem('chatbot_documents');
      localStorage.removeItem('chatbot_description');
    } catch (error) {
      console.error('Error clearing localStorage:', error);
    }
  };

  const handleUpdateDocuments = (newDocuments: DocumentData[]) => {
    console.log('handleUpdateDocuments called with:', newDocuments);
    
    // Validate input
    if (!Array.isArray(newDocuments)) {
      console.error('newDocuments is not an array:', newDocuments);
      return;
    }
    
    setDocuments(newDocuments);
  };

  const handleUpdateDescription = (newDescription: string) => {
    setDescription(newDescription);
  };

    return (
      <main className="min-h-screen bg-gray-50">
        <div className="container mx-auto px-4 py-8">
          <div className="text-center mb-8 relative">
            
            <h1 className="text-4xl font-bold text-gray-900 mb-2">
              Video Q&A & Search
            </h1>
            <p className="text-gray-600">
              Upload a short video (≤3 minutes), ask questions, and jump to matching timestamps.
            </p>
          </div>

          <div className="max-w-6xl mx-auto h-[120vh]">
            <StatelessChatSection
              documents={documents}
              description={description}
              onReset={handleReset}
              onUpdateDocuments={handleUpdateDocuments}
              onUpdateDescription={handleUpdateDescription}
            />
          </div>
        </div>
      </main>
    );
}
