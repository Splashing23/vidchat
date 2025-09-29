// API Configuration
const getApiBaseUrl = () => {
  // In production on Vercel, use the same domain with /api
  if (typeof window !== 'undefined' && window.location.hostname.includes('vercel.app')) {
    return `${window.location.origin}/api`;
  }
  
  // Otherwise use environment variable or default to localhost
  return process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';
};

const apiBaseUrl = getApiBaseUrl();

// Debug logging in development
if (typeof window !== 'undefined' && process.env.NODE_ENV === 'development') {
  console.log('API Base URL:', apiBaseUrl);
}

// Validation: warn if the API URL looks incorrect
if (typeof window !== 'undefined') {
  const currentDomain = window.location.origin;
  
  // If API URL is the same as frontend URL (without /api), warn user
  if (apiBaseUrl === currentDomain) {
    console.warn(
      '⚠️  Warning: NEXT_PUBLIC_API_BASE_URL is set to the same URL as your frontend.',
      '\n   Current value:', apiBaseUrl,
      '\n   If your backend is deployed separately, use that URL.',
      '\n   If using Vercel API routes, add /api to the end.',
      '\n   Examples:',
      '\n   - Separate backend: https://your-backend-project.vercel.app',
      '\n   - API routes: https://vectorless-chatbot.vercel.app/api'
    );
  }
  
  // If it looks like a frontend-only URL (no api subdomain or /api path)
  if (apiBaseUrl.includes('vercel.app') && 
      !apiBaseUrl.includes('/api') && 
      !apiBaseUrl.includes('api.') && 
      !apiBaseUrl.includes('-api') &&
      !apiBaseUrl.includes('-backend')) {
    console.warn(
      '⚠️  Warning: NEXT_PUBLIC_API_BASE_URL might be incorrect.',
      '\n   Current value:', apiBaseUrl,
      '\n   For Vercel deployments, consider:',
      '\n   - Separate backend project: https://your-backend-project.vercel.app',
      '\n   - API routes in same project: https://your-project.vercel.app/api'
    );
  }
}

const config = {
  apiBaseUrl,
};

export default config; 