#!/usr/bin/env node
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const repoRoot = path.resolve(__dirname, '..');
const backendDir = path.join(repoRoot, 'backend');

function run(cmd, args, opts) {
  return new Promise((resolve, reject) => {
    let p;
    try {
      p = spawn(cmd, args, Object.assign({ cwd: backendDir, stdio: 'inherit', shell: false }, opts || {}));
    } catch (err) {
      return reject(err);
    }

    p.on('close', (code) => {
      resolve(code === null ? 0 : code);
    });
    p.on('error', (err) => {
      reject(err);
    });
  });
}

async function main() {
  try {
    if (!fs.existsSync(backendDir)) {
      console.error('backend directory not found at', backendDir);
      process.exit(1);
    }

    // Ensure OPENAI_API_KEY is available either in environment or repo .env.local
    function findKeyInEnvFile(filePath) {
      try {
        if (!fs.existsSync(filePath)) return null;
        const content = fs.readFileSync(filePath, 'utf8');
        const m = content.match(/^\s*OPENAI_API_KEY\s*=\s*(.+)\s*$/m);
        if (m) return m[1].trim().replace(/^"|"$/g, '');
      } catch (err) {
        return null;
      }
      return null;
    }

    if (!process.env.OPENAI_API_KEY) {
      // Check repo-level .env.local and backend/.env
      const repoEnvLocal = path.join(repoRoot, '.env.local');
      const backendEnv = path.join(backendDir, '.env');
      const found = findKeyInEnvFile(repoEnvLocal) || findKeyInEnvFile(backendEnv);
      if (found) {
        process.env.OPENAI_API_KEY = found;
      } else {
        console.error('\nERROR: Missing OpenAI API key.');
        console.error('Set OPENAI_API_KEY in your shell, add it to .env.local in the repository root, or add it to backend/.env.');
        process.exit(1);
      }
    }

    const shPath = path.join(backendDir, 'start.sh');
    const ps1Path = path.join(backendDir, 'start.ps1');

    if (process.platform === 'win32') {
      // Prefer PowerShell script if present
      if (fs.existsSync(ps1Path)) {
        const code = await run('powershell', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ps1Path]);
        if (code === 0) return;
        console.warn('powershell script exited with code', code);
      }

      // Try python uvicorn fallback before attempting bash/WSL to avoid WSL message
      try {
        const code = await run('python', ['-m', 'uvicorn', 'main:app', '--reload', '--host', '0.0.0.0', '--port', '8000']);
        if (code === 0) return;
        console.warn('python uvicorn exited with code', code);
      } catch (err) {
        console.warn('Failed to start uvicorn via python:', err && err.message ? err.message : err);
      }

      // If start.sh exists, try running via bash (WSL/git-bash) as a last resort
      if (fs.existsSync(shPath)) {
        try {
          const code = await run('bash', [shPath]);
          if (code === 0) return;
          console.warn('bash script exited with code', code);
        } catch (err) {
          console.warn('Failed to run bash:', err && err.message ? err.message : err);
        }
      }

      process.exit(1);
    }

    // POSIX: prefer start.sh if present
    if (fs.existsSync(shPath)) {
      try {
        const code = await run('sh', [shPath]);
        if (code === 0) return;
        console.warn('sh script exited with code', code);
      } catch (err) {
        console.warn('Failed to run sh:', err && err.message ? err.message : err);
      }
    }

    // Fallback to python uvicorn
    try {
      const code = await run('python3', ['-m', 'uvicorn', 'main:app', '--reload', '--host', '0.0.0.0', '--port', '8000']);
      if (code === 0) return;
      console.warn('python3 uvicorn exited with code', code);
    } catch (err) {
      console.error('Failed to start uvicorn via python3:', err);
    }

    process.exit(1);
  } catch (err) {
    console.error('Error launching backend:', err && err.message ? err.message : err);
    process.exit(1);
  }
}

main();
