#!/usr/bin/env node
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

function loadKeyFromEnvLocal() {
  const envLocal = path.join(process.cwd(), '.env.local');
  if (!fs.existsSync(envLocal)) return null;
  try {
    const content = fs.readFileSync(envLocal, 'utf8');
    for (const line of content.split(/\r?\n/)) {
      const m = line.match(/^\s*OPENAI_API_KEY\s*=\s*(.+)\s*$/);
      if (m) {
        const val = m[1].trim().replace(/^\"|\"$/g, '');
        if (val.length > 0) return val;
      }
    }
  } catch (err) {
    return null;
  }
  return null;
}

// Determine key: prefer environment, otherwise .env.local
let key = process.env.OPENAI_API_KEY;
if (!key) key = loadKeyFromEnvLocal();

if (!key) {
  console.error('\nERROR: Missing OpenAI API key. Set OPENAI_API_KEY or add .env.local with OPENAI_API_KEY=...');
  process.exit(1);
}

// Set for this process so spawned children inherit it
process.env.OPENAI_API_KEY = key;

// Spawn concurrently so children inherit process.env
// Locate local concurrently binary in node_modules/.bin to avoid relying on npx
const concurrentlyBin = path.join(process.cwd(), 'node_modules', '.bin', process.platform === 'win32' ? 'concurrently.cmd' : 'concurrently');
const args = ['npm run backend', 'next dev --turbopack'];

// Build a single shell command to invoke concurrently with two quoted commands
const quotedArgs = args.map(a => `"${a.replace(/"/g, '\\"')}"`).join(' ');
const cmdString = `"${concurrentlyBin}" ${quotedArgs}`;
const p = spawn(cmdString, { shell: true, stdio: 'inherit', env: process.env });

p.on('close', (code) => process.exit(code === null ? 0 : code));
p.on('error', (err) => {
  console.error('Failed to start dev processes:', err);
  process.exit(1);
});
