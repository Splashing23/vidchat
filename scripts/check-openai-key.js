#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

function fail(msg) {
  console.error('\nERROR: Missing OpenAI API key.');
  console.error(msg);
  console.error('\nSet the OPENAI_API_KEY environment variable in .env.local or your shell.');
  process.exit(1);
}

// Prefer explicit env var
if (process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY.trim().length > 0) {
  process.exit(0);
}

// Also allow a .env.local file at repo root with OPENAI_API_KEY=... (common dev pattern)
const envLocal = path.join(process.cwd(), '.env.local');
if (fs.existsSync(envLocal)) {
  try {
    const content = fs.readFileSync(envLocal, 'utf8');
    for (const line of content.split(/\r?\n/)) {
      const m = line.match(/^\s*OPENAI_API_KEY\s*=\s*(.+)\s*$/);
      if (m) {
        const val = m[1].trim().replace(/^\"|\"$/g, '');
        if (val.length > 0) {
          // set it for this process so subsequent npm-run commands inherit
          process.env.OPENAI_API_KEY = val;
          process.exit(0);
        }
      }
    }
  } catch (err) {
    // fallthrough to error
  }
}

fail('No OPENAI_API_KEY found in environment or .env.local');
