/**
 * Reads <project>/.env (see .env.example). Values in process.env win over the file.
 * A single-quoted value is taken verbatim to its last quote, because the base64 Gemini private key
 * contains characters an unquoted parser would cut.
 */
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const ENV_FILE = join(ROOT, '.env');

function parse(path) {
  if (!existsSync(path)) return {};
  const out = {};
  for (const raw of readFileSync(path, 'utf8').split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#') || !line.includes('=')) continue;
    const i = line.indexOf('=');
    let v = line.slice(i + 1).trim();
    if (v.startsWith("'")) v = v.slice(1, v.lastIndexOf("'"));
    else v = v.split(' #')[0].trim().replace(/^"|"$/g, '');
    out[line.slice(0, i)] = v;
  }
  return out;
}

const FILE_VARS = parse(ENV_FILE);

export function envVal(name, { required = true } = {}) {
  const v = process.env[name] || FILE_VARS[name];
  if (!v && required) throw new Error(`missing env ${name} (looked in process.env and ${ENV_FILE})`);
  return v;
}

/** Vertex-backed GoogleGenAI client options (service account, base64-encoded private key). */
export function geminiClientOptions() {
  return {
    vertexai: true,
    project: envVal('GEMINI_GC_PROJECT_ID'),
    location: envVal('GEMINI_GC_LOCATION'),
    googleAuthOptions: {
      credentials: {
        client_email: envVal('GEMINI_GC_CLIENT_EMAIL'),
        private_key: Buffer.from(envVal('GEMINI_GC_PRIVATE_KEY'), 'base64').toString('utf-8').replace(/\\n/g, '\n'),
      },
    },
  };
}
