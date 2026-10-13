// node tools/gemini_edit.mjs <src> <out.png> <aspect 1:1|9:16|3:4> <prompt.txt>
// One gemini-3-pro-image edit at 4K: fidelity re-render, lettering removal or vertical extension.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import sharp from 'sharp';
import { GoogleGenAI } from '@google/genai';
import { geminiClientOptions } from '../lib/env.mjs';

const [SRC, OUT, ASPECT, PROMPT_FILE] = process.argv.slice(2);
if (!SRC || !OUT || !ASPECT || !PROMPT_FILE) throw new Error('usage: node tools/gemini_edit.mjs <src> <out.png> <aspect> <prompt.txt>');

const buf = await sharp(SRC, { limitInputPixels: false })
  .resize({ width: 2048, height: 2048, fit: 'inside', withoutEnlargement: true }).png().toBuffer();
const ai = new GoogleGenAI(geminiClientOptions());
const res = await ai.models.generateContent({
  model: 'gemini-3-pro-image',
  contents: [{ role: 'user', parts: [{ text: readFileSync(PROMPT_FILE, 'utf8') }, { inlineData: { data: buf.toString('base64'), mimeType: 'image/png' } }] }],
  config: { responseModalities: ['IMAGE'], imageConfig: { aspectRatio: ASPECT, imageSize: '4K' } },
});
const part = res?.candidates?.[0]?.content?.parts?.find(p => p.inlineData?.data);
if (!part) throw new Error(`no image returned (${res?.candidates?.[0]?.finishReason})`);
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, Buffer.from(part.inlineData.data, 'base64'));
const m = await sharp(OUT).metadata();
console.log('->', OUT, `${m.width}x${m.height}`);
