// node tools/bria_expand.mjs <square.png> <out.png> <canvasW> <canvasH> <squareTopY> "<prompt>" [squareLeftX=0]
// Bria Expand (fal.ai): keeps the original pixels and paints the new canvas around them.
import { readFileSync, writeFileSync } from 'node:fs';
import sharp from 'sharp';
import { envVal } from '../lib/env.mjs';

const [SRC, OUT, CW, CH, TOP, PROMPT = '', LEFT = '0'] = process.argv.slice(2);
const cw = Number(CW), ch = Number(CH), top = Number(TOP), left = Number(LEFT);
const key = envVal('FAL_AI_API_KEY');
// The source keeps its aspect and is centred horizontally: its width is the canvas width minus 2 x left.
const meta0 = await sharp(SRC).metadata(); const ow = cw - 2*left; const oh = Math.round(meta0.height * ow / meta0.width);
const png = await sharp(SRC, { limitInputPixels: false }).resize(ow, oh).png().toBuffer();
const body = {
  image_url: `data:image/png;base64,${png.toString('base64')}`,
  canvas_size: [cw, ch], original_image_size: [ow, oh], original_image_location: [left, top],
  prompt: PROMPT, negative_prompt: 'photo, photograph, photorealistic, asphalt, road, pavement, text, letters, logo, watermark, frame, border, duplicate objects, seam',
  sync_mode: true,
};
const h = { Authorization: `Key ${key}`, 'Content-Type': 'application/json' };
const sub = await (await fetch('https://queue.fal.run/fal-ai/bria/expand', { method: 'POST', headers: h, body: JSON.stringify(body) })).json();
if (!sub.request_id) throw new Error('submit failed: ' + JSON.stringify(sub).slice(0, 500));
let st;
for (;;) {
  await new Promise(r => setTimeout(r, 4000));
  st = await (await fetch(sub.status_url, { headers: h })).json();
  if (st.status === 'COMPLETED') break;
  if (st.status && !['IN_QUEUE', 'IN_PROGRESS'].includes(st.status)) throw new Error('status ' + JSON.stringify(st).slice(0, 400));
}
const res = await (await fetch(sub.response_url, { headers: h })).json();
const url = res?.image?.url;
if (!url) throw new Error('no image: ' + JSON.stringify(res).slice(0, 500));
const buf = url.startsWith('data:') ? Buffer.from(url.split(',')[1], 'base64') : Buffer.from(await (await fetch(url)).arrayBuffer());
writeFileSync(OUT, buf);
const m = await sharp(OUT).metadata();
console.log('->', OUT, `${m.width}x${m.height}`, 'seed', res.seed);
