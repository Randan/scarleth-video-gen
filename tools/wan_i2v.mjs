// node tools/wan_i2v.mjs <master.png> <out.mp4> <prompt.txt> [duration=17] [seed]   (9:16 or 16:9 master)
// Wan 3.0 image-to-video of a 9:16 master, used as the motion source for the cover loops.
import { readFileSync, writeFileSync } from 'node:fs';
import sharp from 'sharp';
import { startImageToVideo, pollVideo } from '../lib/wan.mjs';

const [SRC, OUT, PROMPT_FILE, DUR = '17', SEED] = process.argv.slice(2);
const meta = await sharp(SRC).metadata();
const [fw, fh] = meta.width > meta.height ? [1920, 1080] : [1080, 1920];   // 16:9 stage masters or 9:16 phone masters
const jpg = await sharp(SRC, { limitInputPixels: false }).resize(fw, fh, { fit: 'fill' }).jpeg({ quality: 93 }).toBuffer();
const { taskId } = await startImageToVideo({
  prompt: readFileSync(PROMPT_FILE, 'utf8').trim(),
  imageUrl: `data:image/jpeg;base64,${jpg.toString('base64')}`,
  resolution: '1080P', ratio: 'adaptive', duration: Number(DUR), audio: false,
  ...(SEED ? { seed: Number(SEED) } : {}),
});
for (let i = 0; i < 200; i++) {
  await new Promise(r => setTimeout(r, 15000));
  const p = await pollVideo(taskId);
  if (!p.isDone) continue;
  if (p.failed) throw new Error(`Wan failed: ${p.failReason}`);
  writeFileSync(OUT, Buffer.from(await (await fetch(p.videoUrl)).arrayBuffer()));
  console.log('->', OUT, 'task', taskId);
  process.exit(0);
}
throw new Error('Wan timed out');
