/**
 * Wan 3.0 image-to-video client (Alibaba DashScope).
 *
 *   POST {base}/services/aigc/video-generation/video-synthesis   (X-DashScope-Async: enable)
 *   GET  {base}/tasks/{task_id}                                  (poll until SUCCEEDED)
 *
 * The result video_url expires after ~24 h, so download it as soon as the task succeeds.
 * Wan 3.0 has no negative prompt: every restriction has to be phrased in the positive prompt.
 */
import { envVal } from './env.mjs';

export const WAN_MODEL = 'wan3.0-video';
const MIN_DURATION = 2;
const MAX_DURATION = 30;
const MAX_RETRIES = 3;

const sleep = ms => new Promise(r => setTimeout(r, ms));
const baseUrl = () => envVal('WAN_API_BASE_URL').replace(/\/$/, '');
const headers = extra => ({ Authorization: `Bearer ${envVal('WAN_API_KEY')}`, 'Content-Type': 'application/json', ...extra });

async function fetchWithRetry(url, init, label) {
  let lastErr;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt += 1) {
    try {
      const res = await fetch(url, { ...init, signal: AbortSignal.timeout(60_000) });
      if ((res.status === 429 || res.status >= 500) && attempt < MAX_RETRIES) {
        await sleep(2000 * 2 ** attempt);
        continue;
      }
      return res;
    } catch (e) {
      lastErr = e;
      if (attempt < MAX_RETRIES) await sleep(2000 * 2 ** attempt);
    }
  }
  throw new Error(`Wan ${label} failed after ${MAX_RETRIES + 1} attempts: ${String(lastErr)}`);
}

export async function startImageToVideo({ prompt, imageUrl, resolution = '1080P', ratio = 'adaptive', duration = 5, audio = false, seed }) {
  if (!prompt) throw new Error('Wan create needs a non-empty prompt');
  if (!imageUrl) throw new Error('Wan create needs an imageUrl');
  const body = {
    model: WAN_MODEL,
    input: { prompt, media: [{ type: 'first_frame', url: imageUrl }] },
    parameters: {
      resolution, ratio, audio,
      duration: Math.min(Math.max(Math.round(duration), MIN_DURATION), MAX_DURATION),
      enable_thinking: false,
      watermark: false,
      // DashScope rejects seed: null, so the key is omitted unless a seed is given.
      ...(seed !== undefined && seed !== null ? { seed } : {}),
    },
  };
  const res = await fetchWithRetry(`${baseUrl()}/services/aigc/video-generation/video-synthesis`,
    { method: 'POST', headers: headers({ 'X-DashScope-Async': 'enable' }), body: JSON.stringify(body) }, 'create');
  const json = await res.json().catch(() => ({}));
  if (!res.ok || !json?.output?.task_id) {
    throw new Error(`Wan create rejected (${res.status}): ${json?.code ?? ''} ${json?.message ?? JSON.stringify(json).slice(0, 400)}`);
  }
  console.log(`[Wan] task submitted ${json.output.task_id}`);
  return { taskId: json.output.task_id };
}

export async function pollVideo(taskId) {
  const res = await fetchWithRetry(`${baseUrl()}/tasks/${taskId}`, { method: 'GET', headers: headers() }, 'poll');
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(`Wan poll failed (${res.status}) task=${taskId}: ${JSON.stringify(json).slice(0, 400)}`);
  const status = json?.output?.task_status;
  if (status === 'SUCCEEDED') return { isDone: true, videoUrl: json.output.video_url };
  // UNKNOWN is what DashScope returns for an aged-out task id, so it is terminal too.
  if (['FAILED', 'UNKNOWN', 'CANCELED'].includes(status)) {
    return { isDone: true, failed: true, failReason: json.output?.message ?? json.output?.code ?? status };
  }
  return { isDone: false, status };
}
