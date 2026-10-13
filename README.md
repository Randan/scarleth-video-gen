# Scarleth video gen

Animated cover art for the band Scarleth: a Spotify Canvas and an Apple Music Album Motion pair for every
release. Four covers have been done so far:

| Key | Release | Source cover |
|---|---|---|
| `kpd` | Коли прийде день (single) | `covers/Коли Прийде День.png` (1348 px) |
| `bts` | Break the Silence (2011) | `covers/Break The Silence.jpg` (1424 px) |
| `slv` | The Silver Lining (2015) | `t-shirt/3_silver_lining/обложка альбома.jpg` (2000 px) + textless sticker art |
| `vtx` | Vortex (2019) | `covers/Vortex.jpg` (3000 px) + textless sticker art |

Sources live in the band's Google Drive: `~/Library/CloudStorage/GoogleDrive-andrewrandan@gmail.com/Мій диск/Scarleth`.
`scripts/prepare_sources.py` holds the exact paths.

**Status:** 2026-10-09 all 12 videos approved by Andrew. 2026-10-12: a final round of outside feedback (Viktor Morozov)
was applied → `v1`, and a floating-glass Vortex variant was built → `v2` (see Decisions). Earlier: They were delivered to `~/Downloads/Scarleth_canvas/` with
`ІНСТРУКЦІЯ.txt` (Ukrainian upload guide for whoever uploads them; Andrew does not upload himself; copy in `docs/upload_instructions_uk.txt`).
`~/Downloads` is Andrew's scratch area: finished videos always go into their own subfolder there, and he deletes
it when done. The approved renders stay in `work/out/`, and `render_all.sh` rebuilds the folder from them.

## Deliverables per cover (3 files → 12 for 4 covers)

| Platform | File | Size | Length | Lettering |
|---|---|---|---|---|
| Spotify Canvas | `*_Spotify_Canvas_9x16.mp4` | 1080×1920 | 8 s | none: Spotify shows title/artist itself |
| Apple Music, iPhone | `*_AMB_3x4.mp4` | 2048×2732 | 15 s | logo + title inside the art safe area, bottom darkened for Apple's UI text |
| Apple Music, iPad/Mac/TV | `*_AMB_1x1.mp4` | 3840×3840 | 15 s | the cover itself, untouched |

Encoding: H.264 High, yuv420p, 30 fps, BT.709 tags, no audio track. Spotify uses CRF 4 (18–27 MB);
Apple uses 90 Mbps CBR (Apple's window is 45–100). Each encode has ONE keyframe with `ipratio=1`.
Without that, the first frame (an I-frame) is visibly sharper than the last P-frames, and the loop pops.

## Platform rules that shaped the work

**Spotify Canvas** ([guidelines](https://support.spotify.com/us/artists/article/canvas-guidelines/)):
3–8 s (hard limit), 9:16, 720–1080 px tall, MP4/JPG, silent. Keep the subject in the central vertical
band: phones crop the sides, and player controls cover the bottom. A Canvas is attached per TRACK in
Spotify for Artists, not per album.

**Apple Music Album Motion** ([guidelines](https://help.apple.com/itc/albummotionguide/en.lproj/static.html),
[style guide §7](https://help.apple.com/itc/musicstyleguide/en.lproj/static.html)):
- a 3:4 2048×2732 + 1:1 3840×3840 pair of IDENTICAL length, 8–35 s (some distributors still say 15–35),
  45–100 Mbps, 23.976–30 fps, Rec.709/sRGB, no audio track;
- the first frame must equal the static cover; the loop must be seamless, with no cuts, no fade to black,
  and no drop or hold frames;
- 3:4 must be full-frame art, never the 1:1 cover scaled into the frame;
- no frantic flashing, and no upscaled low-res pixels;
- AI must not distort faces, anatomy or titles.

3:4 safe area (measured from Sony Music's Album Motion quick sheet, since Apple's template is not public):
art safe area ≈ x 126–1922, y 429–1708 of 2732. The top ~16% holds navigation and the notch; the bottom
~37% holds the title text and the Play/Shuffle buttons. Delivery goes ONLY through a distributor that
supports motion art. **Once delivered, it cannot be removed without a full takedown and a new UPC.**

## Pipeline

All media goes in `work/<key>/` (gitignored). Prompts are in `prompts/`. Run Node tools with `node`,
and Python with `.venv/bin/python`.

1. **Sources** — `scripts/prepare_sources.py work` copies `orig.png` (+ `clean_src.png`), and registers
   the Vortex textless art straight to `vtx/clean_4k.png`.
2. **4K fidelity re-render** (kpd, bts, slv; Vortex is already 3000 px) —
   `tools/gemini_edit.mjs <orig> <sq_text_4k.png> 1:1 prompts/p_fid_<key>.txt`. For SLV, also run the
   textless sticker art through `p_fid_slv_clean.txt` to get `clean_4k.png`. Gate every Gemini output with
   `scripts/gate.py ref cand` (good: phase response ≥0.85, edge overlap ≥0.9) and look at the crops.
3. **Lettering removal** (kpd, bts) — `tools/gemini_edit.mjs <sq_text_4k> <clean_4k.png> 1:1 prompts/p_clean_<key>.txt`.
   The BTS prompt also removes the ornamental border frame.
4. **Grade back** — `scripts/grade_stills.py work`. Gemini shifts colour, so this re-matches the stills.
5. **9:16 master** (3072×5504, textless):
   - BTS, SLV: `scripts/tall_input.py work <key>`, then `tools/gemini_edit.mjs <tall_input> <tall_raw.png> 9:16 prompts/p_tall_<key>.txt`;
   - KPD: the same Gemini route (the approved take is `tall_raw3.png`). The Bria takes painted the bottom as a road;
   - VTX: `tools/bria_expand.mjs clean_4k.png bria_raw1.png 2304 4096 648 "$(cat prompts/bria_vtx.txt)"`.
     Gemini kept duplicating the spiral and its shards;
   - then `scripts/build_master.py work <key> <raw>` registers the exact clean square into the outpaint,
     grades the outpaint to it and pastes it back with wide feathers (`--seamfix` adds a per-column
     colour+luminance seam correction; the approved masters did not use it);
   - KPD then needs `scripts/kpd_bottom_fix.py work` (luminance-only join correction + bottom darkening).
6. **Lettering layer** — `scripts/text_layer.py work <key>` (RGBA of cover-minus-clean). For VTX 3:4 also run
   `scripts/vortex_word.py work`.
7. **Motion**
   - kpd, bts, slv: `tools/wan_i2v.mjs master_916.png <motion.mp4> prompts/m_<key>.txt 17 [seed]` (Wan 3.0,
     1080×1920, 17 s), then `scripts/analyze.py` to check drift. Camera drift must be about 0 px. Approved
     takes: kpd `motion_s22.mp4` (seed 22), bts/slv `motion_v1.mp4`.
   - vtx: no Wan at all (see decisions below).
8. **Render + deliver** — `bash scripts/render_all.sh [key] [v1|v2]` (default `all v1`) → `~/Downloads/Scarleth_canvas/<version>/`. It runs `composite.py` (kpd/bts/slv) or
   `vtx_sweep.py` (vtx) for spotify → a34 → a11, copies the files to `~/Downloads/Scarleth_canvas`,
   and runs `loopcheck.py`.

QA helpers: `scripts/qa.py` (specs + contact sheet), `scripts/loopcheck.py` (the last→first jump against
normal frame steps). Loops are seamless before encoding. After encoding, a seam up to ~3× the median
step comes from codec noise, and only in moving areas.

## How the composite works (kpd, bts, slv)

Wan re-paints the whole frame. So the render replaces the crisp still ONLY where something really moves.
The motion mask is a per-pixel temporal std, thresholded at 2.5–6 (Wan's texture "boil" sits below ~2.5).
A low-frequency gain map matches the render's colour to the still. The loop cross-fades the tail into the
head (Spotify: 8 s starting at 2 s with a 1 s fade; Apple: 15 s with a 1.5 s fade). Lettering is a static
overlay. The 1:1 is the 4K cover with motion warped in through the master↔square similarity. Use separate
x/y scales: the video is 1080×1920 and the master is 3072×5504, which are not exactly proportional.

## Decisions and review history

**2026-10-12 — Viktor's notes (applied in v1):**
- BTS 3:4: the black outline of the logo/title was eaten by the luminance gate → block matte `outline`
  (bright letters + a ~12 px ring of the cover's own alpha around them).
- KPD 3:4: dark patches between the letters of «ДЕНЬ» (cover background carried in by the matte) → block matte
  `unblend` (white lettering solved against the clean art) + a faint soft shadow for legibility on light sky.
- SLV (all 3): the flower looked doubled/blurred at the loop seam (start of the file) because the linear
  cross-fade mixed two different poses → `composite.py` now cross-fades along optical flow (`flow_blend`, DIS).
  Viktor suggested cutting the first 1–2 s, but Apple needs the first frame to be the cover, so the seam was fixed instead.
- Vortex Apple (15 s): one sweep read as a still image for too long → two sweeps (`APPLE_SWEEPS = [1.5, 9.0]`).
  Spotify (8 s) keeps one.

**v2 — floating glass (Andrew's optional idea):** `scripts/vtx_float.py`. Gemini made a shard-free plate of the
stairwell (`prompts/p_plate_vtx.txt` → `vtx/plate_raw.png`); shards = master pixels that differ from it,
filled and split into ~34 rigid pieces; each drifts ≤0.45% of the frame width and tilts ≤0.9° on a loop-periodic
path that is zero at frame 0; the plate fills what a moving shard uncovers. Static on purpose: the spiral
funnel (the plate redraws it) and the bottom-left photo collage. The flashlight sweep rides on the moved
edges. v2 = v1 with the three Vortex files swapped.

**Earlier (Andrew's calls):**

- **Vortex: no zoom, no morph.** Andrew rejected a breathing push-in and a funnel swirl. The approved
  motion is a still frame where, once per loop, a flashlight-like beam sweeps across fast (1.1 s, Spotify at
  2.6 s, Apple at 5 s), and the glowing glass edges flare as it passes. The lettering does not light up.
  Andrew's optional idea, "the glass floats a little in place", was NOT done: it needs per-shard
  segmentation and risks morphing.
- **Vortex 3:4 word "VORTEX"** must be the clean unblended word (`vortex_word.py`). The generic matte left a
  burnt-in background patch.
- **Silver Lining 3:4 logo** must be taken from the cover pixels at its own place (`cover_patch`, window
  top 389). Re-cutting and re-placing the logo made it look crooked.
- **Break the Silence:** Wan added a few lightning flashes in the sky. Andrew approved them.
- **Коли прийде день:** first Wan take drifted the camera (36 px), seed 11 drifted 5 px, seed 22 is
  locked. Approved.
- Wan-animated faces (Medusa relief, Vortex shard faces) must stay still. The motion mask and the
  procedural Vortex handle this.

## Setup

```bash
npm install
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in the keys
```

Needs `ffmpeg`/`ffprobe` on PATH. Keys: Gemini via Vertex service account (`gemini-3-pro-image`),
Wan 3.0 (DashScope), fal.ai (Bria Expand).

**Machine limit:** 24 GB RAM. Run heavy jobs one at a time (`render_all.sh` does). A 3840² composite
peaks at about 3.8 GB, and three in parallel froze the Mac once.

## Club stage backdrops (16:9) — added 2026-10-12

Four silent 1920×1080 loops, one per release, for the big screen behind the stage: no album titles, the
band logo SCARLETH alone as the focus (upper-middle, clear of the band and the gear below it), 20 s,
30 fps, H.264 CRF 6 (~30–65 Mbps), BT.709. Delivered to `~/Downloads/Scarleth_stage/`.

Why not "the original cover only": the covers are square, and 16:9 Full HD needs 1920 px of width. So the
original art is kept pixel-exact in the centre at 1140 px, and only the sides are painted.

1. 16:9 master (3840×2160, textless): kpd and bts via Gemini (`tall_input`-style padded canvas,
   `prompts/p_wide_<key>.txt`), slv and vtx via Bria Expand (`tools/bria_expand.mjs ... 3840 2160 0 "<prompt>" 840`,
   which centres the square; Gemini cut slv into panels and duplicated vtx). Then `scripts/build_wide.py work <key> <raw>`
   pastes the exact clean square back with wide left/right feathers → `<key>/wide_master.png` + `wide_reg.json`.
2. Logo only: `scripts/logo_rgba.py work <key>` → `<key>/logo_rgba.png`. For slv, Gemini first removed only the
   "the Silver Lining" script (`prompts/p_slv_drop_title.txt` → `slv/sq_logo_only_4k.png`), and the bronze
   ornament stays with the logo.
3. Layout (logo width/height, soft shadow): `scripts/stage_layout.py`. BTS logo sits higher, inside the archway,
   so it does not cover the bell's skull crown.
4. Motion:
   - kpd/bts/slv: `tools/wan_i2v.mjs <key>/wide_master.png <key>/stage_motion.mp4 prompts/m_<key>.txt 22 [seed]`
     (the tool picks 1920×1080 for wide masters). For kpd use `prompts/m_kpd_stage.txt`: it locks the light,
     because the first take grew big sunbeams and brightened the sky by 50%. Approved: kpd `stage_motion2.mp4`
     (seed 22), bts/slv `stage_motion.mp4`. Then `scripts/stage.py work <key> <motion> work/out_stage`.
     It chooses the loop window with the most alike seam frames, divides out Wan's slow global brightening
     (per-channel linear trend, otherwise the light snaps back at the loop point), cross-fades the seam along
     optical flow, and lays the logo on top.
   - vtx: `scripts/vtx_float.py work work/out_stage stage` — floating glass + flashlight sweeps at 1.5 / 8 / 14.5 s, on
     its own 16:9 shard-free plate (`vtx/wide_plate_raw.png` from `prompts/p_plate_vtx.txt`). The logo never lights up.

**Stage round 2 (2026-10-12, Andrew's review):**
- BTS: the cover cut had a clipped bottom and holes → the band's own `logos/logos_old/logo.psd` layer "Group 8"
  (same orange chrome lettering, clean alpha), exported to `work/logos/logo_old_L2.png`.
- SLV: the ornament cut from the cover was ragged → its alpha now comes from the band's vector files, registered
  onto the cover with ECC: letters `poster/shared/silver-lining-scarleth-logo.psd`, ornament
  `logos/Scarleth logos 2025/logo_transparent.psd` (layer 2). The pixels stay the cover's (`scripts/logo_rgba.py`).
- KPD: the left smoke only partly moved (Wan moved it too slowly to pass the motion threshold) and two birds
  merged into one near the tree at 0:14. New take `stage_motion_s31.mp4` (`prompts/m_kpd_stage.txt`: the whole
  smoke cloud rolls at a clearly visible pace). `stage.py` keeps the left-sky region always animated (`ALWAYS`),
  and in the bird area it erases small dark specks from the render every frame and keeps the cover's birds still
  (`BIRDS`). Wan ignores "birds stay still" in the prompt.
- Vortex: approved as is. The floating was then re-scaled to the cover square (`DRIFT` = 0.8% of the square side),
  so the 9:16 / 3:4 / 1:1 canvases float exactly like the stage loop → `~/Downloads/Scarleth_canvas/vortex_float/`.
- Loops: all four stage loops and the three float canvases pass `loopcheck.py`. Seam frames were checked by eye for
  Viktor-style doubling (flower, fire, smoke, shards) and none was found.

**Stage round 3 (2026-10-12, Andrew's review):**
- BTS: the stage lightning looked like threads waving in the wind → new take `stage_motion_s29.mp4` with
  `prompts/m_bts_stage.txt` (sharp, jagged, instant bolts that never sway). Lightning is a discrete event, so
  `stage.py` (FLASHY covers) keeps both seam windows in quiet, un-lit frames and uses a 0.5 s seam, like a
  natural lightning decay. The loop is therefore 15.4 s, not 20 s.
- KPD: (1) the smoke ran backwards for 1–2 s at the loop start: a flow morph from the tail to the head reverses
  directional motion → the smoke region (`ALWAYS`) now uses a plain dissolve at the seam (`PLAIN`). (2) Pale ghost
  birds: the old speck fill (morphological close) painted bird shapes lighter than the sky. Now the open sky in
  the bird box is the still (`bird_gate`: render only on clouds), and Wan birds crossing clouds are erased by a
  compact-blob fill (`clean_cloud_birds`).
- SLV: the ornament was removed completely. The logo is the band's own Silver Lining lettering file
  (`work/logos/sl_logo.png`), clean alpha; checked on black, white and green.

**Stage round 4 (2026-10-12, Andrew's review):**
- BTS: Wan's cloud motion looked odd → the sky inside the archway is the still (`STATIC`), and the lightning
  is put back on top as pure added light: render minus its own running minimum over ±0.5 s, floor 10 (`LIGHT`).
  Between flashes the sky is pixel-identical to the cover.
- KPD: tree foliage morphed → a saturated-green mask of the trees (`work/kpd/trees_mask_1080.npy`) stays still.
  The logo had a dark teal halo (the cover's sky caught in the cut, read as a blue glow) → alpha now comes
  from the band's vector letters (`logo_transparent.psd` layer 3) registered onto the cover (`kpd/letters_alpha.npy`).
- SLV, Vortex: approved.
- All four: loops pass `loopcheck.py`; the logo was checked in frames from the start, middle and end of each video.

**Canvas v3 (2026-10-13, Viktor):** SLV AMB 1x1/3x4 had a 1–2 s "beating" in the haze below-left of the S at the
start of the file: the optical-flow seam morph warped the smoke and light. `composite.py` now morphs along the flow
only around the flower (`FLOWM`) and uses a plain dissolve elsewhere. v3 = v2 + these two SLV files + the
`vortex_float` Vortex canvases (more active shards, approved by Viktor) → `~/Downloads/Scarleth_canvas_v3/` and
`Scarleth_canvas_v3.zip`. Viktor asked for v3 only, as an archive, without older versions.

**Stage v5 (2026-10-13):**
- BTS: Wan's torch flames morphed → torches are procedural now: the cover's own flame warped by an upward-travelling,
  loop-periodic displacement field with brightness flicker (`FIRE`), and the stone stays still. Lightning morphed
  (Wan redraws the bolt each frame) → each flash is ONE frozen bolt image (its brightest frame) played with a strike
  envelope (`ENV`).
- KPD: the right sky (clouds) morphed at ~0:10 → the whole right sky is static, like the trees.
- Vortex: the thick icy lettering Andrew made for the merch (`t-shirt/2_vortex/back.psd`, group "Group 1" →
  `work/logos/vtx_thick.png`). Stage only; the canvases keep the cover's logo.
- All four: loop check, and a start-of-file "beating" check (frame-step energy in the first 1.5 s against the rest,
  per 3×3 region; all ≤1.4×), and logo crops.

**Stage v6 (2026-10-13, Andrew's review of v5):**
- BTS: the procedural torches (v5) read as constant morphing → back to Wan's torches (v4). The real v4 problem was a
  0.5 s flame morph at the loop point: the seam was an optical-flow morph, and the loop started in the render's
  first second, where Wan turns the painted flame into motion. Now BTS uses a plain dissolve at the seam
  (`PLAIN` = everywhere) and never starts before frame 36 (`MIN_OFF`), so the loop is 14.2 s. The frozen-bolt
  lightning from v5 is kept.
- Vortex: matched to Andrew's own backdrop `Drive/Scarleth/vortex-bg-min-logo-3m.mp4`, measured by registering the
  cover onto it: the background is the cover × BGR gain/offset 0.604/-3.1, 0.623/-2.7, 0.573/-7.2 (≈45% darker,
  slightly teal), and the logo has a drop shadow to the lower right (offset 10 px, blur σ 8 px, opacity 0.42 at a
  1280 px wide logo; scaled to ours). Nothing rotates.

**Stage v7 (2026-10-13): Vortex background darkening tuned.** Andrew found the reference level (≈40% darker) too dark.
I swept the darkening strength `BG_S` from 0 to 1 on undarkened probe frames (every 0.5 s, plus the sweep peaks) and
measured two things in a 2–30 px ring around the letters: the median WCAG contrast of the logo against its
surroundings, and the "bright clutter", i.e. the share of ring pixels brighter than half the logo's luminance (glass
edges, light sweeps). The drop shadow keeps the median contrast ≥4.7:1 even with no darkening. Clutter, worst frame,
goes 17% (none) → 13% (0.4) → 10% (0.6) → 6% (1.0). Chosen: `BG_S_DEFAULT = 0.6` (≈24% darker), the least darkening
where the worst frame stays at ≤10% clutter and ≥5.8:1 median contrast. Comparison stills: `work/qa/vtx_dark_levels.jpg`.
BTS v6 approved.

**Canvas v3 loop re-check (2026-10-13):** all 12 files checked globally and locally. SLV AMB 1x1 was borderline on the whole-frame seam (frame 449→450 of the Wan take is itself a larger step), so its loop window now starts at frame 11 of the take (`OFF` override in `composite.py`); re-rendered, and the v3 folder and zip were updated.

**Stage v8 (2026-10-13, Viktor + Andrew on v7):**
- BTS: the lightning was cut in half (the static/added-light box ended at y=600, so the render showed below it) →
  the whole archway behind the bell, down to the floor, is one static backdrop with the added lightning
  (`LIGHT` = 670,0,1260,840). The fragments around the bell now float exactly like the Vortex glass:
  `scripts/float_shards.py` (shared constants), a Gemini shard-free plate (`prompts/p_plate_bts.txt` →
  `bts/wide_plate_raw.png`), 18 pieces.
- KPD: the birds glide and flap (the cover's flock: inpainted out of the still sky, then redrawn per frame with
  a loop-periodic flock drift, per-bird wobble, and a wing flap as a vertical squash, 24 cycles per loop). The rift
  "breathes": its edges spread and close around a hand-traced centre line (5 cycles per loop), with a red glow pulse
  running down it. The logo has a thicker near-black outline (stroke ~15 px at 4096 scale), like the streaming videos.
- Vortex, SLV: unchanged (v7).

**Stage v9 (2026-10-13): KPD only (BTS v8 approved).** Birds: squashing the painted sprites read as morphing, so the
flock is now redrawn as flapping silhouettes at the same places, sizes and colour (body plus two hinged wings, elbow
and tip beat up and down, 24 beats per loop, per-bird phase). The painted birds are inpainted out of the sky.
Rift: warping a band around it deformed the sides → the rift is its own layer (`work/kpd/rift_alpha_1080.npy`:
dark + red pixels connected to the centre line) that only expands outward (0–9%) over the untouched master.
The sides are never warped, only covered a little more and uncovered again. The glow pulse is kept.

**Stage v10 (2026-10-13): KPD.** Wing beats slowed to 12 per loop. The rift alpha now comes from a Gemini segmentation (`prompts/p_mask_kpd_rift.txt` paints the crack magenta → `work/kpd/rift_magenta.png`), eroded 2 px, with the centre line = the outline's per-row middle (`work/kpd/rift_center_1080.npy`). The hand colour mask had caught dark ground left of the crack and dragged it along. Checked: the ground beside the rift does not change across the loop.
