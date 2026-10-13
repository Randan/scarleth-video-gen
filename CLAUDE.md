# Scarleth video gen — working notes for Claude

Read `README.md` first: it holds the specs, the pipeline, and every approved decision.

- Chat with Andrew in Ukrainian. Code, comments, commits and docs are in English.
- This project is ONLY Scarleth animated covers.
- Review routine: Andrew reviews every output before anything ships. Report per platform → format →
  album, with + / − verdicts, the way he reviews.
- Heavy media jobs run ONE AT A TIME (24 GB RAM). Keep costly intermediates in `work/`, never in /tmp.
- Every Gemini/Bria/Wan call costs money. Gate the result (`scripts/gate.py`, `scripts/analyze.py`)
  before building on it, and keep approved takes; do not regenerate them without reason.
- Prove loops with `scripts/loopcheck.py` on the final files. Apple's first frame must equal the cover.
- Finished videos ALWAYS go into a dedicated subfolder of `~/Downloads` (e.g. `~/Downloads/Scarleth_canvas/`).
  It is Andrew's scratch area and he deletes those folders himself, so a missing one is normal;
  keep the masters in `work/`. Andrew does not upload to streaming services himself: add
  `ІНСТРУКЦІЯ.txt` (Ukrainian, see `docs/upload_instructions_uk.txt`) for whoever uploads.
