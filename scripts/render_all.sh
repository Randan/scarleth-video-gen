#!/usr/bin/env bash
# Renders all 12 deliverables from the prepared masters in work/ and copies them to the
# delivery folder with the names the uploader expects. Jobs run ONE AT A TIME on purpose: a 3840x3840
# composite peaks at ~3.8 GB RAM, and three in parallel froze a 24 GB Mac.
#
#   bash scripts/render_all.sh                 # all 12, version v1 (Vortex: flashlight sweep)
#   bash scripts/render_all.sh all v2          # all 12, version v2 (Vortex: sweep + floating glass)
#   bash scripts/render_all.sh vtx v2          # one cover (kpd | bts | slv | vtx)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W="$ROOT/work"; OUT="$W/out"; PY="$ROOT/.venv/bin/python"
ONLY="${1:-all}"; VERSION="${2:-v1}"
DELIVERY="${DELIVERY:-$HOME/Downloads/Scarleth_canvas}/$VERSION"
VTX_SCRIPT=$([[ $VERSION == v2 ]] && echo vtx_float.py || echo vtx_sweep.py)
[[ $VERSION == v2 ]] && OUT="$W/out_v2"

# Approved Wan render per cover (kpd: seed 22; the default-seed and seed-11 takes drifted the camera).
motion() { case $1 in kpd) echo motion_s22.mp4 ;; *) echo motion_v1.mp4 ;; esac; }
name() { case $1 in kpd) echo Koly_Pryide_Den ;; bts) echo Break_The_Silence ;; slv) echo The_Silver_Lining ;; vtx) echo Vortex ;; esac; }

for k in kpd bts slv vtx; do
  [[ "$ONLY" == all || "$ONLY" == "$k" ]] || continue
  for w in spotify a34 a11; do
    if [[ $k == vtx ]]; then "$PY" "$ROOT/scripts/$VTX_SCRIPT" "$W" "$OUT" "$w"
    else "$PY" "$ROOT/scripts/composite.py" "$W" "$k" "$(motion $k)" "$OUT" "$w"; fi
  done
  mkdir -p "$DELIVERY/Spotify" "$DELIVERY/Apple Music"
  cp "$ROOT/docs/upload_instructions_uk.txt" "$DELIVERY/ІНСТРУКЦІЯ.txt"
  cp "$OUT/spotify/${k}_spotify_9x16.mp4" "$DELIVERY/Spotify/Scarleth_$(name $k)_Spotify_Canvas_9x16.mp4"
  cp "$OUT/apple/${k}_apple_3x4.mp4" "$DELIVERY/Apple Music/Scarleth_$(name $k)_AMB_3x4.mp4"
  cp "$OUT/apple/${k}_apple_1x1.mp4" "$DELIVERY/Apple Music/Scarleth_$(name $k)_AMB_1x1.mp4"
done

"$PY" "$ROOT/scripts/loopcheck.py" "$DELIVERY"/Spotify/*.mp4 "$DELIVERY"/Apple\ Music/*.mp4
