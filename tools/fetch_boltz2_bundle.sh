#!/bin/zsh
# Fetch the pinned boltz2-int5 bundle so pass 1 can run locally.
# The revision is the one MODEL_BUNDLES pins; a moving branch is what the
# shard cache token exists to prevent.
B="https://huggingface.co/sokrypton/localfold/resolve/651a25ba8a57c6d18caacf50ab7180b255eb5a82/boltz2-int5"
OUT="/Users/pabloarantes/Downloads/Boltz2_webGPU/refs/bundle-boltz2"
mkdir -p "$OUT"
for f in manifest.json $(for i in $(seq -w 0 23); do echo "weights-$i.int5.bin"; done); do
  [ -s "$OUT/$f" ] && continue
  for attempt in 1 2 3 4 5; do
    code=$(curl -sS -L --max-time 900 -o "$OUT/$f.part" -w '%{http_code}' "$B/$f" 2>/dev/null)
    if [ "$code" = "200" ]; then mv "$OUT/$f.part" "$OUT/$f"; echo "ok   $f"; break; fi
    rm -f "$OUT/$f.part"; echo "retry $f (code=$code, attempt $attempt)"; sleep 20
  done
  [ -s "$OUT/$f" ] || { echo "FAILED $f"; exit 1; }
done
echo "bundle complete: $(du -sh "$OUT" | cut -f1)"
