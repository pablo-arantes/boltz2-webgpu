#!/bin/zsh
# Wait for huggingface.co, then run G0's value half. Up to ~60 minutes.
cd /Users/pabloarantes/Downloads/Boltz2_webGPU
for i in $(seq 1 40); do
  code=$(curl -sS -L -o /dev/null --max-time 20 -w '%{http_code}' https://huggingface.co/ 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "huggingface.co reachable on attempt $i - running G0"
    python3 -I tools/g0_verify.py --out spec/G0-RESULT.md 2>&1
    exit 0
  fi
  echo "attempt $i: hf unreachable (code=$code), waiting 90s"
  sleep 90
done
echo "GAVE UP: huggingface.co unreachable for ~60 minutes; G0 value half still pending"
exit 1
