#!/bin/sh
# Re-run THIS PORT's arm of G8 and of the six-ligand ranking sweep.
#
# Upstream is not re-run: nothing on its side changed, and its five-seed
# control (refs/g8-out-5) and per-ligand values (refs/g8-rank) are on disk.
# Sequential on purpose - one headless Chrome at a time.
set -u
cd /Users/pabloarantes/Downloads/Boltz2_webGPU
P=refs/venv-boltz/bin/python
echo "=== main comparison, 5 seeds, ours only ==="
$P -I tools/g8_compare.py --skip-upstream --seeds=5 --out=refs/g8-today || echo "MAIN FAILED"
i=0
for S in 'CCCc1ccccc1O' 'CCO' 'CC(C)Cc1ccc(cc1)C(C)C(=O)O' 'CC(=O)Oc1ccccc1C(=O)O' 'CN1CCC[C@H]1c1cccnc1' 'c1ccc2c(c1)cccc2O'; do
  i=$((i+1))
  echo "=== ligand $i: $S ==="
  $P -I tools/g8_compare.py --skip-upstream --seeds=1 --smiles="$S" \
     --out="refs/g8-rank-today/lig$i" || echo "LIGAND $i FAILED"
done
echo "=== all done ==="
