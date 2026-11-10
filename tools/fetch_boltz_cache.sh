#!/bin/sh
# Fetch upstream Boltz-2's own runtime assets into an isolated cache directory,
# for the G8 end-to-end comparison and nothing else.
#
# 🔴 THIS IS 6.1 GB AND IT IS NOT NEEDED TO RUN THIS PROJECT. Everything under
# app/ reads a 365 MiB structure bundle and a 6.4 MiB affinity bundle; these
# four files are what UPSTREAM needs to run as an oracle. `mols.tar` in
# particular is the CCD asset set this project deliberately does not use - the
# featuriser oracle was built to need neither it nor ccd.pkl - and it is here
# only because `boltz predict` will not start without it.
#
# 🔴 THE TARGET IS A NEW, EMPTY, ISOLATED DIRECTORY, and the tarball is left for
# boltz itself to extract. Downloaded archives are untrusted input: nothing is
# run from inside this directory and no script here lives in it.
#
# Usage: tools/fetch_boltz_cache.sh <cache-dir>
set -eu
cache=${1:?usage: fetch_boltz_cache.sh <cache-dir>}
mkdir -p "$cache"
# 🔴 `--retry` DOES NOT COVER A STALL INSIDE A TRANSFER, and that is how the
# first attempt at this died: `curl: (56) Recv failure: Operation timed out`
# 1.17 GB into a 2.13 GB file, with `set -e` then aborting the whole script and
# leaving a .part nobody resumed. curl's retry logic applies to a request that
# FAILS TO START or returns an error status; a connection that goes quiet
# half-way through is a successful request that then dies. So the loop is here,
# around curl, with `-C -` resuming from whatever arrived: the unit of progress
# is bytes on disk, not an attempt.
#
# `--speed-limit`/`--speed-time` turn a silent stall into a prompt failure
# (under 10 kB/s for 60 s) so the loop can resume rather than hanging for the
# 100-second TCP timeout, over and over, on a connection that has gone.
fetch() {
  url=$1; name=$2
  if [ -s "$cache/$name" ]; then echo "have     $name"; return; fi
  want=$(curl -fsIL "$url" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{n=$2}END{print n+0}')
  echo "fetching $name  ($want bytes)"
  attempt=0
  while :; do
    attempt=$((attempt + 1))
    have=$([ -f "$cache/$name.part" ] && wc -c < "$cache/$name.part" || echo 0)
    if [ "$want" -gt 0 ] && [ "$have" -ge "$want" ]; then break; fi
    if [ "$attempt" -gt 40 ]; then
      echo "giving up on $name after $attempt attempts at $have/$want bytes" >&2
      return 1
    fi
    echo "  attempt $attempt from $have/$want bytes"
    # 🔴 NOT `set -e`'s PROBLEM TO SOLVE: the failure is expected and handled,
    # so curl's status is consumed here rather than killing the script.
    curl -fL --retry 5 --retry-delay 5 --retry-all-errors \
         --speed-limit 10240 --speed-time 60 \
         -C - -o "$cache/$name.part" "$url" || echo "  interrupted; resuming"
    sleep 2
  done
  mv "$cache/$name.part" "$cache/$name"
  echo "done     $name  $(wc -c < "$cache/$name") bytes"
}
base2=https://huggingface.co/boltz-community/boltz-2/resolve/main
fetch https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl ccd.pkl
fetch $base2/boltz2_conf.ckpt boltz2_conf.ckpt
fetch $base2/boltz2_aff.ckpt  boltz2_aff.ckpt
fetch $base2/mols.tar         mols.tar
echo "cache ready: $cache"
ls -l "$cache"
