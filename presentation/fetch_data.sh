#!/usr/bin/env bash
# Pull the recorded data the figures are drawn from (from the Jetson's run folders) into ./data
set -euo pipefail
cd "$(dirname "$0")"; mkdir -p data out
P=rammp_runs/20260924_202241; F=rammp_runs/deliver_20260924_194800
for f in colour.jpg depth.npy detection.json target.json scoop_log.json under_over.json locate_log.txt; do scp -q jetson:$P/$f data/perceive_$f; done
for f in colour.jpg depth.npy face.json landmarks.json; do scp -q jetson:$F/$f data/face_$f; done
scp -q jetson:rammp_runs/bite_torque_1.json jetson:rammp_runs/rest_torque.json data/
echo "data/ ready; run: for f in f*.py; do python3 \$f; done   (figures -> out/)"
