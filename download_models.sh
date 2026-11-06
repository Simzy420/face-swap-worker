#!/usr/bin/env bash
set -e
mkdir -p /models
cd /models

# InsightFace buffalo_l (detection + recognition)
if [ ! -d "models/buffalo_l" ]; then
  wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  unzip -q buffalo_l.zip -d models/ && rm buffalo_l.zip
fi

# inswapper_128.onnx — swap this URL for your own mirror if it 404s
if [ ! -f "inswapper_128.onnx" ]; then
  wget -q -O inswapper_128.onnx \
    https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
fi

# GFPGAN v1.4 (restoration)
if [ ! -f "GFPGANv1.4.pth" ]; then
  wget -q -O GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
fi

echo "[models] done"
ls -lh /models
