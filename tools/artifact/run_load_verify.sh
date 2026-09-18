#!/bin/bash
set -e
TARGET_MODEL="${1:-/models/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer}"
MODEL_BASENAME=$(basename "$TARGET_MODEL")

docker run --rm --net=none -v /ninfer:/ninfer:ro -v /models:/models:ro ninfer-mtpq4:build bash -c "
set -e
g++ -O3 -std=gnu++20 /ninfer/load_verify.cpp \
    -I/src/src -I/src/include -I/usr/local/cuda/include \
    -I/src/src/targets/qwen3_6/export -I/src/src/targets/qwen3_6_27b/export \
    /build/src/libninfer_engine.a /build/src/libninfer_artifact.a \
    /build/src/libninfer_ops.a /build/src/libninfer_nvfp4_tma.a /build/src/libninfer_core.a \
    /build/src/libninfer_text.a /build/src/libninfer_media_decode.a /build/src/libninfer_media_acquire.a \
    -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart -lavformat -lavcodec -lavutil -lswscale -lcurl -ldl -lpthread \
    -o /tmp/load_verify
/tmp/load_verify /models/$MODEL_BASENAME
"
