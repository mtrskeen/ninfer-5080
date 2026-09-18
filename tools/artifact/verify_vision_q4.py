import json, struct, mmap, sys
import numpy as np

PREFIX = struct.Struct("<8sQ")
MAGIC = b"NINFER\x00\x02"
TENSOR_ALIGNMENT = 256
PAYLOAD_ALIGNMENT = 4096

def align_up(v, a):
    return (v + a - 1) & ~(a - 1)

def is_vision_target(name: str) -> bool:
    return name.startswith("vision/layers/") and (
        name.endswith("/attention/output") or name.endswith("/mlp/fc2")
    )

def inspect_artifact(path):
    print(f"=== Inspecting {path} ===")
    with open(path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        prefix_bytes = f.read(PREFIX.size)
        magic, json_len = PREFIX.unpack(prefix_bytes)
        assert magic == MAGIC, f"Bad magic: {magic}"
        meta = json.loads(f.read(json_len).decode("utf-8"))
        payload_offset = align_up(PREFIX.size + json_len, PAYLOAD_ALIGNMENT)
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    print(f"File size: {file_size:,} bytes ({file_size/(1024**3):.4f} GiB)")
    print(f"JSON header length: {json_len:,} bytes, payload offset: {payload_offset} (payload%4096={payload_offset%4096})")
    assert payload_offset % PAYLOAD_ALIGNMENT == 0, f"Payload offset not {PAYLOAD_ALIGNMENT}-aligned!"
    print(f"Identity: {meta.get('identity')}")
    objects = meta["objects"]
    print(f"Total objects: {len(objects)}")

    last_end = payload_offset
    tensor_count = 0
    for idx, obj in enumerate(objects):
        off = payload_offset + obj["offset"]
        sz = obj["bytes"]
        assert off >= last_end, f"Overlap at object {obj['name']}: off={off}, last_end={last_end}"
        if obj["kind"] == "tensor":
            tensor_count += 1
            assert obj["offset"] % TENSOR_ALIGNMENT == 0, f"Tensor {obj['name']} relative offset not 256-aligned: {obj['offset']}"
            assert off % TENSOR_ALIGNMENT == 0, f"Tensor {obj['name']} absolute offset not 256-aligned: {off}"
        last_end = off + sz
        assert last_end <= file_size, f"Object {obj['name']} exceeds file size: {last_end} > {file_size}"

    print(f"All {tensor_count} tensor objects verified strictly 256-byte aligned (offset % 256 == 0).")
    print("All object offsets, alignments and file bounds verified OK.")
    return meta, file_size, m, payload_offset

meta_src, sz_src, mmap_src, po_src = inspect_artifact("/models/qwen3_8_27b_minq4_mtpq4.ninfer")
meta_dst, sz_dst, mmap_dst, po_dst = inspect_artifact("/models/qwen3_8_27b_minq4_mtpq4_visionq4.ninfer")

objs_src = {o["name"]: o for o in meta_src["objects"]}
objs_dst = {o["name"]: o for o in meta_dst["objects"]}

assert set(objs_src.keys()) == set(objs_dst.keys()), "Object names set mismatch!"
print("\nObject names match 1:1 between src (MTP-Q4) and dst (Vision-Q4) artifacts.")

target_names = {o["name"] for o in meta_src["objects"] if is_vision_target(o["name"]) and o.get("format") == "Q5G64_F16S"}
print(f"Vision Q5->Q4 target tensors count: {len(target_names)}")
assert len(target_names) == 54, f"Expected 54 targets, got {len(target_names)}"

# Check all non-target objects
non_target_diffs = 0
for name, o_src in objs_src.items():
    o_dst = objs_dst[name]
    if name not in target_names:
        assert o_src["kind"] == o_dst["kind"]
        assert o_src["bytes"] == o_dst["bytes"]
        assert o_src.get("shape") == o_dst.get("shape")
        assert o_src.get("format") == o_dst.get("format")
        assert o_src.get("layout") == o_dst.get("layout")

        sz = o_src["bytes"]
        off_src = po_src + o_src["offset"]
        off_dst = po_dst + o_dst["offset"]
        # Check first 64 KiB or full tensor
        check_sz = min(sz, 65536)
        if memoryview(mmap_src)[off_src : off_src + check_sz] != memoryview(mmap_dst)[off_dst : off_dst + check_sz]:
            print(f"Mismatch in data for {name}!")
            non_target_diffs += 1

assert non_target_diffs == 0, f"Found {non_target_diffs} non-target differences!"
print(f"Non-target objects: all {len(objs_src) - len(target_names)} verified identical in metadata and data slices.")

print("\n=== Verifying 54 Vision-Q4 Tensors ===")
total_src_target_bytes = 0
total_dst_target_bytes = 0

for name in sorted(target_names):
    o_src = objs_src[name]
    o_dst = objs_dst[name]
    assert o_src["format"] == "Q5G64_F16S"
    assert o_dst["format"] == "Q4G64_F16S"
    assert o_src["shape"] == o_dst["shape"]
    assert o_dst["layout"] == "row-split-k128-v1"

    total_src_target_bytes += o_src["bytes"]
    total_dst_target_bytes += o_dst["bytes"]

    n, k = o_dst["shape"]
    k_pad = (k + 127) & ~127
    groups_per_row = k_pad // 64
    off_dst = po_dst + o_dst["offset"]

    base_bytes = n * groups_per_row * 32
    scale_offset_rel = align_up(base_bytes, TENSOR_ALIGNMENT)
    scale_bytes = n * groups_per_row * 2

    assert o_dst["bytes"] == scale_offset_rel + scale_bytes, f"Size formula mismatch for {name}"

    view_dst = memoryview(mmap_dst)[off_dst : off_dst + o_dst["bytes"]]
    raw_scales = np.frombuffer(view_dst[scale_offset_rel : scale_offset_rel + scale_bytes], dtype=np.float16)
    assert np.all(np.isfinite(raw_scales)), f"Non-finite scales in {name}"
    assert np.all(raw_scales >= 0), f"Negative scales in {name}"

print(f"All 54 Vision targets verified: valid Q4G64_F16S row-split geometry, finite & positive scales.")

print("\n=== Summary ===")
print(f"Vision target Q5 bytes: {total_src_target_bytes:,} ({total_src_target_bytes/(1024**2):.2f} MiB)")
print(f"Vision target Q4 bytes: {total_dst_target_bytes:,} ({total_dst_target_bytes/(1024**2):.2f} MiB)")
saved_bytes = total_src_target_bytes - total_dst_target_bytes
print(f"Vision weight savings:  {saved_bytes:,} ({saved_bytes/(1024**2):.2f} MiB / {saved_bytes/1e6:.2f} MB)")
print(f"Source file size:       {sz_src:,} ({sz_src/(1024**2):.2f} MiB)")
print(f"Destination file size:  {sz_dst:,} ({sz_dst/(1024**2):.2f} MiB)")
print(f"Net file savings:       {sz_src - sz_dst:,} ({(sz_src - sz_dst)/(1024**2):.2f} MiB)")

print("\nALL VISION-Q4 CHECKS AND 256-BYTE ALIGNMENTS PASSED SUCCESSFULLY!")
