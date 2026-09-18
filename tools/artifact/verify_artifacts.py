import json, struct, mmap, sys
import numpy as np

PREFIX = struct.Struct("<8sQ")
MAGIC = b"NINFER\x00\x02"
TENSOR_ALIGNMENT = 256
PAYLOAD_ALIGNMENT = 4096

MTP_NAMES = [
    "mtp/input_projection",
    "mtp/layer/attention/query_key_gate_value",
    "mtp/layer/attention/output",
    "mtp/layer/mlp/gate_up",
    "mtp/layer/mlp/down",
]

def align_up(v, a):
    return (v + a - 1) & ~(a - 1)

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
            # MANDATORY 256-BYTE ALIGNMENT CHECK
            assert obj["offset"] % TENSOR_ALIGNMENT == 0, f"Tensor {obj['name']} relative offset not 256-aligned: {obj['offset']}"
            assert off % TENSOR_ALIGNMENT == 0, f"Tensor {obj['name']} absolute offset not 256-aligned: {off}"
        last_end = off + sz
        assert last_end <= file_size, f"Object {obj['name']} exceeds file size: {last_end} > {file_size}"

    print(f"All {tensor_count} tensor objects verified strictly 256-byte aligned (offset % 256 == 0).")
    print("All object offsets, alignments and file bounds verified OK.")
    return meta, file_size, m, payload_offset

meta_orig, sz_orig, mmap_orig, po_orig = inspect_artifact("/models/qwen3_8_27b_minq4.ninfer")
meta_new, sz_new, mmap_new, po_new = inspect_artifact("/models/qwen3_8_27b_minq4_mtpq4.ninfer")

objs_orig = {o["name"]: o for o in meta_orig["objects"]}
objs_new = {o["name"]: o for o in meta_new["objects"]}

assert set(objs_orig.keys()) == set(objs_new.keys()), "Object names set mismatch!"
print("\nObject names match 1:1 between original and new artifacts.")

# Check all non-MTP objects
non_mtp_diffs = 0
for name, o_orig in objs_orig.items():
    o_new = objs_new[name]
    if name not in MTP_NAMES:
        assert o_orig["kind"] == o_new["kind"]
        assert o_orig["bytes"] == o_new["bytes"]
        assert o_orig.get("shape") == o_new.get("shape")
        assert o_orig.get("format") == o_new.get("format")
        assert o_orig.get("layout") == o_new.get("layout")
        # In this repacked file, because all tensors are 256-aligned, non-MTP tensors before and after MTP match offsets if sizes aligned!
        sz = o_orig["bytes"]
        off_orig = po_orig + o_orig["offset"]
        off_new = po_new + o_new["offset"]
        check_sz = min(sz, 4096)
        if memoryview(mmap_orig)[off_orig : off_orig + check_sz] != memoryview(mmap_new)[off_new : off_new + check_sz]:
            print(f"Mismatch in data for {name}!")
            non_mtp_diffs += 1

assert non_mtp_diffs == 0, f"Found {non_mtp_diffs} non-MTP differences!"
print(f"Non-MTP objects: all {len(objs_orig) - len(MTP_NAMES)} verified identical in metadata and data slices.")

print("\n=== Checking 5 MTP Tensors ===")
total_orig_mtp_bytes = 0
total_new_mtp_bytes = 0

for name in MTP_NAMES:
    o_orig = objs_orig[name]
    o_new = objs_new[name]
    print(f"\nTensor: {name}")
    print(f"  Orig: format={o_orig['format']}, layout={o_orig['layout']}, shape={o_orig['shape']}, bytes={o_orig['bytes']:,}")
    print(f"  New:  format={o_new['format']}, layout={o_new['layout']}, shape={o_new['shape']}, bytes={o_new['bytes']:,}")
    
    assert o_orig["format"] == "W8G32_F16S"
    assert o_new["format"] == "Q4G64_F16S"
    assert o_orig["shape"] == o_new["shape"]
    assert o_orig["layout"] == "row-split-k128-v1"
    assert o_new["layout"] == "row-split-k128-v1"

    total_orig_mtp_bytes += o_orig["bytes"]
    total_new_mtp_bytes += o_new["bytes"]

    n, k = o_new["shape"]
    groups_per_row = (k + 63) // 64
    off_new = po_new + o_new["offset"]
    
    base_bytes = n * groups_per_row * 32
    scale_offset_rel = align_up(base_bytes, TENSOR_ALIGNMENT)
    scale_bytes = n * groups_per_row * 2
    
    assert o_new["bytes"] == scale_offset_rel + scale_bytes, f"Size formula mismatch for {name}"

    view_new = memoryview(mmap_new)[off_new : off_new + o_new["bytes"]]
    raw_scales = np.frombuffer(view_new[scale_offset_rel : scale_offset_rel + scale_bytes], dtype=np.float16)
    assert np.all(np.isfinite(raw_scales)), f"Non-finite scales in {name}"
    assert np.all(raw_scales >= 0), f"Negative scales in {name}"
    print(f"  Scales check: count={len(raw_scales):,}, min={raw_scales.min():.6e}, max={raw_scales.max():.6e}, mean={raw_scales.mean():.6e} -> OK")

print("\n=== Summary ===")
print(f"Total MTP W8 bytes: {total_orig_mtp_bytes:,} ({total_orig_mtp_bytes/(1024**2):.2f} MiB)")
print(f"Total MTP Q4 bytes: {total_new_mtp_bytes:,} ({total_new_mtp_bytes/(1024**2):.2f} MiB)")
saved_bytes = total_orig_mtp_bytes - total_new_mtp_bytes
print(f"Saved MTP bytes:    {saved_bytes:,} ({saved_bytes/(1024**2):.2f} MiB / {saved_bytes/1e6:.2f} MB)")
print(f"Original file size: {sz_orig:,} ({sz_orig/(1024**2):.2f} MiB)")
print(f"New file size:      {sz_new:,} ({sz_new/(1024**2):.2f} MiB)")
print(f"Net file savings:   {sz_orig - sz_new:,} ({(sz_orig - sz_new)/(1024**2):.2f} MiB)")

print("\nALL 256-BYTE ALIGNMENT AND INTEGRITY CHECKS PASSED!")
