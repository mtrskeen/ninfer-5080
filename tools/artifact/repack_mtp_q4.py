import argparse
import json
import mmap
import os
import struct
import time
import numpy as np

MAGIC = b"NINFER\x00\x02"
PREFIX = struct.Struct("<8sQ")
TENSOR_ALIGNMENT = 256
PLANE_ALIGNMENT = 256
PAYLOAD_ALIGNMENT = 4096

MTP_REQUANT_NAMES = {
    "mtp/input_projection",
    "mtp/layer/attention/query_key_gate_value",
    "mtp/layer/attention/output",
    "mtp/layer/mlp/gate_up",
    "mtp/layer/mlp/down",
}


def align_up(v: int, a: int) -> int:
    return (v + a - 1) & ~(a - 1)


def row_split_geometry(format_name: str, n: int, k: int) -> dict:
    k_pad = align_up(k, 128)
    if format_name == "W8G32_F16S":
        group_size = 32
        bits = 8
    elif format_name == "Q4G64_F16S":
        group_size = 64
        bits = 4
    elif format_name == "Q5G64_F16S":
        group_size = 64
        bits = 5
    else:
        raise ValueError(f"Unsupported format: {format_name}")

    groups_per_row = k_pad // group_size
    base_bytes_per_group = group_size if bits == 8 else group_size // 2
    high_bytes_per_group = 0 if bits in (4, 8) else group_size * (bits - 4) // 8

    base_row_bytes = groups_per_row * base_bytes_per_group
    high_row_bytes = groups_per_row * high_bytes_per_group
    scale_row_bytes = groups_per_row * 2

    base_bytes = n * base_row_bytes
    high_bytes = n * high_row_bytes
    scale_bytes = n * scale_row_bytes

    high_offset = align_up(base_bytes, PLANE_ALIGNMENT)
    scale_offset = high_offset + align_up(high_bytes, PLANE_ALIGNMENT)
    payload_bytes = scale_offset + scale_bytes

    return {
        "n": n,
        "k": k,
        "k_pad": k_pad,
        "group_size": group_size,
        "bits": bits,
        "groups_per_row": groups_per_row,
        "base_row_bytes": base_row_bytes,
        "high_row_bytes": high_row_bytes,
        "scale_row_bytes": scale_row_bytes,
        "base_offset": 0,
        "base_bytes": base_bytes,
        "high_offset": high_offset,
        "high_bytes": high_bytes,
        "scale_offset": scale_offset,
        "scale_bytes": scale_bytes,
        "payload_bytes": payload_bytes,
    }


def dequant_w8g32(payload: memoryview, n: int, k: int) -> np.ndarray:
    geom = row_split_geometry("W8G32_F16S", n, k)
    base_data = np.frombuffer(payload[geom["base_offset"] : geom["base_offset"] + geom["base_bytes"]], dtype=np.int8)
    scale_data = np.frombuffer(payload[geom["scale_offset"] : geom["scale_offset"] + geom["scale_bytes"]], dtype=np.float16)

    codes = base_data.reshape(n, geom["groups_per_row"], 32)
    scales = scale_data.reshape(n, geom["groups_per_row"], 1).astype(np.float32)

    fp32_matrix = (codes.astype(np.float32) * scales).reshape(n, geom["k_pad"])
    return fp32_matrix[:, :k]


def quant_and_encode_q4g64(fp32_matrix: np.ndarray) -> bytes:
    n, k = fp32_matrix.shape
    geom = row_split_geometry("Q4G64_F16S", n, k)

    if geom["k_pad"] != k:
        padded = np.zeros((n, geom["k_pad"]), dtype=np.float32)
        padded[:, :k] = fp32_matrix
    else:
        padded = fp32_matrix

    grouped = padded.reshape(n, geom["groups_per_row"], 64)
    max_abs = np.abs(grouped).max(axis=2)

    qmax = 7
    qmin = -8
    raw_scale = (max_abs.astype(np.float64) / float(qmax)).astype(np.float32)
    scales = raw_scale.astype(np.float16)

    min_subnormal = np.float16(2.0**-24)
    underflow = (scales == 0) & (max_abs > 0)
    if underflow.any():
        scales[underflow] = min_subnormal

    reciprocal = np.zeros_like(max_abs, dtype=np.float32)
    pos = scales > 0
    reciprocal[pos] = (1.0 / scales[pos].astype(np.float64)).astype(np.float32)

    codes = np.clip(
        np.round(grouped * reciprocal[:, :, None]), qmin, qmax
    ).astype(np.int8)

    u = codes.astype(np.uint8) & 0x0F
    low_nibbles = u[:, :, 0::2] | (u[:, :, 1::2] << 4)
    base_bytes = low_nibbles.tobytes()
    scale_bytes = scales.tobytes()

    payload = bytearray(geom["payload_bytes"])
    payload[: len(base_bytes)] = base_bytes
    payload[geom["scale_offset"] : geom["scale_offset"] + len(scale_bytes)] = scale_bytes
    return bytes(payload)


def repack(src_path: str, dst_path: str):
    print(f"Opening source artifact: {src_path}")
    t0 = time.time()
    with open(src_path, "rb") as f_src:
        magic, json_len = PREFIX.unpack(f_src.read(PREFIX.size))
        if magic != MAGIC:
            raise ValueError("Invalid magic in source artifact")
        src_meta = json.loads(f_src.read(json_len).decode("utf-8"))
        src_payload_offset = align_up(PREFIX.size + json_len, PAYLOAD_ALIGNMENT)

        src_mmap = mmap.mmap(f_src.fileno(), 0, access=mmap.ACCESS_READ)

    print(f"Source metadata loaded in {time.time() - t0:.2f}s")
    identity = src_meta["identity"]
    objects = src_meta["objects"]

    print(f"Source model_id: {identity['model_id']}, weights_id: {identity['weights_id']}")
    print(f"Total objects: {len(objects)}")

    new_objects = []
    encoded_payloads = {}

    total_w8_bytes = 0
    total_q4_bytes = 0

    cursor = 0
    for obj in objects:
        name = obj["name"]
        kind = obj["kind"]

        if kind == "tensor":
            cursor = align_up(cursor, TENSOR_ALIGNMENT)
        else:
            cursor = align_up(cursor, 1)

        if name in MTP_REQUANT_NAMES:
            print(f"Requantizing {name} ({obj['format']} -> Q4G64_F16S)...")
            shape = tuple(obj["shape"])
            offset = obj["offset"]
            obj_bytes = obj["bytes"]
            total_w8_bytes += obj_bytes

            raw_view = memoryview(src_mmap)[src_payload_offset + offset : src_payload_offset + offset + obj_bytes]
            fp32_mat = dequant_w8g32(raw_view, shape[0], shape[1])
            q4_payload = quant_and_encode_q4g64(fp32_mat)
            total_q4_bytes += len(q4_payload)

            geom = row_split_geometry("Q4G64_F16S", shape[0], shape[1])
            assert len(q4_payload) == geom["payload_bytes"]

            new_obj = {
                "name": name,
                "kind": "tensor",
                "shape": list(shape),
                "format": "Q4G64_F16S",
                "layout": "row-split-k128-v1",
                "offset": cursor,
                "bytes": len(q4_payload),
            }
            encoded_payloads[name] = q4_payload
            cursor += len(q4_payload)
            new_objects.append(new_obj)
        else:
            new_obj = dict(obj)
            new_obj["offset"] = cursor
            cursor += obj["bytes"]
            new_objects.append(new_obj)

    print(f"MTP W8 size: {total_w8_bytes / (1024*1024):.2f} MB")
    print(f"MTP Q4 size: {total_q4_bytes / (1024*1024):.2f} MB")
    print(f"MTP saved on disk/weights: {(total_w8_bytes - total_q4_bytes) / (1024*1024):.2f} MB")

    new_meta = {
        "identity": {
            "model_id": identity["model_id"],
            "weights_id": identity["weights_id"],
        },
        "objects": new_objects,
    }
    new_json_bytes = json.dumps(new_meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    new_json_len = len(new_json_bytes)
    new_prefix = PREFIX.pack(MAGIC, new_json_len)
    new_payload_offset = align_up(PREFIX.size + new_json_len, PAYLOAD_ALIGNMENT)

    # Double-check all tensor offsets are 256-aligned relative to container beginning and payload
    for obj in new_objects:
        if obj["kind"] == "tensor":
            assert obj["offset"] % TENSOR_ALIGNMENT == 0, f"Tensor {obj['name']} offset not 256 aligned: {obj['offset']}"
            assert (new_payload_offset + obj["offset"]) % TENSOR_ALIGNMENT == 0, f"Tensor {obj['name']} absolute offset not 256 aligned"

    print(f"All {len(new_objects)} objects verified 256-byte aligned. Writing {dst_path}...")
    tmp_dst = dst_path + ".tmp"
    t_write = time.time()
    with open(tmp_dst, "wb") as f_dst:
        f_dst.write(new_prefix)
        f_dst.write(new_json_bytes)
        pad = new_payload_offset - f_dst.tell()
        if pad > 0:
            f_dst.write(b"\x00" * pad)

        total_objs = len(objects)
        for idx, (old_obj, new_obj) in enumerate(zip(objects, new_objects)):
            name = old_obj["name"]
            dest_offset = new_payload_offset + new_obj["offset"]
            cur_pos = f_dst.tell()
            if dest_offset > cur_pos:
                f_dst.write(b"\x00" * (dest_offset - cur_pos))

            if name in encoded_payloads:
                f_dst.write(encoded_payloads[name])
            else:
                src_off = src_payload_offset + old_obj["offset"]
                obj_bytes = old_obj["bytes"]
                chunk_sz = 64 * 1024 * 1024
                for c_start in range(0, obj_bytes, chunk_sz):
                    c_end = min(obj_bytes, c_start + chunk_sz)
                    f_dst.write(memoryview(src_mmap)[src_off + c_start : src_off + c_end])

            if (idx + 1) % 100 == 0 or idx == total_objs - 1:
                print(f"Written {idx + 1}/{total_objs} objects ({(f_dst.tell() / (1024*1024*1024)):.2f} GB)...")

    os.replace(tmp_dst, dst_path)
    print(f"Done! Written {dst_path} in {time.time() - t_write:.2f}s. Total time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repack MTP matrices to Q4G64_F16S with 256-byte alignment")
    parser.add_argument("--src", required=True, help="Path to input .ninfer")
    parser.add_argument("--dst", required=True, help="Path to output .ninfer")
    args = parser.parse_args()
    repack(args.src, args.dst)
