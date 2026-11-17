#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""b70-local: the image encoder for Inkling 2 (Shikigami), in NumPy on the CPU, speaking strata-vision's protocol
(serve/server.py Vision): started as `ink2_vision.py --mmproj <file> [--model <file>] [--threads N] [--max-tokens N]`,
it prints READY, then answers each `ENC <image> <out>` line with `OK <tokens>` (or `ERR <why>`) after writing one
strata-vision record to <out>: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd float32, one row per image token in
raster order; QUIT ends it.  The GPU stays with the language model.

The model (Inkling 2's vision tower, the same arithmetic as transformers' modeling_inkling2.py): the image is padded
(not resized) to 40-pixel cells, one language-model token a cell; each cell is 2 x 2 patches of 20 x 20 pixels; a
patch is a linear map of its 1200 values plus a position embedding resampled bilinearly from a 32 x 32 table; a class
token leads; layer norm, then 23 pre-norm ViT layers (16 heads, 2-D axial rotary positions on interleaved pairs, exact
GELU MLP, per-channel layer scales); a cell's four patch states (4096 values) go through a GELU MLP to 4608.
An image over --max-tokens cells (default 1024) is scaled down to fit first (the reference processor never resizes)."""
import argparse
import math
import os
import struct
import sys

import numpy as np

CELL = 40


def read_gguf(path):
    """-> (metadata dict, {name: float32 array in numpy order}) for a GGUF of F32 / F16 / BF16 tensors."""
    f = open(path, "rb")
    buf = memoryview(np.memmap(path, dtype=np.uint8, mode="r"))
    rd = f.read

    def u32():
        return struct.unpack("<I", rd(4))[0]

    def u64():
        return struct.unpack("<Q", rd(8))[0]

    def string():
        return rd(u64()).decode()

    scal = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def val(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = u32(), u64()
            return [val(et) for _ in range(n)]
        fmt = scal[t]
        return struct.unpack(fmt, rd(struct.calcsize(fmt)))[0]

    if rd(4) != b"GGUF":
        raise ValueError(f"{path} is not a GGUF")
    u32()
    nt, nk = u64(), u64()
    meta = {}
    for _ in range(nk):
        k = string()
        meta[k] = val(u32())
    infos = []
    for _ in range(nt):
        name = string()
        dims = [u64() for _ in range(u32())]
        infos.append((name, dims, u32(), u64()))
    align = int(meta.get("general.alignment", 32))
    base = (f.tell() + align - 1) // align * align
    f.close()
    w = {}
    for name, dims, ty, off in infos:
        n = int(np.prod(dims))
        shape = tuple(reversed(dims))
        if ty == 0:
            a = np.frombuffer(buf, np.float32, n, base + off).reshape(shape).copy()
        elif ty == 1:
            a = np.frombuffer(buf, np.float16, n, base + off).reshape(shape).astype(np.float32)
        elif ty == 30:
            a = (np.frombuffer(buf, np.uint16, n, base + off).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        else:
            raise ValueError(f"{name}: tensor type {ty} is not F32, F16 or BF16")
        w[name] = a
    return meta, w


def erf(x):
    """The error function from the Chebyshev fit of erfc (Numerical Recipes' erfcc, fractional error under 1.2e-7):
    NumPy has no erf, and the MLPs' GELU is the exact one."""
    t = 1.0 / (1.0 + 0.5 * np.abs(x))
    y = t * np.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, 1.0 - y, y - 1.0)


def gelu(x):
    return 0.5 * x * (1.0 + erf(x * np.float32(1.0 / math.sqrt(2.0)))).astype(np.float32)


def layer_norm(x, g, b, eps):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


class Encoder:
    def __init__(self, mmproj, max_tokens):
        meta, self.w = read_gguf(mmproj)
        if meta.get("general.architecture") != "inkling2_vision":
            raise ValueError(f"{mmproj} is not an Inkling 2 vision file")
        p = "inkling2_vision."
        self.D = int(meta[p + "embedding_length"])
        self.L = int(meta[p + "block_count"])
        self.H = int(meta[p + "attention.head_count"])
        self.eps = float(meta[p + "attention.layer_norm_epsilon"])
        self.patch = int(meta[p + "patch_size"])
        self.merge = int(meta[p + "spatial_merge_size"])
        self.side = int(meta[p + "posemb_grid_size"])
        self.theta = float(meta[p + "rope.freq_base"])
        self.n_embd = int(meta[p + "text_embedding_length"])
        self.mean = np.array(meta[p + "image_mean"], np.float32)
        self.std = np.array(meta[p + "image_std"], np.float32)
        self.max_tokens = max_tokens
        assert self.patch * self.merge == CELL

    # ---- the image -> cells of normalized pixels
    def pixels(self, path):
        from PIL import Image, ImageOps
        im = Image.open(path)
        im.load()
        im = ImageOps.exif_transpose(im)
        rgba = im.has_transparency_data
        im = im.convert("RGBA" if rgba else "RGB")
        rows, cols = math.ceil(im.height / CELL), math.ceil(im.width / CELL)
        if rows * cols > self.max_tokens:   # scaled down to fit (the reference processor never resizes)
            s = math.sqrt(self.max_tokens / (rows * cols))
            while True:
                w, h = max(1, int(im.width * s)), max(1, int(im.height * s))
                if math.ceil(h / CELL) * math.ceil(w / CELL) <= self.max_tokens:
                    break
                s *= 0.98
            im = im.resize((w, h), Image.Resampling.LANCZOS)
            rows, cols = math.ceil(im.height / CELL), math.ceil(im.width / CELL)
        a = np.asarray(im).astype(np.int32)                 # H, W, C
        if rgba:   # transparent parts over a checkerboard of 8-pixel squares (204 / 153), as the processor does
            y = np.arange(a.shape[0])[:, None] // 8
            x = np.arange(a.shape[1])[None, :] // 8
            bg = np.where((y + x) % 2 == 0, 204, 153)[:, :, None]
            al = a[:, :, 3:4]
            a = (a[:, :, :3] * al + bg * (255 - al) + 127) // 255
        img = np.full((rows * CELL, cols * CELL, 3), -1.0, np.float32)   # padded with -1 on the 0-255 scale
        img[:a.shape[0], :a.shape[1]] = a
        img = (img / 255.0 - self.mean) / self.std
        cells = img.reshape(rows, CELL, cols, CELL, 3).transpose(0, 2, 1, 3, 4).reshape(rows * cols, CELL, CELL, 3)
        return cells.astype(np.float32), rows, cols

    # ---- the cells -> the language model's rows
    def encode(self, path):
        w, D, H, m, P = self.w, self.D, self.H, self.merge, self.patch
        cells, rows, cols = self.pixels(path)
        n = rows * cols
        # patches, cell-major then the merge block's rows and columns; a patch's values (y, x, channel)
        patches = cells.reshape(n, m, P, m, P, 3).transpose(0, 1, 3, 2, 4, 5).reshape(n * m * m, P * P * 3)
        pe = patches @ w["vision_tower.embeddings.patch_embedding.weight"].T
        # each patch's grid row and column (block order)
        t = np.arange(n * m * m)
        cell, inner = t // (m * m), t % (m * m)
        prow = (cell // cols) * m + inner // m
        pcol = (cell % cols) * m + inner % m
        hp, wp = rows * m, cols * m

        def taps(idx, size):   # bilinear, align_corners=False, border (transformers' vision_utils)
            src = (idx.astype(np.float32) + 0.5) * self.side / size - 0.5
            fl = np.floor(src)
            raw = fl[:, None].astype(np.int64) + np.arange(2)[None, :]
            wt = np.clip(1 - np.abs(src[:, None] - fl[:, None] - np.arange(2)[None, :]), 0, None)
            return np.clip(raw, 0, self.side - 1), wt.astype(np.float32)

        ht, hw = taps(prow, hp)
        wt_, ww = taps(pcol, wp)
        table = w["vision_tower.embeddings.position_embedding.weight"]
        pos = np.zeros_like(pe)
        for i in range(2):
            for j in range(2):
                pos += table[ht[:, i] * self.side + wt_[:, j]] * (hw[:, i] * ww[:, j])[:, None]
        cls = w["vision_tower.embeddings.class_embedding"] + w["vision_tower.embeddings.cls_position_embedding"]
        x = np.concatenate([cls[None, :], pe + pos], 0).astype(np.float32)
        x = layer_norm(x, w["vision_tower.pre_layernorm.weight"], w["vision_tower.pre_layernorm.bias"], self.eps)
        # rotary: the class token at (0, 0), the patches at (row + 1, col + 1); pairs 0..15 the column, 16..31 the row
        hd = D // H
        inv = 1.0 / (self.theta ** (np.arange(0, hd // 2, 2, dtype=np.float32) / (hd // 2)))
        ph = np.concatenate([[0], prow + 1]).astype(np.float32)
        pw = np.concatenate([[0], pcol + 1]).astype(np.float32)
        ang = np.concatenate([pw[:, None] * inv[None, :], ph[:, None] * inv[None, :]], 1)    # S, hd / 2
        cos = np.repeat(np.cos(ang), 2, 1)[:, None, :].astype(np.float32)                   # S, 1, hd
        sin = np.repeat(np.sin(ang), 2, 1)[:, None, :].astype(np.float32)
        S = x.shape[0]

        def rope(v):   # v: S, H, hd; (x1, x2) -> (-x2, x1) on interleaved pairs
            r = np.empty_like(v)
            r[..., 0::2] = -v[..., 1::2]
            r[..., 1::2] = v[..., 0::2]
            return v * cos + r * sin

        sc = np.float32(hd ** -0.5)
        for li in range(self.L):
            p = f"vision_tower.layers.{li}."
            h = layer_norm(x, w[p + "layernorm_before.weight"], w[p + "layernorm_before.bias"], self.eps)
            q = (h @ w[p + "self_attn.q_proj.weight"].T + w[p + "self_attn.q_proj.bias"]).reshape(S, H, hd)
            k = (h @ w[p + "self_attn.k_proj.weight"].T + w[p + "self_attn.k_proj.bias"]).reshape(S, H, hd)
            v = (h @ w[p + "self_attn.v_proj.weight"].T + w[p + "self_attn.v_proj.bias"]).reshape(S, H, hd)
            q, k = rope(q).transpose(1, 0, 2), rope(k).transpose(1, 0, 2)
            v = v.transpose(1, 0, 2)
            o = np.empty((H, S, hd), np.float32)
            for hh in range(H):   # a head at a time: S x S scores
                a = (q[hh] @ k[hh].T) * sc
                a -= a.max(-1, keepdims=True)
                np.exp(a, out=a)
                a /= a.sum(-1, keepdims=True)
                o[hh] = a @ v[hh]
            o = o.transpose(1, 0, 2).reshape(S, D)
            o = o @ w[p + "self_attn.out_proj.weight"].T + w[p + "self_attn.out_proj.bias"]
            x = x + w[p + "lambda_1"] * o
            h = layer_norm(x, w[p + "layernorm_after.weight"], w[p + "layernorm_after.bias"], self.eps)
            h = gelu(h @ w[p + "mlp.fc1.weight"].T + w[p + "mlp.fc1.bias"])
            h = h @ w[p + "mlp.fc2.weight"].T + w[p + "mlp.fc2.bias"]
            x = (x + w[p + "lambda_2"] * h).astype(np.float32)
        y = x[1:].reshape(n, m * m * D)
        y = gelu(y @ w["vision_tower.merger.linear_1.weight"].T)
        y = y @ w["vision_tower.merger.linear_2.weight"].T
        return y.astype(np.float32), rows, cols


def write_record(path, y, rows, cols):
    with open(path + ".part", "wb") as f:
        f.write(struct.pack("<5i", 0x31455653, y.shape[0], cols, rows, y.shape[1]))
        f.write(np.ascontiguousarray(y, np.float32).tobytes())
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--model")                       # the server passes the language model; not read
    ap.add_argument("--threads", type=int)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--min-tokens", type=int)
    ap.add_argument("--gpu", action="store_true")    # accepted, the encoder runs on the CPU
    ap.add_argument("--gpu-pci")
    ap.add_argument("--encode", nargs=2, metavar=("IMAGE", "OUT"), help="encode one image and exit")
    a = ap.parse_args()
    enc = Encoder(a.mmproj, a.max_tokens)
    if a.encode:
        y, r, c = enc.encode(a.encode[0])
        write_record(a.encode[1], y, r, c)
        print(f"OK {y.shape[0]}")
        return 0
    print("READY", flush=True)
    for line in sys.stdin:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "QUIT":
            break
        if parts[0] != "ENC" or len(parts) != 3:
            print("ERR expected: ENC <image> <out>", flush=True)
            continue
        try:
            y, r, c = enc.encode(parts[1])
            write_record(parts[2], y, r, c)
            print(f"OK {y.shape[0]}", flush=True)
        except Exception as e:  # noqa: BLE001 - any failure is the request's, the encoder keeps running
            print(f"ERR {type(e).__name__}: {e}".replace("\n", " "), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
