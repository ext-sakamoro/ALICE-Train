#!/usr/bin/env python3
"""Write the `.pth` ZIP corpus of tests/pth_zip_corpus.rs.

Each archive has the layout `torch.save` writes (`archive/data.pkl` plus one
`archive/data/<key>` entry per storage) with a pickle in the form
`src/pth_loader.rs` reads: a dict of `torch._utils._rebuild_tensor_v2`
reductions over `torch.<Type>Storage(key, location, numel)`. The archives vary
what the ZIP reader has to handle: stored and deflated entries, an extra field
that pads entry data to 64 bytes (as torch's writer does), a storage large
enough for several deflate blocks, two tensors sharing one storage at an
offset, storages written before data.pkl, directory entries with an archive
comment, and ZIP64 records.

The expected values are computed here, from the bytes this script writes, not
by the Rust reader: `expected.tsv` holds, per archive and tensor, the dtype,
the shape and the SHA-256 of the tensor's bytes.

    python3 scripts/gen_pth_zip_corpus.py            write the corpus
    python3 scripts/gen_pth_zip_corpus.py --check    fail if the files differ

The output is deterministic (fixed timestamps, fixed values), so --check
compares byte for byte.
"""
from __future__ import annotations

import hashlib
import io
import pickle
import struct
import sys
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "pth_zip"
DATE = (2020, 1, 1, 0, 0, 0)

# stand-ins for the torch objects, registered under torch's module names so
# pickle writes the same GLOBAL opcodes the reader looks for
_torch = types.ModuleType("torch")
_utils = types.ModuleType("torch._utils")


def _rebuild_tensor_v2(*args):  # pragma: no cover - never called, only pickled by name
    raise RuntimeError("stand-in")


_rebuild_tensor_v2.__module__ = "torch._utils"
_utils._rebuild_tensor_v2 = _rebuild_tensor_v2
_torch._utils = _utils
sys.modules.setdefault("torch", _torch)
sys.modules.setdefault("torch._utils", _utils)

# (storage class, dtype name in the reader, struct format, element size)
DTYPES = {
    "f32": ("FloatStorage", "Float32", "<f", 4),
    "f64": ("DoubleStorage", "Float64", "<d", 8),
    "f16": ("HalfStorage", "Float16", "<e", 2),
    "bf16": ("BFloat16Storage", "BFloat16", None, 2),
    "i64": ("LongStorage", "Int64", "<q", 8),
    "i32": ("IntStorage", "Int32", "<i", 4),
    "i8": ("CharStorage", "Int8", "<b", 1),
    "u8": ("ByteStorage", "Uint8", "<B", 1),
}


def storage_class(name: str) -> type:
    if not hasattr(_torch, name):
        cls = type(name, (), {})
        cls.__module__ = "torch"
        setattr(_torch, name, cls)
    return getattr(_torch, name)


class Storage:
    def __init__(self, kind: str, key: str, numel: int):
        self.kind, self.key, self.numel = kind, key, numel

    def __reduce__(self):
        return (storage_class(DTYPES[self.kind][0]), (self.key, "cpu", self.numel))


class Tensor:
    def __init__(self, storage: Storage, offset: int, shape: tuple):
        self.storage, self.offset, self.shape = storage, offset, shape

    def __reduce__(self):
        stride, acc = [], 1
        for d in reversed(self.shape):
            stride.insert(0, acc)
            acc *= d
        return (_rebuild_tensor_v2, (self.storage, self.offset, self.shape, tuple(stride), False, {}))


def values(kind: str, n: int, seed: int) -> bytes:
    """`n` elements of `kind` from a fixed integer sequence."""
    fmt, size = DTYPES[kind][2], DTYPES[kind][3]
    out = bytearray()
    for i in range(n):
        k = (i * 2654435761 + seed * 40503) % 65521
        if kind == "bf16":
            out += struct.pack("<H", (struct.unpack("<I", struct.pack("<f", (k - 32760) / 97.0))[0] >> 16))
        elif kind in ("f32", "f64", "f16"):
            out += struct.pack(fmt, (k - 32760) / 97.0)
        elif kind == "u8":
            out += struct.pack(fmt, k % 256)
        elif kind == "i8":
            out += struct.pack(fmt, k % 256 - 128)
        else:
            out += struct.pack(fmt, k - 32760)
    assert len(out) == n * size
    return bytes(out)


def pad_extra(offset_of_data: int) -> bytes:
    """An extra field (id 0x4650, as torch writes) that moves entry data to a 64-byte boundary."""
    need = (-(offset_of_data + 4)) % 64
    return struct.pack("<HH", 0x4650, need) + b"Z" * need


def build(spec: dict) -> tuple[bytes, list[tuple[str, str, tuple, str]]]:
    """(archive bytes, [(tensor, dtype, shape, sha256 of its bytes)])."""
    storages: dict[str, tuple[str, bytes]] = {}
    state, rows = {}, []
    for name, kind, key, offset, shape, numel, seed in spec["tensors"]:
        if key not in storages:
            storages[key] = (kind, values(kind, numel, seed))
        st = Storage(kind, key, numel)
        state[name] = Tensor(st, offset, shape)
        count = 1
        for d in shape:
            count *= d
        size = DTYPES[kind][3]
        raw = storages[key][1][offset * size:(offset + count) * size]
        assert len(raw) == count * size
        rows.append((name, DTYPES[kind][1], shape, hashlib.sha256(raw).hexdigest()))
    pkl = pickle.dumps(state, protocol=2)

    buf = io.BytesIO()
    compression = zipfile.ZIP_DEFLATED if spec.get("deflate") else zipfile.ZIP_STORED
    with zipfile.ZipFile(buf, "w", compression=compression) as zf:
        if spec.get("comment"):
            zf.comment = spec["comment"]
        entries = [("archive/data.pkl", pkl)] + [(f"archive/data/{k}", v[1]) for k, v in storages.items()]
        if spec.get("data_first"):
            entries = entries[1:] + entries[:1]
        if spec.get("dirs"):
            for d in ("archive/", "archive/data/"):
                zf.writestr(zipfile.ZipInfo(d, DATE), b"")
        for path, data in entries:
            info = zipfile.ZipInfo(path, DATE)
            info.compress_type = compression
            if spec.get("align"):
                info.extra = pad_extra(buf.tell() + 30 + len(path.encode()))
            with zf.open(info, "w", force_zip64=bool(spec.get("zip64"))) as f:
                f.write(data)
    return buf.getvalue(), rows


SPECS = {
    "stored": {"tensors": [
        ("w", "f32", "0", 0, (3, 4), 12, 1),
        ("b", "f32", "1", 0, (4,), 4, 2),
    ]},
    "deflated": {"deflate": True, "tensors": [
        ("layer.weight", "f16", "0", 0, (8, 8), 64, 3),
        ("layer.bias", "bf16", "1", 0, (8,), 8, 4),
        ("steps", "i64", "2", 0, (1,), 1, 5),
    ]},
    "aligned": {"align": True, "tensors": [
        ("a", "f64", "0", 0, (5,), 5, 6),
        ("b", "i32", "1", 0, (2, 3), 6, 7),
        ("c", "u8", "2", 0, (7,), 7, 8),
        ("d", "i8", "3", 0, (9,), 9, 9),
    ]},
    "large": {"deflate": True, "tensors": [
        ("emb", "f32", "0", 0, (256, 256), 65536, 10),
    ]},
    "shared_offset": {"tensors": [
        ("first", "f32", "0", 0, (4, 2), 20, 11),
        ("second", "f32", "0", 8, (3, 4), 20, 11),
    ]},
    "data_first_dirs_comment": {"deflate": True, "data_first": True, "dirs": True,
                                "comment": b"corpus archive comment", "tensors": [
        ("x", "f32", "0", 0, (6,), 6, 12),
        ("y", "f16", "1", 0, (2, 2), 4, 13),
    ]},
    "zip64": {"zip64": True, "tensors": [
        ("z", "f64", "0", 0, (3, 3), 9, 14),
    ]},
}


def render() -> dict[str, bytes]:
    files, lines = {}, ["# archive\ttensor\tdtype\tshape\tsha256 of the tensor's bytes (written by gen_pth_zip_corpus.py)"]
    for name, spec in SPECS.items():
        data, rows = build(spec)
        files[f"{name}.pth"] = data
        for tensor, dtype, shape, digest in sorted(rows):
            lines.append(f"{name}.pth\t{tensor}\t{dtype}\t{'x'.join(map(str, shape))}\t{digest}")
    files["expected.tsv"] = ("\n".join(lines) + "\n").encode()
    return files


def main(argv: list[str]) -> int:
    files = render()
    if "--check" in argv:
        bad = [n for n, d in files.items() if not (OUT / n).exists() or (OUT / n).read_bytes() != d]
        extra = sorted(p.name for p in OUT.glob("*") if p.name not in files) if OUT.exists() else []
        for n in bad:
            print(f"error: tests/fixtures/pth_zip/{n} differs from the generator", file=sys.stderr)
        for n in extra:
            print(f"error: tests/fixtures/pth_zip/{n} is not written by the generator", file=sys.stderr)
        print(f"gen_pth_zip_corpus: {len(files)} files compared, {len(bad) + len(extra)} problems")
        return 1 if bad or extra or not files else 0
    OUT.mkdir(parents=True, exist_ok=True)
    for n, d in files.items():
        (OUT / n).write_bytes(d)
    print(f"wrote {len(files)} files to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
