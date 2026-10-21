"""Valve VPK (v1/v2) reader and a minimal single-file v1 writer."""

from __future__ import annotations

import os
import struct
import zlib

VPK_SIGNATURE = 0x55AA1234
EMBEDDED = 0x7FFF


class VPKEntry:
    __slots__ = ("path", "crc", "preload", "archive_index", "offset", "length")

    def __init__(self, path, crc, preload, archive_index, offset, length):
        self.path = path
        self.crc = crc
        self.preload = preload
        self.archive_index = archive_index
        self.offset = offset
        self.length = length


def _read_cstr(f) -> str:
    out = bytearray()
    while True:
        c = f.read(1)
        if not c or c == b"\0":
            break
        out += c
    return out.decode("utf-8", errors="replace")


class VPK:
    """Read-only access to a ``*_dir.vpk`` archive set."""

    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.entries: dict[str, VPKEntry] = {}
        with open(dir_path, "rb") as f:
            sig, version, tree_size = struct.unpack("<III", f.read(12))
            if sig != VPK_SIGNATURE:
                raise ValueError(f"{dir_path}: not a VPK file")
            if version == 1:
                header_size = 12
            elif version == 2:
                f.read(16)
                header_size = 28
            else:
                raise ValueError(f"{dir_path}: unsupported VPK version {version}")
            self._data_start = header_size + tree_size
            while True:
                ext = _read_cstr(f)
                if not ext:
                    break
                while True:
                    path = _read_cstr(f)
                    if not path:
                        break
                    while True:
                        name = _read_cstr(f)
                        if not name:
                            break
                        crc, preload_len, arch, off, length, term = struct.unpack("<IHHIIH", f.read(18))
                        if term != 0xFFFF:
                            raise ValueError(f"{dir_path}: corrupt directory entry")
                        preload = f.read(preload_len) if preload_len else b""
                        full = name if ext == " " else f"{name}.{ext}"
                        if path.strip() and path != " ":
                            full = f"{path}/{full}"
                        full = full.lower().replace("\\", "/")
                        self.entries[full] = VPKEntry(full, crc, preload, arch, off, length)

    def __contains__(self, path: str) -> bool:
        return path.lower().replace("\\", "/") in self.entries

    def find(self, predicate):
        return [p for p in self.entries if predicate(p)]

    def read(self, path: str) -> bytes:
        e = self.entries[path.lower().replace("\\", "/")]
        data = e.preload
        if e.length:
            if e.archive_index == EMBEDDED:
                arch_path, off = self.dir_path, self._data_start + e.offset
            else:
                base = self.dir_path[: -len("_dir.vpk")]
                arch_path, off = f"{base}_{e.archive_index:03d}.vpk", e.offset
            with open(arch_path, "rb") as f:
                f.seek(off)
                data += f.read(e.length)
        return data


def write_vpk(out_path: str, files: dict[str, bytes]) -> None:
    """Write a self-contained VPK v1 (all data stored inside the _dir file)."""
    tree: dict[str, dict[str, list[tuple[str, bytes]]]] = {}
    for path, data in files.items():
        path = path.lower().replace("\\", "/")
        directory, _, filename = path.rpartition("/")
        name, dot, ext = filename.rpartition(".")
        if not dot:
            name, ext = filename, " "
        tree.setdefault(ext, {}).setdefault(directory or " ", []).append((name, data))

    tree_bytes = bytearray()
    blobs = bytearray()
    for ext in sorted(tree):
        tree_bytes += ext.encode() + b"\0"
        for directory in sorted(tree[ext]):
            tree_bytes += directory.encode() + b"\0"
            for name, data in sorted(tree[ext][directory]):
                tree_bytes += name.encode() + b"\0"
                crc = zlib.crc32(data) & 0xFFFFFFFF
                tree_bytes += struct.pack("<IHHIIH", crc, 0, EMBEDDED, len(blobs), len(data), 0xFFFF)
                blobs += data
            tree_bytes += b"\0"
        tree_bytes += b"\0"
    tree_bytes += b"\0"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", VPK_SIGNATURE, 1, len(tree_bytes)))
        f.write(tree_bytes)
        f.write(blobs)
