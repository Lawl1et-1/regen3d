"""Fetch selected OBJ files from the BodyParts3D zip without downloading all 136 MB.

The DBCLS server supports HTTP range requests, so the zip is opened as a
seekable remote file: only the central directory and the entries we need are
transferred. Bytes already present in a partial local copy are reused.

    python targets/bp3d_fetch.py FMA7163 FMA5022 ...   (concept ids)
"""
from __future__ import annotations

import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/isa_BP3D_4.0_obj_99.zip"
SRC = Path(__file__).resolve().parents[1] / "assets_src"
CHUNK = 1 << 20


class RemoteFile(io.RawIOBase):
    def __init__(self, url: str, partial: Path | None):
        self.url = url
        req = urllib.request.Request(url, method="HEAD")
        self.size = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])
        self.pos = 0
        self.cache: dict[int, bytes] = {}
        self.local = partial.read_bytes() if partial and partial.exists() else b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _chunk(self, i: int) -> bytes:
        if i in self.cache:
            return self.cache[i]
        lo, hi = i * CHUNK, min(self.size, (i + 1) * CHUNK) - 1
        if hi < len(self.local):
            data = self.local[lo:hi + 1]
        else:
            for attempt in range(30):
                try:
                    req = urllib.request.Request(self.url, headers={"Range": f"bytes={lo}-{hi}"})
                    data = urllib.request.urlopen(req, timeout=60).read()
                    if len(data) == hi - lo + 1:
                        break
                except Exception as e:  # connection resets are frequent on this server
                    print(f"  retry chunk {i}: {e}", flush=True)
                time.sleep(min(2 + attempt, 10))
            else:
                raise RuntimeError(f"chunk {i} failed")
        self.cache[i] = data
        return data

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out = bytearray()
        while len(out) < n:
            i, o = divmod(self.pos + len(out), CHUNK)
            out += self._chunk(i)[o:o + n - len(out)]
        b[:n] = out
        self.pos += n
        return n


def element_files(concepts: list[str]) -> set[str]:
    out = set()
    for line in (SRC / "isa_element_parts.txt").read_text(encoding="utf-8").splitlines()[1:]:
        cid, _, fj = line.split("\t")
        if cid in concepts:
            out.add(fj.strip())
    return out


def main():
    concepts = sys.argv[1:]
    want = element_files(concepts)
    dest = SRC / "obj"
    dest.mkdir(exist_ok=True)
    todo = {w for w in want if not (dest / f"{w}.obj").exists()}
    print(f"{len(want)} element files wanted, {len(todo)} to fetch")
    rf = RemoteFile(URL, SRC / "isa_BP3D_4.0_obj_99.zip")
    z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=CHUNK))
    names = {Path(n).stem: n for n in z.namelist() if n.endswith(".obj")}
    for k, fj in enumerate(sorted(todo)):
        if fj not in names:
            print("missing", fj)
            continue
        (dest / f"{fj}.obj").write_bytes(z.read(names[fj]))
        if k % 20 == 0:
            print(f"  {k + 1}/{len(todo)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
