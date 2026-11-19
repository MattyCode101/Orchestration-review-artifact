#!/usr/bin/env python3
"""Download the FPL 2026 contest benchmarks (release v1.3.0) and verify the archive.

usage: python scripts/fetch_benchmarks.py TARGET_DIR

The archive is about 743 MB. It is downloaded to TARGET_DIR, checked against the SHA-256 that GitHub publishes for the release asset,
and unpacked into TARGET_DIR/fpl26_contest_benchmarks_v1.3.0/. Nothing is unpacked if the checksum differs.
Point ORCH_BENCHMARK_DIR at the unpacked folder. No benchmark file is redistributed by this repository.
"""
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

URL = "https://github.com/Xilinx/fpl26_optimization_contest/releases/download/v1.3.0/fpl26_contest_benchmarks_v1.3.0.tar.gz"
SHA256 = "14d66b7b32171f20be9aa26eb395fdb75b881555854dfa2fa7b870f30d8b71ac"
SIZE = 743289124


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def safe_members(tar: tarfile.TarFile, dest: Path):
    root = dest.resolve()
    for m in tar.getmembers():
        target = (dest / m.name).resolve()
        if root != target and root not in target.parents:
            raise SystemExit(f"unsafe path in archive: {m.name}")
        if m.issym() or m.islnk():
            raise SystemExit(f"link in archive refused: {m.name}")
        yield m


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    archive = target / URL.rsplit("/", 1)[1]
    if not archive.exists() or archive.stat().st_size != SIZE:
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    got = sha256_of(archive)
    if got != SHA256:
        print(f"checksum mismatch: expected {SHA256}, got {got}; nothing unpacked")
        return 1
    print("checksum ok")
    dest = target / "fpl26_contest_benchmarks_v1.3.0"
    dest.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, members=list(safe_members(tar, dest)))
    print(f"unpacked to {dest}; set ORCH_BENCHMARK_DIR to this folder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
