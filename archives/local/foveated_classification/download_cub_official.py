"""Resume and verify the two official CaltechDATA CUB-200-2011 archives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


DEST = Path(r"E:\datasets\CUB-200-2011\original_archives")
FILES = (
    (
        "segmentations.tgz",
        "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1",
        "4d47ba1228eae64f2fa547c47bc65255",
    ),
    (
        "CUB_200_2011.tgz",
        "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
        "97eceeb196236b17998738112f37df78",
    ),
)
CHUNK = 4 * 1024 * 1024
TRANSFER_CHUNK = 64 * 1024


def digest(path: Path, name: str) -> str:
    h = hashlib.new(name)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str, url: str, expected_md5: str) -> dict[str, object]:
    final = DEST / name
    part = DEST / (name + ".part")
    if final.exists():
        actual = digest(final, "md5")
        if actual != expected_md5:
            raise RuntimeError(f"Existing {final} has wrong MD5 {actual}; will not overwrite")
        print(f"VERIFIED_EXISTING {name} {final.stat().st_size} MD5={actual}", flush=True)
        return {"file": str(final), "size": final.stat().st_size, "md5": actual, "sha256": digest(final, "sha256"), "source": url}

    for attempt in range(1, 4):
        offset = part.stat().st_size if part.exists() else 0
        print(f"START {name} attempt={attempt}/3 offset={offset}", flush=True)
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                if response.status != 206 or not match or int(match.group(1)) != offset:
                    raise RuntimeError(f"Unsafe resume response: HTTP {response.status}, Content-Range={content_range!r}")
                total = int(match.group(3))
                if offset > total:
                    raise RuntimeError(f"Partial file exceeds published response size: {offset}>{total}")
                next_report = time.monotonic() + 30
                with part.open("ab") as stream:
                    while True:
                        block = response.read(TRANSFER_CHUNK)
                        if not block:
                            break
                        stream.write(block)
                        offset += len(block)
                        now = time.monotonic()
                        if now >= next_report:
                            print(f"PROGRESS {name} {offset}/{total} ({offset / total:.1%})", flush=True)
                            next_report = now + 30
                if offset != total:
                    raise RuntimeError(f"Incomplete response: {offset}/{total}")
            actual_md5 = digest(part, "md5")
            if actual_md5 != expected_md5:
                raise RuntimeError(f"MD5 mismatch for {name}: expected {expected_md5}, got {actual_md5}")
            sha256 = digest(part, "sha256")
            os.replace(part, final)
            print(f"VERIFIED {name} {total} MD5={actual_md5} SHA256={sha256}", flush=True)
            return {"file": str(final), "size": total, "md5": actual_md5, "sha256": sha256, "source": url}
        except Exception as exc:
            print(f"ATTEMPT_FAILED {name} {attempt}/3 {type(exc).__name__}: {exc}", flush=True)
            if attempt == 3:
                raise
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


def download_parallel(name: str, url: str, expected_md5: str, workers: int = 4) -> dict[str, object]:
    final = DEST / name
    if final.exists():
        return download(name, url, expected_md5)
    probe = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(probe, timeout=60) as response:
        match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match:
            raise RuntimeError("Source does not support verified byte ranges")
        total = int(match.group(1))
    span = (total + workers - 1) // workers

    def fetch(index: int) -> Path:
        start = index * span
        end = min(total, start + span) - 1
        path = DEST / f"{name}.part{index:02d}"
        for attempt in range(1, 4):
            offset = start + (path.stat().st_size if path.exists() else 0)
            if offset > end + 1:
                raise RuntimeError(f"Oversized segment {index}: {offset}>{end + 1}")
            if offset == end + 1:
                return path
            print(f"SEGMENT {name} {index + 1}/{workers} attempt={attempt} offset={offset}", flush=True)
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}", "User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if response.status != 206 or not match or (int(match.group(1)), int(match.group(2)), int(match.group(3))) != (offset, end, total):
                        raise RuntimeError(f"Unexpected segment response: HTTP {response.status}, Content-Range={response.headers.get('Content-Range')!r}")
                    next_report = time.monotonic() + 60
                    with path.open("ab") as stream:
                        while True:
                            block = response.read(TRANSFER_CHUNK)
                            if not block:
                                break
                            stream.write(block)
                            offset += len(block)
                            if time.monotonic() >= next_report:
                                print(f"PROGRESS {name} segment={index + 1}/{workers} {offset - start}/{end - start + 1}", flush=True)
                                next_report = time.monotonic() + 60
                    if offset != end + 1:
                        raise RuntimeError(f"Short segment {index}: {offset - start}/{end - start + 1}")
                return path
            except Exception as exc:
                print(f"SEGMENT_FAILED {name} {index + 1}/{workers} {attempt}/3 {type(exc).__name__}: {exc}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(10 * attempt)
        raise AssertionError("unreachable")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(fetch, range(workers)))
    combined = DEST / (name + ".part")
    if combined.exists():
        raise RuntimeError(f"Refusing to overwrite unrelated existing partial file: {combined}")
    with combined.open("wb") as output:
        for path in parts:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, CHUNK)
    if combined.stat().st_size != total:
        raise RuntimeError(f"Combined size mismatch: {combined.stat().st_size}/{total}")
    actual_md5 = digest(combined, "md5")
    if actual_md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {name}: expected {expected_md5}, got {actual_md5}")
    sha256 = digest(combined, "sha256")
    os.replace(combined, final)
    print(f"VERIFIED {name} {total} MD5={actual_md5} SHA256={sha256}", flush=True)
    return {"file": str(final), "size": total, "md5": actual_md5, "sha256": sha256, "source": url}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    results = [download(*FILES[0]), download_parallel(*FILES[1])]
    manifest = DEST / "official_download_manifest.json"
    temp = manifest.with_suffix(".json.tmp")
    temp.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, manifest)
    print(f"MANIFEST {manifest}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
