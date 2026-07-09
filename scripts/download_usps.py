"""Download USPS files when torchvision's HTTPS verification fails locally."""

from __future__ import annotations

import argparse
import hashlib
import ssl
import urllib.request
from pathlib import Path


FILES = {
    "usps.bz2": (
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
        "ec16c51db3855ca6c91edd34d0e9b197",
    ),
    "usps.t.bz2": (
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
        "8ea070ee2aca1ac39742fdd1ef5ed118",
    ),
}


def md5sum(path: Path) -> str:
    """Return an MD5 digest for a file."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_usps(output_dir: Path, insecure_ssl: bool = False) -> None:
    """Download USPS archives and verify torchvision's expected MD5 hashes."""

    output_dir.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context() if insecure_ssl else None
    for filename, (url, expected_md5) in FILES.items():
        target = output_dir / filename
        if not target.exists():
            print(f"Downloading {filename}...")
            with urllib.request.urlopen(url, context=context) as response:
                target.write_bytes(response.read())
        actual_md5 = md5sum(target)
        if actual_md5 != expected_md5:
            raise RuntimeError(f"{filename} md5 mismatch: got {actual_md5}, expected {expected_md5}")
        print(f"{filename}: {target.stat().st_size} bytes, md5 ok")


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Download USPS dataset archives.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/usps"))
    parser.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Bypass local certificate verification; MD5 hashes are still verified.",
    )
    args = parser.parse_args()
    download_usps(args.output_dir, insecure_ssl=args.insecure_ssl)


if __name__ == "__main__":
    main()
