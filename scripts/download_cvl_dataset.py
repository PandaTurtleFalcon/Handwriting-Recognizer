"""Download CVL dataset archives into the ignored local data folder.

The CVL archives are large and licensed for non-commercial research use, so
this helper never commits data. It queries the Zenodo record metadata, writes a
small manifest, and can download selected archives with checksum verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPResponse
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_URL = "https://zenodo.org/api/records/1492267"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "data" / "cvl"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_ROOT / "zenodo_manifest.json"
FILE_ALIASES = {
    "full": "cvl-database-1-1.zip",
    "cropped": "cvl-database-cropped-1-1.zip",
    "parser": "DkGtDbXmlReader-src.zip",
    "viewer": "GtViewer.zip",
}


@dataclass(frozen=True)
class CvlArchive:
    """One downloadable CVL archive from the Zenodo record."""

    key: str
    size: int
    checksum: str
    url: str

    @property
    def md5(self) -> str | None:
        """Return the expected MD5 digest when Zenodo provides one."""

        algorithm, _, digest = self.checksum.partition(":")
        if algorithm.lower() != "md5" or not digest:
            return None
        return digest


def parse_archives(record: dict[str, Any]) -> dict[str, CvlArchive]:
    """Extract downloadable archive metadata from a Zenodo record JSON."""

    archives: dict[str, CvlArchive] = {}
    for item in record.get("files", []):
        key = str(item.get("key", ""))
        links = item.get("links", {})
        url = str(links.get("self", ""))
        if not key or not url:
            continue
        archives[key] = CvlArchive(
            key=key,
            size=int(item.get("size", 0)),
            checksum=str(item.get("checksum", "")),
            url=url,
        )
    return archives


def fetch_record(record_url: str = DEFAULT_RECORD_URL) -> dict[str, Any]:
    """Fetch the Zenodo CVL record metadata."""

    with urllib.request.urlopen(record_url, timeout=30) as response:
        return json.load(response)


def resolve_archive_keys(requested: list[str], archives: dict[str, CvlArchive]) -> list[str]:
    """Resolve CLI aliases like `full` into concrete Zenodo file keys."""

    resolved = []
    for item in requested:
        key = FILE_ALIASES.get(item, item)
        if key not in archives:
            known = ", ".join(sorted([*FILE_ALIASES, *archives]))
            raise ValueError(f"Unknown CVL archive {item!r}; choose one of: {known}")
        resolved.append(key)
    return resolved


def file_md5(path: Path) -> str:
    """Return the MD5 digest for one local archive."""

    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_is_complete(path: Path, archive: CvlArchive) -> bool:
    """Return whether a local file matches the expected size and checksum."""

    if not path.exists() or path.stat().st_size != archive.size:
        return False
    expected_md5 = archive.md5
    return expected_md5 is None or file_md5(path) == expected_md5


def _response_status(response: HTTPResponse) -> int:
    """Return a urllib response status code across Python response types."""

    status = getattr(response, "status", None)
    if status is not None:
        return int(status)
    getcode = getattr(response, "getcode", None)
    if callable(getcode):
        return int(getcode())
    return int(HTTPStatus.OK)


def _download_to_temporary(
    url: str,
    temporary: Path,
    opener=urllib.request.urlopen,
) -> None:
    """Download a URL to a `.part` file, resuming when the server allows it."""

    existing_size = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
    request = urllib.request.Request(url, headers=headers)
    with opener(request, timeout=30) as response:
        status = _response_status(response)
        if existing_size > 0 and status == HTTPStatus.PARTIAL_CONTENT:
            mode = "ab"
        elif status == HTTPStatus.OK:
            mode = "wb"
        else:
            raise RuntimeError(f"Download failed with HTTP status {status}.")
        with temporary.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)


def download_archive(archive: CvlArchive, output_root: Path) -> Path:
    """Download one CVL archive unless a verified copy already exists."""

    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / archive.key
    if archive_is_complete(destination, archive):
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    _download_to_temporary(archive.url, temporary)
    if archive.size and temporary.stat().st_size != archive.size:
        raise RuntimeError(f"Downloaded {archive.key} has unexpected size: {temporary.stat().st_size}")
    expected_md5 = archive.md5
    if expected_md5 is not None and file_md5(temporary) != expected_md5:
        raise RuntimeError(f"Downloaded {archive.key} failed MD5 verification.")
    temporary.replace(destination)
    return destination


def _safe_extract_path(destination_root: Path, member_name: str) -> Path:
    """Return a zip member destination, rejecting path traversal."""

    target = (destination_root / member_name).resolve()
    root = destination_root.resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"Refusing unsafe zip member path: {member_name}")
    return target


def extract_archive(archive_path: Path, extract_root: Path) -> dict[str, object]:
    """Safely extract one local CVL archive and return a small summary."""

    target_root = extract_root / archive_path.stem
    target_root.mkdir(parents=True, exist_ok=True)
    files = 0
    directories = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = _safe_extract_path(target_root, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                directories += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                destination.write(source.read())
            files += 1
    return {
        "archive": str(archive_path),
        "output_root": str(target_root),
        "files": files,
        "directories": directories,
    }


def manifest_for_archives(archives: dict[str, CvlArchive]) -> dict[str, object]:
    """Return a JSON-serializable manifest for local CVL downloads."""

    return {
        "source": DEFAULT_RECORD_URL,
        "license": "CC BY-NC 3.0 / non-commercial research use",
        "archives": [
            {
                "key": archive.key,
                "size": archive.size,
                "checksum": archive.checksum,
                "url": archive.url,
            }
            for archive in sorted(archives.values(), key=lambda item: item.key)
        ],
    }


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-url", default=DEFAULT_RECORD_URL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        help="Archive alias/key to download. Aliases: full, cropped, parser, viewer. Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print metadata without downloading archives.")
    parser.add_argument("--extract", action="store_true", help="Extract selected downloaded archives after verification.")
    args = parser.parse_args()

    record = fetch_record(args.record_url)
    archives = parse_archives(record)
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = manifest_for_archives(archives)
    manifest_path = args.output_root / DEFAULT_MANIFEST_PATH.name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    selected_keys = resolve_archive_keys(args.archive or ["full"], archives)
    report = {
        "manifest": str(manifest_path),
        "dry_run": bool(args.dry_run),
        "extract": bool(args.extract),
        "selected": selected_keys,
        "downloaded": [],
        "extracted": [],
    }
    if not args.dry_run:
        for key in selected_keys:
            path = download_archive(archives[key], args.output_root)
            report["downloaded"].append(str(path))
            if args.extract:
                report["extracted"].append(extract_archive(path, args.output_root))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
