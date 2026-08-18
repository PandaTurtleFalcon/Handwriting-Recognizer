import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.download_cvl_dataset import (
    CvlArchive,
    archive_is_complete,
    extract_archive,
    manifest_for_archives,
    parse_archives,
    resolve_archive_keys,
)


class DownloadCvlDatasetTests(unittest.TestCase):
    """Regression tests for CVL metadata and download helpers."""

    def test_parse_archives_reads_zenodo_file_metadata(self) -> None:
        """Zenodo file JSON should become keyed archive records."""

        record = {
            "files": [
                {
                    "key": "cvl-database-1-1.zip",
                    "size": 123,
                    "checksum": "md5:abc",
                    "links": {"self": "https://example.test/full.zip"},
                },
                {"key": "missing-url", "links": {}},
            ]
        }

        archives = parse_archives(record)

        self.assertEqual(list(archives), ["cvl-database-1-1.zip"])
        self.assertEqual(archives["cvl-database-1-1.zip"].size, 123)
        self.assertEqual(archives["cvl-database-1-1.zip"].md5, "abc")

    def test_resolve_archive_keys_accepts_aliases_and_exact_keys(self) -> None:
        """CLI users can request friendly aliases or concrete Zenodo names."""

        archives = {
            "cvl-database-1-1.zip": CvlArchive("cvl-database-1-1.zip", 1, "md5:a", "url"),
            "GtViewer.zip": CvlArchive("GtViewer.zip", 1, "md5:b", "url"),
        }

        resolved = resolve_archive_keys(["full", "GtViewer.zip"], archives)

        self.assertEqual(resolved, ["cvl-database-1-1.zip", "GtViewer.zip"])

    def test_resolve_archive_keys_rejects_unknown_archive(self) -> None:
        """Typos should fail before any network download starts."""

        with self.assertRaisesRegex(ValueError, "Unknown CVL archive"):
            resolve_archive_keys(["nope"], {})

    def test_archive_is_complete_checks_size_and_md5(self) -> None:
        """Existing local files should only be reused when verified."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "archive.zip"
            path.write_bytes(b"abc")
            archive = CvlArchive("archive.zip", 3, "md5:900150983cd24fb0d6963f7d28e17f72", "url")
            bad_archive = CvlArchive("archive.zip", 3, "md5:bad", "url")

            self.assertTrue(archive_is_complete(path, archive))
            self.assertFalse(archive_is_complete(path, bad_archive))

    def test_manifest_records_license_and_archive_rows(self) -> None:
        """The local manifest should preserve source and license constraints."""

        archive = CvlArchive("cvl-database-1-1.zip", 123, "md5:abc", "https://example.test/full.zip")

        manifest = manifest_for_archives({"cvl-database-1-1.zip": archive})

        self.assertIn("non-commercial", manifest["license"])
        self.assertEqual(manifest["archives"][0]["key"], "cvl-database-1-1.zip")

    def test_extract_archive_writes_files_under_archive_stem(self) -> None:
        """Archive extraction should keep contents grouped by source archive."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "sample.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("folder/file.txt", "hello")

            report = extract_archive(archive_path, root / "out")

            self.assertEqual(report["files"], 1)
            self.assertEqual((root / "out" / "sample" / "folder" / "file.txt").read_text(encoding="utf-8"), "hello")

    def test_extract_archive_rejects_path_traversal(self) -> None:
        """Malicious zip members must not escape the extraction folder."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "bad.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../escape.txt", "nope")

            with self.assertRaisesRegex(RuntimeError, "unsafe zip member"):
                extract_archive(archive_path, root / "out")


if __name__ == "__main__":
    unittest.main()
