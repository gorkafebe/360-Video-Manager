"""Tests for app/cli.py option parsing and workflow forwarding."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.cli import run_cli


class CLITests(unittest.TestCase):
    def _success_result(self):
        return SimpleNamespace(
            success=True,
            job_id="job-1",
            projection_type="eac",
            confidence=0.9,
            converted_video_path=None,
            upload_result=None,
            manifest_path=None,
            error=None,
        )

    @patch("app.cli.setup_logging")
    @patch("app.cli.process_video_job")
    def test_cli_forwards_category_and_tags(self, mock_process, _mock_logging):
        mock_process.return_value = self._success_result()

        exit_code = run_cli(
            [
                "--url", "https://youtu.be/abc",
                "--upload",
                "--category", "cat-42",
                "--tags", "anxiety, breathing , grounding",
            ]
        )

        self.assertEqual(exit_code, 0)
        opts = mock_process.call_args.args[0]
        self.assertEqual(opts.upload_category, "cat-42")
        self.assertEqual(opts.upload_tags, ["anxiety", "breathing", "grounding"])

    @patch("app.cli.setup_logging")
    @patch("app.cli.process_video_job")
    @patch("core.uploader.create_category")
    def test_cli_new_category_created_when_uploading(
        self, mock_create_category, mock_process, _mock_logging
    ):
        mock_create_category.return_value = "cat-new"
        mock_process.return_value = self._success_result()

        exit_code = run_cli(
            [
                "--url", "https://youtu.be/abc",
                "--upload",
                "--new-category", "Patient John Doe",
            ]
        )

        self.assertEqual(exit_code, 0)
        mock_create_category.assert_called_once_with("Patient John Doe")
        opts = mock_process.call_args.args[0]
        self.assertEqual(opts.upload_category, "cat-new")


if __name__ == "__main__":
    unittest.main()

