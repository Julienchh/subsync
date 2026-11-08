import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.fixtures_gen import (
    evaluate_alignment_accuracy,
    generate_dialogue_bursts,
    generate_distorted_srt,
    generate_ground_truth_srt,
    generate_synthetic_mkv,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = REPO_ROOT / "run.py"


class TestCLIEndToEnd(unittest.TestCase):
    """End-to-end CLI tests invoking `python3 run.py` across sync, batch, and series modes."""

    def test_01_cli_help_and_subcommands_help(self):
        for cmd in (
            [sys.executable, str(RUN_PY), "--help"],
            [sys.executable, str(RUN_PY), "batch", "--help"],
            [sys.executable, str(RUN_PY), "series", "--help"],
            [sys.executable, str(RUN_PY), "--version"],
        ):
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0, f"Command {cmd} failed: {proc.stderr}")
            self.assertTrue(len(proc.stdout.strip()) > 0)

    def test_02_cli_single_sync_command(self):
        with tempfile.TemporaryDirectory(prefix="subsync_cli_sync_") as tmp:
            tmp_dir = Path(tmp)
            bursts = generate_dialogue_bursts(num_bursts=30, duration_s=150.0, seed=101)
            mkv_path = tmp_dir / "Movie.mkv"
            gt_srt = tmp_dir / "Movie.gt.srt"
            in_srt = tmp_dir / "Movie.desync.srt"
            out_srt = tmp_dir / "Movie.synced.srt"

            generate_synthetic_mkv(mkv_path, bursts=bursts, duration_s=150.0)
            generate_ground_truth_srt(gt_srt, bursts, lang="fr")
            generate_distorted_srt(in_srt, bursts, offset_s=2.150, scale_a=1.0, lang="fr")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(RUN_PY),
                    "--cli",
                    "sync",
                    "--sub",
                    str(in_srt),
                    "--ref",
                    str(mkv_path),
                    "--out",
                    str(out_srt),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, f"CLI sync failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
            self.assertTrue(out_srt.is_file(), "Expected synchronized output .srt file")
            metrics = evaluate_alignment_accuracy(out_srt, gt_srt, tolerance_s=0.100)
            self.assertTrue(metrics["passed"], f"Accuracy check failed: {metrics}")

    def test_03_cli_batch_subcommand(self):
        with tempfile.TemporaryDirectory(prefix="subsync_cli_batch_") as tmp:
            tmp_dir = Path(tmp)
            summary_json = tmp_dir / "batch_summary.json"
            offsets = [1.450, -1.950]
            gt_files = {}

            for idx, offset in enumerate(offsets, start=1):
                ep_code = f"S01E0{idx}"
                bursts = generate_dialogue_bursts(num_bursts=28, duration_s=140.0, seed=200 + idx)
                mkv_path = tmp_dir / f"Show.{ep_code}.mkv"
                sub_path = tmp_dir / f"Show.{ep_code}.fr.srt"
                gt_path = tmp_dir / f"gt_{ep_code}.txt"
                generate_synthetic_mkv(mkv_path, bursts=bursts, duration_s=140.0)
                generate_ground_truth_srt(gt_path, bursts, lang="fr")
                generate_distorted_srt(sub_path, bursts, offset_s=offset, scale_a=1.0, lang="fr")
                gt_files[ep_code] = (mkv_path, gt_path)

            proc = subprocess.run(
                [
                    sys.executable,
                    str(RUN_PY),
                    "batch",
                    str(tmp_dir),
                    "--summary-json",
                    str(summary_json),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(proc.returncode, 0, f"CLI batch failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
            self.assertTrue(summary_json.is_file())
            report = json.loads(summary_json.read_text(encoding="utf-8"))
            self.assertEqual(report["succeeded"], 2)

            for ep_code, (mkv_path, gt_path) in gt_files.items():
                synced_srt = tmp_dir / f"{mkv_path.stem}.synced.srt"
                self.assertTrue(synced_srt.is_file(), f"Missing {synced_srt}")
                metrics = evaluate_alignment_accuracy(synced_srt, gt_path, tolerance_s=0.100)
                self.assertTrue(metrics["passed"], f"Batch episode {ep_code} accuracy failed: {metrics}")

    def test_04_cli_series_subcommand_mock_provider(self):
        with tempfile.TemporaryDirectory(prefix="subsync_cli_series_") as tmp:
            tmp_dir = Path(tmp)
            summary_json = tmp_dir / "series_summary.json"
            bursts = generate_dialogue_bursts(num_bursts=28, duration_s=140.0, seed=303)
            mkv_path = tmp_dir / "The Blacklist S01E01 Pilot (1080p x265 10bit Joy).mkv"
            gt_path = tmp_dir / "gt_e01.txt"
            generate_synthetic_mkv(mkv_path, bursts=bursts, duration_s=140.0)
            generate_ground_truth_srt(gt_path, bursts, lang="fr")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(RUN_PY),
                    "series",
                    str(tmp_dir),
                    "--lang",
                    "fr",
                    "--provider",
                    "mock",
                    "--cache-dir",
                    str(tmp_dir / "cache"),
                    "--summary-json",
                    str(summary_json),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(proc.returncode, 0, f"CLI series failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
            synced_srt = tmp_dir / f"{mkv_path.stem}.fr.synced.srt"
            self.assertTrue(synced_srt.is_file(), f"Missing {synced_srt}")
            metrics = evaluate_alignment_accuracy(synced_srt, gt_path, tolerance_s=0.100)
            self.assertTrue(metrics["passed"], f"Series E01 accuracy failed: {metrics}")


if __name__ == "__main__":
    unittest.main()
