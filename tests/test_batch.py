"""Unit and end-to-end verification tests for R2: Unattended Batch Resynchronization."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.fixtures_gen import (
    evaluate_alignment_accuracy,
    generate_dialogue_bursts,
    generate_distorted_srt,
    generate_ground_truth_srt,
    generate_synthetic_mkv,
    generate_synthetic_wav,
)


class TestBatchResynchronization(unittest.TestCase):
    """Verification suite for R2: discovery, multi-tier matching, unattended batch run, and error isolation."""

    @classmethod
    def setUpClass(cls):
        cls.shared_tmp = tempfile.mkdtemp(prefix="subsync_r2_shared_")
        cls.bursts = generate_dialogue_bursts(num_bursts=34, duration_s=150.0, seed=77)
        cls.base_wav = generate_synthetic_wav(
            Path(cls.shared_tmp) / "base_ref.wav", cls.bursts, duration_s=150.0
        )
        cls.base_mkv = generate_synthetic_mkv(
            Path(cls.shared_tmp) / "base_ref.mkv",
            wav_path=cls.base_wav,
            bursts=cls.bursts,
            duration_s=150.0,
            audio_lang="eng",
        )
        cls.gt_srt = generate_ground_truth_srt(
            Path(cls.shared_tmp) / "gt.srt", cls.bursts, lang="en"
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.shared_tmp, ignore_errors=True)

    def test_01_directory_discovery_and_glob_expansion_excluding_synced(self):
        """Verify directory discovery & glob expansion while excluding .synced.srt outputs."""
        from subsync.batch.discovery import discover_media_files, is_synced_output_file

        with tempfile.TemporaryDirectory(prefix="subsync_disc_") as tmp:
            root = Path(tmp)
            (root / "Show.S01E01.mkv").write_bytes(b"video1")
            (root / "Show.S01E02.mp4").write_bytes(b"video2")
            (root / "Show.S01E01.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nHi\n", encoding="utf-8")
            (root / "Show.S01E02.fr.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nSalut\n", encoding="utf-8")
            # Already-synced output files that must be excluded
            (root / "Show.S01E01.synced.srt").write_text("synced", encoding="utf-8")
            (root / "Show.S01E02.fr.synced.srt").write_text("synced", encoding="utf-8")

            self.assertTrue(is_synced_output_file(root / "Show.S01E01.synced.srt"))
            self.assertTrue(is_synced_output_file(root / "Show.S01E02.fr.synced.srt"))
            self.assertFalse(is_synced_output_file(root / "Show.S01E01.srt"))

            videos, subs = discover_media_files(ref_dir=root)
            self.assertEqual([p.name for p in videos], ["Show.S01E01.mkv", "Show.S01E02.mp4"])
            self.assertEqual([p.name for p in subs], ["Show.S01E01.srt", "Show.S01E02.fr.srt"])

            # Glob expansion check
            g_videos, g_subs = discover_media_files(
                ref_pattern=str(root / "*.mkv"),
                sub_pattern=str(root / "*.srt"),
            )
            self.assertEqual([p.name for p in g_videos], ["Show.S01E01.mkv"])
            self.assertEqual([p.name for p in g_subs], ["Show.S01E01.srt", "Show.S01E02.fr.srt"])

    def test_02_multi_tier_episode_and_stem_matching(self):
        """Verify S01E01, 1x01, 101 scene numbering, release tag disambiguation, and movie stem matching."""
        from subsync.batch.matcher import extract_episode_info, match_files

        # 1. Mixed episode formats: S01E01, 1x02, 103
        videos = [
            Path("/media/The.Blacklist.S01E01.Pilot.1080p.x265.Joy.mkv"),
            Path("/media/The.Blacklist.S01E02.The.Freelancer.720p.HDTV.LOL.mkv"),
            Path("/media/The.Blacklist.S01E03.Wujing.1080p.WEB-DL.mkv"),
        ]
        # Multiple candidates for E02 (LOL vs DIMENSION, French vs Spanish) to test disambiguation
        subs = [
            Path("/subs/The Blacklist - 01x01 - Pilot.French.srt"),
            Path("/subs/The.Blacklist.1x02.DIMENSION.French.srt"),
            Path("/subs/The.Blacklist.1x02.HDTV.LOL.French.srt"),
            Path("/subs/The.Blacklist.1x02.HDTV.LOL.Spanish.srt"),
            Path("/subs/The.Blacklist.103.Wujing.fr.srt"),
        ]

        info_103 = extract_episode_info(subs[4])
        self.assertEqual((info_103.season, info_103.episode), (1, 3))

        report = match_files(videos=videos, subtitles=subs, mode="auto", target_lang="fr")
        self.assertEqual(len(report.paired), 3)

        by_ep = {pair.episode: pair for pair in report.paired}
        self.assertEqual(by_ep[1].subtitle.name, "The Blacklist - 01x01 - Pilot.French.srt")
        # E02 should disambiguate to LOL + French over DIMENSION or Spanish
        self.assertEqual(by_ep[2].subtitle.name, "The.Blacklist.1x02.HDTV.LOL.French.srt")
        self.assertEqual(by_ep[3].subtitle.name, "The.Blacklist.103.Wujing.fr.srt")

        # 2. Tier 2 Movie / non-episodic stem matching
        movie_videos = [
            Path("/movies/Inception.2010.1080p.BluRay.x264.mkv"),
            Path("/movies/Interstellar.2014.2160p.UHD.BluRay.x265.mkv"),
        ]
        movie_subs = [
            Path("/movies/Interstellar.2014.fr.srt"),
            Path("/movies/Inception.2010.en.srt"),
        ]
        movie_report = match_files(videos=movie_videos, subtitles=movie_subs, mode="auto")
        self.assertEqual(len(movie_report.paired), 2)
        self.assertEqual(movie_report.paired[0].video.name, "Inception.2010.1080p.BluRay.x264.mkv")
        self.assertEqual(movie_report.paired[0].subtitle.name, "Inception.2010.en.srt")
        self.assertEqual(movie_report.paired[1].video.name, "Interstellar.2014.2160p.UHD.BluRay.x265.mkv")
        self.assertEqual(movie_report.paired[1].subtitle.name, "Interstellar.2014.fr.srt")

    def test_03_unattended_4_episode_batch_accuracy(self):
        """Run unattended batch across 4 episodes with offsets (+1.50s, -2.10s, +3.25s, -0.85s) and verify <= 0.100s."""
        from subsync.batch.runner import run_batch

        offsets = [
            (1, 1.500),
            (2, -2.100),
            (3, 3.250),
            (4, -0.850),
        ]

        with tempfile.TemporaryDirectory(prefix="subsync_batch_4ep_") as tmp:
            batch_dir = Path(tmp)
            for ep_num, offset_s in offsets:
                ep_mkv = batch_dir / f"The.Blacklist.S01E0{ep_num}.1080p.mkv"
                shutil.copy2(self.base_mkv, ep_mkv)
                ep_sub = batch_dir / f"The.Blacklist.01x0{ep_num}.en.srt"
                generate_distorted_srt(ep_sub, self.bursts, offset_s=offset_s, scale_a=1.0, lang="en")

            summary = run_batch(
                ref_dir=batch_dir,
                sub_dir=batch_dir,
                out_pattern="{ref_dir}/{ref_name}.synced.srt",
                continue_on_error=True,
            )

            self.assertEqual(summary.total, 4)
            self.assertEqual(summary.succeeded, 4)
            self.assertEqual(summary.failed, 0)

            for res in summary.results:
                self.assertIn(res.status, ("OK", "SUCCESS"))
                self.assertTrue(res.succeeded)
                self.assertIsNotNone(res.out_path)
                self.assertTrue(Path(res.out_path).is_file())
                eval_res = evaluate_alignment_accuracy(
                    res.out_path, self.gt_srt, tolerance_s=0.100, max_mae_s=0.050
                )
                self.assertTrue(eval_res["passed"])
                self.assertLessEqual(eval_res["max_error_s"], 0.100)

    def test_04_error_isolation_corrupt_middle_video(self):
        """Inject 1 corrupt video in the middle of a 4-episode batch; verify 3 succeed and 1 is FAILED."""
        from subsync.batch.runner import run_batch

        episode_specs = [
            (1, 1.500, False),
            (2, -2.100, True),   # Episode 2 is corrupt/unreadable
            (3, 3.250, False),
            (4, -0.850, False),
        ]

        with tempfile.TemporaryDirectory(prefix="subsync_batch_corrupt_") as tmp:
            batch_dir = Path(tmp)
            for ep_num, offset_s, is_corrupt in episode_specs:
                ep_mkv = batch_dir / f"Show.S01E0{ep_num}.mkv"
                if is_corrupt:
                    ep_mkv.write_bytes(b"NOT_A_VALID_MATROSKA_OR_MEDIA_CONTAINER_STREAM_0000")
                else:
                    shutil.copy2(self.base_mkv, ep_mkv)
                ep_sub = batch_dir / f"Show.S01E0{ep_num}.srt"
                generate_distorted_srt(ep_sub, self.bursts, offset_s=offset_s, scale_a=1.0, lang="en")

            summary = run_batch(
                ref_dir=batch_dir,
                sub_dir=batch_dir,
                out_pattern="{ref_dir}/{ref_name}.synced.srt",
                continue_on_error=True,
            )

            self.assertEqual(summary.total, 4)
            self.assertEqual(summary.succeeded, 3)
            self.assertEqual(summary.failed, 1)

            by_ep = {r.episode: r for r in summary.results}
            self.assertEqual(by_ep[2].status, "FAILED")
            self.assertFalse(by_ep[2].succeeded)
            self.assertTrue(bool(by_ep[2].error))

            for valid_ep in (1, 3, 4):
                res = by_ep[valid_ep]
                self.assertIn(res.status, ("OK", "SUCCESS"))
                self.assertTrue(res.succeeded)
                self.assertTrue(Path(res.out_path).is_file())
                eval_res = evaluate_alignment_accuracy(
                    res.out_path, self.gt_srt, tolerance_s=0.100
                )
                self.assertTrue(eval_res["passed"])

    def test_05_atomic_in_place_with_backup_and_summary_json(self):
        """Verify atomic --in-place replacement with --backup (.bak) and --summary-json export."""
        from subsync.batch.runner import run_batch

        with tempfile.TemporaryDirectory(prefix="subsync_batch_inplace_") as tmp:
            work_dir = Path(tmp)
            ep_mkv = work_dir / "Episode.S01E01.mkv"
            shutil.copy2(self.base_mkv, ep_mkv)

            ep_sub = work_dir / "Episode.S01E01.srt"
            generate_distorted_srt(ep_sub, self.bursts, offset_s=2.350, scale_a=1.0, lang="en")
            original_distorted_text = ep_sub.read_text(encoding="utf-8")

            summary_json_path = work_dir / "batch_summary.json"

            summary = run_batch(
                ref_dir=work_dir,
                sub_dir=work_dir,
                in_place=True,
                backup=True,
                summary_json=summary_json_path,
            )

            self.assertEqual(summary.total, 1)
            self.assertEqual(summary.succeeded, 1)

            # Backup file must exist and match the pre-sync distorted subtitle content
            bak_path = work_dir / "Episode.S01E01.srt.bak"
            self.assertTrue(bak_path.is_file())
            self.assertEqual(bak_path.read_text(encoding="utf-8"), original_distorted_text)

            # In-place subtitle file must now be aligned within <= 0.100s of ground truth
            eval_res = evaluate_alignment_accuracy(ep_sub, self.gt_srt, tolerance_s=0.100)
            self.assertTrue(eval_res["passed"])

            # Summary JSON must exist and contain valid structured execution metrics
            self.assertTrue(summary_json_path.is_file())
            report_data = json.loads(summary_json_path.read_text(encoding="utf-8"))
            self.assertEqual(report_data["total"], 1)
            self.assertEqual(report_data["succeeded"], 1)
            self.assertEqual(report_data["failed"], 0)
            self.assertEqual(len(report_data["results"]), 1)
            self.assertAlmostEqual(report_data["results"][0]["offset_seconds"], 2.350, delta=0.100)


if __name__ == "__main__":
    unittest.main()
