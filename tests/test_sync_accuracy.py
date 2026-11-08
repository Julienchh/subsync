"""Unit and end-to-end verification tests for R1: Core Synchronization & 0.1s Accuracy."""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.fixtures_gen import (
    GROUND_TRUTH_CASES,
    evaluate_alignment_accuracy,
    generate_dialogue_bursts,
    generate_distorted_srt,
    generate_fixture_bundle,
    generate_ground_truth_srt,
    parse_srt_file,
    write_srt_file,
)


def _try_import_subsync():
    """Attempt to import subsync and pysubs2; return (subsync, pysubs2, error)."""
    try:
        import pysubs2
        import subsync
        from subsync import OutputFile, RefFile, SubFile, SyncController, SyncTask

        return subsync, pysubs2, None
    except Exception as exc:
        return None, None, exc


class TestFixturesGeneratorUnit(unittest.TestCase):
    """Self-test for the deterministic synthetic media and subtitle fixture generator."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="subsync_test_fixtures_")
        cls.bundle = generate_fixture_bundle(
            cls.tmpdir, num_bursts=32, duration_s=150.0, seed=42, include_mkv=True
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_bursts_count_and_bounds(self):
        bursts = self.bundle["bursts"]
        self.assertGreaterEqual(len(bursts), 25)
        self.assertLessEqual(len(bursts), 60)
        prev_end = 0.0
        for start_s, end_s in bursts:
            self.assertGreater(start_s, prev_end)
            self.assertGreater(end_s - start_s, 0.8)
            prev_end = end_s
        self.assertLess(prev_end, self.bundle["duration_s"])

    def test_wav_and_mkv_files_created(self):
        wav_path = Path(self.bundle["wav_path"])
        mkv_path = Path(self.bundle["mkv_path"])
        self.assertTrue(wav_path.is_file())
        self.assertGreater(wav_path.stat().st_size, 100_000)
        self.assertTrue(mkv_path.is_file())
        self.assertGreater(mkv_path.stat().st_size, 10_000)

    def test_ground_truth_and_distorted_srt_files(self):
        gt_en = parse_srt_file(self.bundle["gt_en_srt"])
        gt_fr = parse_srt_file(self.bundle["gt_fr_srt"])
        self.assertEqual(len(gt_en), len(self.bundle["bursts"]))
        self.assertEqual(len(gt_fr), len(self.bundle["bursts"]))
        self.assertIn("Ça doit être bon", gt_fr[0]["text"])

        # Perfect recovery from known ground-truth parameters must achieve <= 0.002s error
        for case_key, case_info in self.bundle["cases"].items():
            dist_events = parse_srt_file(case_info["distorted_srt"])
            self.assertEqual(len(dist_events), len(gt_en))
            a = case_info["a"]
            b = case_info["b"]
            recovered = [
                (ev["start_s"] * a + b, ev["end_s"] * a + b, ev["text"])
                for ev in dist_events
            ]
            rec_path = os.path.join(self.tmpdir, f"recovered_{case_key}.srt")
            write_srt_file(rec_path, recovered)
            eval_res = evaluate_alignment_accuracy(
                rec_path, self.bundle["gt_en_srt"], tolerance_s=0.100
            )
            self.assertTrue(eval_res["passed"])
            self.assertLessEqual(eval_res["max_error_s"], 0.010)
            self.assertLessEqual(eval_res["mean_abs_error_s"], 0.005)


class TestSyncAccuracy(unittest.TestCase):
    """End-to-end verification of R1: Core Synchronization Modernization & <= 0.100s Accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="subsync_r1_accuracy_")
        cls.bundle = generate_fixture_bundle(
            cls.tmpdir, num_bursts=35, duration_s=160.0, seed=101, include_mkv=True
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _require_subsync(self):
        subsync_mod, pysubs2_mod, err = _try_import_subsync()
        if err is not None:
            self.fail(f"Failed to import subsync / pysubs2 cleanly: {err}")
        return subsync_mod, pysubs2_mod

    def _run_sync_and_verify(
        self,
        distorted_srt_path: str,
        ref_media_path: str,
        gt_srt_path: str,
        out_name: str,
        sub_lang: str = "eng",
        ref_lang: str = "eng",
        tolerance_s: float = 0.100,
    ):
        subsync_mod, _ = self._require_subsync()
        from subsync import SyncController, SyncTask

        out_srt_path = os.path.join(self.tmpdir, out_name)
        if os.path.exists(out_srt_path):
            os.remove(out_srt_path)

        task = SyncTask(
            sub={"path": distorted_srt_path, "lang": sub_lang},
            ref={"path": ref_media_path, "lang": ref_lang},
            out={"path": out_srt_path, "enc": "UTF-8"},
        )

        job_results = []
        job_errors = []

        controller = SyncController(
            onJobEnd=lambda t, status, res: job_results.append((status, res)),
            onError=lambda t, src, err: job_errors.append((src, err)),
        )
        controller.configure(overwrite=True, minEffort=0.5)
        controller.synchronize(task, interactive=False)
        finished = controller.wait()
        self.assertTrue(finished, "SyncController was unexpectedly terminated")

        if not os.path.exists(out_srt_path) and job_results:
            _, res = job_results[-1]
            if res and getattr(res, "path", None) and os.path.exists(res.path):
                out_srt_path = res.path

        if not os.path.exists(out_srt_path):
            try:
                controller.saveSynchronizedSubtitles(path=out_srt_path, task=task)
            except Exception as exc:
                self.fail(
                    f"Synchronized subtitle file was not created at {out_srt_path}: {exc}; "
                    f"errors={job_errors}, results={job_results}"
                )

        self.assertTrue(
            os.path.isfile(out_srt_path),
            f"Expected synchronized output file at {out_srt_path} (errors={job_errors})",
        )

        if job_results:
            status, result = job_results[-1]
            self.assertTrue(
                getattr(status, "correlated", True),
                f"Expected status.correlated=True, got {status}",
            )
            self.assertTrue(
                getattr(result, "success", True),
                f"Expected result.success=True, got {result}",
            )

        eval_res = evaluate_alignment_accuracy(
            out_srt_path,
            gt_srt_path,
            tolerance_s=tolerance_s,
            max_mae_s=0.050,
            raise_on_failure=True,
        )
        self.assertLessEqual(eval_res["max_error_s"], 0.100)
        self.assertLessEqual(eval_res["mean_abs_error_s"], 0.050)
        return eval_res, out_srt_path

    def test_01_clean_package_import(self):
        """Verify clean import of subsync, SyncController, SyncTask, and pysubs2."""
        import pysubs2
        import subsync
        from subsync import OutputFile, RefFile, SubFile, SyncController, SyncTask

        self.assertIsNotNone(subsync)
        self.assertIsNotNone(SyncController)
        self.assertIsNotNone(SyncTask)
        self.assertIsNotNone(SubFile)
        self.assertIsNotNone(RefFile)
        self.assertIsNotNone(OutputFile)
        self.assertIsNotNone(pysubs2)

    def test_02_case1_pure_positive_offset_2_350s(self):
        """Case 1: Pure positive offset +2.350s (a = 1.0) within <= 0.100s."""
        case = self.bundle["cases"]["case1_pos_offset"]
        eval_res, _ = self._run_sync_and_verify(
            distorted_srt_path=case["distorted_srt"],
            ref_media_path=self.bundle["wav_path"],
            gt_srt_path=self.bundle["gt_en_srt"],
            out_name="synced_case1_pos.srt",
        )
        self.assertTrue(eval_res["passed"])

    def test_03_case2_pure_negative_offset_1_800s(self):
        """Case 2: Pure negative offset -1.800s (a = 1.0) within <= 0.100s."""
        case = self.bundle["cases"]["case2_neg_offset"]
        eval_res, _ = self._run_sync_and_verify(
            distorted_srt_path=case["distorted_srt"],
            ref_media_path=self.bundle["wav_path"],
            gt_srt_path=self.bundle["gt_en_srt"],
            out_name="synced_case2_neg.srt",
        )
        self.assertTrue(eval_res["passed"])

    def test_04_case3_large_offset_8_420s(self):
        """Case 3: Large positive offset +8.420s (a = 1.0) within <= 0.100s."""
        case = self.bundle["cases"]["case3_large_offset"]
        eval_res, _ = self._run_sync_and_verify(
            distorted_srt_path=case["distorted_srt"],
            ref_media_path=self.bundle["wav_path"],
            gt_srt_path=self.bundle["gt_en_srt"],
            out_name="synced_case3_large.srt",
        )
        self.assertTrue(eval_res["passed"])

    def test_05_case4_subsecond_offset_0_650s(self):
        """Case 4: Sub-second negative offset -0.650s (a = 1.0) within <= 0.100s."""
        case = self.bundle["cases"]["case4_subsec_offset"]
        eval_res, _ = self._run_sync_and_verify(
            distorted_srt_path=case["distorted_srt"],
            ref_media_path=self.bundle["wav_path"],
            gt_srt_path=self.bundle["gt_en_srt"],
            out_name="synced_case4_subsec.srt",
        )
        self.assertTrue(eval_res["passed"])

    def test_06_case5_framerate_drift_25_over_23_976(self):
        """Case 5: Framerate drift 25.0 / 23.976 (a = 1.042709, b = +1.500s) within <= 0.100s."""
        case = self.bundle["cases"]["case5_framerate_drift"]
        eval_res, _ = self._run_sync_and_verify(
            distorted_srt_path=case["distorted_srt"],
            ref_media_path=self.bundle["wav_path"],
            gt_srt_path=self.bundle["gt_en_srt"],
            out_name="synced_case5_drift.srt",
        )
        self.assertTrue(eval_res["passed"])

    def test_07_cross_language_english_mkv_to_french_srt(self):
        """Cross-language alignment: English audio .mkv aligned with French .srt subtitles."""
        case = self.bundle["cases"]["case1_pos_offset"]
        eval_res, synced_path = self._run_sync_and_verify(
            distorted_srt_path=case["distorted_fr_srt"],
            ref_media_path=self.bundle["mkv_path"],
            gt_srt_path=self.bundle["gt_fr_srt"],
            out_name="synced_crosslang_fr.srt",
            sub_lang="fra",
            ref_lang="eng",
        )
        self.assertTrue(eval_res["passed"])
        synced_text = Path(synced_path).read_text(encoding="utf-8")
        self.assertIn("Ça doit être bon", synced_text)


if __name__ == "__main__":
    unittest.main()
