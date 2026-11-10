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

    def test_08_wall_to_wall_dense_dialogue_act_alignment(self):
        """Wall-to-wall 85s abutting dialogue act (0.00s gaps, S01E08 regression) locks via vocal attacks without +5.2s edge aliasing."""
        from subsync.synchro import aligner, vad

        frame_hz = 100
        total_dur_s = 420.0
        ref_env = [0.0] * int(total_dur_s * frame_hz)
        ref_ivs = []
        sub_ivs = []

        def _add_speech(s: float, e: float, amp: float = 0.92):
            ref_ivs.append((s, e))
            i0 = max(0, int(round(s * frame_hz)))
            i1 = min(len(ref_env), int(round(e * frame_hz)))
            for k in range(i0, i1):
                ref_env[k] = max(ref_env[k], amp)

        # Acts 0 & 1: true offset +1.30s
        for base_t in (20.0, 95.0):
            t_cur = base_t
            for idx in range(12):
                dur = 1.8 + (idx % 3) * 0.5
                sub_ivs.append((t_cur, t_cur + dur))
                _add_speech(t_cur + 1.30, t_cur + dur + 1.30)
                t_cur += dur + 0.85 + (idx % 2) * 0.45

        # Act 2 (S01E08 minute-20 pattern): 32 abutting subtitles (gap = 0.00s) over 83s at true offset +2.50s,
        # with short first utterance + trailing laugh [288.2..291.2s] before dead silence [291.3..297.0s].
        t_cur = 200.0
        for idx in range(32):
            dur = 2.1 + (idx % 4) * 0.35
            s_sub, e_sub = t_cur, t_cur + dur
            sub_ivs.append((s_sub, e_sub))
            if idx == 0:
                _add_speech(s_sub + 2.50, s_sub + 2.95)
            else:
                _add_speech(s_sub + 2.50, e_sub + 2.50 - 0.24)
            t_cur = e_sub
        # Trailing laugh right after Act 2 that would fool pure boxcar NCC into shifting +2.7s late
        _add_speech(t_cur + 2.80, t_cur + 5.60, amp=0.55)

        # Act 3: true offset +2.50s
        t_cur = 305.0
        for idx in range(14):
            dur = 1.9 + (idx % 3) * 0.4
            sub_ivs.append((t_cur, t_cur + dur))
            _add_speech(t_cur + 2.50, t_cur + dur + 2.50)
            t_cur += dur + 0.80

        ref_ivs.sort()
        res = aligner.align_intervals_and_envelopes(
            sub_intervals=sub_ivs,
            ref_envelope=ref_env,
            ref_intervals=ref_ivs,
            frame_hz=frame_hz,
        )
        self.assertTrue(res.correlated)
        # Check midpoint of the dense wall-to-wall act (t_sub = 240.0s -> true t_ref = 242.50s)
        pred_mid = res.formula.getY(240.0)
        self.assertLessEqual(abs(pred_mid - 242.50), 0.10, f"Dense act drifted: got {pred_mid - 240.0:+.3f}s instead of +2.50s")

    def test_09_second_order_anti_spike_regularization(self):
        """Interior short act with periodic decoy peak cannot create an isolated ping-pong offset spike."""
        from subsync.synchro import aligner

        frame_hz = 100
        total_dur_s = 460.0
        ref_env = [0.0] * int(total_dur_s * frame_hz)
        ref_ivs = []
        sub_ivs = []

        def _add_speech(s: float, e: float, amp: float = 0.90):
            ref_ivs.append((s, e))
            i0 = max(0, int(round(s * frame_hz)))
            i1 = min(len(ref_env), int(round(e * frame_hz)))
            for k in range(i0, i1):
                ref_env[k] = max(ref_env[k], amp)

        # 5 acts all truly at +1.20s, with a rhythmic decoy at +4.60s on Act 2
        act_starts = [20.0, 105.0, 200.0, 260.0, 355.0]
        for a_idx, t0 in enumerate(act_starts):
            t_cur = t0
            n_cues = 6 if a_idx == 2 else 14
            for c_idx in range(n_cues):
                dur = 1.6 + (c_idx % 3) * 0.35
                sub_ivs.append((t_cur, t_cur + dur))
                _add_speech(t_cur + 1.20, t_cur + dur + 1.20, amp=0.80 if a_idx == 2 else 0.95)
                if a_idx == 2:
                    _add_speech(t_cur + 4.60, t_cur + dur + 4.60, amp=0.95)
                t_cur += dur + (1.60 if a_idx == 2 else 0.85)

        ref_ivs.sort()
        res = aligner.align_intervals_and_envelopes(
            sub_intervals=sub_ivs,
            ref_envelope=ref_env,
            ref_intervals=ref_ivs,
            frame_hz=frame_hz,
        )
        self.assertTrue(res.correlated)
        pred_act2 = res.formula.getY(210.0)
        self.assertLessEqual(abs(pred_act2 - 211.20), 0.10, f"Isolated spike was not suppressed: {pred_act2 - 210.0:+.3f}s")

    def test_10_stage4_verify_and_heal_executes_and_heals(self):
        """Stage 4 Subtitles.verify_and_heal_with_vad runs cleanly without error and heals an unaligned dialogue window."""
        from subsync.compat_pysubs2 import SSAEvent
        from subsync.subtitle import Subtitles

        frame_hz = 100
        total_dur_s = 360.0
        ref_env = [0.0] * int(total_dur_s * frame_hz)
        ref_ivs = []
        subs = Subtitles()
        gt_starts = []

        for w_idx, base_t in enumerate((20.0, 95.0, 175.0, 260.0)):
            t_cur = base_t
            injected_err = -1.40 if w_idx == 2 else 0.0
            for c_idx in range(10):
                dur = 1.7 + (c_idx % 3) * 0.3
                s_gt, e_gt = t_cur, t_cur + dur
                ref_ivs.append((s_gt, e_gt))
                for k in range(int(round(s_gt * frame_hz)), int(round(e_gt * frame_hz))):
                    ref_env[k] = 0.95
                ev = SSAEvent(
                    start=int(round((s_gt + injected_err) * 1000.0)),
                    end=int(round((e_gt + injected_err) * 1000.0)),
                    text=f"Line {w_idx}-{c_idx}",
                )
                subs.events.append(ev)
                gt_starts.append((w_idx, s_gt))
                t_cur += dur + 1.50

        healed_events, verified_wins, total_wins = subs.verify_and_heal_with_vad(
            ref_env, ref_intervals=ref_ivs, frame_hz=frame_hz
        )
        self.assertEqual(healed_events, 10)
        self.assertGreaterEqual(verified_wins, 4)
        for ev, (w_idx, s_gt) in zip(subs.events, gt_starts):
            err = abs((ev.start / 1000.0) - s_gt)
            self.assertLessEqual(err, 0.080, f"Window {w_idx} residual error too high: {err:.3f}s")

    def test_11_commercial_break_staircase_preserves_unit_slope(self):
        """4-plateau commercial-break staircase (+0s -> +10s -> +20s -> +30s across 8 acts) keeps a = 1.0."""
        from subsync.synchro import aligner

        frame_hz = 100
        total_dur_s = 1200.0
        ref_env = [0.0] * int(total_dur_s * frame_hz)
        ref_ivs = []
        sub_ivs = []

        # 8 acts grouped into 4 plateaus of 2 acts each: offsets 0.0s, +10.0s, +20.0s, +30.0s
        plateau_offsets = [0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0]
        for a_idx, off_b in enumerate(plateau_offsets):
            t_cur = 25.0 + a_idx * 130.0
            for c_idx in range(14):
                dur = 1.65 + (c_idx % 4) * 0.40
                s_sub, e_sub = t_cur, t_cur + dur
                sub_ivs.append((s_sub, e_sub))
                s_ref, e_ref = s_sub + off_b, e_sub + off_b
                ref_ivs.append((s_ref, e_ref))
                for k in range(int(round(s_ref * frame_hz)), int(round(e_ref * frame_hz))):
                    ref_env[k] = 0.95
                t_cur += dur + 0.90 + (c_idx % 2) * 0.35

        res = aligner.align_intervals_and_envelopes(
            sub_intervals=sub_ivs,
            ref_envelope=ref_env,
            ref_intervals=ref_ivs,
            frame_hz=frame_hz,
            allow_drift=True,
        )
        self.assertTrue(res.correlated)
        self.assertAlmostEqual(res.a, 1.0, places=5, msg=f"False linear drift fitted to commercial staircase: a={res.a}")

    def test_12_sdh_non_speech_cues_excluded_from_envelope(self):
        """Pure SDH music/SFX cues ([MUSIQUE], ♪♪, (Soupire)) are excluded from intervals while mixed cues remain."""
        from subsync.compat_pysubs2 import SSAEvent
        from subsync.subtitle import Subtitles
        from subsync.synchro import vad

        subs = Subtitles()
        subs.events = [
            SSAEvent(start=1000, end=3000, text="[MUSIQUE DRAMATIQUE]"),
            SSAEvent(start=4000, end=6000, text="♪♪ La la ♪♪"),
            SSAEvent(start=7000, end=9000, text="(Soupire)"),
            SSAEvent(start=10000, end=12500, text="[Il soupire] Bonjour Elizabeth."),
            SSAEvent(start=14000, end=16000, text="On doit partir maintenant."),
        ]
        ivs = vad.subtitles_to_intervals(subs, update_global_state=False)
        self.assertEqual(len(ivs), 2)
        self.assertAlmostEqual(ivs[0][0], 10.0, places=2)
        self.assertAlmostEqual(ivs[1][0], 14.0, places=2)
        self.assertEqual(len(subs.events), 5)

    def test_13_destack_zero_duration_clusters_s09e02_regression(self):
        """Stacked 0ms Mandarin translation cues (S09E02) are unstacked into non-overlapping >=350ms cues."""
        from subsync.compat_pysubs2 import SSAEvent
        from subsync.subtitle import Subtitles, _destack_zero_duration_clusters

        subs = Subtitles()
        subs.events.append(SSAEvent(start=717000, end=719235, text="Previous English line."))
        for idx in range(13):
            subs.events.append(SSAEvent(start=787247, end=787247, text=f"Mandarin line {idx + 1} translated."))
        subs.events.append(SSAEvent(start=787247, end=789500, text="Next English line after Mandarin scene."))

        fixed = _destack_zero_duration_clusters(
            subs.events,
            ref_intervals=[(721.27, 745.0), (746.0, 780.51)],
        )
        self.assertEqual(fixed, 13)
        for i in range(len(subs.events)):
            dur_ms = subs.events[i].end - subs.events[i].start
            self.assertGreaterEqual(dur_ms, 350, f"Cue {i} duration < 350ms: {dur_ms}ms")
            if i > 0:
                self.assertGreaterEqual(
                    subs.events[i].start,
                    subs.events[i - 1].end,
                    f"Cue {i} overlaps cue {i - 1}",
                )

    def test_14_expanded_french_credit_watermarks_cleaned(self):
        """French translator credit lines ('Sous-titres traduits par :', 'Traduction :', 'Resynchro') are stripped."""
        from subsync.compat_pysubs2 import SSAEvent
        from subsync.subtitle import Subtitles

        subs = Subtitles()
        subs.events = [
            SSAEvent(start=1000, end=3000, text="Bonjour Raymond."),
            SSAEvent(start=4000, end=6000, text="Sous-titres traduits par : Sorrento"),
            SSAEvent(start=7000, end=9000, text="Traduction : Equipe Addic7ed"),
            SSAEvent(start=10000, end=48000, text="Resynchro , corrected by Norvegian"),
            SSAEvent(start=50000, end=52000, text="   "),
        ]
        subs.cleanWatermarks()
        self.assertEqual(len(subs.events), 1)
        self.assertEqual(subs.events[0].text, "Bonjour Raymond.")


if __name__ == "__main__":
    unittest.main()
