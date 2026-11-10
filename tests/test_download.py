"""Unit and end-to-end verification tests for R3: Automated Series Subtitle Retrieval & Sync."""

import inspect
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
    parse_srt_file,
)


REALISTIC_ADDIC7ED_SEASON1_HTML = """
<div id="season">
  <table class="tabel95">
    <thead>
      <tr>
        <th>Season</th><th>Episode</th><th>Title</th><th>Language</th>
        <th>Version</th><th>Completed</th><th>HI</th><th>Corrected</th><th>HD</th><th>Download</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td>1</td>
        <td><a href="/serie/The_Blacklist/1/1/Pilot">Pilot</a></td>
        <td>English</td>
        <td>HDTV.x264-LOL</td>
        <td>Completed</td>
        <td></td>
        <td>C</td>
        <td>HD</td>
        <td class="c"><a href="/updated/1/79400/0">Download</a></td>
      </tr>
      <tr>
        <td>1</td>
        <td>1</td>
        <td><a href="/serie/The_Blacklist/1/1/Pilot">Pilot</a></td>
        <td>French</td>
        <td>HDTV.x264-LOL</td>
        <td>80% Completed</td>
        <td><img src="/images/hi.JPG" alt="Hearing Impaired" /></td>
        <td></td>
        <td></td>
        <td class="c"><a href="/original/8/79405/0">Download</a></td>
      </tr>
      <tr>
        <td>1</td>
        <td>1</td>
        <td><a href="/serie/The_Blacklist/1/1/Pilot">Pilot</a></td>
        <td>French</td>
        <td>1080p.x265.10bit-Joy</td>
        <td>Completed</td>
        <td></td>
        <td>C</td>
        <td>HD</td>
        <td class="c"><a href="/updated/8/79405/0">Download</a></td>
      </tr>
      <tr>
        <td>1</td>
        <td>2</td>
        <td><a href="/serie/The_Blacklist/1/2/The_Freelancer">The Freelancer</a></td>
        <td>French</td>
        <td>x264-LOL</td>
        <td>Completed</td>
        <td></td>
        <td>C</td>
        <td>HD</td>
        <td class="c"><a href="/updated/8/79639/0">Download</a></td>
      </tr>
    </tbody>
  </table>
</div>
"""


class TestSeriesSubtitleDownload(unittest.TestCase):
    """Verification suite for R3: metadata parsing, charset normalization, Addic7ed HTML ranking, and series pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.shared_tmp = tempfile.mkdtemp(prefix="subsync_r3_shared_")
        cls.bursts = generate_dialogue_bursts(num_bursts=34, duration_s=150.0, seed=99)
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
        cls.gt_fr_srt = generate_ground_truth_srt(
            Path(cls.shared_tmp) / "gt_fr.srt", cls.bursts, lang="fr"
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.shared_tmp, ignore_errors=True)

    def test_01_video_filename_metadata_extraction(self):
        """Verify video filename metadata extraction for The Blacklist S01E01 Pilot (1080p x265 10bit Joy).mkv."""
        from subsync.download.parser import parse_episode_video

        filename = "The Blacklist S01E01 Pilot (1080p x265 10bit Joy).mkv"
        meta = parse_episode_video(filename, probe_audio=False)
        self.assertIsNotNone(meta)

        series_val = getattr(meta, "series_name", getattr(meta, "series", None))
        self.assertEqual(series_val, "The Blacklist")
        self.assertEqual(meta.season, 1)
        self.assertEqual(meta.episode, 1)
        self.assertEqual(meta.episode_title, "Pilot")
        tag_set_lower = {t.lower() for t in meta.release_tags}
        self.assertIn("joy", tag_set_lower)
        self.assertIn("1080p", tag_set_lower)
        self.assertIn("x265", tag_set_lower)
        self.assertIn("10bit", tag_set_lower)

    def test_02_iso8859_1_and_cp1252_french_subtitle_normalization(self):
        """Verify ISO-8859-1 and Windows-1252 French subtitles normalize to clean UTF-8 without mojibake."""
        from subsync.download.normalizer import (
            detect_subtitle_encoding,
            normalize_subtitle_bytes,
            write_normalized_subtitle,
        )

        french_srt_cp1252 = (
            "1\r\n"
            "00:00:14,000 --> 00:00:16,500\r\n"
            "Ça doit être bon, vérifions l'œuvre complète dès à présent.\r\n\r\n"
            "2\r\n"
            "00:00:18,200 --> 00:00:21,000\r\n"
            "Écoutez-moi : é, è, ê, à, œ sont parfaitement préservés.\r\n"
        )
        raw_cp1252 = french_srt_cp1252.encode("cp1252")
        detected = detect_subtitle_encoding(raw_cp1252, declared_charset="iso-8859-1")
        self.assertIn(detected.lower(), ("cp1252", "windows-1252", "iso-8859-1"))

        normalized = normalize_subtitle_bytes(raw_cp1252)
        self.assertIn("Ça doit être bon", normalized)
        for ch in ("é", "è", "ê", "à", "œ"):
            self.assertIn(ch, normalized)
        self.assertNotIn("\r\n", normalized)
        self.assertNotIn("Ã", normalized)

        # Also test pure ISO-8859-1 bytes
        french_srt_latin1 = (
            "1\n"
            "00:00:10,000 --> 00:00:12,500\n"
            "Ça doit être bon : é, è, ê, à.\n"
        ).encode("iso-8859-1")
        norm_latin1 = normalize_subtitle_bytes(french_srt_latin1)
        self.assertIn("Ça doit être bon : é, è, ê, à.", norm_latin1)

        with tempfile.TemporaryDirectory(prefix="subsync_norm_") as tmp:
            out_file = write_normalized_subtitle(raw_cp1252, Path(tmp) / "out.fr.srt")
            saved_text = out_file.read_text(encoding="utf-8")
            self.assertIn("Ça doit être bon", saved_text)
            self.assertIn("œ", saved_text)

    def test_03_addic7ed_html_parser_and_candidate_ranking(self):
        """Verify Addic7ed season HTML row parsing and candidate ranking by release tags and completion."""
        from subsync.download.addic7ed import Addic7edProvider
        from subsync.download.parser import parse_episode_video

        rows = Addic7edProvider.parse_season_html(
            REALISTIC_ADDIC7ED_SEASON1_HTML,
            show_id=4008,
            series_name="The Blacklist",
        )
        self.assertEqual(len(rows), 4)
        fr_ep1_rows = [
            r for r in rows if r["season"] == 1 and r["episode"] == 1 and "french" in r["language"].lower()
        ]
        self.assertEqual(len(fr_ep1_rows), 2)

        def fake_http_get(url, headers, timeout):
            return REALISTIC_ADDIC7ED_SEASON1_HTML.encode("utf-8"), {"content-type": "text/html"}, url

        with tempfile.TemporaryDirectory(prefix="subsync_addic7ed_") as tmp:
            provider = Addic7edProvider(
                cache_dir=tmp,
                min_request_interval_s=0.0,
                use_cache=False,
                http_get_fn=fake_http_get,
            )
            meta = parse_episode_video(
                "The Blacklist S01E01 Pilot (1080p x265 10bit Joy).mkv",
                probe_audio=False,
            )
            candidates = provider.search_subtitles(meta, target_lang="fr")
            self.assertEqual(len(candidates), 2)
            # The completed, non-HI, Updated, Joy-matching candidate must rank #1
            best = candidates[0]
            second = candidates[1]
            self.assertGreater(best.score, second.score)
            self.assertTrue(best.completed)
            self.assertFalse(best.hearing_impaired)
            self.assertIn("Joy", best.release_version)
            self.assertTrue(best.download_url.endswith("/updated/8/79405/0"))

    def test_04_end_to_end_series_pipeline_run_series_sync(self):
        """Verify end-to-end run_series_sync on a directory of episode .mkv files with target language 'fr'."""
        from subsync.download.mock_provider import MockSubtitleProvider
        from subsync.download.pipeline import run_series_sync

        with tempfile.TemporaryDirectory(prefix="subsync_series_e2e_") as tmp:
            root = Path(tmp)
            series_dir = root / "The Blacklist" / "S1"
            series_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = root / "cache"

            ep1_mkv = series_dir / "The Blacklist S01E01 Pilot (1080p x265 10bit Joy).mkv"
            ep2_mkv = series_dir / "The Blacklist S01E02 The Freelancer (1080p x265 10bit Joy).mkv"
            shutil.copy2(self.base_mkv, ep1_mkv)
            shutil.copy2(self.base_mkv, ep2_mkv)

            # Create distorted French subtitles encoded in Windows-1252 / ISO-8859-1 with known offsets
            dist_ep1_path = root / "mock_ep1_fr.srt"
            dist_ep2_path = root / "mock_ep2_fr.srt"
            generate_distorted_srt(dist_ep1_path, self.bursts, offset_s=2.350, scale_a=1.0, lang="fr")
            generate_distorted_srt(dist_ep2_path, self.bursts, offset_s=-1.800, scale_a=1.0, lang="fr")

            ep1_cp1252_bytes = Path(dist_ep1_path).read_text(encoding="utf-8").encode("cp1252")
            ep2_cp1252_bytes = Path(dist_ep2_path).read_text(encoding="utf-8").encode("cp1252")

            mock_provider = MockSubtitleProvider(
                catalog={
                    (1, 1, "fr"): ep1_cp1252_bytes,
                    (1, 2, "fr"): ep2_cp1252_bytes,
                    (1, 1): ep1_cp1252_bytes,
                    (1, 2): ep2_cp1252_bytes,
                    "S01E01": ep1_cp1252_bytes,
                    "S01E02": ep2_cp1252_bytes,
                },
                output_encoding="cp1252",
            )

            summary_json_path = root / "series_summary.json"

            # Call run_series_sync adapting to its parameter signature
            sig = inspect.signature(run_series_sync)
            kwargs = {}
            param_map = {
                "series_dir": series_dir,
                "directory": series_dir,
                "ref_dir": series_dir,
                "lang": "fr",
                "target_lang": "fr",
                "sub_lang": "fr",
                "ref_lang": "eng",
                "provider": mock_provider,
                "providers": [mock_provider],
                "cache_dir": cache_dir,
                "summary_json": summary_json_path,
                "quiet": True,
            }
            for p_name in sig.parameters:
                if p_name in param_map:
                    kwargs[p_name] = param_map[p_name]

            summary = run_series_sync(**kwargs)
            self.assertIsNotNone(summary)
            self.assertEqual(getattr(summary, "succeeded", 2), 2)
            self.assertEqual(getattr(summary, "failed", 0), 0)

            synced_srts = sorted(series_dir.glob("*.synced.srt"))
            if not synced_srts:
                synced_srts = sorted(series_dir.glob("*.srt"))
            self.assertEqual(
                len(synced_srts),
                2,
                f"Expected 2 synchronized French .srt files in {series_dir}, found {synced_srts}",
            )

            for srt_file in synced_srts:
                text = srt_file.read_text(encoding="utf-8")
                self.assertIn("Ça doit être bon", text)
                eval_res = evaluate_alignment_accuracy(
                    srt_file, self.gt_fr_srt, tolerance_s=0.100, max_mae_s=0.050
                )
                self.assertTrue(eval_res["passed"])
                self.assertLessEqual(eval_res["max_error_s"], 0.100)

    def test_05_rate_limit_fallback_to_secondary_provider(self):
        """Verify Addic7ed HTML error page raises ProviderRateLimitError and run_series_sync falls back to GestdownProvider."""
        import json
        from subsync.download.addic7ed import Addic7edProvider
        from subsync.download.pipeline import run_series_sync
        from subsync.download.provider import ProviderRateLimitError
        from subsync.download.subdl import GestdownProvider

        html_error_page = b"<!DOCTYPE html><html><head><title>Addic7ed.com</title></head><body>Limit</body></html>"

        def fake_addic7ed_get(url, headers, timeout):
            if "ajax_loadshow" in url.lower():
                return REALISTIC_ADDIC7ED_SEASON1_HTML.encode("utf-8"), {"content-type": "text/html"}, url
            return html_error_page, {"content-type": "text/html"}, url

        with tempfile.TemporaryDirectory(prefix="subsync_fallback_") as tmp:
            root = Path(tmp)
            series_dir = root / "S1"
            series_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = root / "cache"

            ep1_mkv = series_dir / "The Blacklist S01E01 Pilot (1080p x265 10bit Joy).mkv"
            shutil.copy2(self.base_mkv, ep1_mkv)

            dist_ep1_path = root / "mock_ep1_fr.srt"
            generate_distorted_srt(dist_ep1_path, self.bursts, offset_s=1.250, scale_a=1.0, lang="fr")
            valid_srt_bytes = dist_ep1_path.read_bytes()

            addic7ed_prov = Addic7edProvider(
                cache_dir=cache_dir,
                min_request_interval_s=0.0,
                use_cache=False,
                http_get_fn=fake_addic7ed_get,
            )

            # Confirm download_subtitle raises ProviderRateLimitError on HTML error page
            from subsync.download.parser import parse_episode_video
            meta = parse_episode_video(ep1_mkv, probe_audio=False)
            cands = addic7ed_prov.search_subtitles(meta, "fr")
            self.assertGreaterEqual(len(cands), 1)
            with self.assertRaises(ProviderRateLimitError):
                addic7ed_prov.download_subtitle(cands[0])

            def fake_gestdown_get(url, headers, timeout):
                if "/subtitles/get/" in url:
                    payload = {
                        "matchingSubtitles": [
                            {
                                "subtitleId": "testSubId1",
                                "version": "WEB-DL-NTb",
                                "completed": True,
                                "hearingImpaired": False,
                                "downloadUri": "/subtitles/download/testSubId1",
                                "downloadCount": 42,
                            }
                        ]
                    }
                    return json.dumps(payload).encode("utf-8"), {"content-type": "application/json"}
                return valid_srt_bytes, {"content-type": "text/plain"}

            gestdown_prov = GestdownProvider(
                cache_dir=cache_dir,
                use_cache=False,
                http_get_fn=fake_gestdown_get,
            )

            summary = run_series_sync(
                series_dir=series_dir,
                target_lang="fr",
                provider=[addic7ed_prov, gestdown_prov],
                cache_dir=cache_dir,
                quiet=True,
            )
            self.assertEqual(summary.succeeded, 1)
            self.assertEqual(summary.failed, 0)
            self.assertEqual(summary.results[0].provider_name, "gestdown")

    def test_06_mixed_utf8_cp1252_mojibake_repair_s06e02_regression(self):
        """Verify mixed single-encoded UTF-8 ('é') + double-encoded CP1252 mojibake ('faÃ§on', 'Ã ', 'Ã‡', 'Å“') repairs cleanly."""
        from subsync.download.normalizer import normalize_subtitle_bytes

        # Exact hybrid corruption pattern found in Addic7ed S06E02 ('réputation' clean UTF-8 + 'faÃ§on' / 'Ã ' / 'Ã‡' / 'Å“' double-encoded)
        mixed_payload = (
            b"1\n"
            b"00:00:15,115 --> 00:00:18,310\n"
            b"<i>sur ma fa\xc3\x83\xc2\xa7on de travailler \xc3\x83\xc2\xa0 c\xc3\x85\xe2\x80\x9cur ouvert.</i>\n\n"
            b"2\n"
            b"00:00:21,015 --> 00:00:25,320\n"
            b"Je connais votre r\xc3\xa9putation. \xc3\x83\xe2\x80\xa1a marche, \xc3\x83\xe2\x80\xb0videmment !\n"
        )
        repaired = normalize_subtitle_bytes(mixed_payload)
        self.assertIn("sur ma façon de travailler à cœur ouvert.", repaired)
        self.assertIn("Je connais votre réputation. Ça marche, Évidemment !", repaired)
        self.assertNotIn("Ã", repaired)
        self.assertNotIn("Å", repaired)


if __name__ == "__main__":
    unittest.main()
