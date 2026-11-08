"""Deterministic offline mock subtitle provider for unit and E2E testing."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from subsync.download.parser import SeriesEpisodeMetadata
from subsync.download.provider import (
    ProviderError,
    SubtitleCandidate,
    SubtitleProvider,
    normalize_lang,
    score_subtitle_candidate,
)

_FRENCH_SAMPLE_LINES = [
    "Ça doit être bon, nous sommes prêts à commencer.",
    "Où êtes-vous allés pendant tout ce temps ?",
    "Je ne pense pas que ce soit une coïncidence.",
    "Écoutez-moi bien, chaque seconde compte maintenant.",
    "Nous avons vérifié le dossier de Raymond Reddington.",
    "Il faut prévenir l'équipe avant qu'il ne soit trop tard.",
    "C'est exactement ce qu'il voulait nous faire croire.",
    "Vous êtes sûr que personne ne nous a suivis ?",
    "Regardez ces documents, tout était prévu dès le début.",
    "D'accord, allons-y doucement et restons prudents.",
]

_ENGLISH_SAMPLE_LINES = [
    "Everything looks ready, we can begin the operation now.",
    "Where have you been during all this time?",
    "I do not think this is a simple coincidence.",
    "Listen carefully, every single second counts right now.",
    "We checked the entire file on Raymond Reddington.",
    "We need to warn the team before it is too late.",
    "That is对中国 exactly what he wanted us to believe.".replace("对中国", ""),
    "Are you certain that nobody followed us here?",
    "Look at these records, everything was planned from the start.",
    "All right, let us move carefully and stay alert.",
]

_SPANISH_SAMPLE_LINES = [
    "Todo está listo, podemos comenzar la operación ahora.",
    "¿Dónde has estado durante todo este tiempo?",
    "No creo que esto sea una simple coincidencia.",
    "Escucha con atención, cada segundo cuenta ahora mismo.",
    "Hemos revisado todo el expediente del caso.",
]

_SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9\.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9\.]+)")


def format_srt_timestamp(seconds: float) -> str:
    """Format a timestamp in seconds into SRT `HH:MM:SS,mmm` format."""
    clamped = max(0.0, float(seconds))
    total_ms = int(round(clamped * 1000.0))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt_from_intervals(
    intervals: Sequence[Tuple[float, float]],
    lang_code: str = "fr",
    offset_s: float = 0.0,
    speed_factor: float = 1.0,
) -> str:
    """
    Construct a valid `.srt` subtitle document from `(start_s, end_s)` speech intervals.
    If `offset_s` is provided, shifts each event timestamp by `offset_s`.
    """
    if lang_code.startswith("fr"):
        lines_pool = _FRENCH_SAMPLE_LINES
    elif lang_code.startswith("es"):
        lines_pool = _SPANISH_SAMPLE_LINES
    else:
        lines_pool = _ENGLISH_SAMPLE_LINES

    blocks: List[str] = []
    idx = 1
    for i, (raw_start, raw_end) in enumerate(intervals):
        start_t = (raw_start * speed_factor) + offset_s
        end_t = (raw_end * speed_factor) + offset_s
        if end_t <= 0.05:
            continue
        start_t = max(0.0, start_t)
        if end_t - start_t < 0.25:
            end_t = start_t + 1.5
        text_line = lines_pool[i % len(lines_pool)]
        blocks.append(
            f"{idx}\n{format_srt_timestamp(start_t)} --> {format_srt_timestamp(end_t)}\n{text_line}\n"
        )
        idx += 1

    if not blocks:
        blocks.append(
            f"1\n00:00:02,000 --> 00:00:04,500\n{lines_pool[0]}\n"
        )

    return "\n".join(blocks)


def extract_speech_intervals_for_mock(
    video_path: Path,
    max_duration_s: float = 600.0,
    noise_db: float = -35.0,
    min_silence_s: float = 0.25,
) -> List[Tuple[float, float]]:
    """
    Inspect `video_path` using `ffmpeg` silencedetect to find real speech bursts
    so synthetic mock subtitles match the actual audio timeline of the video.
    """
    if not video_path.exists() or not video_path.is_file():
        return []

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "info",
        "-t",
        str(max_duration_s),
        "-i",
        str(video_path),
        "-vn",
        "-sn",
        "-af",
        f"silencedetect=n={noise_db}dB:d={min_silence_s}",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        stderr = proc.stderr or ""
    except Exception:
        return []

    # Parse silence_start and silence_end markers into speech intervals
    events: List[Tuple[str, float]] = []
    for line in stderr.splitlines():
        m_start = _SILENCE_START_RE.search(line)
        if m_start:
            events.append(("start", float(m_start.group(1))))
            continue
        m_end = _SILENCE_END_RE.search(line)
        if m_end:
            events.append(("end", float(m_end.group(1))))

    speech_intervals: List[Tuple[float, float]] = []
    speech_cursor = 0.0
    for kind, ts in events:
        if kind == "start":
            if ts - speech_cursor >= 0.35:
                speech_intervals.append((speech_cursor, min(ts, speech_cursor + 6.0)))
            speech_cursor = ts
        elif kind == "end":
            speech_cursor = ts

    # Filter out intervals too short or split overly long continuous intervals
    refined: List[Tuple[float, float]] = []
    for s_start, s_end in speech_intervals:
        dur = s_end - s_start
        if dur < 0.35:
            continue
        if dur <= 5.5:
            refined.append((round(s_start, 3), round(s_end, 3)))
        else:
            # Break long continuous non-silent blocks into 3.0s dialogue chunks
            cur = s_start
            while cur + 1.0 < s_end and len(refined) < 120:
                chunk_end = min(cur + 3.0, s_end)
                refined.append((round(cur, 3), round(chunk_end, 3)))
                cur = chunk_end + 0.5
        if len(refined) >= 120:
            break

    return refined


class MockSubtitleProvider(SubtitleProvider):
    """
    Deterministic offline subtitle provider for testing.

    Can either serve subtitles from an in-memory `catalog` / `mock_dir`,
    or dynamically generate realistic `.srt` subtitles aligned to the target
    video's speech bursts (with an optional artificial `offset_s` and `output_encoding`
    such as `'iso-8859-1'` or `'utf-8'`).
    """

    name = "mock"

    def __init__(
        self,
        catalog: Optional[Dict[Any, Union[bytes, str, Path]]] = None,
        mock_dir: Optional[Union[Path, str]] = None,
        offset_s: float = 0.0,
        speed_factor: float = 1.0,
        output_encoding: str = "utf-8",
        available_episodes: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        self.catalog: Dict[Any, Union[bytes, str, Path]] = dict(catalog or {})
        self.mock_dir: Optional[Path] = Path(mock_dir) if mock_dir else None
        self.offset_s = float(offset_s)
        self.speed_factor = float(speed_factor)
        self.output_encoding = output_encoding
        self.available_episodes = set(available_episodes) if available_episodes is not None else None
        self._generated_payloads: Dict[str, bytes] = {}

    def _lookup_custom_payload(
        self,
        meta: SeriesEpisodeMetadata,
        lang_code: str,
    ) -> Optional[bytes]:
        keys_to_check = [
            (meta.series_name.lower(), meta.season, meta.episode, lang_code),
            (meta.season, meta.episode, lang_code),
            (meta.season, meta.episode),
            f"S{meta.season:02d}E{meta.episode:02d}",
            meta.video_path.stem,
        ]
        for key in keys_to_check:
            if key in self.catalog:
                val = self.catalog[key]
                if isinstance(val, bytes):
                    return val
                if isinstance(val, Path):
                    return val.read_bytes()
                if isinstance(val, str):
                    return val.encode(self.output_encoding, errors="replace")

        if self.mock_dir and self.mock_dir.exists():
            patterns = [
                f"*{meta.episode_code}*.{lang_code}.srt",
                f"*{meta.episode_code}*.srt",
                f"{meta.video_path.stem}.srt",
            ]
            for pat in patterns:
                matches = sorted(self.mock_dir.glob(pat))
                if matches:
                    return matches[0].read_bytes()

        return None

    def search_subtitles(
        self,
        meta: SeriesEpisodeMetadata,
        target_lang: str,
    ) -> List[SubtitleCandidate]:
        lang_spec = normalize_lang(target_lang)
        if self.available_episodes is not None and (meta.season, meta.episode) not in self.available_episodes:
            return []

        custom_bytes = self._lookup_custom_payload(meta, lang_spec.iso639_1)
        if custom_bytes is not None:
            payload_bytes = custom_bytes
        else:
            intervals = extract_speech_intervals_for_mock(meta.video_path)
            if not intervals:
                # Default synthetic 25-event speech timeline
                intervals = [
                    (2.0 + i * 6.0, 4.2 + i * 6.0)
                    for i in range(25)
                ]
            srt_text = build_srt_from_intervals(
                intervals=intervals,
                lang_code=lang_spec.iso639_1,
                offset_s=self.offset_s,
                speed_factor=self.speed_factor,
            )
            payload_bytes = srt_text.encode(self.output_encoding, errors="replace")

        release_ver = " ".join(meta.release_tags) if meta.release_tags else "MOCK-LOL"
        dl_url = (
            f"mock://subtitles/{meta.series_name.replace(' ', '_')}/"
            f"S{meta.season:02d}E{meta.episode:02d}/{lang_spec.iso639_1}.srt"
        )
        self._generated_payloads[dl_url] = payload_bytes

        score = score_subtitle_candidate(
            meta=meta,
            release_version=release_ver,
            completed=True,
            hearing_impaired=False,
        )

        return [
            SubtitleCandidate(
                provider_name=self.name,
                series_name=meta.series_name,
                season=meta.season,
                episode=meta.episode,
                language=lang_spec.iso639_1,
                release_version=release_ver,
                download_url=dl_url,
                referer=None,
                hearing_impaired=False,
                completed=True,
                score=score,
                extra_metadata={
                    "mock_offset_s": self.offset_s,
                    "charset": self.output_encoding,
                    "video_path": str(meta.video_path),
                },
            )
        ]

    def download_subtitle(self, candidate: SubtitleCandidate) -> bytes:
        if candidate.download_url in self._generated_payloads:
            return self._generated_payloads[candidate.download_url]

        # Re-generate on the fly if download_subtitle is called directly with a constructed candidate
        video_path_str = candidate.extra_metadata.get("video_path")
        video_path = Path(video_path_str) if video_path_str else Path(f"{candidate.series_name}.mkv")
        meta = SeriesEpisodeMetadata(
            video_path=video_path,
            series_name=candidate.series_name,
            season=candidate.season,
            episode=candidate.episode,
        )
        found = self.search_subtitles(meta, candidate.language)
        if not found:
            raise ProviderError(f"Mock subtitle not found for {candidate.download_url}")
        return self._generated_payloads[found[0].download_url]
