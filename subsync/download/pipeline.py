"""End-to-end series subtitle download and synchronization orchestrator."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from subsync.download.addic7ed import Addic7edProvider
from subsync.download.mock_provider import (
    MockSubtitleProvider,
    extract_speech_intervals_for_mock,
    format_srt_timestamp,
)
from subsync.download.normalizer import (
    SubtitleNormalizationError,
    normalize_subtitle_bytes,
)
from subsync.download.opensubtitles import OpenSubtitlesProvider
from subsync.download.parser import (
    SeriesEpisodeMetadata,
    parse_episode_video,
    scan_series_directory,
)
from subsync.download.provider import (
    ProviderError,
    ProviderRateLimitError,
    SubtitleCandidate,
    SubtitleProvider,
    normalize_lang,
)
from subsync.download.subdl import (
    GestdownProvider,
    PodnapisiProvider,
    SubDLProvider,
    TVSubtitlesProvider,
)

DEFAULT_OUT_PATTERN = "{ref_dir}/{ref_name}.{lang}.synced.srt"

_SRT_BLOCK_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{2,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{2,3})"
)


@dataclass
class SeriesSyncConfig:
    """Configuration for automated series subtitle download and synchronization."""

    series_dir: Path
    target_lang: str = "fr"
    ref_lang: Optional[str] = None
    provider: str = "auto"
    opensubtitles_api_key: Optional[str] = None
    out_pattern: str = DEFAULT_OUT_PATTERN
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "subsync")
    save_raw_subtitle: bool = False
    recursive: bool = False
    max_episodes: Optional[int] = None
    episodes_filter: Optional[List[int]] = None
    season_filter: Optional[int] = None
    dry_run: bool = False
    skip_if_exists: bool = False
    task_timeout: float = 300.0
    hearing_impaired: bool = False
    summary_json: Optional[Path] = None
    mock_offset_s: float = 0.0
    workers: int = 0
    quiet: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.series_dir, Path):
            self.series_dir = Path(self.series_dir)
        if not isinstance(self.cache_dir, Path):
            self.cache_dir = Path(self.cache_dir)
        if self.summary_json is not None and not isinstance(self.summary_json, Path):
            self.summary_json = Path(self.summary_json)


@dataclass
class EpisodeSyncResult:
    """Execution result for a single series episode."""

    video_path: Path
    series_name: str
    season: int
    episode: int
    episode_title: Optional[str] = None
    audio_lang: str = "eng"
    target_lang: str = "fr"
    status: str = "PENDING"
    provider_name: Optional[str] = None
    candidate_release: Optional[str] = None
    download_url: Optional[str] = None
    raw_subtitle_path: Optional[Path] = None
    output_subtitle_path: Optional[Path] = None
    offset_seconds: Optional[float] = None
    offset_start_s: Optional[float] = None
    offset_mid_s: Optional[float] = None
    offset_end_s: Optional[float] = None
    act_count: int = 1
    offset_evolution: Optional[str] = None
    recap_status: str = "Non"
    dropped_recap_lines: int = 0
    first_sub_time_s: Optional[float] = None
    first_sub_text: Optional[str] = None
    speed_factor: Optional[float] = None
    correlation_score: Optional[float] = None
    raw_correlation_score: Optional[float] = None
    post_correlation_score: Optional[float] = None
    raw_vocal_score: Optional[float] = None
    post_vocal_score: Optional[float] = None
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def episode_code(self) -> str:
        return f"S{self.season:02d}E{self.episode:02d}"

    @property
    def matched_release_display(self) -> str:
        prov = self.provider_name or "-"
        rel = self.candidate_release or ""
        if rel and rel != "-":
            return f"{prov} ({rel})"
        return prov

    @property
    def first_sub_display(self) -> str:
        if self.first_sub_time_s is not None and self.first_sub_text:
            snippet = self.first_sub_text[:48]
            return f"{self.first_sub_time_s:.2f}s : « {snippet} »"
        return "-"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "series_name": self.series_name,
            "season": self.season,
            "episode": self.episode,
            "episode_code": self.episode_code,
            "episode_title": self.episode_title,
            "audio_lang": self.audio_lang,
            "target_lang": self.target_lang,
            "status": self.status,
            "provider_name": self.provider_name,
            "candidate_release": self.candidate_release,
            "download_url": self.download_url,
            "raw_subtitle_path": str(self.raw_subtitle_path) if self.raw_subtitle_path else None,
            "output_subtitle_path": str(self.output_subtitle_path) if self.output_subtitle_path else None,
            "offset_seconds": round(float(self.offset_seconds), 4) if self.offset_seconds is not None else None,
            "offset_start_s": round(float(self.offset_start_s), 4) if self.offset_start_s is not None else None,
            "offset_mid_s": round(float(self.offset_mid_s), 4) if self.offset_mid_s is not None else None,
            "offset_end_s": round(float(self.offset_end_s), 4) if self.offset_end_s is not None else None,
            "act_count": int(self.act_count),
            "offset_evolution": self.offset_evolution,
            "recap_status": self.recap_status,
            "dropped_recap_lines": int(self.dropped_recap_lines),
            "first_sub_time_s": round(float(self.first_sub_time_s), 3) if self.first_sub_time_s is not None else None,
            "first_sub_text": self.first_sub_text,
            "speed_factor": round(float(self.speed_factor), 6) if self.speed_factor is not None else None,
            "correlation_score": round(float(self.correlation_score), 4) if self.correlation_score is not None else None,
            "raw_correlation_score": round(float(self.raw_correlation_score), 4) if self.raw_correlation_score is not None else None,
            "post_correlation_score": round(float(self.post_correlation_score), 4) if self.post_correlation_score is not None else None,
            "raw_vocal_score": round(float(self.raw_vocal_score), 4) if self.raw_vocal_score is not None else None,
            "post_vocal_score": round(float(self.post_vocal_score), 4) if self.post_vocal_score is not None else None,
            "elapsed_seconds": round(float(self.elapsed_seconds), 3),
            "error": self.error,
        }


@dataclass
class SeriesSyncSummary:
    """Aggregated summary of a `run_series_sync` execution across a series directory."""

    series_dir: Path
    target_lang: str
    provider: str
    total_episodes: int
    succeeded: int
    skipped: int
    failed: int
    dry_run: bool = False
    elapsed_seconds: float = 0.0
    results: List[EpisodeSyncResult] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        if self.total_episodes == 0:
            return 1
        if self.failed == 0:
            return 0
        if self.succeeded > 0 or self.skipped > 0:
            return 2
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_dir": str(self.series_dir),
            "target_lang": self.target_lang,
            "provider": self.provider,
            "total_episodes": self.total_episodes,
            "succeeded": self.succeeded,
            "skipped": self.skipped,
            "failed": self.failed,
            "dry_run": self.dry_run,
            "elapsed_seconds": round(float(self.elapsed_seconds), 3),
            "exit_code": self.exit_code,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def format_markdown(self) -> str:
        """Render a rich Markdown recap table identical to the detailed post-sync audit report."""
        lines: List[str] = [
            "| Épisode | Version retenue | Récap « Précédemment » | Évolution Décalage (Début → Milieu → Fin) | Corrélation IA (Avant → Après) | 1re réplique synchronisée |",
            "| :--- | :--- | :--- | :--- | :---: | :--- |",
        ]
        for r in self.results:
            ep_label = f"**{r.episode_code}** — *{r.episode_title}*" if r.episode_title else f"**{r.episode_code}**"
            ver = f"`{r.matched_release_display}`" if r.provider_name else f"`{r.status}`"
            recap = r.recap_status or "Non"
            if r.offset_evolution:
                off_evo = f"`{r.offset_evolution}`"
            elif r.offset_seconds is not None:
                off_evo = f"`{r.offset_seconds:+.2f}s (constant)`"
            else:
                off_evo = "-"
            if r.raw_correlation_score is not None and r.correlation_score is not None:
                corr_str = f"`{r.raw_correlation_score * 100.0:.1f}%` → **{r.correlation_score * 100.0:.1f}%**"
            elif r.correlation_score is not None:
                corr_str = f"**{r.correlation_score * 100.0:.1f}%**"
            else:
                corr_str = "-"
            first_str = r.first_sub_display if r.status == "SUCCESS" else (r.error or r.status)
            lines.append(f"| {ep_label} | {ver} | {recap} | {off_evo} | {corr_str} | {first_str} |")
        return "\n".join(lines)

    def format_table(self) -> str:
        """Render a rich console recap table showing version, recap handling, multi-act offset evolution, IA score, and 1st line."""
        headers = [
            "Épisode",
            "Version retenue",
            "Récap « Précédemment »",
            "Évolution Décalage (Début -> Milieu -> Fin)",
            "Corr. IA (Avant -> Après)",
            "1re réplique synchronisée",
        ]
        rows: List[List[str]] = []
        for r in self.results:
            title_short = (r.episode_title or "")[:20]
            ep_col = f"{r.episode_code} {title_short}".strip()
            ver_col = r.matched_release_display[:34] if r.provider_name else r.status
            recap_col = (r.recap_status or "Non")[:24]
            if r.offset_evolution:
                off_col = r.offset_evolution
            elif r.offset_seconds is not None:
                off_col = f"{r.offset_seconds:+.2f}s (constant)"
            else:
                off_col = "-"
            if r.raw_correlation_score is not None and r.correlation_score is not None:
                corr_col = f"{r.raw_correlation_score * 100.0:.1f}% -> {r.correlation_score * 100.0:.1f}%"
            elif r.correlation_score is not None:
                corr_col = f"{r.correlation_score * 100.0:.1f}%"
            else:
                corr_col = "-"
            first_col = r.first_sub_display if r.status == "SUCCESS" else (r.error or r.status)
            rows.append([ep_col, ver_col, recap_col, off_col, corr_col, first_col])

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if len(cell) > col_widths[i]:
                    col_widths[i] = len(cell)

        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        head_sep = "+" + "+".join("=" * (w + 2) for w in col_widths) + "+"

        def _fmt_row(cells: Sequence[str]) -> str:
            parts = []
            for i, cell in enumerate(cells):
                if i == 4:
                    parts.append(f" {cell:>{col_widths[i]}} ")
                else:
                    parts.append(f" {cell:<{col_widths[i]}} ")
            return "|" + "|".join(parts) + "|"

        lines: List[str] = [
            "",
            f"=== RÉCAPITULATIF SYNCHRONISATION HYBRIDE (MATH + IA) : {self.series_dir} ===",
            sep,
            _fmt_row(headers),
            head_sep,
        ]
        for row in rows:
            lines.append(_fmt_row(row))
        lines.append(sep)
        lines.append(
            f"Total: {self.total_episodes} | Succeeded: {self.succeeded} | "
            f"Skipped: {self.skipped} | Failed: {self.failed} | "
            f"Elapsed: {self.elapsed_seconds:.2f}s"
        )
        return "\n".join(lines)


def format_series_output_path(
    meta: SeriesEpisodeMetadata,
    target_lang: str,
    out_pattern: str = DEFAULT_OUT_PATTERN,
) -> Path:
    """
    Format the output subtitle file path using template placeholders:
      `{ref_dir}`, `{ref_name}`, `{ref_path}`, `{video_dir}`, `{video_name}`,
      `{lang}`, `{sub_lang}`, `{lang3}`, `{ref_lang}`, `{audio_lang}`,
      `{series}`, `{series_name}`, `{season}`, `{episode}`, `{episode_code}`.
    """
    lang_spec = normalize_lang(target_lang)
    ref_dir = str(meta.video_path.parent)
    ref_name = meta.video_path.stem

    context = {
        "ref_dir": ref_dir,
        "video_dir": ref_dir,
        "sub_dir": ref_dir,
        "ref_name": ref_name,
        "video_name": ref_name,
        "sub_name": ref_name,
        "ref_path": str(meta.video_path),
        "video_path": str(meta.video_path),
        "lang": lang_spec.iso639_1,
        "sub_lang": lang_spec.iso639_1,
        "lang2": lang_spec.iso639_1,
        "lang3": lang_spec.iso639_2,
        "ref_lang": meta.audio_lang,
        "audio_lang": meta.audio_lang,
        "series": meta.series_name,
        "series_name": meta.series_name,
        "season": meta.season,
        "episode": meta.episode,
        "episode_code": meta.episode_code,
        "title": meta.episode_title or "",
    }
    try:
        rendered = out_pattern.format(**context)
    except KeyError:
        rendered = f"{ref_dir}/{ref_name}.{lang_spec.iso639_1}.synced.srt"
    return Path(rendered)


def _parse_ts_seconds(h: str, m: str, s: str, ms: str) -> float:
    millis = int(ms.ljust(3, "0")[:3])
    return int(h) * 3600.0 + int(m) * 60.0 + int(s) + (millis / 1000.0)


def parse_srt_events(srt_text: str) -> List[Tuple[float, float]]:
    """Extract `(start_s, end_s)` intervals from an SRT subtitle document."""
    intervals: List[Tuple[float, float]] = []
    for match in _SRT_BLOCK_TIME_RE.finditer(srt_text):
        start_s = _parse_ts_seconds(match.group(1), match.group(2), match.group(3), match.group(4))
        end_s = _parse_ts_seconds(match.group(5), match.group(6), match.group(7), match.group(8))
        if end_s > start_s:
            intervals.append((start_s, end_s))
    return intervals


def shift_srt_text(
    srt_text: str,
    offset_seconds: float = 0.0,
    speed_factor: float = 1.0,
    formula: Optional[Any] = None,
) -> str:
    """
    Transform all SRT timing lines in `srt_text` by:
      `t_new = max(0.0, formula.getY(t_old))` when `formula.piecewise_anchors` is present,
      or `t_new = max(0.0, t_old * speed_factor + offset_seconds)`.
    """
    has_piecewise = bool(
        formula is not None
        and getattr(formula, "piecewise_anchors", None)
        and hasattr(formula, "getY")
    )

    def _replace_match(match: re.Match) -> str:
        start_s = _parse_ts_seconds(match.group(1), match.group(2), match.group(3), match.group(4))
        end_s = _parse_ts_seconds(match.group(5), match.group(6), match.group(7), match.group(8))
        if has_piecewise:
            new_start = max(0.0, float(formula.getY(start_s)))
            new_end = max(new_start + 0.05, float(formula.getY(end_s)))
        else:
            new_start = max(0.0, start_s * speed_factor + offset_seconds)
            new_end = max(new_start + 0.05, end_s * speed_factor + offset_seconds)
        return f"{format_srt_timestamp(new_start)} --> {format_srt_timestamp(new_end)}"

    transformed = _SRT_BLOCK_TIME_RE.sub(_replace_match, srt_text)
    if not transformed.endswith("\n"):
        transformed += "\n"
    return transformed


def _interval_overlap_score(
    sub_intervals: Sequence[Tuple[float, float]],
    audio_starts: Sequence[float],
    audio_ends: Sequence[float],
    offset_s: float,
    speed_factor: float = 1.0,
) -> float:
    """
    Compute total temporal overlap (in seconds) plus boundary alignment bonus
    between shifted subtitle intervals `[s * speed + offset, e * speed + offset]`
    and detected audio speech intervals `[a_start, a_end]`.
    """
    total_score = 0.0
    n_audio = len(audio_starts)
    for s_start, s_end in sub_intervals:
        t0 = s_start * speed_factor + offset_s
        t1 = s_end * speed_factor + offset_s
        if t1 <= 0.0:
            continue
        # Binary search first audio interval that can overlap [t0, t1]
        idx = bisect.bisect_right(audio_ends, t0)
        while idx < n_audio:
            a0 = audio_starts[idx]
            if a0 >= t1 + 0.15:
                break
            a1 = audio_ends[idx]
            ov = min(t1, a1) - max(t0, a0)
            if ov > 0.0:
                total_score += ov
            # Bonus for sharp speech onset alignment within 100ms
            onset_diff = abs(t0 - a0)
            if onset_diff < 0.10:
                total_score += 0.25 * (1.0 - onset_diff / 0.10)
            idx += 1
    return total_score


def _fallback_vad_synchronize_pair(
    video_path: Path,
    raw_srt_path: Path,
    out_srt_path: Path,
    max_offset_s: float = 30.0,
) -> Dict[str, Any]:
    """
    Built-in high-precision (<= 0.005s resolution) VAD interval cross-correlator
    used when `subsync.synchro.modern_engine` is not present.
    """
    srt_text = raw_srt_path.read_text(encoding="utf-8")
    sub_intervals = parse_srt_events(srt_text)
    audio_intervals = extract_speech_intervals_for_mock(video_path, max_duration_s=900.0)

    if not sub_intervals or not audio_intervals:
        out_srt_path.parent.mkdir(parents=True, exist_ok=True)
        out_srt_path.write_text(srt_text, encoding="utf-8", newline="\n")
        return {
            "offset_seconds": 0.0,
            "speed_factor": 1.0,
            "correlation_score": 1.0,
            "output_path": str(out_srt_path),
        }

    # Restrict subtitle intervals to the analyzed audio window (+ margin)
    max_audio_t = audio_intervals[-1][1] + max_offset_s + 5.0
    sub_window = [(s, e) for s, e in sub_intervals if s <= max_audio_t][:150]
    if not sub_window:
        sub_window = sub_intervals[:150]

    audio_starts = [a for a, _ in audio_intervals]
    audio_ends = [b for _, b in audio_intervals]

    # Stage 1: Coarse grid search at 25ms (0.025s) resolution + direct onset difference candidates
    candidate_offsets = set()
    num_coarse = int(round((2.0 * max_offset_s) / 0.025)) + 1
    for i in range(num_coarse):
        candidate_offsets.add(round(-max_offset_s + i * 0.025, 4))

    # Add exact onset differences (a_start - s_start) within [-max_offset_s, +max_offset_s]
    for s_start, _ in sub_window[:60]:
        idx_low = bisect.bisect_left(audio_starts, s_start - max_offset_s)
        idx_high = bisect.bisect_right(audio_starts, s_start + max_offset_s)
        for j in range(idx_low, idx_high):
            candidate_offsets.add(round(audio_starts[j] - s_start, 4))

    best_offset = 0.0
    best_score = _interval_overlap_score(sub_window, audio_starts, audio_ends, 0.0)

    for cand in sorted(candidate_offsets):
        sc = _interval_overlap_score(sub_window, audio_starts, audio_ends, cand)
        if sc > best_score + 1e-6 or (abs(sc - best_score) <= 1e-6 and abs(cand) < abs(best_offset)):
            best_score = sc
            best_offset = cand

    # Stage 2: Fine search around best_offset at 2ms (0.002s) resolution
    fine_best_offset = best_offset
    fine_best_score = best_score
    for step in range(-25, 26):
        cand = round(best_offset + step * 0.002, 4)
        if abs(cand) > max_offset_s:
            continue
        sc = _interval_overlap_score(sub_window, audio_starts, audio_ends, cand)
        if sc > fine_best_score + 1e-6 or (
            abs(sc - fine_best_score) <= 1e-6 and abs(cand) < abs(fine_best_offset)
        ):
            fine_best_score = sc
            fine_best_offset = cand

    total_sub_dur = sum(e - s for s, e in sub_window) or 1.0
    corr_score = min(1.0, max(0.0, fine_best_score / total_sub_dur))

    synced_text = shift_srt_text(srt_text, offset_seconds=fine_best_offset, speed_factor=1.0)
    out_srt_path.parent.mkdir(parents=True, exist_ok=True)
    out_srt_path.write_text(synced_text, encoding="utf-8", newline="\n")

    return {
        "offset_seconds": round(fine_best_offset, 4),
        "speed_factor": 1.0,
        "correlation_score": round(corr_score, 4),
        "output_path": str(out_srt_path),
    }


from contextlib import contextmanager

_REF_ENVELOPE_CACHE: Dict[
    Tuple[str, Optional[int], Optional[str], int],
    Tuple[List[float], List[Tuple[float, float]]],
] = {}
_MAX_ENVELOPE_CACHE_ITEMS = 4


def get_cached_speech_envelope(
    video_path: Union[Path, str],
    stream_index: Optional[int] = None,
    stream_type: Optional[str] = None,
    encoding: str = "utf-8",
    fps: Optional[float] = None,
    frame_hz: int = 100,
) -> Tuple[List[float], List[Tuple[float, float]]]:
    """Extract or retrieve from in-memory cache the 100 Hz reference speech envelope."""
    from subsync.synchro import vad

    key = (str(Path(video_path).resolve()), stream_index, stream_type, int(frame_hz))
    cached = _REF_ENVELOPE_CACHE.get(key)
    if cached is not None:
        return cached

    orig_fn = getattr(vad, "_orig_extract_speech_envelope", vad.extract_speech_envelope)
    env, intervals = orig_fn(
        str(video_path),
        stream_index=stream_index,
        stream_type=stream_type,
        encoding=encoding,
        fps=fps,
        frame_hz=frame_hz,
    )
    if len(_REF_ENVELOPE_CACHE) >= _MAX_ENVELOPE_CACHE_ITEMS:
        oldest_key = next(iter(_REF_ENVELOPE_CACHE))
        _REF_ENVELOPE_CACHE.pop(oldest_key, None)
    _REF_ENVELOPE_CACHE[key] = (env, intervals)
    return env, intervals


@contextmanager
def _cached_reference_envelope_context(video_path: Path):
    """Temporarily wrap `vad.extract_speech_envelope` so multi-candidate evaluation extracts audio once."""
    try:
        from subsync.synchro import vad
    except ImportError:
        yield
        return

    if getattr(vad, "_orig_extract_speech_envelope", None) is not None:
        # Already wrapped by an outer context
        yield
        return

    orig_fn = vad.extract_speech_envelope
    vad._orig_extract_speech_envelope = orig_fn
    target_resolved = str(Path(video_path).resolve())

    def _cached_extract(
        path: str,
        stream_index: Optional[int] = None,
        stream_type: Optional[str] = None,
        encoding: str = "utf-8",
        fps: Optional[float] = None,
        frame_hz: int = vad.FRAME_HZ,
    ):
        try:
            if str(Path(path).resolve()) == target_resolved:
                return get_cached_speech_envelope(
                    path,
                    stream_index=stream_index,
                    stream_type=stream_type,
                    encoding=encoding,
                    fps=fps,
                    frame_hz=frame_hz,
                )
        except Exception:
            pass
        return orig_fn(
            path,
            stream_index=stream_index,
            stream_type=stream_type,
            encoding=encoding,
            fps=fps,
            frame_hz=frame_hz,
        )

    vad.extract_speech_envelope = _cached_extract
    try:
        yield
    finally:
        vad.extract_speech_envelope = orig_fn
        delattr(vad, "_orig_extract_speech_envelope")


_SRT_FULL_BLOCK_RE = re.compile(
    r"(?:^\d+\s*\n)?"
    r"(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{2,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,\.](\d{2,3})[^\n]*\n"
    r"(.*?)(?=\n\s*\n|\Z)",
    re.MULTILINE | re.DOTALL,
)

_WATERMARK_RE = re.compile(
    r"(addic7ed|subsynchro|tvsubtitles|opensubtitles|podnapisi|subdl|gestdown|"
    r"sync(?:hro(?:nis[ée])?)?\s*(?:par|by)|traduit\s*par|traduction\s*:|"
    r"sous-titres\s*(?:par|fr)|www\.|http://|https://)",
    re.IGNORECASE,
)

_RECAP_KEYWORDS = ("précédemment", "precedemment", "previously on", "previously")


def _parse_srt_blocks_with_text(srt_text: str) -> List[Tuple[float, float, str]]:
    """Parse `(start_s, end_s, clean_text)` blocks from an SRT document, excluding watermarks."""
    blocks: List[Tuple[float, float, str]] = []
    for match in _SRT_FULL_BLOCK_RE.finditer(srt_text or ""):
        start_s = _parse_ts_seconds(match.group(1), match.group(2), match.group(3), match.group(4))
        end_s = _parse_ts_seconds(match.group(5), match.group(6), match.group(7), match.group(8))
        raw_txt = match.group(9) or ""
        clean_txt = re.sub(r"<[^>]+>", "", raw_txt)
        clean_txt = re.sub(r"\{[^}]+\}", "", clean_txt)
        clean_txt = " ".join(clean_txt.split()).strip()
        if not clean_txt:
            continue
        if _WATERMARK_RE.search(clean_txt):
            continue
        blocks.append((start_s, end_s, clean_txt))
    return blocks


def _extract_act_offsets(formula: Any, window_offsets: Any, default_offset: float, speed_factor: float = 1.0) -> List[float]:
    """Return ordered per-act offsets `b_i` across the episode."""
    win_bs: List[float] = []
    if window_offsets and isinstance(window_offsets, (list, tuple)):
        for item in window_offsets:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                win_bs.append(float(item[2]))
            elif isinstance(item, (int, float)):
                win_bs.append(float(item))
    if not win_bs and formula is not None:
        anchors = getattr(formula, "piecewise_anchors", None)
        slope_a = float(getattr(formula, "a", speed_factor) or speed_factor)
        if anchors and isinstance(anchors, (list, tuple)):
            for pt in anchors:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    px, py = float(pt[0]), float(pt[1])
                    win_bs.append(py - slope_a * px)
    return win_bs if win_bs else [float(default_offset)]


def extract_recap_and_dialogue_summary(
    raw_srt_text: str,
    synced_srt_text: str,
    offset_seconds: float,
    speed_factor: float = 1.0,
    formula: Any = None,
    window_offsets: Any = None,
) -> Dict[str, Any]:
    """
    Compute rich post-sync diagnostics for an episode:
      - `offset_start_s`, `offset_mid_s`, `offset_end_s`, `act_count`, `offset_evolution`
      - `recap_status` ("Non", "Purge auto (N lignes)", "Inclus (Vidéo + Sub)")
      - `dropped_recap_lines`
      - `first_sub_time_s`, `first_sub_text`
    """
    act_bs = _extract_act_offsets(formula, window_offsets, offset_seconds, speed_factor)
    act_count = len(act_bs)
    off_start = float(act_bs[0])
    off_mid = float(act_bs[act_count // 2])
    off_end = float(act_bs[-1])
    spread = max(act_bs) - min(act_bs) if act_count >= 2 else 0.0

    if act_count >= 2 and spread >= 0.15:
        offset_evolution = f"{off_start:+.2f}s -> {off_mid:+.2f}s -> {off_end:+.2f}s ({act_count} actes)"
    else:
        offset_evolution = f"{offset_seconds:+.2f}s (constant)"

    raw_blocks = _parse_srt_blocks_with_text(raw_srt_text)
    synced_blocks = _parse_srt_blocks_with_text(synced_srt_text)

    has_piecewise = bool(
        formula is not None
        and getattr(formula, "piecewise_anchors", None)
        and hasattr(formula, "getY")
    )

    dropped_lines = 0
    has_recap_word_raw = False
    for st, et, txt in raw_blocks:
        if st <= 90.0 and any(k in txt.lower() for k in _RECAP_KEYWORDS):
            has_recap_word_raw = True
        if st <= 120.0:
            if has_piecewise:
                new_st = float(formula.getY(st))
                new_et = float(formula.getY(et))
            else:
                new_st = st * speed_factor + off_start
                new_et = et * speed_factor + off_start
            if new_st < -0.30 or new_et <= 0.0:
                dropped_lines += 1

    # Also compare block count difference if synced_blocks dropped early lines
    if dropped_lines == 0 and len(raw_blocks) > len(synced_blocks):
        diff = len(raw_blocks) - len(synced_blocks)
        if diff > 0 and (off_start < -2.0 or has_recap_word_raw):
            dropped_lines = diff

    has_recap_word_synced = any(
        st <= 65.0 and any(k in txt.lower() for k in _RECAP_KEYWORDS)
        for st, _, txt in synced_blocks[:10]
    )

    if dropped_lines > 0:
        recap_status = f"Purge auto ({dropped_lines} lignes)"
    elif has_recap_word_synced or has_recap_word_raw:
        recap_status = "Inclus (Vidéo + Sub)"
    else:
        recap_status = "Non"

    first_sub_time_s: Optional[float] = None
    first_sub_text: Optional[str] = None
    if synced_blocks:
        first_sub_time_s = round(max(0.0, float(synced_blocks[0][0])), 3)
        first_sub_text = synced_blocks[0][2]

    return {
        "offset_start_s": round(off_start, 4),
        "offset_mid_s": round(off_mid, 4),
        "offset_end_s": round(off_end, 4),
        "act_count": act_count,
        "offset_evolution": offset_evolution,
        "recap_status": recap_status,
        "dropped_recap_lines": dropped_lines,
        "first_sub_time_s": first_sub_time_s,
        "first_sub_text": first_sub_text,
    }


def _compute_window_spread(formula: Any, window_offsets: Any, speed_factor: float = 1.0) -> float:
    """Compute the spread (max - min) of local window offsets in seconds across an episode."""
    win_bs: List[float] = []
    if window_offsets and isinstance(window_offsets, (list, tuple)):
        for item in window_offsets:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                win_bs.append(float(item[2]))
            elif isinstance(item, (int, float)):
                win_bs.append(float(item))
    if not win_bs and formula is not None:
        anchors = getattr(formula, "piecewise_anchors", None)
        slope_a = float(getattr(formula, "a", speed_factor) or speed_factor)
        if anchors and isinstance(anchors, (list, tuple)):
            for pt in anchors:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    px, py = float(pt[0]), float(pt[1])
                    win_bs.append(py - slope_a * px)
    if len(win_bs) >= 2:
        return max(win_bs) - min(win_bs)
    return 0.0


def _extract_sync_metrics(raw_result: Any, out_path: Path) -> Dict[str, Any]:
    """Normalize the return value of any `synchronize_pair` or custom `sync_fn`."""
    offset_s = 0.0
    speed_factor = 1.0
    corr = 1.0
    raw_corr = None
    post_corr = None
    raw_vocal = None
    post_vocal = None
    formula = None
    window_offsets = None

    if isinstance(raw_result, dict):
        formula = raw_result.get("formula")
        window_offsets = raw_result.get("window_offsets")
        if raw_result.get("raw_factor") is not None:
            raw_corr = float(raw_result["raw_factor"])
        if raw_result.get("post_factor") is not None:
            post_corr = float(raw_result["post_factor"])
        if raw_result.get("raw_vocal_score") is not None:
            raw_vocal = float(raw_result["raw_vocal_score"])
        if raw_result.get("post_vocal_score") is not None:
            post_vocal = float(raw_result["post_vocal_score"])
        for k in ("offset_seconds", "offset_s", "offset", "shift", "delay", "b"):
            if raw_result.get(k) is not None:
                offset_s = float(raw_result[k])
                break
        for k in ("speed_factor", "scale", "drift", "slope", "a"):
            if raw_result.get(k) is not None:
                speed_factor = float(raw_result[k])
                break
        for k in ("correlation_score", "correlation", "score", "factor", "r2", "confidence"):
            if raw_result.get(k) is not None:
                corr = float(raw_result[k])
                break
        if raw_result.get("success") is False or raw_result.get("correlated") is False:
            err_msg = raw_result.get("error") or "Synchronization failed to correlate"
            raise RuntimeError(str(err_msg))
    elif isinstance(raw_result, (int, float)):
        offset_s = float(raw_result)
    elif isinstance(raw_result, tuple) and len(raw_result) >= 1:
        if isinstance(raw_result[0], (int, float)):
            offset_s = float(raw_result[0])
        if len(raw_result) >= 2 and isinstance(raw_result[1], (int, float)):
            speed_factor = float(raw_result[1])
    elif raw_result is not None:
        formula = getattr(raw_result, "formula", None)
        window_offsets = getattr(raw_result, "window_offsets", None)
        if getattr(raw_result, "raw_factor", None) is not None:
            raw_corr = float(getattr(raw_result, "raw_factor"))
        if getattr(raw_result, "post_factor", None) is not None:
            post_corr = float(getattr(raw_result, "post_factor"))
        if getattr(raw_result, "raw_vocal_score", None) is not None:
            raw_vocal = float(getattr(raw_result, "raw_vocal_score"))
        if getattr(raw_result, "post_vocal_score", None) is not None:
            post_vocal = float(getattr(raw_result, "post_vocal_score"))
        for attr in ("offset_seconds", "offset_s", "offset", "shift", "delay", "b"):
            val = getattr(raw_result, attr, None)
            if val is not None:
                offset_s = float(val)
                break
        for attr in ("speed_factor", "scale", "drift", "slope", "a"):
            val = getattr(raw_result, attr, None)
            if val is not None:
                speed_factor = float(val)
                break
        for attr in ("correlation_score", "correlation", "score", "factor", "r2", "confidence"):
            val = getattr(raw_result, attr, None)
            if val is not None:
                corr = float(val)
                break
        succ = getattr(raw_result, "success", getattr(raw_result, "correlated", True))
        if succ is False:
            err_msg = getattr(raw_result, "error", None) or "Synchronization failed to correlate"
            raise RuntimeError(str(err_msg))

    window_spread_s = _compute_window_spread(formula, window_offsets, speed_factor)

    return {
        "offset_seconds": offset_s,
        "speed_factor": speed_factor,
        "correlation_score": corr,
        "raw_correlation_score": raw_corr,
        "post_correlation_score": post_corr if post_corr is not None else corr,
        "raw_vocal_score": raw_vocal,
        "post_vocal_score": post_vocal,
        "formula": formula,
        "window_offsets": window_offsets,
        "window_spread_s": window_spread_s,
        "output_path": str(out_path),
    }


def _call_flexible_sync_fn(
    fn: Callable[..., Any],
    video_path: Path,
    raw_srt_path: Path,
    out_srt_path: Path,
    ref_lang: str,
    sub_lang: str,
    timeout: float,
) -> Dict[str, Any]:
    """
    Invoke `fn` (either `subsync.synchro.modern_engine.synchronize_pair` or a user-supplied
    `sync_fn`) by inspecting its parameter names so any signature convention works seamlessly.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    param_names = [p.name for p in params]
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)

    name_map: Dict[str, Any] = {
        "sub_path": raw_srt_path,
        "subtitle_path": raw_srt_path,
        "sub": raw_srt_path,
        "in_sub": raw_srt_path,
        "ref_path": video_path,
        "reference_path": video_path,
        "video_path": video_path,
        "ref": video_path,
        "video": video_path,
        "out_path": out_srt_path,
        "output_path": out_srt_path,
        "out": out_srt_path,
        "dst_path": out_srt_path,
        "ref_lang": ref_lang,
        "audio_lang": ref_lang,
        "sub_lang": sub_lang,
        "lang": sub_lang,
        "target_lang": sub_lang,
        "timeout": timeout,
        "task_timeout": timeout,
    }

    kwargs: Dict[str, Any] = {}
    for pname in param_names:
        if pname in name_map:
            kwargs[pname] = name_map[pname]

    if kwargs:
        res = fn(**kwargs)
    elif len(params) >= 3 or any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        # Standard positional fallback: check if first arg looks like video or sub
        first_name = param_names[0].lower() if param_names else ""
        if "ref" in first_name or "vid" in first_name:
            res = fn(video_path, raw_srt_path, out_srt_path)
        else:
            res = fn(raw_srt_path, video_path, out_srt_path)
    elif len(params) == 2:
        res = fn(raw_srt_path, video_path)
    elif has_var_keyword:
        res = fn(
            sub_path=raw_srt_path,
            ref_path=video_path,
            out_path=out_srt_path,
            ref_lang=ref_lang,
            sub_lang=sub_lang,
        )
    else:
        res = fn(raw_srt_path, video_path, out_srt_path)

    metrics = _extract_sync_metrics(res, out_srt_path)
    # If custom sync_fn returned an offset without writing out_srt_path yet, write it now
    if not out_srt_path.exists():
        raw_text = raw_srt_path.read_text(encoding="utf-8")
        synced_text = shift_srt_text(
            raw_text,
            offset_seconds=metrics["offset_seconds"],
            speed_factor=metrics["speed_factor"],
            formula=metrics.get("formula"),
        )
        out_srt_path.parent.mkdir(parents=True, exist_ok=True)
        out_srt_path.write_text(synced_text, encoding="utf-8", newline="\n")

    return metrics


def synchronize_episode_subtitle(
    video_path: Path,
    raw_srt_path: Path,
    out_srt_path: Path,
    ref_lang: str = "eng",
    sub_lang: str = "fr",
    timeout: float = 300.0,
    sync_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """
    Synchronize `raw_srt_path` against `video_path` and write aligned subtitle to `out_srt_path`.
    Uses `sync_fn` if supplied, otherwise delegates to `subsync.synchro.modern_engine.synchronize_pair`
    with automatic fallback to the built-in 2ms VAD cross-correlation engine.
    """
    if sync_fn is not None:
        return _call_flexible_sync_fn(
            sync_fn,
            video_path=video_path,
            raw_srt_path=raw_srt_path,
            out_srt_path=out_srt_path,
            ref_lang=ref_lang,
            sub_lang=sub_lang,
            timeout=timeout,
        )

    for mod_name in ("subsync.synchro.modern_engine", "subsync.synchro.aligner"):
        try:
            mod = __import__(mod_name, fromlist=["synchronize_pair"])
            sync_pair_fn = getattr(mod, "synchronize_pair", None)
            if callable(sync_pair_fn):
                return _call_flexible_sync_fn(
                    sync_pair_fn,
                    video_path=video_path,
                    raw_srt_path=raw_srt_path,
                    out_srt_path=out_srt_path,
                    ref_lang=ref_lang,
                    sub_lang=sub_lang,
                    timeout=timeout,
                )
        except ImportError:
            continue

    return _fallback_vad_synchronize_pair(
        video_path=video_path,
        raw_srt_path=raw_srt_path,
        out_srt_path=out_srt_path,
    )


def create_provider_chain(
    provider: Union[str, SubtitleProvider, Sequence[SubtitleProvider]],
    cache_dir: Path,
    opensubtitles_api_key: Optional[str] = None,
    prefer_hearing_impaired: bool = False,
    mock_offset_s: float = 0.0,
) -> List[SubtitleProvider]:
    """Instantiate an ordered list of `SubtitleProvider` instances for a given provider spec."""
    if isinstance(provider, SubtitleProvider):
        return [provider]
    if isinstance(provider, (list, tuple)) and all(isinstance(p, SubtitleProvider) for p in provider):
        return list(provider)

    prov_key = str(provider or "auto").strip().lower()

    if prov_key == "addic7ed":
        return [
            Addic7edProvider(
                cache_dir=cache_dir,
                prefer_hearing_impaired=prefer_hearing_impaired,
            )
        ]
    if prov_key == "opensubtitles":
        return [
            OpenSubtitlesProvider(
                api_key=opensubtitles_api_key,
                cache_dir=cache_dir,
                prefer_hearing_impaired=prefer_hearing_impaired,
            )
        ]
    if prov_key == "gestdown":
        return [
            GestdownProvider(
                cache_dir=cache_dir,
                prefer_hearing_impaired=prefer_hearing_impaired,
            )
        ]
    if prov_key == "podnapisi":
        return [
            PodnapisiProvider(
                cache_dir=cache_dir,
                prefer_hearing_impaired=prefer_hearing_impaired,
            )
        ]
    if prov_key == "tvsubtitles":
        return [
            TVSubtitlesProvider(
                cache_dir=cache_dir,
                prefer_hearing_impaired=prefer_hearing_impaired,
            )
        ]
    if prov_key == "subdl":
        return [
            SubDLProvider(
                cache_dir=cache_dir,
                prefer_hearing_impaired=prefer_hearing_impaired,
            )
        ]
    if prov_key == "mock":
        return [MockSubtitleProvider(offset_s=mock_offset_s)]

    if prov_key in {"auto", "all"}:
        chain: List[SubtitleProvider] = [
            Addic7edProvider(
                cache_dir=cache_dir,
                prefer_hearing_impaired=prefer_hearing_impaired,
            ),
            GestdownProvider(
                cache_dir=cache_dir,
                prefer_hearing_impaired=prefer_hearing_impaired,
            ),
        ]
        os_prov = OpenSubtitlesProvider(
            api_key=opensubtitles_api_key,
            cache_dir=cache_dir,
            prefer_hearing_impaired=prefer_hearing_impaired,
        )
        if os_prov.is_configured:
            chain.append(os_prov)
        chain.append(
            PodnapisiProvider(
                cache_dir=cache_dir,
                prefer_hearing_impaired=prefer_hearing_impaired,
            )
        )
        return chain

    raise ValueError(
        f"Unknown subtitle provider '{provider}'. "
        "Supported: addic7ed, gestdown, opensubtitles, podnapisi, tvsubtitles, subdl, mock, auto"
    )


_PREFERRED_RELEASE_TAGS = ("web-dl", "webrip", "web", "bluray", "bdrip", "proper", "repack")


def _is_candidate_cached(prov: SubtitleProvider, cand: SubtitleCandidate) -> bool:
    """Return True if `cand` is already present in `prov`'s local subtitle cache."""
    cpath_fn = getattr(prov, "_candidate_cache_path", None)
    if callable(cpath_fn) and getattr(prov, "use_cache", False):
        try:
            cpath = cpath_fn(cand)
            return bool(cpath and cpath.exists() and cpath.stat().st_size > 0)
        except Exception:
            return False
    return False


def _has_preferred_release_tag(cand: SubtitleCandidate) -> bool:
    """Return True if candidate release version indicates a clean WEB-DL / BluRay / PROPER master."""
    rel_lower = (cand.release_version or "").lower()
    return any(tag in rel_lower for tag in _PREFERRED_RELEASE_TAGS)


def _candidate_alignment_quality(metrics: Dict[str, Any], cand: SubtitleCandidate) -> float:
    """
    Compute composite alignment quality score for candidate selection:
      - Higher correlation score is better.
      - Large multi-act window offset spread (`window_spread_s`) is penalized so clean
        WEB-DL/BluRay transfers win over HDTV commercial-break cuts.
      - Minor tie-breaker for preferred release tags and metadata ranking score.
    """
    corr = float(metrics.get("correlation_score") or 0.0)
    spread = float(metrics.get("window_spread_s") or 0.0)
    spread_penalty = 0.005 * max(0.0, spread - 2.0) + 0.012 * max(0.0, spread - 10.0)
    tag_bonus = 0.003 if _has_preferred_release_tag(cand) else 0.0
    rank_bonus = min(0.002, max(0.0, float(getattr(cand, "score", 0.0) or 0.0)) * 1e-5)
    return corr - spread_penalty + tag_bonus + rank_bonus


def _should_evaluate_more_candidates(
    metrics: Dict[str, Any],
    remaining_cands: Sequence[Tuple[SubtitleProvider, SubtitleCandidate]],
) -> bool:
    """
    Determine whether additional subtitle candidates in `candidates[:3]` should be
    evaluated against the already-extracted reference speech envelope.
    """
    if not remaining_cands:
        return False

    corr = float(metrics.get("correlation_score") or 0.0)
    spread = float(metrics.get("window_spread_s") or 0.0)

    # Low correlation (< 0.38) always triggers evaluation of backup candidates
    if corr < 0.38:
        return True

    # Large multi-act commercial-break drift (> 10.0s) triggers evaluation when another
    # cached or downloadable candidate (especially WEB-DL / BluRay / PROPER) is available
    if spread > 10.0:
        return True

    # If another candidate in top 3 is already cached on disk AND is a preferred release
    # while current candidate has moderate multi-act drift (> 3.5s), compare for free
    if spread > 3.5:
        for prov, cand in remaining_cands:
            if _is_candidate_cached(prov, cand) and _has_preferred_release_tag(cand):
                return True

    return False


def _process_single_episode(
    idx: int,
    total_episodes: int,
    meta: SeriesEpisodeMetadata,
    providers: Sequence[SubtitleProvider],
    lang_spec: Any,
    effective_audio_lang: str,
    out_path: Path,
    resolved_cache_dir: Path,
    save_raw_subtitle: bool = False,
    dry_run: bool = False,
    skip_if_exists: bool = False,
    task_timeout: float = 300.0,
    sync_fn: Optional[Callable[..., Any]] = None,
    quiet: bool = False,
) -> EpisodeSyncResult:
    """
    Download, normalize, and synchronize subtitles for a single episode, evaluating up to
    the top 3 subtitle candidates against a cached reference audio envelope when needed.
    """
    ep_start = time.monotonic()
    result = EpisodeSyncResult(
        video_path=meta.video_path,
        series_name=meta.series_name,
        season=meta.season,
        episode=meta.episode,
        episode_title=meta.episode_title,
        audio_lang=effective_audio_lang,
        target_lang=lang_spec.iso639_1,
        output_subtitle_path=out_path,
    )

    if skip_if_exists and out_path.exists() and out_path.stat().st_size > 0:
        result.status = "SKIPPED"
        result.elapsed_seconds = time.monotonic() - ep_start
        if not quiet:
            print(
                f"[{idx}/{total_episodes}] {meta.episode_code} ({meta.episode_title or meta.video_path.name}): "
                f"SKIPPED (already exists: {out_path.name})",
                file=sys.stderr,
            )
        return result

    discovered_pairs: List[Tuple[SubtitleProvider, SubtitleCandidate]] = []
    provider_errors: List[str] = []
    dl_errors: List[str] = []

    for prov in providers:
        try:
            cands = prov.search_subtitles(meta, lang_spec.iso639_1)
        except Exception as p_exc:
            provider_errors.append(f"{prov.name}: {p_exc}")
            continue

        if not cands:
            continue

        for c in cands:
            discovered_pairs.append((prov, c))

        if dry_run:
            break

    if not discovered_pairs:
        err_detail = "; ".join(provider_errors) if provider_errors else "No matching subtitles found"
        raise ProviderError(
            f"No '{lang_spec.iso639_1}' subtitles found for {meta.series_name} {meta.episode_code} ({err_detail})"
        )

    if dry_run:
        best_prov, best_cand = discovered_pairs[0]
        result.status = "DRY_RUN"
        result.provider_name = best_prov.name
        result.candidate_release = best_cand.release_version
        result.download_url = best_cand.download_url
        result.elapsed_seconds = time.monotonic() - ep_start
        if not quiet:
            print(
                f"[{idx}/{total_episodes}] {meta.episode_code}: DRY_RUN "
                f"(would download {best_cand.release_version} from {best_prov.name} -> {out_path.name})",
                file=sys.stderr,
            )
        return result

    raw_cache_dir = resolved_cache_dir / "raw"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    raw_srt_path = raw_cache_dir / f"{meta.video_path.stem}.{lang_spec.iso639_1}.raw.srt"

    rate_limited_providers = set()
    evaluated_candidates: List[Dict[str, Any]] = []
    sync_errors: List[str] = []

    # Cache the reference audio speech envelope in memory across all candidate evaluations
    with _cached_reference_envelope_context(meta.video_path):
        max_eval_candidates = 3
        for pair_idx, (prov, cand) in enumerate(discovered_pairs):
            if len(evaluated_candidates) >= max_eval_candidates:
                break

            if id(prov) in rate_limited_providers and not _is_candidate_cached(prov, cand):
                continue

            try:
                raw_bytes = prov.download_subtitle(cand)
                charset_hint = cand.extra_metadata.get("charset")
                normalized_srt_text = normalize_subtitle_bytes(
                    raw_bytes,
                    declared_charset=charset_hint,
                )
            except ProviderRateLimitError as rl_exc:
                dl_errors.append(f"{prov.name}({cand.release_version}): {rl_exc}")
                rate_limited_providers.add(id(prov))
                continue
            except Exception as dl_exc:
                dl_errors.append(f"{prov.name}({cand.release_version}): {dl_exc}")
                continue

            cand_idx = len(evaluated_candidates)
            cand_raw_path = (
                raw_srt_path
                if cand_idx == 0
                else raw_cache_dir / f"{meta.video_path.stem}.{lang_spec.iso639_1}.cand{cand_idx}.raw.srt"
            )
            cand_out_path = (
                out_path
                if cand_idx == 0
                else raw_cache_dir / f"{meta.video_path.stem}.{lang_spec.iso639_1}.cand{cand_idx}.synced.srt"
            )
            cand_raw_path.write_text(normalized_srt_text, encoding="utf-8", newline="\n")

            try:
                sync_metrics = synchronize_episode_subtitle(
                    video_path=meta.video_path,
                    raw_srt_path=cand_raw_path,
                    out_srt_path=cand_out_path,
                    ref_lang=effective_audio_lang,
                    sub_lang=lang_spec.iso639_2,
                    timeout=task_timeout,
                    sync_fn=sync_fn,
                )
                synced_text = (
                    cand_out_path.read_text(encoding="utf-8")
                    if cand_out_path.exists()
                    else shift_srt_text(
                        normalized_srt_text,
                        offset_seconds=float(sync_metrics.get("offset_seconds") or 0.0),
                        speed_factor=float(sync_metrics.get("speed_factor") or 1.0),
                        formula=sync_metrics.get("formula"),
                    )
                )
                quality = _candidate_alignment_quality(sync_metrics, cand)
                evaluated_candidates.append(
                    {
                        "prov": prov,
                        "cand": cand,
                        "raw_text": normalized_srt_text,
                        "synced_text": synced_text,
                        "metrics": sync_metrics,
                        "quality": quality,
                    }
                )
            except Exception as sync_exc:
                sync_errors.append(f"{prov.name}({cand.release_version}): {sync_exc}")
                continue
            finally:
                if cand_idx > 0:
                    for tmp_p in (cand_raw_path, cand_out_path):
                        try:
                            if tmp_p.exists():
                                tmp_p.unlink()
                        except OSError:
                            pass

            remaining = discovered_pairs[pair_idx + 1 : pair_idx + 1 + (max_eval_candidates - len(evaluated_candidates))]
            if not _should_evaluate_more_candidates(sync_metrics, remaining):
                break

    if not evaluated_candidates:
        if dl_errors and not sync_errors:
            all_errs = dl_errors + provider_errors
            raise ProviderError(
                f"All subtitle candidate downloads failed for {meta.episode_code}: {'; '.join(all_errs)}"
            )
        raise RuntimeError(
            f"Synchronization failed for all candidates of {meta.episode_code}: {'; '.join(sync_errors + dl_errors)}"
        )

    best_entry = max(evaluated_candidates, key=lambda item: item["quality"])
    chosen_prov = best_entry["prov"]
    chosen_cand = best_entry["cand"]
    chosen_raw_text = best_entry["raw_text"]
    chosen_synced_text = best_entry["synced_text"]
    chosen_metrics = best_entry["metrics"]

    raw_srt_path.write_text(chosen_raw_text, encoding="utf-8", newline="\n")
    if save_raw_subtitle:
        sidecar_raw = meta.video_path.with_name(
            f"{meta.video_path.stem}.{lang_spec.iso639_1}.raw.srt"
        )
        sidecar_raw.write_text(chosen_raw_text, encoding="utf-8", newline="\n")
        result.raw_subtitle_path = sidecar_raw
    else:
        result.raw_subtitle_path = raw_srt_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(chosen_synced_text, encoding="utf-8", newline="\n")

    result.provider_name = chosen_prov.name
    result.candidate_release = chosen_cand.release_version
    result.download_url = chosen_cand.download_url
    result.status = "SUCCESS"
    result.offset_seconds = float(chosen_metrics.get("offset_seconds") or 0.0)
    result.speed_factor = float(chosen_metrics.get("speed_factor") or 1.0)
    result.correlation_score = float(chosen_metrics.get("correlation_score") or 1.0)
    result.raw_correlation_score = chosen_metrics.get("raw_correlation_score")
    result.post_correlation_score = chosen_metrics.get("post_correlation_score")
    result.raw_vocal_score = chosen_metrics.get("raw_vocal_score")
    result.post_vocal_score = chosen_metrics.get("post_vocal_score")

    rich_diag = extract_recap_and_dialogue_summary(
        raw_srt_text=chosen_raw_text,
        synced_srt_text=chosen_synced_text,
        offset_seconds=result.offset_seconds,
        speed_factor=result.speed_factor,
        formula=chosen_metrics.get("formula"),
        window_offsets=chosen_metrics.get("window_offsets"),
    )
    result.offset_start_s = rich_diag["offset_start_s"]
    result.offset_mid_s = rich_diag["offset_mid_s"]
    result.offset_end_s = rich_diag["offset_end_s"]
    result.act_count = rich_diag["act_count"]
    result.offset_evolution = rich_diag["offset_evolution"]
    result.recap_status = rich_diag["recap_status"]
    result.dropped_recap_lines = rich_diag["dropped_recap_lines"]
    result.first_sub_time_s = rich_diag["first_sub_time_s"]
    result.first_sub_text = rich_diag["first_sub_text"]
    result.elapsed_seconds = time.monotonic() - ep_start

    if not quiet:
        print(
            f"[{idx}/{total_episodes}] {meta.episode_code} ({meta.episode_title or ''}): "
            f"SUCCESS via {chosen_prov.name} [{chosen_cand.release_version}] "
            f"offset={result.offset_evolution} corr={result.correlation_score * 100.0:.1f}% "
            f"recap={result.recap_status} -> {out_path.name}",
            file=sys.stderr,
        )

    return result


def _process_episode_in_worker(task_payload: Dict[str, Any]) -> EpisodeSyncResult:
    """Isolated worker-process entry point for multi-core episode synchronization."""
    ep_start = time.monotonic()
    idx = int(task_payload["idx"])
    total_episodes = int(task_payload["total_episodes"])
    meta: SeriesEpisodeMetadata = task_payload["meta"]
    lang_spec = task_payload["lang_spec"]
    effective_audio_lang = str(task_payload["effective_audio_lang"])
    out_path: Path = task_payload["out_path"]
    resolved_cache_dir: Path = task_payload["resolved_cache_dir"]
    quiet = bool(task_payload["quiet"])

    try:
        providers = create_provider_chain(
            provider=task_payload["provider"],
            cache_dir=resolved_cache_dir,
            opensubtitles_api_key=task_payload.get("opensubtitles_api_key"),
            prefer_hearing_impaired=bool(task_payload.get("hearing_impaired", False)),
            mock_offset_s=float(task_payload.get("mock_offset_s", 0.0)),
        )
        return _process_single_episode(
            idx=idx,
            total_episodes=total_episodes,
            meta=meta,
            providers=providers,
            lang_spec=lang_spec,
            effective_audio_lang=effective_audio_lang,
            out_path=out_path,
            resolved_cache_dir=resolved_cache_dir,
            save_raw_subtitle=bool(task_payload.get("save_raw_subtitle", False)),
            dry_run=bool(task_payload.get("dry_run", False)),
            skip_if_exists=bool(task_payload.get("skip_if_exists", False)),
            task_timeout=float(task_payload.get("task_timeout", 300.0)),
            sync_fn=None,
            quiet=quiet,
        )
    except Exception as exc:
        if not quiet:
            print(
                f"[{idx}/{total_episodes}] {meta.episode_code}: FAILED - {exc}",
                file=sys.stderr,
            )
        return EpisodeSyncResult(
            video_path=meta.video_path,
            series_name=meta.series_name,
            season=meta.season,
            episode=meta.episode,
            episode_title=meta.episode_title,
            audio_lang=effective_audio_lang,
            target_lang=lang_spec.iso639_1,
            output_subtitle_path=out_path,
            status="FAILED",
            error=str(exc),
            elapsed_seconds=time.monotonic() - ep_start,
        )


def run_series_sync(
    series_dir: Union[Path, str],
    target_lang: str = "fr",
    provider: Union[str, SubtitleProvider, Sequence[SubtitleProvider]] = "auto",
    out_pattern: str = DEFAULT_OUT_PATTERN,
    cache_dir: Optional[Union[Path, str]] = None,
    max_episodes: Optional[int] = None,
    episodes_filter: Optional[Sequence[int]] = None,
    season_filter: Optional[int] = None,
    dry_run: bool = False,
    skip_if_exists: bool = False,
    sync_fn: Optional[Callable[..., Any]] = None,
    ref_lang: Optional[str] = None,
    opensubtitles_api_key: Optional[str] = None,
    save_raw_subtitle: bool = False,
    recursive: bool = False,
    task_timeout: float = 300.0,
    hearing_impaired: bool = False,
    summary_json: Optional[Union[Path, str]] = None,
    mock_offset_s: float = 0.0,
    workers: int = 0,
    quiet: bool = False,
    config: Optional[SeriesSyncConfig] = None,
) -> SeriesSyncSummary:
    """
    Scan `series_dir` for TV series episodes, download matching subtitles in `target_lang`,
    normalize character encoding to UTF-8, synchronize each subtitle against the episode
    video audio to <= 0.1s accuracy, and write synchronized `.srt` files.

    Each episode is isolated inside its own `try...except` block so a single failure
    never aborts the unattended season run.
    """
    start_wall = time.monotonic()

    if config is not None:
        series_dir = config.series_dir
        target_lang = config.target_lang
        ref_lang = config.ref_lang
        provider = config.provider
        opensubtitles_api_key = config.opensubtitles_api_key
        out_pattern = config.out_pattern
        cache_dir = config.cache_dir
        save_raw_subtitle = config.save_raw_subtitle
        recursive = config.recursive
        max_episodes = config.max_episodes
        episodes_filter = config.episodes_filter
        season_filter = config.season_filter
        dry_run = config.dry_run
        skip_if_exists = config.skip_if_exists
        task_timeout = config.task_timeout
        hearing_impaired = config.hearing_impaired
        summary_json = config.summary_json
        mock_offset_s = config.mock_offset_s
        workers = config.workers
        quiet = config.quiet

    root_dir = Path(series_dir)
    resolved_cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "subsync")
    lang_spec = normalize_lang(target_lang)

    providers = create_provider_chain(
        provider=provider,
        cache_dir=resolved_cache_dir,
        opensubtitles_api_key=opensubtitles_api_key,
        prefer_hearing_impaired=hearing_impaired,
        mock_offset_s=mock_offset_s,
    )
    provider_label = (
        provider if isinstance(provider, str) else ",".join(getattr(p, "name", "custom") for p in providers)
    )

    episodes = scan_series_directory(
        root_dir,
        recursive=recursive,
        probe_audio=(ref_lang is None),
    )

    if season_filter is not None:
        episodes = [ep for ep in episodes if ep.season == int(season_filter)]
    if episodes_filter is not None:
        ep_set = {int(e) for e in episodes_filter}
        episodes = [ep for ep in episodes if ep.episode in ep_set]
    if max_episodes is not None and int(max_episodes) >= 0:
        episodes = episodes[: int(max_episodes)]

    results: List[EpisodeSyncResult] = []
    succeeded = 0
    skipped = 0
    failed = 0

    cpu_cores = os.cpu_count() or 4
    resolved_workers = int(workers) if int(workers) > 0 else min(6, max(1, cpu_cores // 4))
    use_multiproc = (
        resolved_workers > 1
        and len(episodes) > 1
        and not dry_run
        and sync_fn is None
        and isinstance(provider, str)
    )

    if use_multiproc:
        from concurrent.futures import ProcessPoolExecutor

        # Pre-warm season HTML index caches and primary candidate downloads sequentially
        # in the parent process so worker processes never trigger HTTP rate-limiting on Addic7ed.
        for meta in episodes:
            for prov in providers:
                try:
                    cands = prov.search_subtitles(meta, lang_spec.iso639_1)
                    if cands:
                        prov.download_subtitle(cands[0])
                        break
                except Exception:
                    continue

        payloads = []
        for idx, meta in enumerate(episodes, start=1):
            effective_audio_lang = ref_lang or meta.audio_lang or "eng"
            out_path = format_series_output_path(meta, lang_spec.iso639_1, out_pattern)
            payloads.append(
                {
                    "idx": idx,
                    "total_episodes": len(episodes),
                    "meta": meta,
                    "provider": provider,
                    "opensubtitles_api_key": opensubtitles_api_key,
                    "hearing_impaired": hearing_impaired,
                    "mock_offset_s": mock_offset_s,
                    "lang_spec": lang_spec,
                    "effective_audio_lang": effective_audio_lang,
                    "out_path": out_path,
                    "resolved_cache_dir": resolved_cache_dir,
                    "save_raw_subtitle": save_raw_subtitle,
                    "dry_run": dry_run,
                    "skip_if_exists": skip_if_exists,
                    "task_timeout": task_timeout,
                    "quiet": quiet,
                }
            )

        with ProcessPoolExecutor(max_workers=min(resolved_workers, len(episodes))) as pool:
            for result in pool.map(_process_episode_in_worker, payloads):
                if result.status == "SKIPPED":
                    skipped += 1
                elif result.status == "SUCCESS":
                    succeeded += 1
                else:
                    failed += 1
                results.append(result)
    else:
        for idx, meta in enumerate(episodes, start=1):
            ep_start = time.monotonic()
            effective_audio_lang = ref_lang or meta.audio_lang or "eng"
            out_path = format_series_output_path(meta, lang_spec.iso639_1, out_pattern)

            try:
                result = _process_single_episode(
                    idx=idx,
                    total_episodes=len(episodes),
                    meta=meta,
                    providers=providers,
                    lang_spec=lang_spec,
                    effective_audio_lang=effective_audio_lang,
                    out_path=out_path,
                    resolved_cache_dir=resolved_cache_dir,
                    save_raw_subtitle=save_raw_subtitle,
                    dry_run=dry_run,
                    skip_if_exists=skip_if_exists,
                    task_timeout=task_timeout,
                    sync_fn=sync_fn,
                    quiet=quiet,
                )
                if result.status == "SKIPPED":
                    skipped += 1
                else:
                    succeeded += 1
            except Exception as exc:
                result = EpisodeSyncResult(
                    video_path=meta.video_path,
                    series_name=meta.series_name,
                    season=meta.season,
                    episode=meta.episode,
                    episode_title=meta.episode_title,
                    audio_lang=effective_audio_lang,
                    target_lang=lang_spec.iso639_1,
                    output_subtitle_path=out_path,
                    status="FAILED",
                    error=str(exc),
                    elapsed_seconds=time.monotonic() - ep_start,
                )
                failed += 1
                if not quiet:
                    print(
                        f"[{idx}/{len(episodes)}] {meta.episode_code}: FAILED - {exc}",
                        file=sys.stderr,
                    )

            results.append(result)

    total_elapsed = time.monotonic() - start_wall
    summary = SeriesSyncSummary(
        series_dir=root_dir,
        target_lang=lang_spec.iso639_1,
        provider=str(provider_label),
        total_episodes=len(episodes),
        succeeded=succeeded,
        skipped=skipped,
        failed=failed,
        dry_run=dry_run,
        elapsed_seconds=total_elapsed,
        results=results,
    )

    if summary_json is not None:
        json_path = Path(summary_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(summary.to_json(indent=2), encoding="utf-8")

    try:
        reports_dir = resolved_cache_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "last_series_summary.json").write_text(summary.to_json(indent=2), encoding="utf-8")
    except Exception:
        pass

    if not quiet:
        print(summary.format_table(), file=sys.stderr)

    return summary
