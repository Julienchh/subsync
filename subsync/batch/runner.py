"""Unattended batch synchronization execution engine with per-task error isolation."""

from __future__ import annotations

import concurrent.futures
import inspect
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from subsync.batch.discovery import discover_media_files
from subsync.batch.matcher import (
    EpisodeInfo,
    MatchedPair,
    MatchReport,
    extract_episode_info,
    match_files,
)

logger = logging.getLogger(__name__)


class _ConditionalPatternField:
    """Supports legacy `{if:field:value}` and `{if_not:field:value}` syntax in output patterns."""

    def __init__(self, values: Dict[str, str], inverted: bool = False) -> None:
        self._values = values
        self._inverted = inverted

    def __format__(self, fmt: str) -> str:
        if ":" not in fmt:
            return ""
        key, val = fmt.split(":", 1)
        is_set = bool(self._values.get(key, ""))
        if is_set ^ self._inverted:
            return val
        return ""


@dataclass
class BatchConfig:
    """Configuration for unattended batch resynchronization."""

    ref_dir: Optional[Path] = None
    sub_dir: Optional[Path] = None
    ref_pattern: Optional[Union[str, Sequence[str]]] = None
    sub_pattern: Optional[Union[str, Sequence[str]]] = None
    out_dir: Optional[Path] = None
    out_pattern: str = "{ref_dir}/{ref_name}.synced.srt"
    in_place: bool = False
    backup: bool = False
    skip_if_exists: bool = False
    recursive: bool = False
    matcher_mode: str = "auto"
    lang: Optional[str] = None
    sub_lang: Optional[str] = None
    ref_lang: Optional[str] = None
    task_timeout: float = 300.0
    continue_on_error: bool = True
    dry_run: bool = False
    jobs: int = 1
    summary_json: Optional[Path] = None
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.ref_dir is not None and not isinstance(self.ref_dir, Path):
            self.ref_dir = Path(self.ref_dir)
        if self.sub_dir is not None and not isinstance(self.sub_dir, Path):
            self.sub_dir = Path(self.sub_dir)
        if self.out_dir is not None and not isinstance(self.out_dir, Path):
            self.out_dir = Path(self.out_dir)
        if self.summary_json is not None and not isinstance(self.summary_json, Path):
            self.summary_json = Path(self.summary_json)
        if self.lang and not self.sub_lang:
            self.sub_lang = self.lang
        elif self.sub_lang and not self.lang:
            self.lang = self.sub_lang
        if self.jobs < 1:
            self.jobs = 1


@dataclass
class BatchJobResult:
    """Execution outcome for a single (video, subtitle) synchronization job."""

    video_path: Path
    sub_path: Path
    out_path: Optional[Path] = None
    status: str = "OK"  # "OK" | "FAILED" | "SKIPPED" | "DRY_RUN"
    offset_seconds: float = 0.0
    speed_factor: float = 1.0
    correlation: float = 0.0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    match_tier: Optional[int] = None
    match_score: Optional[float] = None
    backup_path: Optional[Path] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        video_path: Union[str, Path],
        sub_path: Union[str, Path],
        out_path: Optional[Union[str, Path]] = None,
        status: str = "OK",
        offset_seconds: Optional[float] = None,
        speed_factor: Optional[float] = None,
        correlation: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        error: Optional[str] = None,
        season: Optional[int] = None,
        episode: Optional[int] = None,
        match_tier: Optional[int] = None,
        match_score: Optional[float] = None,
        backup_path: Optional[Union[str, Path]] = None,
        details: Optional[Dict[str, Any]] = None,
        *,
        # Convenient aliases for keyword initialization
        offset: Optional[float] = None,
        b: Optional[float] = None,
        factor: Optional[float] = None,
        a: Optional[float] = None,
        score: Optional[float] = None,
        elapsed: Optional[float] = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.sub_path = Path(sub_path)
        self.out_path = Path(out_path) if out_path is not None else None
        self.status = status
        self.offset_seconds = float(
            offset_seconds
            if offset_seconds is not None
            else (offset if offset is not None else (b if b is not None else 0.0))
        )
        self.speed_factor = float(
            speed_factor
            if speed_factor is not None
            else (factor if factor is not None else (a if a is not None else 1.0))
        )
        self.correlation = float(
            correlation
            if correlation is not None
            else (score if score is not None else 0.0)
        )
        self.elapsed_seconds = float(
            elapsed_seconds
            if elapsed_seconds is not None
            else (elapsed if elapsed is not None else 0.0)
        )
        self.error = error
        self.season = season
        self.episode = episode
        self.match_tier = match_tier
        self.match_score = match_score
        self.backup_path = Path(backup_path) if backup_path is not None else None
        self.details = details or {}

    @property
    def ref_path(self) -> Path:
        return self.video_path

    @property
    def video(self) -> Path:
        return self.video_path

    @property
    def ref(self) -> Path:
        return self.video_path

    @property
    def subtitle_path(self) -> Path:
        return self.sub_path

    @property
    def subtitle(self) -> Path:
        return self.sub_path

    @property
    def sub(self) -> Path:
        return self.sub_path

    @property
    def output_path(self) -> Optional[Path]:
        return self.out_path

    @property
    def offset(self) -> float:
        return self.offset_seconds

    @property
    def b(self) -> float:
        return self.offset_seconds

    @property
    def factor(self) -> float:
        return self.speed_factor

    @property
    def a(self) -> float:
        return self.speed_factor

    @property
    def score(self) -> float:
        return self.correlation

    @property
    def elapsed(self) -> float:
        return self.elapsed_seconds

    @property
    def succeeded(self) -> bool:
        return self.status in ("OK", "SUCCESS", "SKIPPED", "DRY_RUN")

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "video_path": str(self.video_path),
            "ref_path": str(self.video_path),
            "sub_path": str(self.sub_path),
            "out_path": str(self.out_path) if self.out_path is not None else None,
            "status": self.status,
            "offset_seconds": round(self.offset_seconds, 6),
            "b": round(self.offset_seconds, 6),
            "speed_factor": round(self.speed_factor, 6),
            "a": round(self.speed_factor, 6),
            "correlation": round(self.correlation, 6),
            "score": round(self.correlation, 6),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "error": self.error,
            "season": self.season,
            "episode": self.episode,
            "match_tier": self.match_tier,
            "match_score": self.match_score,
            "backup_path": str(self.backup_path) if self.backup_path is not None else None,
        }
        if self.details:
            for k in (
                "offset_start_s",
                "offset_mid_s",
                "offset_end_s",
                "act_count",
                "offset_evolution",
                "recap_status",
                "dropped_recap_lines",
                "first_sub_time_s",
                "first_sub_text",
            ):
                if k in self.details:
                    payload[k] = self.details[k]
        return payload


@dataclass
class BatchSummary:
    """Aggregate report for a batch resynchronization run."""

    results: List[BatchJobResult] = field(default_factory=list)
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    dry_run_count: int = 0
    unmatched_videos: List[Path] = field(default_factory=list)
    unmatched_subtitles: List[Path] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.results and self.total == 0:
            self.recompute_counts()

    def recompute_counts(self) -> None:
        self.total = len(self.results)
        self.succeeded = sum(1 for r in self.results if r.status == "OK")
        self.failed = sum(1 for r in self.results if r.status == "FAILED")
        self.skipped = sum(1 for r in self.results if r.status == "SKIPPED")
        self.dry_run_count = sum(1 for r in self.results if r.status == "DRY_RUN")

    @property
    def jobs(self) -> List[BatchJobResult]:
        return self.results

    @property
    def ok(self) -> int:
        return self.succeeded

    @property
    def ok_count(self) -> int:
        return self.succeeded

    @property
    def failed_count(self) -> int:
        return self.failed

    @property
    def skipped_count(self) -> int:
        return self.skipped

    @property
    def dry_run(self) -> int:
        return self.dry_run_count

    @property
    def elapsed_seconds(self) -> float:
        return self.total_elapsed_seconds

    @property
    def exit_code(self) -> int:
        if self.failed == 0:
            return 0
        if self.succeeded > 0 or self.skipped > 0 or self.dry_run_count > 0:
            return 2
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "ok": self.succeeded,
            "failed": self.failed,
            "skipped": self.skipped,
            "dry_run": self.dry_run_count,
            "total_elapsed_seconds": round(self.total_elapsed_seconds, 4),
            "unmatched_videos": [str(p) for p in self.unmatched_videos],
            "unmatched_subtitles": [str(p) for p in self.unmatched_subtitles],
            "results": [r.to_dict() for r in self.results],
        }


def format_output_path(
    video_path: Union[str, Path],
    sub_path: Union[str, Path],
    out_dir: Optional[Union[str, Path]] = None,
    out_pattern: str = "{ref_dir}/{ref_name}.synced.srt",
    in_place: bool = False,
    lang: Optional[str] = None,
    sub_lang: Optional[str] = None,
    ref_lang: Optional[str] = None,
) -> Path:
    """Format the output subtitle path for a given `(video_path, sub_path)` pair.

    Supports template variables:
    `{ref_dir}`, `{ref_name}`, `{ref_path}`, `{sub_dir}`, `{sub_name}`,
    `{sub_path}`, `{lang}`, `{sub_lang}`, `{ref_lang}` (plus `{if:field:val}`
    and `{if_not:field:val}`).
    """
    v_path = Path(video_path)
    s_path = Path(sub_path)

    if in_place:
        return s_path

    effective_pattern = out_pattern or "{ref_dir}/{ref_name}.synced.srt"

    # Infer language tag from subtitle filename if not explicitly passed
    inferred_sub_lang: Optional[str] = None
    if not lang and not sub_lang:
        sub_info = extract_episode_info(s_path)
        if sub_info.language_tag:
            inferred_sub_lang = str(sub_info.language_tag)

    eff_sub_lang = sub_lang or lang or inferred_sub_lang or ""
    eff_lang = lang or sub_lang or inferred_sub_lang or ""
    eff_ref_lang = ref_lang or ""

    ref_dir_str = str(Path(out_dir)) if out_dir is not None else str(v_path.parent)
    sub_dir_str = str(Path(out_dir)) if out_dir is not None else str(s_path.parent)

    fields: Dict[str, str] = {
        "ref_dir": ref_dir_str,
        "ref_name": v_path.stem,
        "ref_path": str(v_path),
        "ref_no": "1",
        "ref_lang": eff_ref_lang,
        "ref_lang2": eff_ref_lang[:2] if eff_ref_lang else "",
        "sub_dir": sub_dir_str,
        "sub_name": s_path.stem,
        "sub_path": str(s_path),
        "sub_no": "1",
        "sub_lang": eff_sub_lang,
        "sub_lang2": eff_sub_lang[:2] if eff_sub_lang else "",
        "lang": eff_lang,
    }

    formatted = effective_pattern.format(
        **fields,
        **{
            "if": _ConditionalPatternField(fields, inverted=False),
            "if_not": _ConditionalPatternField(fields, inverted=True),
        },
    )

    # Clean up accidental double dots when {lang} is empty (e.g. "{ref_name}.{lang}.synced.srt")
    formatted = re.sub(r"\.{2,}", ".", formatted)

    out_path = Path(formatted)
    if out_dir is not None and "{ref_dir}" not in effective_pattern and "{sub_dir}" not in effective_pattern:
        if not out_path.is_absolute():
            out_path = Path(out_dir) / out_path

    return out_path


def _default_synchronize_pair(
    sub_path: Path,
    ref_path: Path,
    out_path: Path,
    sub_lang: Optional[str] = None,
    ref_lang: Optional[str] = None,
) -> Any:
    """Invoke `subsync.synchro.modern_engine.synchronize_pair`."""
    from subsync.synchro.modern_engine import synchronize_pair

    return synchronize_pair(
        sub_path=sub_path,
        ref_path=ref_path,
        out_path=out_path,
        sub_lang=sub_lang,
        ref_lang=ref_lang,
    )


def _invoke_sync_callable(
    sync_callable: Callable[..., Any],
    sub_path: Path,
    ref_path: Path,
    out_path: Path,
    sub_lang: Optional[str] = None,
    ref_lang: Optional[str] = None,
) -> Any:
    """Call `sync_callable` with flexible signature adaptation for unit test hooks."""
    kwargs: Dict[str, Any] = {
        "sub_path": sub_path,
        "ref_path": ref_path,
        "out_path": out_path,
        "sub_lang": sub_lang,
        "ref_lang": ref_lang,
    }
    try:
        sig = inspect.signature(sync_callable)
        params = sig.parameters
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if has_var_kw:
            return sync_callable(**kwargs)

        accepted_kwargs: Dict[str, Any] = {}
        # Map standard parameter names and common aliases
        alias_map = {
            "sub_path": sub_path,
            "subtitle_path": sub_path,
            "sub": sub_path,
            "subtitle": sub_path,
            "ref_path": ref_path,
            "video_path": ref_path,
            "ref": ref_path,
            "video": ref_path,
            "out_path": out_path,
            "output_path": out_path,
            "out": out_path,
            "output": out_path,
            "sub_lang": sub_lang,
            "lang": sub_lang,
            "ref_lang": ref_lang,
        }
        for p_name, p_obj in params.items():
            if p_name in alias_map and p_obj.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                accepted_kwargs[p_name] = alias_map[p_name]

        if accepted_kwargs:
            return sync_callable(**accepted_kwargs)

        # Fallback for positional-only functions (e.g. `lambda s, r, o: ...`)
        pos_params = [
            p
            for p in params.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(pos_params) == 3:
            return sync_callable(sub_path, ref_path, out_path)
        if len(pos_params) == 2:
            return sync_callable(sub_path, ref_path)
        return sync_callable(sub_path, ref_path, out_path, sub_lang, ref_lang)
    except (ValueError, TypeError):
        return sync_callable(
            sub_path=sub_path,
            ref_path=ref_path,
            out_path=out_path,
            sub_lang=sub_lang,
            ref_lang=ref_lang,
        )


def _extract_sync_metrics(raw_result: Any) -> Tuple[float, float, float]:
    """Extract `(offset_seconds, speed_factor, correlation)` from a sync function return value."""
    if raw_result is None or raw_result is True:
        return 0.0, 1.0, 1.0
    if raw_result is False:
        raise RuntimeError("Synchronization returned False")

    if isinstance(raw_result, (int, float)):
        return float(raw_result), 1.0, 1.0

    if isinstance(raw_result, tuple):
        if len(raw_result) >= 3:
            return float(raw_result[0]), float(raw_result[1]), float(raw_result[2])
        if len(raw_result) == 2:
            return float(raw_result[0]), float(raw_result[1]), 1.0
        if len(raw_result) == 1:
            return float(raw_result[0]), 1.0, 1.0

    if isinstance(raw_result, dict):
        if raw_result.get("success") is False or raw_result.get("ok") is False or raw_result.get("status") == "FAILED":
            err_msg = raw_result.get("error") or "Synchronization failed"
            raise RuntimeError(str(err_msg))
        offset = raw_result.get(
            "offset_seconds",
            raw_result.get("offset", raw_result.get("b", raw_result.get("shift", 0.0))),
        )
        factor = raw_result.get(
            "speed_factor",
            raw_result.get("factor", raw_result.get("a", raw_result.get("scale", 1.0))),
        )
        corr = raw_result.get(
            "correlation",
            raw_result.get("score", raw_result.get("max_score", raw_result.get("confidence", 1.0))),
        )
        return float(offset or 0.0), float(factor or 1.0), float(corr or 0.0)

    # Object / dataclass return value
    for fail_attr in ("success", "succeeded", "ok"):
        if hasattr(raw_result, fail_attr) and getattr(raw_result, fail_attr) is False:
            err_msg = getattr(raw_result, "error", None) or "Synchronization failed"
            raise RuntimeError(str(err_msg))
    if getattr(raw_result, "status", None) == "FAILED":
        err_msg = getattr(raw_result, "error", None) or "Synchronization failed"
        raise RuntimeError(str(err_msg))

    offset = 0.0
    for attr in ("offset_seconds", "offset", "b", "shift", "delay"):
        val = getattr(raw_result, attr, None)
        if val is not None:
            offset = float(val)
            break

    factor = 1.0
    for attr in ("speed_factor", "factor", "a", "scale", "drift", "slope"):
        val = getattr(raw_result, attr, None)
        if val is not None:
            factor = float(val)
            break

    corr = 0.0
    for attr in ("correlation", "score", "max_score", "confidence", "points"):
        val = getattr(raw_result, attr, None)
        if val is not None:
            corr = float(val)
            break

    return offset, factor, corr


def _run_with_timeout(
    fn: Callable[[], Any],
    timeout: Optional[float],
) -> Any:
    """Execute `fn()` with an optional timeout in seconds using a daemon thread."""
    if timeout is None or timeout <= 0:
        return fn()

    import threading

    box: Dict[str, Any] = {}

    def _target() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:
            box["error"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        raise TimeoutError(f"Task timed out after {timeout:.1f}s")
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _normalize_pair_item(
    item: Union[MatchedPair, Tuple[Union[str, Path], Union[str, Path]]]
) -> MatchedPair:
    if isinstance(item, MatchedPair):
        return item
    v_path, s_path = Path(item[0]), Path(item[1])
    v_info = extract_episode_info(v_path)
    s_info = extract_episode_info(s_path)
    return MatchedPair(
        video=v_path,
        subtitle=s_path,
        tier=1,
        method="explicit",
        score=1.0,
        season=v_info.season if v_info.season is not None else s_info.season,
        episode=v_info.episode if v_info.episode is not None else s_info.episode,
        video_info=v_info,
        subtitle_info=s_info,
    )


def _execute_single_job(
    pair: MatchedPair,
    cfg: BatchConfig,
    sync_callable: Callable[..., Any],
) -> BatchJobResult:
    """Execute a single pair synchronization with strict error isolation and atomic in-place handling."""
    start_time = time.monotonic()
    video_path = pair.video
    sub_path = pair.subtitle

    sub_lang = cfg.sub_lang or cfg.lang
    if not sub_lang and pair.subtitle_info and pair.subtitle_info.language_tag:
        sub_lang = str(pair.subtitle_info.language_tag)

    out_path = format_output_path(
        video_path=video_path,
        sub_path=sub_path,
        out_dir=cfg.out_dir,
        out_pattern=cfg.out_pattern,
        in_place=cfg.in_place,
        lang=cfg.lang,
        sub_lang=sub_lang,
        ref_lang=cfg.ref_lang,
    )

    season = pair.season
    episode = pair.episode
    if season is None and pair.video_info:
        season = pair.video_info.season
    if episode is None and pair.video_info:
        episode = pair.video_info.episode

    # 1. Dry-run mode
    if cfg.dry_run:
        return BatchJobResult(
            video_path=video_path,
            sub_path=sub_path,
            out_path=out_path,
            status="DRY_RUN",
            offset_seconds=0.0,
            speed_factor=1.0,
            correlation=0.0,
            elapsed_seconds=0.0,
            season=season,
            episode=episode,
            match_tier=pair.tier,
            match_score=pair.score,
        )

    # 2. Skip-if-exists mode
    if cfg.skip_if_exists and not cfg.in_place and out_path.exists():
        logger.info("Skipping existing output subtitle: %s", out_path)
        return BatchJobResult(
            video_path=video_path,
            sub_path=sub_path,
            out_path=out_path,
            status="SKIPPED",
            offset_seconds=0.0,
            speed_factor=1.0,
            correlation=0.0,
            elapsed_seconds=0.0,
            season=season,
            episode=episode,
            match_tier=pair.tier,
            match_score=pair.score,
        )

    tmp_path: Optional[Path] = None
    backup_path: Optional[Path] = None

    try:
        if not video_path.exists():
            raise FileNotFoundError(f"Reference media file not found: {video_path}")
        if not sub_path.exists():
            raise FileNotFoundError(f"Subtitle file not found: {sub_path}")

        if cfg.in_place:
            parent_dir = sub_path.parent
            parent_dir.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{sub_path.stem}.subsync_tmp_",
                suffix=sub_path.suffix or ".srt",
                dir=str(parent_dir),
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            # Pre-populate tempfile so even mock sync_fns that don't write to disk allow atomic replace
            shutil.copy2(sub_path, tmp_path)
            target_write_path = tmp_path
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            target_write_path = out_path

        def _task() -> Any:
            return _invoke_sync_callable(
                sync_callable,
                sub_path=sub_path,
                ref_path=video_path,
                out_path=target_write_path,
                sub_lang=sub_lang,
                ref_lang=cfg.ref_lang,
            )

        raw_res = _run_with_timeout(_task, cfg.task_timeout)
        offset_sec, speed_fac, corr = _extract_sync_metrics(raw_res)

        rich_details: Dict[str, Any] = {}
        try:
            from subsync.download.pipeline import extract_recap_and_dialogue_summary

            raw_text_before = sub_path.read_text(encoding="utf-8", errors="replace") if sub_path.exists() else ""
            synced_file = tmp_path if (cfg.in_place and tmp_path is not None) else out_path
            synced_text_after = (
                synced_file.read_text(encoding="utf-8", errors="replace")
                if synced_file is not None and synced_file.exists()
                else ""
            )
            formula_obj = raw_res.get("formula") if isinstance(raw_res, dict) else getattr(raw_res, "formula", None)
            win_offs = (
                raw_res.get("window_offsets")
                if isinstance(raw_res, dict)
                else getattr(raw_res, "window_offsets", None)
            )
            rich_details = extract_recap_and_dialogue_summary(
                raw_srt_text=raw_text_before,
                synced_srt_text=synced_text_after,
                offset_seconds=offset_sec,
                speed_factor=speed_fac,
                formula=formula_obj,
                window_offsets=win_offs,
            )
        except Exception:
            pass

        if cfg.in_place and tmp_path is not None:
            if cfg.backup:
                backup_path = sub_path.with_name(sub_path.name + ".bak")
                shutil.copy2(sub_path, backup_path)
            os.replace(str(tmp_path), str(sub_path))
            tmp_path = None

        elapsed = time.monotonic() - start_time
        return BatchJobResult(
            video_path=video_path,
            sub_path=sub_path,
            out_path=out_path,
            status="OK",
            offset_seconds=offset_sec,
            speed_factor=speed_fac,
            correlation=corr,
            elapsed_seconds=elapsed,
            error=None,
            season=season,
            episode=episode,
            match_tier=pair.tier,
            match_score=pair.score,
            backup_path=backup_path,
            details=rich_details,
        )

    except Exception as exc:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        elapsed = time.monotonic() - start_time
        logger.warning(
            "Batch task failed for video=%s, sub=%s: %s",
            video_path,
            sub_path,
            exc,
        )
        return BatchJobResult(
            video_path=video_path,
            sub_path=sub_path,
            out_path=out_path,
            status="FAILED",
            offset_seconds=0.0,
            speed_factor=1.0,
            correlation=0.0,
            elapsed_seconds=elapsed,
            error=str(exc),
            season=season,
            episode=episode,
            match_tier=pair.tier,
            match_score=pair.score,
        )


def run_batch(
    config: Optional[
        Union[
            BatchConfig,
            MatchReport,
            Sequence[Union[MatchedPair, Tuple[Union[str, Path], Union[str, Path]]]],
            str,
            Path,
        ]
    ] = None,
    *,
    pairs: Optional[
        Sequence[Union[MatchedPair, Tuple[Union[str, Path], Union[str, Path]]]]
    ] = None,
    videos: Optional[Sequence[Union[str, Path]]] = None,
    subtitles: Optional[Sequence[Union[str, Path]]] = None,
    ref_dir: Optional[Union[str, Path]] = None,
    sub_dir: Optional[Union[str, Path]] = None,
    ref_pattern: Optional[Union[str, Sequence[str]]] = None,
    sub_pattern: Optional[Union[str, Sequence[str]]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    out_pattern: Optional[str] = None,
    in_place: Optional[bool] = None,
    backup: Optional[bool] = None,
    skip_if_exists: Optional[bool] = None,
    recursive: Optional[bool] = None,
    matcher_mode: Optional[str] = None,
    lang: Optional[str] = None,
    sub_lang: Optional[str] = None,
    ref_lang: Optional[str] = None,
    task_timeout: Optional[float] = None,
    continue_on_error: Optional[bool] = None,
    dry_run: Optional[bool] = None,
    jobs: Optional[int] = None,
    summary_json: Optional[Union[str, Path]] = None,
    sync_fn: Optional[Callable[..., Any]] = None,
    progress_callback: Optional[Callable[[BatchJobResult, int, int], None]] = None,
) -> BatchSummary:
    """Run unattended batch subtitle synchronization with strict per-task error isolation.

    Can be invoked with a `BatchConfig`, a `MatchReport`, an explicit list of `pairs`,
    explicit `videos`/`subtitles` lists, or directory/glob parameters (`ref_dir`, `sub_dir`,
    `ref_pattern`, `sub_pattern`).
    """
    batch_start = time.monotonic()

    unmatched_videos: List[Path] = []
    unmatched_subtitles: List[Path] = []

    if isinstance(config, BatchConfig):
        cfg = BatchConfig(**vars(config))
    elif isinstance(config, MatchReport):
        cfg = BatchConfig()
        if pairs is None:
            pairs = config.paired
        unmatched_videos = list(config.unmatched_videos)
        unmatched_subtitles = list(config.unmatched_subtitles)
    elif isinstance(config, (str, Path)):
        cfg = BatchConfig(ref_dir=Path(config))
    elif isinstance(config, Sequence):
        cfg = BatchConfig()
        if pairs is None:
            pairs = config
    else:
        cfg = BatchConfig()

    # Apply explicit keyword overrides onto cfg
    if ref_dir is not None:
        cfg.ref_dir = Path(ref_dir)
    if sub_dir is not None:
        cfg.sub_dir = Path(sub_dir)
    if ref_pattern is not None:
        cfg.ref_pattern = ref_pattern
    if sub_pattern is not None:
        cfg.sub_pattern = sub_pattern
    if out_dir is not None:
        cfg.out_dir = Path(out_dir)
    if out_pattern is not None:
        cfg.out_pattern = out_pattern
    if in_place is not None:
        cfg.in_place = in_place
    if backup is not None:
        cfg.backup = backup
    if skip_if_exists is not None:
        cfg.skip_if_exists = skip_if_exists
    if recursive is not None:
        cfg.recursive = recursive
    if matcher_mode is not None:
        cfg.matcher_mode = matcher_mode
    if lang is not None:
        cfg.lang = lang
        if not cfg.sub_lang:
            cfg.sub_lang = lang
    if sub_lang is not None:
        cfg.sub_lang = sub_lang
        if not cfg.lang:
            cfg.lang = sub_lang
    if ref_lang is not None:
        cfg.ref_lang = ref_lang
    if task_timeout is not None:
        cfg.task_timeout = task_timeout
    if continue_on_error is not None:
        cfg.continue_on_error = continue_on_error
    if dry_run is not None:
        cfg.dry_run = dry_run
    if jobs is not None:
        cfg.jobs = max(1, int(jobs))
    if summary_json is not None:
        cfg.summary_json = Path(summary_json)

    # Resolve pairs if not directly supplied
    resolved_pairs: List[MatchedPair] = []
    if pairs is not None:
        resolved_pairs = [_normalize_pair_item(p) for p in pairs]
    else:
        if videos is not None and subtitles is not None:
            disc_videos = [Path(v) for v in videos]
            disc_subs = [Path(s) for s in subtitles]
        else:
            disc_videos, disc_subs = discover_media_files(
                ref_dir=cfg.ref_dir,
                sub_dir=cfg.sub_dir,
                ref_pattern=cfg.ref_pattern,
                sub_pattern=cfg.sub_pattern,
                recursive=cfg.recursive,
            )
        report = match_files(
            videos=disc_videos,
            subtitles=disc_subs,
            mode=cfg.matcher_mode,
            target_lang=cfg.sub_lang or cfg.lang,
        )
        resolved_pairs = report.paired
        unmatched_videos = report.unmatched_videos
        unmatched_subtitles = report.unmatched_subtitles

    effective_sync_fn = sync_fn or _default_synchronize_pair
    total_jobs = len(resolved_pairs)
    results: List[BatchJobResult] = []

    if total_jobs == 0:
        summary = BatchSummary(
            results=[],
            total=0,
            succeeded=0,
            failed=0,
            skipped=0,
            dry_run_count=0,
            unmatched_videos=unmatched_videos,
            unmatched_subtitles=unmatched_subtitles,
            total_elapsed_seconds=time.monotonic() - batch_start,
        )
        if cfg.summary_json is not None:
            from subsync.batch.reporter import export_summary_json

            export_summary_json(summary, cfg.summary_json)
        return summary

    if cfg.jobs <= 1 or cfg.dry_run:
        # Sequential execution
        for idx, pair in enumerate(resolved_pairs, start=1):
            res = _execute_single_job(pair, cfg, effective_sync_fn)
            results.append(res)
            if progress_callback is not None:
                try:
                    progress_callback(res, idx, total_jobs)
                except Exception as cb_exc:
                    logger.debug("Progress callback error: %s", cb_exc)
            if res.status == "FAILED" and not cfg.continue_on_error:
                break
    else:
        # Parallel execution via ThreadPoolExecutor while preserving input pair order
        ordered_results: Dict[int, BatchJobResult] = {}
        completed_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.jobs) as pool:
            future_to_idx = {
                pool.submit(_execute_single_job, pair, cfg, effective_sync_fn): idx
                for idx, pair in enumerate(resolved_pairs)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                pair = resolved_pairs[idx]
                try:
                    res = future.result()
                except Exception as exc:
                    res = BatchJobResult(
                        video_path=pair.video,
                        sub_path=pair.subtitle,
                        status="FAILED",
                        error=str(exc),
                        season=pair.season,
                        episode=pair.episode,
                    )
                ordered_results[idx] = res
                completed_count += 1
                if progress_callback is not None:
                    try:
                        progress_callback(res, completed_count, total_jobs)
                    except Exception as cb_exc:
                        logger.debug("Progress callback error: %s", cb_exc)

        results = [ordered_results[i] for i in sorted(ordered_results.keys())]

    total_elapsed = time.monotonic() - batch_start
    summary = BatchSummary(
        results=results,
        unmatched_videos=unmatched_videos,
        unmatched_subtitles=unmatched_subtitles,
        total_elapsed_seconds=total_elapsed,
    )
    summary.recompute_counts()

    if cfg.summary_json is not None:
        from subsync.batch.reporter import export_summary_json

        export_summary_json(summary, cfg.summary_json)

    return summary
