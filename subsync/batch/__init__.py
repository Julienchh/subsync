"""Unattended batch resynchronization subsystem (`subsync.batch`)."""

from subsync.batch.discovery import (
    SUBTITLE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    discover_media_files,
    is_subtitle_file,
    is_synced_output_file,
    is_video_file,
)
from subsync.batch.matcher import (
    EpisodeInfo,
    LanguageTag,
    MatchedPair,
    MatchReport,
    extract_episode_info,
    match_files,
    normalize_language,
)
from subsync.batch.reporter import (
    BatchProgressReporter,
    create_progress_callback,
    export_summary_json,
    format_correlation,
    format_offset,
    format_speed_factor,
    format_summary,
    format_summary_table,
    print_summary,
    print_summary_table,
)
from subsync.batch.runner import (
    BatchConfig,
    BatchJobResult,
    BatchSummary,
    format_output_path,
    run_batch,
)

__all__ = [
    # discovery
    "VIDEO_EXTENSIONS",
    "SUBTITLE_EXTENSIONS",
    "discover_media_files",
    "is_video_file",
    "is_subtitle_file",
    "is_synced_output_file",
    # matcher
    "EpisodeInfo",
    "LanguageTag",
    "MatchedPair",
    "MatchReport",
    "extract_episode_info",
    "match_files",
    "normalize_language",
    # runner
    "BatchConfig",
    "BatchJobResult",
    "BatchSummary",
    "format_output_path",
    "run_batch",
    # reporter
    "BatchProgressReporter",
    "create_progress_callback",
    "export_summary_json",
    "format_correlation",
    "format_offset",
    "format_speed_factor",
    "format_summary",
    "format_summary_table",
    "print_summary",
    "print_summary_table",
]
