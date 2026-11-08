"""Automated TV series subtitle download, normalization, and synchronization package."""

from subsync.download.addic7ed import Addic7edProvider, BUILTIN_SHOW_IDS
from subsync.download.mock_provider import (
    MockSubtitleProvider,
    build_srt_from_intervals,
    format_srt_timestamp,
)
from subsync.download.normalizer import (
    SubtitleNormalizationError,
    detect_subtitle_encoding,
    normalize_subtitle_bytes,
    validate_subtitle_text,
    write_normalized_subtitle,
)
from subsync.download.opensubtitles import OpenSubtitlesProvider
from subsync.download.parser import (
    SeriesEpisodeMetadata,
    parse_episode_video,
    probe_audio_language,
    scan_series_directory,
)
from subsync.download.pipeline import (
    DEFAULT_OUT_PATTERN,
    EpisodeSyncResult,
    SeriesSyncConfig,
    SeriesSyncSummary,
    create_provider_chain,
    format_series_output_path,
    parse_srt_events,
    run_series_sync,
    shift_srt_text,
    synchronize_episode_subtitle,
)
from subsync.download.provider import (
    NormalizedLanguage,
    ProviderAuthError,
    ProviderError,
    ProviderNetworkError,
    ProviderRateLimitError,
    SubtitleCandidate,
    SubtitleProvider,
    matches_language,
    normalize_lang,
    score_subtitle_candidate,
)
from subsync.download.subdl import (
    GestdownProvider,
    PodnapisiProvider,
    SubDLProvider,
    TVSubtitlesProvider,
)

__all__ = [
    "Addic7edProvider",
    "BUILTIN_SHOW_IDS",
    "DEFAULT_OUT_PATTERN",
    "EpisodeSyncResult",
    "GestdownProvider",
    "MockSubtitleProvider",
    "NormalizedLanguage",
    "OpenSubtitlesProvider",
    "PodnapisiProvider",
    "ProviderAuthError",
    "ProviderError",
    "ProviderNetworkError",
    "ProviderRateLimitError",
    "SeriesEpisodeMetadata",
    "SeriesSyncConfig",
    "SeriesSyncSummary",
    "SubDLProvider",
    "SubtitleCandidate",
    "SubtitleNormalizationError",
    "SubtitleProvider",
    "TVSubtitlesProvider",
    "build_srt_from_intervals",
    "create_provider_chain",
    "detect_subtitle_encoding",
    "format_series_output_path",
    "format_srt_timestamp",
    "matches_language",
    "normalize_lang",
    "normalize_subtitle_bytes",
    "parse_episode_video",
    "parse_srt_events",
    "probe_audio_language",
    "run_series_sync",
    "scan_series_directory",
    "score_subtitle_candidate",
    "shift_srt_text",
    "synchronize_episode_subtitle",
    "validate_subtitle_text",
    "write_normalized_subtitle",
]
