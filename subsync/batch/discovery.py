"""Media and subtitle file discovery for unattended batch synchronization."""

from __future__ import annotations

import glob
import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: Set[str] = {
    ".mkv",
    ".mp4",
    ".avi",
    ".mov",
    ".m4v",
    ".webm",
    ".ts",
    ".flv",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".wav",
    ".mp3",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
}

SUBTITLE_EXTENSIONS: Set[str] = {
    ".srt",
    ".ass",
    ".ssa",
    ".vtt",
    ".sub",
}

_NATURAL_SORT_RE = re.compile(r"(\d+)")
_SYNCED_FILE_RE = re.compile(r"(?:^|[._-])synced(?:\.[a-z0-9]{2,5})?$", re.IGNORECASE)


def _natural_sort_key(path: Path) -> List[Union[int, str]]:
    """Produce a key for natural sorting of file paths (e.g. E2 before E10)."""
    parts = _NATURAL_SORT_RE.split(str(path).lower())
    return [int(part) if part.isdigit() else part for part in parts]


def is_video_file(path: Union[str, Path]) -> bool:
    """Return True if path has a recognized video or audio extension."""
    p = Path(path)
    return p.suffix.lower() in VIDEO_EXTENSIONS


def is_subtitle_file(path: Union[str, Path]) -> bool:
    """Return True if path has a recognized subtitle extension."""
    p = Path(path)
    return p.suffix.lower() in SUBTITLE_EXTENSIONS


def is_synced_output_file(path: Union[str, Path]) -> bool:
    """Return True if path is an already-generated `.synced.*` subtitle output file."""
    p = Path(path)
    name_lower = p.name.lower()
    if ".synced." in name_lower:
        return True
    return bool(_SYNCED_FILE_RE.search(p.stem))


def _normalize_patterns(
    pattern: Optional[Union[str, Path, Sequence[Union[str, Path]]]]
) -> List[str]:
    if pattern is None:
        return []
    if isinstance(pattern, (str, Path)):
        return [str(pattern)]
    return [str(item) for item in pattern if item is not None]


def _pattern_explicitly_targets_synced(pattern_str: str) -> bool:
    """Check whether a user-supplied pattern explicitly asks for `.synced` files."""
    return ".synced" in pattern_str.lower()


def _scan_directory(
    directory: Path,
    allowed_extensions: Set[str],
    recursive: bool = False,
) -> List[Path]:
    """Scan a directory for files matching `allowed_extensions`."""
    if not directory.exists():
        logger.warning("Directory does not exist: %s", directory)
        return []
    if directory.is_file():
        return [directory] if directory.suffix.lower() in allowed_extensions else []

    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    matched: List[Path] = []
    for entry in iterator:
        try:
            if entry.is_file() and entry.suffix.lower() in allowed_extensions:
                matched.append(entry)
        except OSError as exc:
            logger.warning("Failed to inspect path %s: %s", entry, exc)
    return matched


def _expand_patterns(
    patterns: Sequence[str],
    base_dir: Optional[Path],
    allowed_extensions: Set[str],
    recursive: bool = False,
    exclude_synced: bool = False,
) -> List[Path]:
    """Expand one or more glob patterns or direct file/directory paths."""
    results: List[Path] = []
    for raw_pat in patterns:
        explicit_synced = _pattern_explicitly_targets_synced(raw_pat)
        pat_path = Path(raw_pat)

        candidates: List[Path] = []
        # Direct existing file or directory check
        if pat_path.exists():
            if pat_path.is_file():
                candidates.append(pat_path)
            elif pat_path.is_dir():
                candidates.extend(_scan_directory(pat_path, allowed_extensions, recursive=recursive))
        else:
            # Try expanding relative to base_dir if provided and pattern is relative
            if base_dir is not None and not pat_path.is_absolute():
                rel_target = base_dir / raw_pat
                if rel_target.exists():
                    if rel_target.is_file():
                        candidates.append(rel_target)
                    elif rel_target.is_dir():
                        candidates.extend(
                            _scan_directory(rel_target, allowed_extensions, recursive=recursive)
                        )
                else:
                    glob_expr = str(base_dir / ("**/" + raw_pat if recursive and "**" not in raw_pat else raw_pat))
                    candidates.extend(Path(p) for p in glob.glob(glob_expr, recursive=True))

            # Also expand raw_pat directly against current working directory / filesystem
            if not candidates:
                if recursive and "**" not in raw_pat and not pat_path.is_absolute():
                    glob_matches = glob.glob(raw_pat, recursive=True) + glob.glob(
                        str(Path("**") / raw_pat), recursive=True
                    )
                else:
                    glob_matches = glob.glob(raw_pat, recursive=True)
                candidates.extend(Path(p) for p in glob_matches)

        for cand in candidates:
            try:
                if not cand.is_file():
                    continue
            except OSError:
                continue
            if cand.suffix.lower() not in allowed_extensions:
                continue
            if exclude_synced and not explicit_synced and is_synced_output_file(cand):
                continue
            results.append(cand)

    return results


def discover_media_files(
    ref_dir: Optional[Union[str, Path]] = None,
    sub_dir: Optional[Union[str, Path]] = None,
    ref_pattern: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    sub_pattern: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    recursive: bool = False,
    include_synced: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Discover reference video/audio files and subtitle files from directories or globs.

    Parameters
    ----------
    ref_dir : str or Path, optional
        Directory containing reference video/audio files.
    sub_dir : str or Path, optional
        Directory containing subtitle files. Defaults to `ref_dir` when `ref_dir`
        is provided and neither `sub_dir` nor `sub_pattern` is specified.
    ref_pattern : str, Path, or sequence of str/Path, optional
        Glob pattern(s) for reference files (e.g. ``"*.mkv"``, ``"season1/*.mkv"``).
    sub_pattern : str, Path, or sequence of str/Path, optional
        Glob pattern(s) for subtitle files (e.g. ``"*.srt"``, ``"subs/*.srt"``).
    recursive : bool, default False
        If True, scan directories recursively.
    include_synced : bool, default False
        If True, do not exclude `.synced.*` files during directory discovery.

    Returns
    -------
    Tuple[List[Path], List[Path]]
        Naturally sorted, deduplicated `(video_paths, subtitle_paths)`.
    """
    ref_dir_path = Path(ref_dir) if ref_dir is not None else None
    sub_dir_path = Path(sub_dir) if sub_dir is not None else None

    # Default sub_dir to ref_dir if only ref_dir was passed
    if sub_dir_path is None and sub_pattern is None and ref_dir_path is not None:
        sub_dir_path = ref_dir_path

    ref_patterns = _normalize_patterns(ref_pattern)
    sub_patterns = _normalize_patterns(sub_pattern)

    discovered_refs: List[Path] = []
    if ref_patterns:
        discovered_refs.extend(
            _expand_patterns(
                ref_patterns,
                base_dir=ref_dir_path,
                allowed_extensions=VIDEO_EXTENSIONS,
                recursive=recursive,
                exclude_synced=False,
            )
        )
    elif ref_dir_path is not None:
        discovered_refs.extend(
            _scan_directory(ref_dir_path, VIDEO_EXTENSIONS, recursive=recursive)
        )

    discovered_subs: List[Path] = []
    if sub_patterns:
        discovered_subs.extend(
            _expand_patterns(
                sub_patterns,
                base_dir=sub_dir_path or ref_dir_path,
                allowed_extensions=SUBTITLE_EXTENSIONS,
                recursive=recursive,
                exclude_synced=not include_synced,
            )
        )
    elif sub_dir_path is not None:
        raw_subs = _scan_directory(
            sub_dir_path, SUBTITLE_EXTENSIONS, recursive=recursive
        )
        for sub_file in raw_subs:
            if not include_synced and is_synced_output_file(sub_file):
                continue
            discovered_subs.append(sub_file)

    # Deduplicate while preserving natural sort order
    unique_refs: List[Path] = []
    seen_refs: Set[Path] = set()
    for p in sorted(discovered_refs, key=_natural_sort_key):
        resolved = p.resolve() if p.exists() else p
        if resolved not in seen_refs:
            seen_refs.add(resolved)
            unique_refs.append(p)

    unique_subs: List[Path] = []
    seen_subs: Set[Path] = set()
    for p in sorted(discovered_subs, key=_natural_sort_key):
        resolved = p.resolve() if p.exists() else p
        if resolved not in seen_subs:
            seen_subs.add(resolved)
            unique_subs.append(p)

    if not unique_refs:
        logger.warning(
            "No reference media files discovered (ref_dir=%s, ref_pattern=%s)",
            ref_dir,
            ref_pattern,
        )
    if not unique_subs:
        logger.warning(
            "No subtitle files discovered (sub_dir=%s, sub_pattern=%s)",
            sub_dir,
            sub_pattern,
        )

    return unique_refs, unique_subs
