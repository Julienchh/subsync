"""Video filename and directory parser for TV series episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

DEFAULT_VIDEO_EXTENSIONS = {
    ".mkv",
    ".mp4",
    ".avi",
    ".m4v",
    ".webm",
    ".mov",
    ".mpg",
    ".mpeg",
    ".ts",
    ".m2ts",
    ".wmv",
    ".flv",
    ".ogv",
    ".vob",
    ".3gp",
    ".divx",
    ".rmvb",
}

# ISO-639-1 (2-letter) to ISO-639-2/T or B (3-letter) audio tag normalization
_AUDIO_LANG_MAP = {
    "en": "eng",
    "eng": "eng",
    "english": "eng",
    "fr": "fra",
    "fre": "fra",
    "fra": "fra",
    "french": "fra",
    "es": "spa",
    "spa": "spa",
    "spanish": "spa",
    "de": "ger",
    "deu": "ger",
    "ger": "ger",
    "german": "ger",
    "it": "ita",
    "ita": "ita",
    "italian": "ita",
    "pt": "por",
    "por": "por",
    "pob": "por",
    "nl": "dut",
    "nld": "dut",
    "dut": "dut",
    "pl": "pol",
    "pol": "pol",
    "ru": "rus",
    "rus": "rus",
    "ar": "ara",
    "ara": "ara",
    "ja": "jpn",
    "jpn": "jpn",
    "ko": "kor",
    "kor": "kor",
    "zh": "chi",
    "zho": "chi",
    "chi": "chi",
    "tr": "tur",
    "tur": "tur",
    "sv": "swe",
    "swe": "swe",
}

# Common release/technical tokens used in scene and P2P filenames
_KNOWN_RELEASE_TOKEN_RE = re.compile(
    r"^(?:"
    r"2160p|1080p|1080i|720p|576p|480p|4k|uhd|hd|sd|"
    r"x264|x265|h\.?264|h\.?265|hevc|avc|xvid|divx|vp9|av1|"
    r"10bit|8bit|10-bit|8-bit|hdr|hdr10|dv|dolby\.?vision|"
    r"web-?dl|webrip|web|bluray|blu-ray|bdrip|brrip|hdtv|pdtv|dvdrip|hdrip|remux|"
    r"aac\d*(?:\.\d+)?|ac3|eac3|ddp?\d*(?:\.\d+)?|dd\d*(?:\.\d+)?|dts(?:-?hd)?|truehd|atmos|flac|mp3|opus|"
    r"5\.1|7\.1|2\.0|dual|multi|vostfr|french|truefrench|subfrench|english|eng|ita|spa|ger|"
    r"proper|repack|rerip|internal|limited|extended|uncut|unrated|directors\.?cut|"
    r"amzn|nf|dsnp|atvp|hmax|pcok|hulu|bbc|itvx|"
    r"lol|dimension|sva|avs|fleet|killers|bae|ntb|flux|ion10|cakes|ggez|joy|rarbg|psa|qxr|tigole|ettv|eztv|afg|fov|immerse|batv|tla"
    r")$",
    re.IGNORECASE,
)

# Regex patterns for matching season & episode numbers in filenames
# Pattern 1: S01E01, s1e02, S01E01-E02, S01E01E02, S01.E01
_PAT_SXXEYY = re.compile(
    r"(?<![A-Za-z0-9])[Ss](\d{1,2})[\.\-\s_]*[Ee](\d{1,3})"
    r"(?:[\.\-\s_]*[Ee]?(\d{1,3}))?(?![A-Za-z0-9])"
)

# Pattern 2: 1x02, 01x02, 01x02-03
_PAT_X_EP = re.compile(
    r"(?<![A-Za-z0-9])(\d{1,2})[xX](\d{1,3})"
    r"(?:[\.\-\s_]*(?:[xX]|[Ee])?(\d{1,3}))?(?![A-Za-z0-9])"
)

# Pattern 3: Season 1 Episode 02 / Saison 1 Episode 2
_PAT_SEASON_WORD = re.compile(
    r"(?<![A-Za-z0-9])(?:Season|Saison|Series)[\.\-\s_]*(\d{1,2})"
    r"[\.\-\s_]+(?:Episode|Ep)[\.\-\s_]*(\d{1,3})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Pattern 4: Episode-only in filename (E01, Ep01, Episode 01) when inside a season folder
_PAT_EP_ONLY = re.compile(
    r"(?<![A-Za-z0-9])(?:Episode|Ep|E)[\.\-\s_]*(\d{1,3})(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# Directory name patterns indicating a Season folder (e.g. "S1", "S01", "Season 1", "Saison 01")
_SEASON_ONLY_DIR_RE = re.compile(
    r"^[\s\._\-\[\(]*(?:Season|Saison|Series|S)[\s\._\-]*(\d{1,2})[\s\._\-\]\)]*$",
    re.IGNORECASE,
)

# Directory name containing both Series Name and Season suffix (e.g. "The Blacklist S01", "The Blacklist - Season 1")
_SERIES_AND_SEASON_DIR_RE = re.compile(
    r"^(.+?)[\s\._\-\[\(]+(?:Season|Saison|Series|S)[\s\._\-]*(\d{1,2})[\s\._\-\]\)]*$",
    re.IGNORECASE,
)


@dataclass
class SeriesEpisodeMetadata:
    """Extracted metadata for a single TV series episode video file."""

    video_path: Path
    series_name: str
    season: int
    episode: int
    episode_title: Optional[str] = None
    release_tags: List[str] = field(default_factory=list)
    audio_lang: str = "eng"
    extra_episodes: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.video_path, Path):
            self.video_path = Path(self.video_path)
        self.season = int(self.season)
        self.episode = int(self.episode)
        if self.release_tags is None:
            self.release_tags = []

    @property
    def episode_code(self) -> str:
        """Canonical episode identifier such as 'S01E01'."""
        return f"S{self.season:02d}E{self.episode:02d}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to a JSON-friendly dictionary."""
        return {
            "video_path": str(self.video_path),
            "series_name": self.series_name,
            "season": self.season,
            "episode": self.episode,
            "episode_code": self.episode_code,
            "episode_title": self.episode_title,
            "release_tags": list(self.release_tags),
            "audio_lang": self.audio_lang,
            "extra_episodes": list(self.extra_episodes),
        }


def _clean_title_tokens(raw_text: str) -> str:
    """Clean separators (dots, underscores, dashes) into human-readable spaces."""
    if not raw_text:
        return ""
    cleaned = raw_text.replace(".", " ").replace("_", " ")
    # Collapse surrounding dashes/colons
    cleaned = re.sub(r"^[\s\-:–—]+|[\s\-:–—]+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_tags_and_title(remainder: str) -> Tuple[Optional[str], List[str]]:
    """
    Separate the episode title and release tags from the portion of the filename
    following the season/episode token.

    Example inputs:
      - 'Pilot (1080p x265 10bit Joy)' -> ('Pilot', ['1080p', 'x265', '10bit', 'Joy'])
      - '.The.Freelancer.1080p.WEB-DL.DD5.1.H.264-LOL' -> ('The Freelancer', ['1080p', 'WEB-DL', 'DD5.1', 'H.264', 'LOL'])
      - ' - Berlin Conclusion (1080p x265 10bit Joy)' -> ('Berlin Conclusion', ['1080p', 'x265', '10bit', 'Joy'])
    """
    if not remainder:
        return None, []

    bracket_tags: List[str] = []

    def _collect_bracket(match: re.Match) -> str:
        inner = match.group(1)
        tokens = [
            t.strip()
            for t in re.split(r"[\s,;_]+", inner)
            if t.strip()
        ]
        bracket_tags.extend(tokens)
        return " "

    # Extract parenthetical or bracketed blocks: (1080p x265 10bit Joy) or [720p HDTV]
    without_brackets = re.sub(r"[\(\[]([^\)\]]+)[\)\]]", _collect_bracket, remainder)

    # Normalize common compound codecs before tokenizing (H.264 -> H264, DD5.1 -> DD5_1)
    normalized = re.sub(r"\b([Hh])\.?(26[45])\b", r"\1\2", without_brackets)
    normalized = re.sub(r"\b(DDP?|AAC|AC3|DTS)[\.\s]?(5\.1|7\.1|2\.0)\b", r"\1\2", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(5|7|2)\.(1|0)\b", r"\1_\2", normalized)

    raw_tokens = [
        tok for tok in re.split(r"[\.\s_]+", normalized.strip()) if tok and tok not in {"-", "–", "—"}
    ]

    title_tokens: List[str] = []
    inline_tags: List[str] = []
    in_release_section = False

    for tok in raw_tokens:
        restored_tok = tok.replace("_", ".")
        # Check hyphenated group suffix like 'x264-LOL' or 'WEB-DL-DIMENSION'
        if "-" in restored_tok and not restored_tok.upper().startswith("WEB-DL") and not restored_tok.upper().startswith("BLU-RAY"):
            parts = [p for p in restored_tok.split("-") if p]
            if any(_KNOWN_RELEASE_TOKEN_RE.match(p) for p in parts):
                in_release_section = True
                inline_tags.extend(parts)
                continue

        if _KNOWN_RELEASE_TOKEN_RE.match(restored_tok):
            in_release_section = True
            inline_tags.append(restored_tok)
        elif in_release_section:
            # Once release tags have started, trailing tokens (such as release group name) are tags
            inline_tags.extend([p for p in restored_tok.split("-") if p])
        else:
            title_tokens.append(restored_tok)

    all_tags: List[str] = []
    seen_lower = set()
    for tag in inline_tags + bracket_tags:
        clean_tag = tag.strip(".-_[]()")
        if clean_tag and clean_tag.lower() not in seen_lower:
            seen_lower.add(clean_tag.lower())
            all_tags.append(clean_tag)

    ep_title = _clean_title_tokens(" ".join(title_tokens))
    return (ep_title if ep_title else None), all_tags


def _infer_series_and_season_from_parents(video_path: Path) -> Tuple[Optional[str], Optional[int]]:
    """
    Inspect parent and grandparent directories to infer the series name and/or season number
    when the video filename omits them (e.g. '/mnt/f/Séries/The Blacklist/S1/S01E01.mkv').
    """
    inferred_season: Optional[int] = None
    inferred_series: Optional[str] = None

    ignored_dir_names = {
        "video",
        "videos",
        "tv",
        "series",
        "séries",
        "shows",
        "media",
        "downloads",
        "mnt",
        "home",
        "tmp",
        "var",
        "usr",
        "episodes",
        "subs",
        "subtitles",
    }

    current = video_path.resolve().parent if video_path.exists() else video_path.parent
    for _ in range(4):
        if not current or current.name == "" or current == current.parent:
            break
        dir_name = current.name.strip()
        if dir_name.lower() in ignored_dir_names or len(dir_name) <= 1:
            current = current.parent
            continue

        # Check if directory is purely a season folder like 'S1', 'S01', 'Season 1', 'Saison 1'
        m_season_only = _SEASON_ONLY_DIR_RE.match(dir_name)
        if m_season_only:
            if inferred_season is None:
                inferred_season = int(m_season_only.group(1))
            current = current.parent
            continue

        # Check if directory has both series name and season like 'The Blacklist S1' or 'The Blacklist - Season 1'
        m_both = _SERIES_AND_SEASON_DIR_RE.match(dir_name)
        if m_both:
            candidate_series = _clean_title_tokens(m_both.group(1))
            if candidate_series and candidate_series.lower() not in ignored_dir_names:
                inferred_series = candidate_series
                if inferred_season is None:
                    inferred_season = int(m_both.group(2))
                break

        # Otherwise, treat this directory name as the series name
        candidate_series = _clean_title_tokens(dir_name)
        if candidate_series and candidate_series.lower() not in ignored_dir_names:
            inferred_series = candidate_series
            break

        current = current.parent

    return inferred_series, inferred_season


def probe_audio_language(
    video_path: Union[Path, str],
    default_lang: str = "eng",
    timeout: float = 8.0,
) -> str:
    """
    Probe the primary audio stream language of a video file via `ffprobe`.
    Falls back quickly to `default_lang` ('eng') if ffprobe fails, times out,
    or no language tag is embedded.
    """
    path = Path(video_path)
    if not path.exists() or not path.is_file():
        return default_lang

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index:stream_disposition=default:stream_tags=language",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return default_lang
        data = json.loads(proc.stdout)
        streams = data.get("streams") or []
        if not streams:
            return default_lang

        # Prefer stream marked as default disposition if it has a valid language tag
        ordered_streams = sorted(
            streams,
            key=lambda s: int((s.get("disposition") or {}).get("default", 0)),
            reverse=True,
        )
        for stream in ordered_streams:
            tags = stream.get("tags") or {}
            # Case-insensitive lookup for 'language' tag
            raw_lang = None
            for k, v in tags.items():
                if k.lower() == "language" and isinstance(v, str):
                    raw_lang = v.strip().lower()
                    break
            if raw_lang and raw_lang not in {"und", "unk", "unknown", "mul", "zxx", ""}:
                return _AUDIO_LANG_MAP.get(raw_lang, raw_lang)
    except Exception:
        return default_lang

    return default_lang


def parse_episode_video(
    video_path: Union[Path, str],
    probe_audio: bool = True,
) -> Optional[SeriesEpisodeMetadata]:
    """
    Extract series metadata (series_name, season, episode, episode_title, release_tags, audio_lang)
    from a video file path.

    Supports naming conventions such as:
      - 'The Blacklist S01E01 Pilot (1080p x265 10bit Joy).mkv'
      - 'Show.Name.1x02.Title.mkv'
      - 'The.Blacklist.S01E02.The.Freelancer.1080p.WEB-DL.DD5.1.H.264-LOL.mkv'
      - '/path/to/The Blacklist/S1/S01E01 Pilot.mkv' (falls back to parent/grandparent directory)
    """
    path = Path(video_path)
    stem = path.stem

    season: Optional[int] = None
    episode: Optional[int] = None
    extra_episodes: List[int] = []
    prefix_str = ""
    suffix_str = ""

    # Try Pattern 1: S01E01
    m = _PAT_SXXEYY.search(stem)
    if not m:
        # Try Pattern 2: 1x02
        m = _PAT_X_EP.search(stem)
    if not m:
        # Try Pattern 3: Season 1 Episode 2
        m = _PAT_SEASON_WORD.search(stem)

    if m:
        season = int(m.group(1))
        episode = int(m.group(2))
        if m.lastindex and m.lastindex >= 3 and m.group(3):
            extra_ep = int(m.group(3))
            if extra_ep > episode:
                extra_episodes = list(range(episode + 1, extra_ep + 1))
            elif extra_ep != episode:
                extra_episodes = [extra_ep]
        prefix_str = stem[: m.start()]
        suffix_str = stem[m.end() :]
    else:
        # Try Pattern 4: Episode-only (E01 / Ep 01 / Episode 01) with season from parent folder
        m_ep = _PAT_EP_ONLY.search(stem)
        if not m_ep:
            return None
        episode = int(m_ep.group(1))
        prefix_str = stem[: m_ep.start()]
        suffix_str = stem[m_ep.end() :]

    series_name = _clean_title_tokens(prefix_str)
    parent_series, parent_season = _infer_series_and_season_from_parents(path)

    if season is None:
        season = parent_season if parent_season is not None else 1

    if not series_name:
        series_name = parent_series or "Unknown Series"

    episode_title, release_tags = _extract_tags_and_title(suffix_str)
    audio_lang = probe_audio_language(path) if probe_audio else "eng"

    return SeriesEpisodeMetadata(
        video_path=path,
        series_name=series_name,
        season=season,
        episode=episode,
        episode_title=episode_title,
        release_tags=release_tags,
        audio_lang=audio_lang,
        extra_episodes=extra_episodes,
    )


def scan_series_directory(
    series_dir: Union[Path, str],
    recursive: bool = False,
    probe_audio: bool = True,
    video_extensions: Optional[Sequence[str]] = None,
) -> List[SeriesEpisodeMetadata]:
    """
    Scan a directory for TV series episode video files and return parsed
    `SeriesEpisodeMetadata` items sorted by `(season, episode, filename)`.
    """
    root = Path(series_dir)
    if not root.exists() or not root.is_dir():
        return []

    allowed_exts = (
        {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in video_extensions}
        if video_extensions
        else DEFAULT_VIDEO_EXTENSIONS
    )

    iterator = root.rglob("*") if recursive else root.iterdir()
    episodes: List[SeriesEpisodeMetadata] = []

    for entry in iterator:
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in allowed_exts:
            continue
        meta = parse_episode_video(entry, probe_audio=probe_audio)
        if meta is not None:
            episodes.append(meta)

    episodes.sort(key=lambda item: (item.season, item.episode, item.video_path.name))
    return episodes
