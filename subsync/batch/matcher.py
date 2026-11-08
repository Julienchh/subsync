"""Multi-tier episode and filename matching engine for batch synchronization."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

# Canonical ISO 639-1 language code mapping from 2-letter, 3-letter, and full names
_LANGUAGE_SYNONYMS: Dict[str, str] = {
    # French
    "fr": "fr",
    "fre": "fr",
    "fra": "fr",
    "french": "fr",
    "francais": "fr",
    "français": "fr",
    "vf": "fr",
    "vostfr": "fr",
    # English
    "en": "en",
    "eng": "en",
    "english": "en",
    # Spanish
    "es": "es",
    "spa": "es",
    "esp": "es",
    "spanish": "es",
    "espanol": "es",
    "español": "es",
    # German
    "de": "de",
    "ger": "de",
    "deu": "de",
    "german": "de",
    "deutsch": "de",
    # Italian
    "it": "it",
    "ita": "it",
    "italian": "it",
    "italiano": "it",
    # Portuguese
    "pt": "pt",
    "por": "pt",
    "pob": "pt",
    "ptbr": "pt",
    "portuguese": "pt",
    # Russian
    "ru": "ru",
    "rus": "ru",
    "russian": "ru",
    # Dutch
    "nl": "nl",
    "dut": "nl",
    "nld": "nl",
    "dutch": "nl",
    # Polish
    "pl": "pl",
    "pol": "pl",
    "polish": "pl",
    # Japanese
    "ja": "ja",
    "jpn": "ja",
    "jap": "ja",
    "japanese": "ja",
    # Chinese
    "zh": "zh",
    "chi": "zh",
    "zho": "zh",
    "chs": "zh",
    "cht": "zh",
    "chinese": "zh",
    # Arabic
    "ar": "ar",
    "ara": "ar",
    "arabic": "ar",
    # Korean
    "ko": "ko",
    "kor": "ko",
    "korean": "ko",
    # Swedish
    "sv": "sv",
    "swe": "sv",
    "swedish": "sv",
    # Danish
    "da": "da",
    "dan": "da",
    "danish": "da",
    # Norwegian
    "no": "no",
    "nor": "no",
    "norwegian": "no",
    # Finnish
    "fi": "fi",
    "fin": "fi",
    "finnish": "fi",
    # Czech
    "cs": "cs",
    "cze": "cs",
    "ces": "cs",
    "czech": "cs",
    # Hungarian
    "hu": "hu",
    "hun": "hu",
    "hungarian": "hu",
    # Romanian
    "ro": "ro",
    "rum": "ro",
    "ron": "ro",
    "romanian": "ro",
    # Turkish
    "tr": "tr",
    "tur": "tr",
    "turkish": "tr",
    # Greek
    "el": "el",
    "gre": "el",
    "ell": "el",
    "greek": "el",
    # Hebrew
    "he": "he",
    "heb": "he",
    "hebrew": "he",
    # Ukrainian
    "uk": "uk",
    "ukr": "uk",
    "ukrainian": "uk",
}

# Unambiguous language names/3-letter codes that can appear mid-suffix (e.g. .LOL.French.C.updated.Addic7ed.com.srt)
_UNAMBIGUOUS_MID_LANG_TOKENS: Set[str] = {
    "fre",
    "fra",
    "french",
    "francais",
    "français",
    "vostfr",
    "eng",
    "english",
    "spa",
    "spanish",
    "ger",
    "deu",
    "german",
    "ita",
    "italian",
    "por",
    "portuguese",
    "rus",
    "russian",
    "dut",
    "nld",
    "dutch",
    "pol",
    "polish",
    "jpn",
    "japanese",
    "chi",
    "zho",
    "chinese",
    "ara",
    "arabic",
    "kor",
    "korean",
}

_SUB_MODIFIER_TAGS: Set[str] = {
    "hi",
    "sdh",
    "cc",
    "forced",
    "synced",
    "updated",
    "addic7ed",
    "com",
    "completed",
    "proper",
    "repack",
    "rerip",
    "retail",
    "subrip",
    "srt",
}


def normalize_language(lang: Optional[str]) -> Optional[str]:
    """Normalize a language tag (e.g. 'fre', 'fra', 'French', 'fr') to canonical 2-letter code."""
    if not lang:
        return None
    cleaned = lang.strip().lower().lstrip(".")
    return _LANGUAGE_SYNONYMS.get(cleaned, cleaned if len(cleaned) in (2, 3) else None)


class LanguageTag(str):
    """Language tag string that compares equal across ISO 639-1/639-2/name synonyms."""

    raw: str
    canonical: str

    def __new__(cls, value: str, canonical: Optional[str] = None) -> "LanguageTag":
        norm = canonical or normalize_language(value) or value.strip().lower()
        obj = super().__new__(cls, norm)
        obj.raw = value.strip().lower()
        obj.canonical = norm
        return obj

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            if super().__eq__(other):
                return True
            other_norm = normalize_language(other)
            if other_norm is not None and other_norm == self.canonical:
                return True
            return self.raw == other.strip().lower()
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()


# Known technical and release group tokens
_KNOWN_TECH_TAGS: Dict[str, str] = {
    "2160p": "2160p",
    "4k": "4K",
    "1080p": "1080p",
    "1080i": "1080i",
    "720p": "720p",
    "576p": "576p",
    "480p": "480p",
    "360p": "360p",
    "web-dl": "WEB-DL",
    "webdl": "WEB-DL",
    "webrip": "WEBRip",
    "web": "WEB",
    "bluray": "BluRay",
    "blu-ray": "BluRay",
    "bdrip": "BDRip",
    "brrip": "BRRip",
    "hdtv": "HDTV",
    "pdtv": "PDTV",
    "dvdrip": "DVDRip",
    "remux": "REMUX",
    "amzn": "AMZN",
    "nf": "NF",
    "dsnp": "DSNP",
    "hmax": "HMAX",
    "atvp": "ATVP",
    "hulu": "HULU",
    "x264": "x264",
    "h264": "H264",
    "h.264": "H264",
    "x265": "x265",
    "h265": "H265",
    "h.265": "H265",
    "hevc": "HEVC",
    "avc": "AVC",
    "xvid": "XviD",
    "divx": "DivX",
    "10bit": "10bit",
    "10-bit": "10bit",
    "8bit": "8bit",
    "hdr": "HDR",
    "hdr10": "HDR10",
    "dv": "DV",
    "aac": "AAC",
    "ac3": "AC3",
    "eac3": "EAC3",
    "dd5.1": "DD5.1",
    "ddp5.1": "DDP5.1",
    "dts": "DTS",
    "truehd": "TrueHD",
    "atmos": "Atmos",
    "flac": "FLAC",
    "proper": "PROPER",
    "repack": "REPACK",
    "rerip": "RERIP",
    "internal": "INTERNAL",
    "extended": "EXTENDED",
    "unrated": "UNRATED",
}

_KNOWN_RELEASE_GROUPS: Set[str] = {
    "lol",
    "dimension",
    "joy",
    "sva",
    "avs",
    "killers",
    "fleet",
    "immerse",
    "2hd",
    "ntb",
    "ion10",
    "rarbg",
    "yts",
    "yify",
    "ettv",
    "eztv",
    " PSA",
    "psa",
    "qxr",
    "tigole",
    "utr",
    "vyndros",
    "rovers",
    "sparks",
    "amiable",
    "geckos",
    "deflate",
    "bamboozle",
    "flux",
    "cakes",
    "ggwp",
    "playweb",
    " successfulcrab",
    "syncopy",
    "fov",
    "tla",
    "river",
    "afg",
    "memento",
    "mtb",
    "bae",
    "strife",
    "fgt",
    "evo",
    "goki",
}

# Episode extraction regexes in priority order:
# 1. S01E01, s1e1, S01.E01, S01_E01, S01E01-E02, S01E01E02, S01E01-02
_RE_SXXEYY = re.compile(
    r"(?i)(?:^|(?<![a-z0-9]))s(\d{1,2})[\s._-]*e(\d{1,3})"
    r"((?:[\s._-]*[-–][\s._-]*[eE]?\d{1,3}|[\s._-]*[eE]\d{1,3})*)"
    r"(?=$|(?![a-z0-9]))"
)

# 2. 01x01, 1x01, 01x01-02, 01x01-01x02
_RE_NXX = re.compile(
    r"(?i)(?:^|(?<![a-z0-9]))(\d{1,2})x(\d{2,3})"
    r"(?:[\s._-]*[-–][\s._-]*(?:\d{1,2}x)?(\d{2,3}))?"
    r"(?=$|(?![a-z0-9]))"
)

# 3. Season 1 Episode 1, Saison 1 Episode 1
_RE_SEASON_EPISODE = re.compile(
    r"(?i)(?:^|(?<![a-z0-9]))(?:season|saison)[\s._-]*(\d{1,2})[\s._,-]*"
    r"(?:episode|ep|e)[\s._-]*(\d{1,3})"
    r"(?:[\s._-]*[-–][\s._-]*(?:episode|ep|e)?[\s._-]*(\d{1,3}))?"
    r"(?=$|(?![a-z0-9]))"
)

# 4. E01, Ep01, Ep.01, Episode 01 (standalone without season)
_RE_EP_ONLY = re.compile(
    r"(?i)(?:^|[\s._\-\[\(])(?:ep(?:isode)?[\s._-]*(\d{1,3})|e(\d{2,3}))"
    r"(?:[\s._-]*[-–][\s._-]*[eE]?(?:p)?(\d{1,3}))?"
    r"(?=$|[\s._\-\]\)])"
)

# 5. Technical tokens to mask before testing 3-digit scene numbering (101, 102, etc.)
_RE_TECH_NOISE_FOR_3DIGIT = re.compile(
    r"(?i)(?:^|(?<![a-z0-9]))(?:"
    r"2160[pi]?|1080[pi]?|720[pi]?|576[pi]?|480[pi]?|360[pi]?|240[pi]?|"
    r"[xh][._-]?26[45]|26[45]|10[._-]?bit|8[._-]?bit|"
    r"(?:ddp?|aac|ac3|dts)?[._-]?[257]\.[01]|"
    r"(?:19|20)\d{2}"
    r")(?=$|(?![a-z0-9]))"
)

_RE_3DIGIT_SCENE = re.compile(
    r"(?:^|[\s._\-\[\(])([1-9])(0[1-9]|[1-4]\d)(?=$|[\s._\-\]\)])"
)

_EXCLUDED_3DIGIT_NUMBERS: Set[str] = {
    "100",
    "200",
    "240",
    "264",
    "265",
    "300",
    "360",
    "400",
    "404",
    "420",
    "422",
    "444",
    "480",
    "500",
    "512",
    "576",
    "600",
    "700",
    "720",
    "800",
    "900",
}


@dataclass
class EpisodeInfo:
    """Extracted metadata from a video or subtitle filename."""

    series_title: str
    season: Optional[int]
    episode: Optional[int]
    extra_episodes: List[int] = field(default_factory=list)
    release_tags: Set[str] = field(default_factory=set)
    language_tag: Optional[str] = None
    normalized_stem: str = ""

    @property
    def has_episode(self) -> bool:
        return self.episode is not None

    @property
    def all_episodes(self) -> List[int]:
        if self.episode is None:
            return []
        return [self.episode] + [ep for ep in self.extra_episodes if ep != self.episode]


@dataclass
class MatchedPair:
    """A matched (video, subtitle) file pair with matching metadata."""

    video: Path
    subtitle: Path
    tier: int = 1
    method: str = "episode"
    score: float = 1.0
    season: Optional[int] = None
    episode: Optional[int] = None
    video_info: Optional[EpisodeInfo] = None
    subtitle_info: Optional[EpisodeInfo] = None

    @property
    def video_path(self) -> Path:
        return self.video

    @property
    def ref_path(self) -> Path:
        return self.video

    @property
    def ref(self) -> Path:
        return self.video

    @property
    def sub_path(self) -> Path:
        return self.subtitle

    @property
    def subtitle_path(self) -> Path:
        return self.subtitle

    @property
    def sub(self) -> Path:
        return self.subtitle

    def __iter__(self) -> Iterator[Path]:
        yield self.video
        yield self.subtitle

    def __getitem__(self, index: int) -> Path:
        return (self.video, self.subtitle)[index]

    def __len__(self) -> int:
        return 2


@dataclass
class MatchReport:
    """Result of matching candidate video files with candidate subtitle files."""

    paired: List[MatchedPair] = field(default_factory=list)
    unmatched_videos: List[Path] = field(default_factory=list)
    unmatched_subtitles: List[Path] = field(default_factory=list)

    @property
    def pairs(self) -> List[MatchedPair]:
        return self.paired


def _extract_language_suffix(stem: str) -> Tuple[str, Optional[LanguageTag]]:
    """Extract trailing or suffix language tag from filename stem and return (cleaned_stem, language_tag)."""
    working = stem
    # Strip trailing .synced if present
    if working.lower().endswith(".synced"):
        working = working[: -len(".synced")]

    # Check trailing dot/underscore/hyphen separated tokens for language codes
    # Examples:
    #   Show.S01E01.fr
    #   Show.S01E01.fre
    #   Show.S01E01.fr.hi
    #   The Blacklist - 01x01 - Pilot.LOL.French.C.updated.Addic7ed.com
    dot_parts = working.split(".")
    if len(dot_parts) >= 2:
        # Check from the end, skipping modifier tags like .hi, .sdh, .forced, .updated, .Addic7ed, .com, .C
        idx = len(dot_parts) - 1
        while idx >= 1:
            tok_lower = dot_parts[idx].strip().lower()
            if tok_lower in _SUB_MODIFIER_TAGS or (len(tok_lower) == 1 and tok_lower.isalpha()):
                idx -= 1
                continue
            if tok_lower in _LANGUAGE_SYNONYMS:
                # If idx is the last meaningful token, 2-letter/3-letter/full name are all valid.
                # If idx is followed by Addic7ed/modifier tags, allow if it's in _LANGUAGE_SYNONYMS.
                is_at_end = idx == len(dot_parts) - 1 or all(
                    p.strip().lower() in _SUB_MODIFIER_TAGS
                    or (len(p.strip()) == 1 and p.strip().isalpha())
                    for p in dot_parts[idx + 1 :]
                )
                if is_at_end or tok_lower in _UNAMBIGUOUS_MID_LANG_TOKENS:
                    lang_tag = LanguageTag(tok_lower, _LANGUAGE_SYNONYMS[tok_lower])
                    kept_parts = dot_parts[:idx]
                    return ".".join(kept_parts), lang_tag
            break

    # Also check underscore or hyphen suffix at the very end (e.g. movie_fr, movie-eng)
    m_end = re.search(r"[._-]([a-zA-Z]{2,10})$", working)
    if m_end:
        cand = m_end.group(1).lower()
        if cand in _LANGUAGE_SYNONYMS:
            lang_tag = LanguageTag(cand, _LANGUAGE_SYNONYMS[cand])
            return working[: m_end.start()], lang_tag

    return working, None


def _parse_extra_episodes(primary_ep: int, tail_str: str) -> List[int]:
    """Parse multi-episode tail such as '-E02', 'E02E03', '-02', '-E03'."""
    if not tail_str:
        return []
    extra: List[int] = []
    # Find all numbers and whether they were preceded by a range hyphen
    tokens = re.findall(r"([-–])?[\s._-]*[eE]?(\d{1,3})", tail_str)
    last_ep = primary_ep
    for hyphen, num_str in tokens:
        ep_num = int(num_str)
        if hyphen and ep_num > last_ep and (ep_num - last_ep) <= 12:
            for r_ep in range(last_ep + 1, ep_num + 1):
                if r_ep not in extra and r_ep != primary_ep:
                    extra.append(r_ep)
        elif ep_num != primary_ep and ep_num not in extra:
            extra.append(ep_num)
        last_ep = ep_num
    return extra


def _extract_release_tags(text_after_ep: str, full_stem: str) -> Set[str]:
    """Extract release group and technical tags (e.g. LOL, DIMENSION, Joy, 1080p, WEB-DL, BluRay)."""
    tags: Set[str] = set()
    search_space = text_after_ep if text_after_ep else full_stem

    # 1. Check multi-token hyphenated tech tags first (WEB-DL, Blu-Ray, H.264, H.265, DD5.1, DDP5.1, 10-bit)
    for pattern, canonical in [
        (r"(?i)\bweb[\s._-]?dl\b", "WEB-DL"),
        (r"(?i)\bblu[\s._-]?ray\b", "BluRay"),
        (r"(?i)\bh\.264\b", "H264"),
        (r"(?i)\bh\.265\b", "H265"),
        (r"(?i)\bddp?5\.1\b", "DD5.1"),
        (r"(?i)\b10-bit\b", "10bit"),
    ]:
        if re.search(pattern, full_stem):
            tags.add(canonical)

    # 2. Extract tokens inside parentheses or brackets in search_space, e.g. "(1080p x265 10bit Joy)"
    for bracket_match in re.finditer(r"[\(\[]([^\)\]]+)[\)\]]", search_space):
        inner = bracket_match.group(1)
        for raw_tok in re.split(r"[\s._,-]+", inner):
            if not raw_tok:
                continue
            low = raw_tok.lower()
            if low in _KNOWN_TECH_TAGS:
                tags.add(raw_tok if low in ("1080p", "720p", "480p", "2160p", "x264", "x265", "10bit") else _KNOWN_TECH_TAGS[low])
            elif low not in _LANGUAGE_SYNONYMS and low not in _SUB_MODIFIER_TAGS and len(raw_tok) >= 2:
                tags.add(raw_tok)

    # 3. Extract hyphenated release group at the end of the stem or before Addic7ed tail, e.g. "x264-LOL", "HDTV-DIMENSION"
    for group_match in re.finditer(
        r"(?:x264|x265|h264|h265|h\.264|h\.265|hevc|xvid|hdtv|web-?dl|webrip|bluray|720p|1080p|480p)-([A-Za-z0-9]{2,15})\b",
        search_space,
        re.IGNORECASE,
    ):
        grp = group_match.group(1)
        if grp.lower() not in _LANGUAGE_SYNONYMS and grp.lower() not in _SUB_MODIFIER_TAGS:
            tags.add(grp)

    # Also check trailing `-GROUP` at the end of `search_space`
    m_trailing_grp = re.search(r"-([A-Za-z0-9]{2,15})$", search_space.strip())
    if m_trailing_grp:
        grp = m_trailing_grp.group(1)
        if (
            grp.lower() not in _LANGUAGE_SYNONYMS
            and grp.lower() not in _SUB_MODIFIER_TAGS
            and not grp.isdigit()
        ):
            tags.add(grp)

    # 4. Scan individual tokens in search_space for known tech tags, known release groups, or Addic7ed release tokens
    # In Addic7ed: "The Blacklist - 01x01 - Pilot.LOL.French.C.updated.Addic7ed.com"
    for raw_tok in re.split(r"[\s._,\(\)\[\]-]+", search_space):
        if not raw_tok:
            continue
        low = raw_tok.lower()
        if low in _KNOWN_TECH_TAGS:
            tags.add(_KNOWN_TECH_TAGS[low])
        elif low in _KNOWN_RELEASE_GROUPS:
            tags.add(raw_tok)

    # 5. Check dot-separated tokens after episode title (e.g. Pilot.LOL or Pilot.SVA)
    if "." in search_space:
        dot_tokens = [t.strip() for t in search_space.split(".") if t.strip()]
        for tok in dot_tokens[1:]:
            # Strip leading hyphen if like "x264-LOL"
            sub_toks = [st for st in tok.split("-") if st]
            for st in sub_toks:
                low = st.lower()
                if low in _KNOWN_TECH_TAGS:
                    tags.add(_KNOWN_TECH_TAGS[low])
                elif (
                    low not in _LANGUAGE_SYNONYMS
                    and low not in _SUB_MODIFIER_TAGS
                    and len(st) >= 2
                    and st.isalnum()
                    and (st.isupper() or low in _KNOWN_RELEASE_GROUPS)
                ):
                    tags.add(st)

    return tags


def _clean_series_title(raw_prefix: str, fallback_parent: Optional[str] = None) -> str:
    """Clean the filename substring preceding the episode identifier into a series title."""
    cleaned = re.sub(r"[._]+", " ", raw_prefix)
    # Remove bracketed prefixes like "[SubsPlease]"
    cleaned = re.sub(r"^\s*\[[^\]]+\]\s*", "", cleaned)
    cleaned = re.sub(r"[\s\-–:(\[]+$", "", cleaned)
    cleaned = re.sub(r"^[\s\-–:)\]]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned and fallback_parent:
        parent_clean = re.sub(r"[._]+", " ", fallback_parent).strip()
        if not re.match(r"^(?:s|season|saison)\s*\d+$", parent_clean, re.IGNORECASE):
            cleaned = parent_clean
    return cleaned


def _build_normalized_stem(stem_no_lang: str) -> str:
    """Build a normalized stem stripped of language, brackets, and release/technical suffixes."""
    s = stem_no_lang
    # Remove parenthesized/bracketed blocks unless they are just a 4-digit year
    def _replace_brackets(m: re.Match) -> str:
        content = m.group(1).strip()
        if re.fullmatch(r"(?:19|20)\d{2}", content):
            return f" {content} "
        return " "

    s = re.sub(r"[\(\[]([^\)\]]*)[\)\]]", _replace_brackets, s)
    # Remove Addic7ed boilerplate
    s = re.sub(
        r"(?i)\b(?:c\.updated|updated|addic7ed(?:\.com)?|synced|proper|repack|rerip)\b",
        " ",
        s,
    )
    # Remove multi-token tech patterns
    s = re.sub(r"(?i)\b(?:web[\s._-]?dl|blu[\s._-]?ray|h\.26[45]|ddp?[257]\.[01]|10-bit)\b", " ", s)

    tokens = re.split(r"[\s._,\-:;]+", s)
    kept: List[str] = []
    for idx, tok in enumerate(tokens):
        if not tok:
            continue
        low = tok.lower()
        if low in _KNOWN_TECH_TAGS or low in _SUB_MODIFIER_TAGS:
            continue
        if idx > 0 and low in _KNOWN_RELEASE_GROUPS:
            continue
        if idx > 0 and low in _UNAMBIGUOUS_MID_LANG_TOKENS:
            continue
        kept.append(low)

    if not kept:
        # Fallback to basic alphanumeric lowercase
        fallback = re.sub(r"[^a-z0-9]+", " ", stem_no_lang.lower()).strip()
        return re.sub(r"\s+", " ", fallback)

    return " ".join(kept)


def extract_episode_info(path: Union[str, Path]) -> EpisodeInfo:
    """Extract structured episode, series, release tag, and language metadata from a file path.

    Handles:
    - ``S01E01``, ``s1e1``, ``S01.E01``, ``S01E01-E02``, ``S01E01E02``
    - ``01x01``, ``1x01``
    - ``Season 1 Episode 1``
    - ``E01``, ``Ep01``
    - 3-digit scene numbers like ``101`` (excluding ``1080p``, ``720p``, ``480p``, ``264``, ``265``, ``10bit``, ``5.1``)
    - Language suffixes like ``.fr.srt``, ``.fre.srt``, ``.fra.srt``, ``.en.srt``, ``.eng.srt``, ``.french.srt``
    """
    p = Path(path)
    raw_stem = p.stem
    stem_no_lang, lang_tag = _extract_language_suffix(raw_stem)

    season: Optional[int] = None
    episode: Optional[int] = None
    extra_episodes: List[int] = []
    match_span: Optional[Tuple[int, int]] = None

    # 1. Try S01E01 / s1e1 / S01.E01 / S01E01-E02 / S01E01E02
    m = _RE_SXXEYY.search(stem_no_lang)
    if m:
        season = int(m.group(1))
        episode = int(m.group(2))
        extra_episodes = _parse_extra_episodes(episode, m.group(3) or "")
        match_span = m.span()

    # 2. Try 01x01 / 1x01
    if match_span is None:
        m = _RE_NXX.search(stem_no_lang)
        if m:
            s_val, e_val = int(m.group(1)), int(m.group(2))
            # Guard against resolutions like 1920x1080 (already prevented by \d{1,2})
            season = s_val
            episode = e_val
            if m.group(3):
                extra_episodes = _parse_extra_episodes(episode, f"-{m.group(3)}")
            match_span = m.span()

    # 3. Try "Season 1 Episode 1"
    if match_span is None:
        m = _RE_SEASON_EPISODE.search(stem_no_lang)
        if m:
            season = int(m.group(1))
            episode = int(m.group(2))
            if m.group(3):
                extra_episodes = _parse_extra_episodes(episode, f"-{m.group(3)}")
            match_span = m.span()

    # 4. Try standalone E01 / Ep01 / Episode 01
    if match_span is None:
        m = _RE_EP_ONLY.search(stem_no_lang)
        if m:
            ep_str = m.group(1) or m.group(2)
            episode = int(ep_str)
            season = None
            if m.group(3):
                extra_episodes = _parse_extra_episodes(episode, f"-{m.group(3)}")
            match_span = m.span()

    # 5. Try 3-digit scene number like 101 (when not 1080p/720p/480p/264/265/10bit/5.1)
    if match_span is None:
        masked = _RE_TECH_NOISE_FOR_3DIGIT.sub(
            lambda mt: " " * (mt.end() - mt.start()), stem_no_lang
        )
        for m3 in _RE_3DIGIT_SCENE.finditer(masked):
            full_3digit = m3.group(1) + m3.group(2)
            if full_3digit in _EXCLUDED_3DIGIT_NUMBERS:
                continue
            season = int(m3.group(1))
            episode = int(m3.group(2))
            match_span = m3.span()
            break

    if match_span is not None:
        prefix = stem_no_lang[: match_span[0]]
        suffix = stem_no_lang[match_span[1] :]
        series_title = _clean_series_title(prefix)
        release_tags = _extract_release_tags(suffix, stem_no_lang)
    else:
        series_title = _clean_series_title(stem_no_lang)
        release_tags = _extract_release_tags(stem_no_lang, stem_no_lang)

    normalized_stem = _build_normalized_stem(stem_no_lang)

    return EpisodeInfo(
        series_title=series_title,
        season=season,
        episode=episode,
        extra_episodes=extra_episodes,
        release_tags=release_tags,
        language_tag=lang_tag,
        normalized_stem=normalized_stem,
    )


def _normalize_title_for_comparison(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _series_titles_compatible(v_title: str, s_title: str) -> Tuple[bool, float]:
    """Check whether video and subtitle series titles are compatible.

    If either title is empty (e.g. ``S01E01.srt``), they are considered compatible.
    If both are non-empty, they must share significant similarity so that multiple
    different shows in the same directory do not cross-match on ``(season, episode)``.
    """
    v_norm = _normalize_title_for_comparison(v_title)
    s_norm = _normalize_title_for_comparison(s_title)
    if not v_norm or not s_norm:
        return True, 0.5
    if v_norm == s_norm:
        return True, 1.0
    if v_norm in s_norm or s_norm in v_norm:
        return True, 0.9
    ratio = difflib.SequenceMatcher(None, v_norm, s_norm).ratio()
    if ratio >= 0.65:
        return True, ratio
    return False, 0.0


def _canonicalize_tag_for_overlap(tag: str) -> str:
    low = tag.lower().strip()
    if low in ("h264", "x264", "h.264", "avc"):
        return "h264"
    if low in ("h265", "x265", "h.265", "hevc"):
        return "h265"
    if low in ("webdl", "web-dl"):
        return "web-dl"
    if low in ("bluray", "blu-ray", "bdrip", "brrip"):
        return "bluray"
    return low


def _compute_release_tag_overlap(v_tags: Set[str], s_tags: Set[str]) -> int:
    if not v_tags or not s_tags:
        return 0
    v_norm = {_canonicalize_tag_for_overlap(t) for t in v_tags}
    s_norm = {_canonicalize_tag_for_overlap(t) for t in s_tags}
    return len(v_norm & s_norm)


def _language_preference_score(
    sub_lang: Optional[str], target_lang: Optional[str]
) -> float:
    """Score a subtitle's language tag against `target_lang`."""
    if not target_lang:
        return 0.0
    norm_target = normalize_language(target_lang)
    if sub_lang is None:
        # Untagged subtitle is acceptable but less preferred than an explicit match
        return 10.0
    norm_sub = normalize_language(sub_lang)
    if norm_sub == norm_target or sub_lang == target_lang:
        return 1000.0
    return -500.0


def _episodes_conflict(v_info: EpisodeInfo, s_info: EpisodeInfo) -> bool:
    """Return True if both files have episode numbers that conflict with each other."""
    if not v_info.has_episode or not s_info.has_episode:
        return False
    if (
        v_info.season is not None
        and s_info.season is not None
        and v_info.season != s_info.season
    ):
        return True
    v_eps = set(v_info.all_episodes)
    s_eps = set(s_info.all_episodes)
    return v_eps.isdisjoint(s_eps)


def match_files(
    videos: Sequence[Union[str, Path]],
    subtitles: Sequence[Union[str, Path]],
    mode: str = "auto",
    target_lang: Optional[str] = None,
    fuzzy_threshold: float = 0.6,
) -> MatchReport:
    """Match video files with subtitle files using a 3-tier matching strategy.

    Parameters
    ----------
    videos : sequence of Path or str
        Candidate video/audio reference files.
    subtitles : sequence of Path or str
        Candidate subtitle files.
    mode : {'auto', 'episode', 'name', 'stem', 'fuzzy'}, default 'auto'
        Matching strategy mode:
        - ``'auto'``: Runs Tier 1 (episode), Tier 2 (normalized stem), Tier 3 (fuzzy).
        - ``'episode'``: Runs Tier 1 (episode).
        - ``'name'`` or ``'stem'``: Runs Tier 2 (normalized exact stem).
        - ``'fuzzy'``: Runs Tier 1, Tier 2, and Tier 3 (fuzzy similarity).
    target_lang : str, optional
        Preferred subtitle language code (e.g. ``'fr'``, ``'fre'``, ``'en'``) for
        disambiguating multiple subtitle candidates for the same episode.
    fuzzy_threshold : float, default 0.6
        Minimum similarity ratio for Tier 3 fuzzy matching.

    Returns
    -------
    MatchReport
        Contains `paired`, `unmatched_videos`, and `unmatched_subtitles`.
    """
    video_paths = [Path(v) for v in videos]
    sub_paths = [Path(s) for s in subtitles]

    v_infos: Dict[int, EpisodeInfo] = {
        i: extract_episode_info(vp) for i, vp in enumerate(video_paths)
    }
    s_infos: Dict[int, EpisodeInfo] = {
        j: extract_episode_info(sp) for j, sp in enumerate(sub_paths)
    }

    matched_v_indices: Set[int] = set()
    matched_s_indices: Set[int] = set()
    paired_by_v_idx: Dict[int, MatchedPair] = {}

    mode_norm = (mode or "auto").strip().lower()

    # Determine whether only a single season exists across all videos/subtitles
    known_seasons = {
        info.season
        for info in list(v_infos.values()) + list(s_infos.values())
        if info.season is not None
    }
    single_season_context = len(known_seasons) <= 1

    # -------------------------------------------------------------------------
    # Tier 1: Canonical (season, episode) matching
    # -------------------------------------------------------------------------
    if mode_norm in ("auto", "episode", "fuzzy"):
        for v_idx, v_path in enumerate(video_paths):
            v_info = v_infos[v_idx]
            if not v_info.has_episode:
                continue

            v_eps = set(v_info.all_episodes)
            best_s_idx: Optional[int] = None
            best_score = float("-inf")
            best_similarity = 0.0

            for s_idx, s_path in enumerate(sub_paths):
                if s_idx in matched_s_indices:
                    continue
                s_info = s_infos[s_idx]
                if not s_info.has_episode:
                    continue

                # Check season compatibility
                if v_info.season is not None and s_info.season is not None:
                    if v_info.season != s_info.season:
                        continue
                    season_bonus = 20.0
                elif single_season_context:
                    season_bonus = 10.0
                else:
                    continue

                # Check episode overlap
                s_eps = set(s_info.all_episodes)
                if v_info.episode == s_info.episode:
                    ep_bonus = 50.0
                elif not v_eps.isdisjoint(s_eps):
                    ep_bonus = 30.0
                else:
                    continue

                # Check series title compatibility (prevents mixing different shows in same folder)
                compatible, title_sim = _series_titles_compatible(
                    v_info.series_title, s_info.series_title
                )
                if not compatible:
                    continue

                lang_score = _language_preference_score(s_info.language_tag, target_lang)
                tag_overlap = _compute_release_tag_overlap(
                    v_info.release_tags, s_info.release_tags
                )
                stem_sim = difflib.SequenceMatcher(
                    None, v_info.normalized_stem, s_info.normalized_stem
                ).ratio()

                total_score = (
                    lang_score
                    + (tag_overlap * 25.0)
                    + ep_bonus
                    + season_bonus
                    + (title_sim * 10.0)
                    + stem_sim
                )

                if total_score > best_score:
                    best_score = total_score
                    best_s_idx = s_idx
                    best_similarity = stem_sim

            if best_s_idx is not None:
                matched_v_indices.add(v_idx)
                matched_s_indices.add(best_s_idx)
                s_info = s_infos[best_s_idx]
                paired_by_v_idx[v_idx] = MatchedPair(
                    video=v_path,
                    subtitle=sub_paths[best_s_idx],
                    tier=1,
                    method="episode",
                    score=round(max(best_similarity, 1.0), 4),
                    season=v_info.season if v_info.season is not None else s_info.season,
                    episode=v_info.episode,
                    video_info=v_info,
                    subtitle_info=s_info,
                )

    # -------------------------------------------------------------------------
    # Tier 2: Normalized exact stem matching
    # -------------------------------------------------------------------------
    if mode_norm in ("auto", "name", "stem", "exact", "fuzzy"):
        for v_idx, v_path in enumerate(video_paths):
            if v_idx in matched_v_indices:
                continue
            v_info = v_infos[v_idx]
            if not v_info.normalized_stem:
                continue

            best_s_idx = None
            best_score = float("-inf")

            for s_idx, s_path in enumerate(sub_paths):
                if s_idx in matched_s_indices:
                    continue
                s_info = s_infos[s_idx]
                if _episodes_conflict(v_info, s_info):
                    continue

                if v_info.normalized_stem == s_info.normalized_stem:
                    lang_score = _language_preference_score(
                        s_info.language_tag, target_lang
                    )
                    tag_overlap = _compute_release_tag_overlap(
                        v_info.release_tags, s_info.release_tags
                    )
                    total_score = lang_score + (tag_overlap * 25.0)
                    if total_score > best_score:
                        best_score = total_score
                        best_s_idx = s_idx

            if best_s_idx is not None:
                matched_v_indices.add(v_idx)
                matched_s_indices.add(best_s_idx)
                s_info = s_infos[best_s_idx]
                paired_by_v_idx[v_idx] = MatchedPair(
                    video=v_path,
                    subtitle=sub_paths[best_s_idx],
                    tier=2,
                    method="stem",
                    score=1.0,
                    season=v_info.season,
                    episode=v_info.episode,
                    video_info=v_info,
                    subtitle_info=s_info,
                )

    # -------------------------------------------------------------------------
    # Tier 3: Fuzzy token similarity matching (difflib.SequenceMatcher)
    # -------------------------------------------------------------------------
    if mode_norm in ("auto", "fuzzy"):
        fuzzy_candidates: List[Tuple[float, float, int, int]] = []
        for v_idx in range(len(video_paths)):
            if v_idx in matched_v_indices:
                continue
            v_info = v_infos[v_idx]
            if not v_info.normalized_stem:
                continue

            for s_idx in range(len(sub_paths)):
                if s_idx in matched_s_indices:
                    continue
                s_info = s_infos[s_idx]
                if not s_info.normalized_stem:
                    continue
                if _episodes_conflict(v_info, s_info):
                    continue
                compatible, _ = _series_titles_compatible(
                    v_info.series_title, s_info.series_title
                )
                if not compatible and v_info.has_episode and s_info.has_episode:
                    continue

                seq_ratio = difflib.SequenceMatcher(
                    None, v_info.normalized_stem, s_info.normalized_stem
                ).ratio()
                v_tokens = set(v_info.normalized_stem.split())
                s_tokens = set(s_info.normalized_stem.split())
                union_len = len(v_tokens | s_tokens)
                token_ratio = (len(v_tokens & s_tokens) / union_len) if union_len else 0.0
                sim_ratio = max(seq_ratio, (seq_ratio + token_ratio) / 2.0)

                if sim_ratio >= fuzzy_threshold:
                    lang_score = _language_preference_score(
                        s_info.language_tag, target_lang
                    )
                    tag_overlap = _compute_release_tag_overlap(
                        v_info.release_tags, s_info.release_tags
                    )
                    rank_score = (sim_ratio * 100.0) + (lang_score * 0.1) + (tag_overlap * 5.0)
                    fuzzy_candidates.append((rank_score, sim_ratio, v_idx, s_idx))

        # Greedily assign highest-scoring fuzzy pairs
        fuzzy_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, sim_ratio, v_idx, s_idx in fuzzy_candidates:
            if v_idx in matched_v_indices or s_idx in matched_s_indices:
                continue
            matched_v_indices.add(v_idx)
            matched_s_indices.add(s_idx)
            v_info = v_infos[v_idx]
            s_info = s_infos[s_idx]
            paired_by_v_idx[v_idx] = MatchedPair(
                video=video_paths[v_idx],
                subtitle=sub_paths[s_idx],
                tier=3,
                method="fuzzy",
                score=round(sim_ratio, 4),
                season=v_info.season if v_info.season is not None else s_info.season,
                episode=v_info.episode if v_info.episode is not None else s_info.episode,
                video_info=v_info,
                subtitle_info=s_info,
            )

    paired = [paired_by_v_idx[i] for i in sorted(paired_by_v_idx.keys())]
    unmatched_videos = [
        vp for i, vp in enumerate(video_paths) if i not in matched_v_indices
    ]
    unmatched_subtitles = [
        sp for j, sp in enumerate(sub_paths) if j not in matched_s_indices
    ]

    return MatchReport(
        paired=paired,
        unmatched_videos=unmatched_videos,
        unmatched_subtitles=unmatched_subtitles,
    )
