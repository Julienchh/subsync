"""Base subtitle provider interface, candidate models, and language normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from subsync.download.normalizer import normalize_subtitle_bytes
from subsync.download.parser import SeriesEpisodeMetadata


class ProviderError(RuntimeError):
    """Base exception for subtitle provider errors."""


class ProviderNetworkError(ProviderError):
    """Raised when an online subtitle provider cannot be reached due to network/DNS/HTTP errors."""


class ProviderRateLimitError(ProviderError):
    """Raised when a subtitle provider enforces rate limiting or daily download quotas."""


class ProviderAuthError(ProviderError):
    """Raised when a subtitle provider requires an API key or valid credentials."""


@dataclass(frozen=True)
class NormalizedLanguage:
    """
    Canonical representation of a subtitle/audio language across ISO standards
    and provider-specific identifiers (Addic7ed, OpenSubtitles, Podnapisi).

    Supports both attribute access (`lang.iso639_1`) and dictionary-style access
    (`lang['iso639_1']`, `lang.get('iso639_1')`).
    """

    iso639_1: str
    iso639_2: str
    iso639_2b: str
    addic7ed_name: str
    addic7ed_id: int
    aliases: Tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.addic7ed_name

    @property
    def code2(self) -> str:
        return self.iso639_1

    @property
    def code3(self) -> str:
        return self.iso639_2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iso639_1": self.iso639_1,
            "iso639_2": self.iso639_2,
            "iso639_2b": self.iso639_2b,
            "addic7ed_name": self.addic7ed_name,
            "addic7ed_id": self.addic7ed_id,
            "name": self.addic7ed_name,
        }

    def __getitem__(self, key: str) -> Any:
        data = self.to_dict()
        if key in data:
            return data[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.to_dict()

    def keys(self) -> Sequence[str]:
        return list(self.to_dict().keys())

    def items(self) -> Sequence[Tuple[str, Any]]:
        return list(self.to_dict().items())

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())


_LANGUAGE_REGISTRY: List[NormalizedLanguage] = [
    NormalizedLanguage(
        iso639_1="fr",
        iso639_2="fra",
        iso639_2b="fre",
        addic7ed_name="French",
        addic7ed_id=8,
        aliases=("fr", "fra", "fre", "french", "français", "francais", "vostfr", "fr-fr", "fr_fr", "french (canada)", "french (canadian)"),
    ),
    NormalizedLanguage(
        iso639_1="en",
        iso639_2="eng",
        iso639_2b="eng",
        addic7ed_name="English",
        addic7ed_id=1,
        aliases=("en", "eng", "english", "en-us", "en_us", "en-gb", "en_gb", "anglais"),
    ),
    NormalizedLanguage(
        iso639_1="es",
        iso639_2="spa",
        iso639_2b="spa",
        addic7ed_name="Spanish",
        addic7ed_id=4,
        aliases=("es", "spa", "spanish", "español", "espanol", "espagnol", "castellano", "spanish (spain)", "spanish (latin america)"),
    ),
    NormalizedLanguage(
        iso639_1="de",
        iso639_2="deu",
        iso639_2b="ger",
        addic7ed_name="German",
        addic7ed_id=11,
        aliases=("de", "ger", "deu", "german", "deutsch", "allemand"),
    ),
    NormalizedLanguage(
        iso639_1="it",
        iso639_2="ita",
        iso639_2b="ita",
        addic7ed_name="Italian",
        addic7ed_id=7,
        aliases=("it", "ita", "italian", "italiano", "italien"),
    ),
    NormalizedLanguage(
        iso639_1="pt",
        iso639_2="por",
        iso639_2b="por",
        addic7ed_name="Portuguese",
        addic7ed_id=9,
        aliases=("pt", "por", "portuguese", "português", "portugues", "pt-pt"),
    ),
    NormalizedLanguage(
        iso639_1="pt-br",
        iso639_2="pob",
        iso639_2b="pob",
        addic7ed_name="Portuguese (Brazilian)",
        addic7ed_id=10,
        aliases=("pt-br", "pt_br", "pob", "pb", "brazilian", "portuguese (brazilian)", "portuguese-brazilian"),
    ),
    NormalizedLanguage(
        iso639_1="nl",
        iso639_2="nld",
        iso639_2b="dut",
        addic7ed_name="Dutch",
        addic7ed_id=17,
        aliases=("nl", "dut", "nld", "dutch", "nederlands"),
    ),
    NormalizedLanguage(
        iso639_1="pl",
        iso639_2="pol",
        iso639_2b="pol",
        addic7ed_name="Polish",
        addic7ed_id=23,
        aliases=("pl", "pol", "polish", "polski"),
    ),
    NormalizedLanguage(
        iso639_1="ru",
        iso639_2="rus",
        iso639_2b="rus",
        addic7ed_name="Russian",
        addic7ed_id=19,
        aliases=("ru", "rus", "russian", "русский"),
    ),
    NormalizedLanguage(
        iso639_1="ar",
        iso639_2="ara",
        iso639_2b="ara",
        addic7ed_name="Arabic",
        addic7ed_id=38,
        aliases=("ar", "ara", "arabic", "العربية"),
    ),
    NormalizedLanguage(
        iso639_1="zh",
        iso639_2="zho",
        iso639_2b="chi",
        addic7ed_name="Chinese (Simplified)",
        addic7ed_id=24,
        aliases=("zh", "chi", "zho", "chinese", "chs", "cht", "cmn", "zh-cn", "zh-tw"),
    ),
    NormalizedLanguage(
        iso639_1="ja",
        iso639_2="jpn",
        iso639_2b="jpn",
        addic7ed_name="Japanese",
        addic7ed_id=32,
        aliases=("ja", "jpn", "japanese", "jp"),
    ),
    NormalizedLanguage(
        iso639_1="ko",
        iso639_2="kor",
        iso639_2b="kor",
        addic7ed_name="Korean",
        addic7ed_id=42,
        aliases=("ko", "kor", "korean", "kr"),
    ),
    NormalizedLanguage(
        iso639_1="tr",
        iso639_2="tur",
        iso639_2b="tur",
        addic7ed_name="Turkish",
        addic7ed_id=16,
        aliases=("tr", "tur", "turkish", "türkçe", "turkce"),
    ),
    NormalizedLanguage(
        iso639_1="sv",
        iso639_2="swe",
        iso639_2b="swe",
        addic7ed_name="Swedish",
        addic7ed_id=18,
        aliases=("sv", "swe", "swedish", "svenska"),
    ),
    NormalizedLanguage(
        iso639_1="el",
        iso639_2="ell",
        iso639_2b="gre",
        addic7ed_name="Greek",
        addic7ed_id=27,
        aliases=("el", "gre", "ell", "greek"),
    ),
    NormalizedLanguage(
        iso639_1="ro",
        iso639_2="ron",
        iso639_2b="rum",
        addic7ed_name="Romanian",
        addic7ed_id=26,
        aliases=("ro", "rum", "ron", "romanian"),
    ),
    NormalizedLanguage(
        iso639_1="cs",
        iso639_2="ces",
        iso639_2b="cze",
        addic7ed_name="Czech",
        addic7ed_id=14,
        aliases=("cs", "cze", "ces", "czech"),
    ),
    NormalizedLanguage(
        iso639_1="hu",
        iso639_2="hun",
        iso639_2b="hun",
        addic7ed_name="Hungarian",
        addic7ed_id=20,
        aliases=("hu", "hun", "hungarian", "magyar"),
    ),
    NormalizedLanguage(
        iso639_1="he",
        iso639_2="heb",
        iso639_2b="heb",
        addic7ed_name="Hebrew",
        addic7ed_id=22,
        aliases=("he", "heb", "hebrew"),
    ),
    NormalizedLanguage(
        iso639_1="da",
        iso639_2="dan",
        iso639_2b="dan",
        addic7ed_name="Danish",
        addic7ed_id=29,
        aliases=("da", "dan", "danish"),
    ),
    NormalizedLanguage(
        iso639_1="fi",
        iso639_2="fin",
        iso639_2b="fin",
        addic7ed_name="Finnish",
        addic7ed_id=28,
        aliases=("fi", "fin", "finnish"),
    ),
    NormalizedLanguage(
        iso639_1="no",
        iso639_2="nor",
        iso639_2b="nor",
        addic7ed_name="Norwegian",
        addic7ed_id=30,
        aliases=("no", "nor", "nob", "nno", "norwegian"),
    ),
]

_ALIAS_TO_LANGUAGE: Dict[str, NormalizedLanguage] = {}
for _entry in _LANGUAGE_REGISTRY:
    _ALIAS_TO_LANGUAGE[_entry.iso639_1.lower()] = _entry
    _ALIAS_TO_LANGUAGE[_entry.iso639_2.lower()] = _entry
    _ALIAS_TO_LANGUAGE[_entry.iso639_2b.lower()] = _entry
    _ALIAS_TO_LANGUAGE[_entry.addic7ed_name.lower()] = _entry
    for _alias in _entry.aliases:
        _ALIAS_TO_LANGUAGE[_alias.lower()] = _entry


def normalize_lang(lang: str) -> NormalizedLanguage:
    """
    Normalize a language code or name (e.g., 'fr', 'fra', 'fre', 'french', 'français')
    into a canonical `NormalizedLanguage` descriptor containing `iso639_1`, `iso639_2`,
    `addic7ed_name`, and `addic7ed_id`.
    """
    if not lang or not isinstance(lang, str):
        raise ValueError("Language code must be a non-empty string")

    cleaned = lang.strip().lower()
    if cleaned in _ALIAS_TO_LANGUAGE:
        return _ALIAS_TO_LANGUAGE[cleaned]

    # Strip parenthetical region qualifiers such as 'French (Canadian)' -> 'french'
    base_cleaned = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]\s*", "", cleaned).strip()
    if base_cleaned in _ALIAS_TO_LANGUAGE:
        return _ALIAS_TO_LANGUAGE[base_cleaned]

    # Handle hyphen/underscore region codes such as 'fr-CA' or 'en_AU'
    if "-" in base_cleaned or "_" in base_cleaned:
        prefix = re.split(r"[-_]", base_cleaned)[0]
        if prefix in _ALIAS_TO_LANGUAGE:
            return _ALIAS_TO_LANGUAGE[prefix]

    # Fallback custom NormalizedLanguage for uncommon codes
    code2 = base_cleaned[:2] if len(base_cleaned) >= 2 else base_cleaned
    code3 = base_cleaned[:3] if len(base_cleaned) >= 3 else base_cleaned
    return NormalizedLanguage(
        iso639_1=code2,
        iso639_2=code3,
        iso639_2b=code3,
        addic7ed_name=lang.strip().title(),
        addic7ed_id=0,
        aliases=(cleaned,),
    )


def matches_language(candidate_lang: str, target_lang: str) -> bool:
    """Return True if `candidate_lang` represents the same language as `target_lang`."""
    if not candidate_lang or not target_lang:
        return False
    norm_cand = normalize_lang(candidate_lang)
    norm_target = normalize_lang(target_lang)
    if norm_cand.iso639_1 == norm_target.iso639_1 or norm_cand.iso639_2 == norm_target.iso639_2:
        return True
    if norm_target.addic7ed_name.lower() in candidate_lang.strip().lower():
        return True
    return False


@dataclass
class SubtitleCandidate:
    """Represents a subtitle release candidate discovered on an online provider."""

    provider_name: str
    series_name: str
    season: int
    episode: int
    language: str
    release_version: str
    download_url: str
    referer: Optional[str] = None
    hearing_impaired: bool = False
    completed: bool = True
    score: float = 0.0
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate metadata to a JSON-compatible dictionary."""
        return {
            "provider_name": self.provider_name,
            "series_name": self.series_name,
            "season": self.season,
            "episode": self.episode,
            "language": self.language,
            "release_version": self.release_version,
            "download_url": self.download_url,
            "referer": self.referer,
            "hearing_impaired": self.hearing_impaired,
            "completed": self.completed,
            "score": round(float(self.score), 3),
            "extra_metadata": dict(self.extra_metadata),
        }


# Equivalence classes for release tag matching (e.g. HDTV/WEB/BluRay families)
_SOURCE_EQUIVALENTS = [
    {"hdtv", "pdtv", "lol", "dimension", "sva", "avs", "killers", "fleet", "fov", "immerse", "batv"},
    {"web", "web-dl", "webdl", "webrip", "amzn", "nf", "ntb", "flux", "ion10", "joy"},
    {"bluray", "blu-ray", "bdrip", "brrip", "remux", "demand", "rov"},
]


def score_subtitle_candidate(
    meta: SeriesEpisodeMetadata,
    release_version: str,
    completed: bool = True,
    hearing_impaired: bool = False,
    prefer_hearing_impaired: bool = False,
    is_updated: bool = True,
    download_count: int = 0,
) -> float:
    """
    Compute a deterministic ranking score for a subtitle candidate against `meta`.
    Prioritizes:
      1. Completed status (+50.0 vs +10.0)
      2. Hearing-impaired preference match (+15.0 vs -5.0)
      3. Updated release over original (+10.0)
      4. Release tag overlap and source family similarity (+8.0 per direct match, +3.0 per family match)
      5. Download popularity tie-breaker (up to +5.0)
    """
    score = 50.0 if completed else 10.0

    if hearing_impaired == prefer_hearing_impaired:
        score += 15.0
    else:
        score -= 5.0

    if is_updated:
        score += 10.0

    video_tokens = {
        tok.lower().strip(".-_[]()")
        for tok in (meta.release_tags or [])
        if tok.strip(".-_[]()")
    }
    version_tokens = {
        tok.lower().strip(".-_[]()")
        for tok in re.split(r"[\s\.,_\-\+/;\(\)\[\]]+", release_version or "")
        if tok.strip(".-_[]()")
    }

    if video_tokens and version_tokens:
        direct_matches = video_tokens & version_tokens
        score += 8.0 * len(direct_matches)

        for group in _SOURCE_EQUIVALENTS:
            if (video_tokens & group) and (version_tokens & group):
                score += 3.0

    if download_count > 0:
        score += min(5.0, float(download_count) / 1000.0)

    return round(score, 3)


class SubtitleProvider(ABC):
    """Abstract base interface for subtitle providers."""

    name: str = "base"

    @abstractmethod
    def search_subtitles(
        self,
        meta: SeriesEpisodeMetadata,
        target_lang: str,
    ) -> List[SubtitleCandidate]:
        """
        Query the provider for subtitles matching `meta` (series, season, episode)
        and `target_lang`. Returns candidates sorted by descending `score`.
        """

    @abstractmethod
    def download_subtitle(self, candidate: SubtitleCandidate) -> bytes:
        """
        Download raw subtitle bytes for the given `candidate`.
        """

    def download_and_normalize(
        self,
        candidate: SubtitleCandidate,
        declared_charset: Optional[str] = None,
    ) -> str:
        """
        Download the subtitle for `candidate` and normalize it into validated UTF-8 SRT text.
        """
        raw_bytes = self.download_subtitle(candidate)
        charset = declared_charset or candidate.extra_metadata.get("charset")
        return normalize_subtitle_bytes(raw_bytes, declared_charset=charset)
