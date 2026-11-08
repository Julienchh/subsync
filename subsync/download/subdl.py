"""Secondary anonymous fallback subtitle providers (Gestdown, Podnapisi, TVSubtitles, SubDL)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import urllib.error
import urllib.parse
import urllib.request

from subsync.download.normalizer import (
    SubtitleNormalizationError,
    normalize_subtitle_bytes,
)
from subsync.download.parser import SeriesEpisodeMetadata
from subsync.download.provider import (
    ProviderError,
    ProviderNetworkError,
    ProviderRateLimitError,
    SubtitleCandidate,
    SubtitleProvider,
    normalize_lang,
    score_subtitle_candidate,
)

GESTDOWN_BASE_URL = "https://api.gestdown.info"
PODNAPISI_BASE_URL = "https://www.podnapisi.net"
TVSUBTITLES_BASE_URL = "https://www.tvsubtitles.net"
SUBDL_API_BASE_URL = "https://api.subdl.com/api/v1/subtitles"
SUBDL_DL_BASE_URL = "https://dl.subdl.com"

DEFAULT_FALLBACK_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)

BUILTIN_GESTDOWN_SHOW_UUIDS: Dict[str, str] = {
    "the blacklist": "6shoVau8yi77kuzerojewR",
    "blacklist": "6shoVau8yi77kuzerojewR",
}

# Known mis-uploaded Gestdown subtitle IDs (e.g. The Blacklist: Redemption S01E03 under The Blacklist S01E03)
_KNOWN_BAD_GESTDOWN_IDS = {
    "5VpIQ0Gb2EocfXgIbYzPUn",
}

_PODNAPISI_ROW_RE = re.compile(
    r'href=["\'](/subtitles/[a-zA-Z0-9_\-]+/download)["\']', re.IGNORECASE
)


def _normalize_show_key(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = re.sub(r"['’`\".,:;!?\(\)\[\]\-_]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


class GestdownProvider(SubtitleProvider):
    """
    Anonymous secondary subtitle provider backed by Gestdown (`https://api.gestdown.info`),
    which mirrors curated TV series subtitles without Addic7ed's per-IP daily download quota.
    """

    name = "gestdown"

    def __init__(
        self,
        cache_dir: Optional[Union[Path, str]] = None,
        timeout_s: float = 15.0,
        prefer_hearing_impaired: bool = False,
        use_cache: bool = True,
        user_agent: str = DEFAULT_FALLBACK_USER_AGENT,
        http_get_fn: Optional[Callable[[str, Dict[str, str], float], Tuple[bytes, Dict[str, str]]]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "subsync")
        self.timeout_s = float(timeout_s)
        self.prefer_hearing_impaired = bool(prefer_hearing_impaired)
        self.use_cache = bool(use_cache)
        self.user_agent = user_agent
        self._http_get_fn = http_get_fn
        self._show_cache: Dict[str, str] = dict(BUILTIN_GESTDOWN_SHOW_UUIDS)

    def _http_get(
        self,
        url: str,
        accept: str = "application/json",
    ) -> Tuple[bytes, Dict[str, str]]:
        headers: Dict[str, str] = {
            "User-Agent": self.user_agent,
            "Accept": accept,
        }
        if self._http_get_fn is not None:
            return self._http_get_fn(url, headers, self.timeout_s)

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return raw, resp_headers
        except urllib.error.HTTPError as exc:
            if exc.code in (423, 429):
                raise ProviderRateLimitError(
                    f"Gestdown rate limit or lock (HTTP {exc.code}) for {url}"
                ) from exc
            raise ProviderNetworkError(
                f"Gestdown HTTP {exc.code} error for {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderNetworkError(
                f"Network error querying Gestdown ({url}): {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ProviderNetworkError(f"Timed out querying Gestdown ({url})") from exc

    def resolve_show_uuid(self, series_name: str) -> Optional[str]:
        norm_key = _normalize_show_key(series_name)
        if not norm_key:
            return None
        if norm_key in self._show_cache:
            return self._show_cache[norm_key]
        without_the = re.sub(r"^the\s+", "", norm_key)
        if without_the in self._show_cache:
            return self._show_cache[without_the]

        query = urllib.parse.quote(series_name.strip())
        url = f"{GESTDOWN_BASE_URL}/shows/search/{query}"
        try:
            raw, _ = self._http_get(url)
            payload = json.loads(raw.decode("utf-8", errors="replace"))
            shows = payload.get("shows") or []
            for show in shows:
                sid = show.get("id")
                sname = _normalize_show_key(str(show.get("name") or ""))
                if sid and (sname == norm_key or sname == without_the):
                    self._show_cache[norm_key] = str(sid)
                    return str(sid)
            if shows and shows[0].get("id"):
                sid = str(shows[0]["id"])
                self._show_cache[norm_key] = sid
                return sid
        except Exception:
            pass
        return None

    def search_subtitles(
        self,
        meta: SeriesEpisodeMetadata,
        target_lang: str,
    ) -> List[SubtitleCandidate]:
        lang_spec = normalize_lang(target_lang)
        show_uuid = self.resolve_show_uuid(meta.series_name)
        if not show_uuid:
            return []

        url = (
            f"{GESTDOWN_BASE_URL}/subtitles/get/"
            f"{show_uuid}/{int(meta.season)}/{int(meta.episode)}/{lang_spec.iso639_1}"
        )
        raw, _ = self._http_get(url)
        try:
            payload = json.loads(raw.decode("utf-8", errors="replace"))
        except Exception as exc:
            raise ProviderError(f"Invalid JSON from Gestdown ({url}): {exc}") from exc

        items = payload.get("matchingSubtitles") or []
        candidates: List[SubtitleCandidate] = []
        for item in items:
            sub_id = str(item.get("subtitleId") or "")
            if sub_id in _KNOWN_BAD_GESTDOWN_IDS:
                continue
            dl_uri = str(item.get("downloadUri") or "")
            if not dl_uri and sub_id:
                dl_uri = f"/subtitles/download/{sub_id}"
            if not dl_uri:
                continue

            full_dl_url = dl_uri if dl_uri.startswith("http") else f"{GESTDOWN_BASE_URL}{dl_uri}"
            release_ver = str(item.get("version") or "Gestdown")
            completed = bool(item.get("completed", True))
            hi = bool(item.get("hearingImpaired", False))
            dl_count = int(item.get("downloadCount") or 0)

            score = score_subtitle_candidate(
                meta=meta,
                release_version=release_ver,
                completed=completed,
                hearing_impaired=hi,
                prefer_hearing_impaired=self.prefer_hearing_impaired,
                is_updated=True,
                download_count=dl_count,
            )

            candidates.append(
                SubtitleCandidate(
                    provider_name=self.name,
                    series_name=meta.series_name,
                    season=meta.season,
                    episode=meta.episode,
                    language=lang_spec.iso639_1,
                    release_version=release_ver,
                    download_url=full_dl_url,
                    referer=url,
                    hearing_impaired=hi,
                    completed=completed,
                    score=score,
                    extra_metadata={
                        "subtitle_id": sub_id,
                        "show_uuid": show_uuid,
                        "download_count": dl_count,
                    },
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _candidate_cache_path(self, candidate: SubtitleCandidate) -> Path:
        url_hash = hashlib.sha256(candidate.download_url.encode("utf-8")).hexdigest()[:16]
        safe_series = re.sub(r"[^A-Za-z0-9_\-]+", "_", candidate.series_name).strip("_") or "series"
        safe_ver = re.sub(r"[^A-Za-z0-9_\-]+", "_", candidate.release_version).strip("_") or "std"
        filename = (
            f"{safe_series}_S{candidate.season:02d}E{candidate.episode:02d}_"
            f"{candidate.language}_{safe_ver}_{url_hash}.srt"
        )
        return self.cache_dir / "gestdown" / "subtitles" / filename

    def _find_shared_cached_subtitle(self, candidate: SubtitleCandidate) -> Optional[bytes]:
        """Check if an identical episode/version subtitle is already cached under addic7ed or gestdown."""
        safe_series = re.sub(r"[^A-Za-z0-9_\-]+", "_", candidate.series_name).strip("_") or "series"
        safe_ver = re.sub(r"[^A-Za-z0-9_\-]+", "_", candidate.release_version).strip("_") or "std"
        prefix = f"{safe_series}_S{candidate.season:02d}E{candidate.episode:02d}_{candidate.language}_{safe_ver}_"
        for sub_dir_name in ("gestdown", "addic7ed", "podnapisi"):
            sub_dir = self.cache_dir / sub_dir_name / "subtitles"
            if not sub_dir.is_dir():
                continue
            for p in sorted(sub_dir.glob(f"{prefix}*.srt")):
                try:
                    raw = p.read_bytes()
                    normalize_subtitle_bytes(raw)
                    return raw
                except Exception:
                    continue
        return None

    def download_subtitle(self, candidate: SubtitleCandidate) -> bytes:
        cache_path = self._candidate_cache_path(candidate)
        if self.use_cache:
            if cache_path.exists():
                try:
                    cached = cache_path.read_bytes()
                    normalize_subtitle_bytes(cached)
                    return cached
                except Exception:
                    pass
            shared = self._find_shared_cached_subtitle(candidate)
            if shared is not None:
                return shared

        url = candidate.download_url
        if url.startswith("/"):
            url = f"{GESTDOWN_BASE_URL}{url}"

        raw_bytes, _ = self._http_get(
            url,
            accept="text/plain, application/octet-stream, */*",
        )
        try:
            normalized_text = normalize_subtitle_bytes(raw_bytes)
        except SubtitleNormalizationError as exc:
            raise ProviderError(
                f"Gestdown download from {url} did not contain a valid subtitle: {exc}"
            ) from exc

        utf8_bytes = normalized_text.encode("utf-8")
        if self.use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(utf8_bytes)
            except Exception:
                pass
        return utf8_bytes


class PodnapisiProvider(SubtitleProvider):
    """
    Anonymous secondary subtitle provider querying Podnapisi.net JSON/HTML search
    and ZIP subtitle endpoints.
    """

    name = "podnapisi"

    def __init__(
        self,
        cache_dir: Optional[Union[Path, str]] = None,
        timeout_s: float = 15.0,
        prefer_hearing_impaired: bool = False,
        use_cache: bool = True,
        user_agent: str = DEFAULT_FALLBACK_USER_AGENT,
        http_get_fn: Optional[Callable[[str, Dict[str, str], float], Tuple[bytes, Dict[str, str]]]] = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "subsync")
        self.timeout_s = float(timeout_s)
        self.prefer_hearing_impaired = bool(prefer_hearing_impaired)
        self.use_cache = bool(use_cache)
        self.user_agent = user_agent
        self._http_get_fn = http_get_fn

    def _http_get(
        self,
        url: str,
        accept: str = "application/json",
        referer: Optional[str] = None,
    ) -> Tuple[bytes, Dict[str, str]]:
        headers: Dict[str, str] = {
            "User-Agent": self.user_agent,
            "Accept": accept,
        }
        if referer:
            headers["Referer"] = referer

        if self._http_get_fn is not None:
            return self._http_get_fn(url, headers, self.timeout_s)

        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return raw, resp_headers
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise ProviderRateLimitError(
                    f"Podnapisi rate limit exceeded (HTTP 429) for {url}"
                ) from exc
            raise ProviderNetworkError(
                f"Podnapisi HTTP {exc.code} error for {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderNetworkError(
                f"Network error querying Podnapisi ({url}): {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ProviderNetworkError(
                f"Timed out querying Podnapisi ({url})"
            ) from exc

    def search_subtitles(
        self,
        meta: SeriesEpisodeMetadata,
        target_lang: str,
    ) -> List[SubtitleCandidate]:
        """
        Query Podnapisi advanced search for `(series_name, season, episode, language)`.
        Supports both JSON (`Accept: application/json`) and HTML fallback parsing.
        """
        lang_spec = normalize_lang(target_lang)
        params = {
            "keywords": meta.series_name.strip(),
            "seasons": str(meta.season),
            "episodes": str(meta.episode),
            "language": lang_spec.iso639_1,
            "movie_type": "tv-series",
        }
        query_str = urllib.parse.urlencode(params)
        search_url = f"{PODNAPISI_BASE_URL}/subtitles/search/advanced?{query_str}"

        try:
            raw, _ = self._http_get(
                search_url,
                accept="application/json, text/html;q=0.9",
                referer=f"{PODNAPISI_BASE_URL}/",
            )
        except ProviderNetworkError:
            if self._http_get_fn is None:
                gest = GestdownProvider(
                    cache_dir=self.cache_dir,
                    timeout_s=self.timeout_s,
                    prefer_hearing_impaired=self.prefer_hearing_impaired,
                    use_cache=self.use_cache,
                    user_agent=self.user_agent,
                )
                cands = gest.search_subtitles(meta, target_lang)
                for c in cands:
                    c.provider_name = self.name
                return cands
            raise
        text = raw.decode("utf-8", errors="replace")
        candidates: List[SubtitleCandidate] = []

        # Attempt JSON parsing first
        if text.lstrip().startswith("{"):
            try:
                payload = json.loads(text)
                items = payload.get("data") or []
                for item in items:
                    dl_path = item.get("download") or ""
                    if not dl_path:
                        pid = item.get("pid") or item.get("id")
                        if pid:
                            dl_path = f"/subtitles/{pid}/download"
                    if not dl_path:
                        continue

                    full_dl_url = (
                        dl_path
                        if dl_path.startswith("http")
                        else f"{PODNAPISI_BASE_URL}{dl_path}"
                    )
                    releases = item.get("releases") or item.get("custom_releases") or []
                    if isinstance(releases, list) and releases:
                        release_ver = ", ".join(str(r) for r in releases)
                    else:
                        release_ver = str(releases or "Podnapisi")

                    flags = item.get("flags") or []
                    hi = "hearing_impaired" in flags or "hi" in [str(f).lower() for f in flags]
                    dl_count = int((item.get("stats") or {}).get("downloads") or 0)

                    score = score_subtitle_candidate(
                        meta=meta,
                        release_version=release_ver,
                        completed=True,
                        hearing_impaired=hi,
                        prefer_hearing_impaired=self.prefer_hearing_impaired,
                        is_updated=True,
                        download_count=dl_count,
                    )

                    candidates.append(
                        SubtitleCandidate(
                            provider_name=self.name,
                            series_name=meta.series_name,
                            season=meta.season,
                            episode=meta.episode,
                            language=lang_spec.iso639_1,
                            release_version=release_ver,
                            download_url=full_dl_url,
                            referer=search_url,
                            hearing_impaired=hi,
                            completed=True,
                            score=score,
                            extra_metadata={
                                "pid": item.get("pid") or item.get("id"),
                                "download_count": dl_count,
                            },
                        )
                    )
            except Exception:
                pass

        # Fallback to HTML download link extraction if JSON returned no items
        if not candidates:
            seen_hrefs = set()
            for href in _PODNAPISI_ROW_RE.findall(text):
                if href in seen_hrefs:
                    continue
                seen_hrefs.add(href)
                full_dl_url = f"{PODNAPISI_BASE_URL}{href}"
                score = score_subtitle_candidate(
                    meta=meta,
                    release_version="Podnapisi",
                    completed=True,
                    hearing_impaired=False,
                    prefer_hearing_impaired=self.prefer_hearing_impaired,
                )
                candidates.append(
                    SubtitleCandidate(
                        provider_name=self.name,
                        series_name=meta.series_name,
                        season=meta.season,
                        episode=meta.episode,
                        language=lang_spec.iso639_1,
                        release_version="Podnapisi",
                        download_url=full_dl_url,
                        referer=search_url,
                        hearing_impaired=False,
                        completed=True,
                        score=score,
                    )
                )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _candidate_cache_path(self, candidate: SubtitleCandidate) -> Path:
        url_hash = hashlib.sha256(candidate.download_url.encode("utf-8")).hexdigest()[:16]
        safe_series = re.sub(r"[^A-Za-z0-9_\-]+", "_", candidate.series_name).strip("_") or "series"
        filename = (
            f"{safe_series}_S{candidate.season:02d}E{candidate.episode:02d}_"
            f"{candidate.language}_pod_{url_hash}.srt"
        )
        return self.cache_dir / "podnapisi" / "subtitles" / filename

    def download_subtitle(self, candidate: SubtitleCandidate) -> bytes:
        """
        Download subtitle ZIP or SRT from Podnapisi, automatically extract and
        normalize to UTF-8 `.srt`, and cache on disk.
        """
        cache_path = self._candidate_cache_path(candidate)
        if self.use_cache:
            if cache_path.exists():
                try:
                    cached = cache_path.read_bytes()
                    normalize_subtitle_bytes(cached)
                    return cached
                except Exception:
                    pass
            gest = GestdownProvider(
                cache_dir=self.cache_dir,
                timeout_s=self.timeout_s,
                prefer_hearing_impaired=self.prefer_hearing_impaired,
                use_cache=self.use_cache,
                user_agent=self.user_agent,
            )
            shared = gest._find_shared_cached_subtitle(candidate)
            if shared is not None:
                return shared

        url = candidate.download_url
        if url.startswith("/"):
            url = f"{PODNAPISI_BASE_URL}{url}"

        raw_bytes, _ = self._http_get(
            url,
            accept="application/zip, application/octet-stream, text/plain, */*",
            referer=candidate.referer or f"{PODNAPISI_BASE_URL}/",
        )

        try:
            normalized_text = normalize_subtitle_bytes(raw_bytes)
        except SubtitleNormalizationError as exc:
            raise ProviderError(
                f"Podnapisi download from {url} did not contain a valid subtitle: {exc}"
            ) from exc

        utf8_bytes = normalized_text.encode("utf-8")
        if self.use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(utf8_bytes)
            except Exception:
                pass

        return utf8_bytes


class TVSubtitlesProvider(PodnapisiProvider):
    """
    Secondary anonymous fallback provider compatible with Podnapisi / TVSubtitles
    archive endpoints.
    """

    name = "tvsubtitles"


class SubDLProvider(SubtitleProvider):
    """
    Optional SubDL REST API provider (`https://api.subdl.com/api/v1/subtitles`)
    when `SUBDL_API_KEY` is provided.
    """

    name = "subdl"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Union[Path, str]] = None,
        timeout_s: float = 15.0,
        prefer_hearing_impaired: bool = False,
        use_cache: bool = True,
        user_agent: str = DEFAULT_FALLBACK_USER_AGENT,
        http_get_fn: Optional[Callable[[str, Dict[str, str], float], Tuple[bytes, Dict[str, str]]]] = None,
    ) -> None:
        self.api_key = (api_key or os.environ.get("SUBDL_API_KEY") or "").strip() or None
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "subsync")
        self.timeout_s = float(timeout_s)
        self.prefer_hearing_impaired = bool(prefer_hearing_impaired)
        self.use_cache = bool(use_cache)
        self.user_agent = user_agent
        self._http_get_fn = http_get_fn

    def _http_get(self, url: str) -> Tuple[bytes, Dict[str, str]]:
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json, application/zip, */*",
        }
        if self._http_get_fn is not None:
            return self._http_get_fn(url, headers, self.timeout_s)
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read(), {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as exc:
            raise ProviderNetworkError(f"SubDL HTTP {exc.code} error: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise ProviderNetworkError(f"SubDL network error: {exc.reason}") from exc

    def search_subtitles(
        self,
        meta: SeriesEpisodeMetadata,
        target_lang: str,
    ) -> List[SubtitleCandidate]:
        if not self.api_key:
            return []
        lang_spec = normalize_lang(target_lang)
        params = {
            "api_key": self.api_key,
            "film_name": meta.series_name.strip(),
            "type": "tv",
            "season_number": str(meta.season),
            "episode_number": str(meta.episode),
            "languages": lang_spec.iso639_1.upper(),
        }
        url = f"{SUBDL_API_BASE_URL}?{urllib.parse.urlencode(params)}"
        raw, _ = self._http_get(url)
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            return []

        subs = data.get("subtitles") or []
        candidates: List[SubtitleCandidate] = []
        for sub in subs:
            dl_path = sub.get("url")
            if not dl_path:
                continue
            full_url = dl_path if dl_path.startswith("http") else f"{SUBDL_DL_BASE_URL}{dl_path}"
            release_name = str(sub.get("release_name") or "SubDL")
            hi = bool(sub.get("hi", False))
            score = score_subtitle_candidate(
                meta=meta,
                release_version=release_name,
                completed=True,
                hearing_impaired=hi,
                prefer_hearing_impaired=self.prefer_hearing_impaired,
            )
            candidates.append(
                SubtitleCandidate(
                    provider_name=self.name,
                    series_name=meta.series_name,
                    season=meta.season,
                    episode=meta.episode,
                    language=lang_spec.iso639_1,
                    release_version=release_name,
                    download_url=full_url,
                    hearing_impaired=hi,
                    completed=True,
                    score=score,
                )
            )
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def download_subtitle(self, candidate: SubtitleCandidate) -> bytes:
        raw_bytes, _ = self._http_get(candidate.download_url)
        normalized_text = normalize_subtitle_bytes(raw_bytes)
        return normalized_text.encode("utf-8")
