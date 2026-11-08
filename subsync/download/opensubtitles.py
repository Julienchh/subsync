"""OpenSubtitles.com REST API v2 subtitle provider."""

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
    ProviderAuthError,
    ProviderError,
    ProviderNetworkError,
    ProviderRateLimitError,
    SubtitleCandidate,
    SubtitleProvider,
    normalize_lang,
    score_subtitle_candidate,
)

OPENSUBTITLES_API_BASE = "https://api.opensubtitles.com/api/v1"
DEFAULT_OS_USER_AGENT = "Subsync v0.18"


class OpenSubtitlesProvider(SubtitleProvider):
    """
    Subtitle provider for the official OpenSubtitles.com REST API v2.

    Requires an API key passed via `api_key` parameter, CLI `--opensubtitles-api-key`,
    or `OPENSUBTITLES_API_KEY` environment variable.
    """

    name = "opensubtitles"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Union[Path, str]] = None,
        timeout_s: float = 15.0,
        prefer_hearing_impaired: bool = False,
        use_cache: bool = True,
        user_agent: str = DEFAULT_OS_USER_AGENT,
        http_request_fn: Optional[
            Callable[[str, str, Dict[str, str], Optional[bytes], float], Tuple[bytes, Dict[str, str]]]
        ] = None,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("OPENSUBTITLES_API_KEY")
            or os.environ.get("SUBSYNC_OPENSUBTITLES_API_KEY")
            or ""
        ).strip()
        self.api_key: Optional[str] = resolved_key if resolved_key else None
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "subsync")
        self.timeout_s = float(timeout_s)
        self.prefer_hearing_impaired = bool(prefer_hearing_impaired)
        self.use_cache = bool(use_cache)
        self.user_agent = user_agent
        self._http_request_fn = http_request_fn

    @property
    def is_configured(self) -> bool:
        """Return True if an OpenSubtitles API key is available."""
        return bool(self.api_key)

    def _request(
        self,
        method: str,
        url: str,
        payload: Optional[Dict[str, Any]] = None,
        require_auth: bool = True,
    ) -> Tuple[bytes, Dict[str, str]]:
        if require_auth and not self.api_key:
            raise ProviderAuthError(
                "OpenSubtitles REST API v2 requires an API key "
                "(--opensubtitles-api-key or OPENSUBTITLES_API_KEY env var)."
            )

        headers: Dict[str, str] = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Api-Key"] = self.api_key

        body_bytes: Optional[bytes] = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body_bytes = json.dumps(payload).encode("utf-8")

        if self._http_request_fn is not None:
            return self._http_request_fn(method, url, headers, body_bytes, self.timeout_s)

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return raw, resp_headers
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise ProviderAuthError(
                    f"OpenSubtitles authentication rejected (HTTP {exc.code}): valid Api-Key required"
                ) from exc
            if exc.code == 429:
                raise ProviderRateLimitError(
                    "OpenSubtitles API rate limit exceeded (HTTP 429)"
                ) from exc
            raise ProviderNetworkError(
                f"OpenSubtitles HTTP {exc.code} error for {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderNetworkError(
                f"Network error querying OpenSubtitles ({url}): {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ProviderNetworkError(
                f"Timed out querying OpenSubtitles ({url})"
            ) from exc

    def search_subtitles(
        self,
        meta: SeriesEpisodeMetadata,
        target_lang: str,
    ) -> List[SubtitleCandidate]:
        """Search OpenSubtitles REST API v2 for episode subtitles."""
        if not self.api_key:
            raise ProviderAuthError(
                "OpenSubtitles REST API v2 requires an API key "
                "(--opensubtitles-api-key or OPENSUBTITLES_API_KEY env var)."
            )

        lang_spec = normalize_lang(target_lang)
        params = {
            "query": meta.series_name.strip().lower(),
            "season_number": str(meta.season),
            "episode_number": str(meta.episode),
            "languages": lang_spec.iso639_1,
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{OPENSUBTITLES_API_BASE}/subtitles?{query_string}"

        raw, _ = self._request("GET", url, require_auth=True)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"Invalid JSON response from OpenSubtitles: {exc}") from exc

        items = payload.get("data") or []
        candidates: List[SubtitleCandidate] = []

        for item in items:
            attrs = item.get("attributes") or {}
            files = attrs.get("files") or []
            if not files:
                continue

            file_info = files[0] or {}
            file_id = file_info.get("file_id")
            if file_id is None:
                continue

            release = (
                attrs.get("release")
                or file_info.get("file_name")
                or "OpenSubtitles"
            )
            hi = bool(attrs.get("hearing_impaired", False))
            dl_count = int(attrs.get("download_count") or 0)
            direct_link = attrs.get("download_url") or ""

            score = score_subtitle_candidate(
                meta=meta,
                release_version=str(release),
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
                    release_version=str(release),
                    download_url=direct_link or f"{OPENSUBTITLES_API_BASE}/download#{file_id}",
                    referer=None,
                    hearing_impaired=hi,
                    completed=True,
                    score=score,
                    extra_metadata={
                        "file_id": int(file_id),
                        "file_name": file_info.get("file_name"),
                        "download_count": dl_count,
                        "ratings": attrs.get("ratings"),
                        "direct_link": direct_link,
                    },
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _candidate_cache_path(self, candidate: SubtitleCandidate) -> Path:
        file_id = candidate.extra_metadata.get("file_id") or hashlib.sha256(
            candidate.download_url.encode("utf-8")
        ).hexdigest()[:12]
        safe_series = re.sub(r"[^A-Za-z0-9_\-]+", "_", candidate.series_name).strip("_") or "series"
        filename = (
            f"{safe_series}_S{candidate.season:02d}E{candidate.episode:02d}_"
            f"{candidate.language}_os_{file_id}.srt"
        )
        return self.cache_dir / "opensubtitles" / "subtitles" / filename

    def download_subtitle(self, candidate: SubtitleCandidate) -> bytes:
        """
        Obtain a temporary download link via `POST /api/v1/download` (or use direct URL)
        and download the subtitle bytes.
        """
        cache_path = self._candidate_cache_path(candidate)
        if self.use_cache and cache_path.exists():
            try:
                cached = cache_path.read_bytes()
                normalize_subtitle_bytes(cached)
                return cached
            except Exception:
                pass

        direct_link = candidate.extra_metadata.get("direct_link")
        if not direct_link and candidate.download_url.startswith("http") and "/api/v1/download#" not in candidate.download_url:
            direct_link = candidate.download_url

        if not direct_link:
            file_id = candidate.extra_metadata.get("file_id")
            if file_id is None:
                m = re.search(r"#(\d+)$", candidate.download_url)
                if m:
                    file_id = int(m.group(1))
            if file_id is None:
                raise ProviderError("OpenSubtitles candidate missing 'file_id' for download")

            dl_resp_bytes, _ = self._request(
                "POST",
                f"{OPENSUBTITLES_API_BASE}/download",
                payload={"file_id": int(file_id)},
                require_auth=True,
            )
            try:
                dl_info = json.loads(dl_resp_bytes.decode("utf-8"))
            except Exception as exc:
                raise ProviderError(f"Invalid JSON from OpenSubtitles /download: {exc}") from exc

            direct_link = dl_info.get("link")
            if not direct_link:
                raise ProviderError(f"OpenSubtitles /download did not return a link: {dl_info}")

        raw_bytes, _ = self._request("GET", str(direct_link), require_auth=False)
        try:
            normalized_text = normalize_subtitle_bytes(raw_bytes)
        except SubtitleNormalizationError as exc:
            raise ProviderError(f"Downloaded OpenSubtitles payload is invalid: {exc}") from exc

        utf8_bytes = normalized_text.encode("utf-8")
        if self.use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(utf8_bytes)
            except Exception:
                pass

        return utf8_bytes
