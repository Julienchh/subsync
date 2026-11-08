"""Addic7ed anonymous subtitle provider with show ID resolution, season table parsing, and disk caching."""

from __future__ import annotations

import hashlib
import html as html_lib
import json
from pathlib import Path
import re
import time
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
    matches_language,
    normalize_lang,
    score_subtitle_candidate,
)

ADDIC7ED_BASE_URL = "https://www.addic7ed.com"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)

# Built-in fast lookup map for popular series -> Addic7ed show ID
BUILTIN_SHOW_IDS: Dict[str, int] = {
    "the blacklist": 4008,
    "blacklist": 4008,
    "breaking bad": 126,
    "better call saul": 5023,
    "game of thrones": 1245,
    "house of the dragon": 8867,
    "stranger things": 5794,
    "the mandalorian": 7561,
    "the last of us": 9062,
    "the boys": 7374,
    "mr robot": 5196,
    "dark": 6543,
    "sherlock": 763,
    "fargo": 4381,
    "true detective": 4206,
    "westworld": 5920,
    "succession": 6856,
    "severance": 8667,
    "arcane": 8598,
    "chernobyl": 7316,
    "peaky blinders": 4017,
    "prison break": 16,
    "lost": 15,
    "friends": 73,
    "the office": 100,
    "dexter": 23,
    "the wire": 154,
    "the sopranos": 175,
    "vikings": 3437,
    "narcos": 5343,
    "mindhunter": 6437,
    "ozark": 6352,
    "the witcher": 7639,
    "wednesday": 9015,
    "house": 12,
    "house md": 12,
    "person of interest": 1906,
    "fringe": 135,
    "homeland": 1927,
    "the americans": 3511,
    "mad men": 125,
    "boardwalk empire": 1020,
    "suits": 1766,
    "white collar": 584,
    "mentalist": 211,
    "the mentalist": 211,
    "castle": 274,
    "bones": 36,
    "criminal minds": 43,
    "ncis": 82,
    "supernatural": 20,
    "doctor who": 94,
    "the walking dead": 1152,
    "arrow": 3005,
    "the flash": 4565,
    "daredevil": 5133,
    "the punisher": 6493,
    "loki": 8418,
    "wandavision": 8228,
    "andor": 8905,
}

_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_SHOW_ID_URL_RE = re.compile(r"/show/(\d+)")
_SERIE_LINK_RE = re.compile(r'href=["\'](/serie/[^"\']+)["\']', re.IGNORECASE)
_OPTION_SHOW_RE = re.compile(
    r'<option\s+value=["\'](\d+)["\'][^>]*>([^<]+)</option>', re.IGNORECASE
)
_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD_RE = re.compile(r"<td\b[^>]*>(.*?)</td>", re.IGNORECASE | re.DOTALL)
_DL_HREF_RE = re.compile(
    r'href=["\']((?:/updated/|/original/)[^"\']+)["\']', re.IGNORECASE
)
_CHARSET_RE = re.compile(r"charset=([A-Za-z0-9_\-]+)", re.IGNORECASE)

# Known mis-uploaded subtitle links on Addic7ed (e.g. The Blacklist: Redemption S01E03 uploaded under Show 4008 S01E03)
_KNOWN_BAD_DOWNLOAD_HREFS = {
    "/updated/8/79913/15",
    "/original/8/79913/15",
}


def _normalize_show_key(name: str) -> str:
    """Normalize a series title for dictionary and fuzzy show lookup."""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"['’`\".,:;!?\(\)\[\]\-_]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _strip_html_tags(raw_html: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    text = _TAG_STRIP_RE.sub("", raw_html or "")
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


class Addic7edProvider(SubtitleProvider):
    """
    Zero-authentication subtitle provider for Addic7ed.com.

    Features:
      - Built-in fast lookup map for common shows (including The Blacklist -> 4008)
        plus persistent JSON disk cache (`addic7ed_shows.json`).
      - Dynamic show ID resolution via `/search.php` (302 redirect and result links)
        and `/ajax_getShows.php`.
      - Season table fetching via `/ajax_loadShow.php?show={id}&season={season}&langs=&hd=0&hi=0`
        with local disk caching.
      - Candidate scoring by completion state, HI flag, `/updated/` vs `/original/`,
        and release group similarity.
      - Polite rate limiting and local disk caching of downloaded `.srt` files.
    """

    name = "addic7ed"

    def __init__(
        self,
        cache_dir: Optional[Union[Path, str]] = None,
        min_request_interval_s: float = 1.0,
        timeout_s: float = 15.0,
        prefer_hearing_impaired: bool = False,
        use_cache: bool = True,
        http_get_fn: Optional[Callable[[str, Dict[str, str], float], Tuple[bytes, Dict[str, str], str]]] = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else (Path.home() / ".cache" / "subsync")
        self.min_request_interval_s = max(0.0, float(min_request_interval_s))
        self.timeout_s = float(timeout_s)
        self.prefer_hearing_impaired = bool(prefer_hearing_impaired)
        self.use_cache = bool(use_cache)
        self.user_agent = user_agent
        self._http_get_fn = http_get_fn
        self._last_request_ts: float = 0.0
        self._show_cache: Dict[str, int] = dict(BUILTIN_SHOW_IDS)
        self._load_show_cache()

    @property
    def shows_cache_path(self) -> Path:
        return self.cache_dir / "addic7ed_shows.json"

    def _load_show_cache(self) -> None:
        if not self.use_cache:
            return
        try:
            if self.shows_cache_path.exists():
                data = json.loads(self.shows_cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(k, str) and isinstance(v, int):
                            self._show_cache[_normalize_show_key(k)] = v
        except Exception:
            pass

    def _save_show_cache(self) -> None:
        if not self.use_cache:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.shows_cache_path.write_text(
                json.dumps(self._show_cache, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _respect_rate_limit(self) -> None:
        if self.min_request_interval_s <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        if self._last_request_ts > 0 and elapsed < self.min_request_interval_s:
            time.sleep(self.min_request_interval_s - elapsed)
        self._last_request_ts = time.monotonic()

    def _http_get(
        self,
        url: str,
        referer: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[bytes, Dict[str, str], str]:
        """
        Perform an HTTP GET request with browser headers.
        Returns `(body_bytes, headers_lower_dict, final_url)`.
        """
        headers: Dict[str, str] = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        if referer:
            headers["Referer"] = referer
        if extra_headers:
            headers.update(extra_headers)

        if self._http_get_fn is not None:
            return self._http_get_fn(url, headers, self.timeout_s)

        self._respect_rate_limit()
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read()
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                final_url = resp.geturl() or url
                return body, resp_headers, final_url
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise ProviderRateLimitError(
                    f"Addic7ed rate limit exceeded (HTTP 429) for {url}"
                ) from exc
            raise ProviderNetworkError(
                f"Addic7ed HTTP {exc.code} error fetching {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ProviderNetworkError(
                f"Network unreachable querying Addic7ed ({url}): {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise ProviderNetworkError(
                f"Timed out querying Addic7ed ({url})"
            ) from exc

    def resolve_show_id(self, series_name: str) -> Optional[int]:
        """
        Resolve a TV series title to its numeric Addic7ed show ID.
        Checks:
          1. Built-in show map & persistent disk cache (`addic7ed_shows.json`).
          2. `https://www.addic7ed.com/search.php?search=<query>&Submit=Search`
             (handles 302 redirect to `/show/<id>` and search results page links).
          3. `https://www.addic7ed.com/ajax_getShows.php`.
        """
        norm_key = _normalize_show_key(series_name)
        if not norm_key:
            return None

        if norm_key in self._show_cache:
            return self._show_cache[norm_key]

        # Also check without leading "the "
        without_the = re.sub(r"^the\s+", "", norm_key)
        if without_the in self._show_cache:
            return self._show_cache[without_the]

        # Dynamic Strategy 1: search.php
        query = urllib.parse.quote_plus(series_name.strip())
        search_url = f"{ADDIC7ED_BASE_URL}/search.php?search={query}&Submit=Search"
        try:
            body, _, final_url = self._http_get(search_url, referer=f"{ADDIC7ED_BASE_URL}/")
            m_url = _SHOW_ID_URL_RE.search(final_url)
            if m_url:
                show_id = int(m_url.group(1))
                self._show_cache[norm_key] = show_id
                self._save_show_cache()
                return show_id

            html_text = body.decode("utf-8", errors="replace")
            m_body = _SHOW_ID_URL_RE.search(html_text)
            if m_body:
                show_id = int(m_body.group(1))
                self._show_cache[norm_key] = show_id
                self._save_show_cache()
                return show_id

            # If search.php returns links to /serie/<Show_Name>/... follow the first matching link
            serie_links = _SERIE_LINK_RE.findall(html_text)
            if serie_links:
                serie_path = serie_links[0]
                if not serie_path.startswith("/"):
                    serie_path = "/" + serie_path
                ep_body, _, ep_final_url = self._http_get(
                    f"{ADDIC7ED_BASE_URL}{serie_path}",
                    referer=search_url,
                )
                m_ep = _SHOW_ID_URL_RE.search(ep_final_url) or _SHOW_ID_URL_RE.search(
                    ep_body.decode("utf-8", errors="replace")
                )
                if m_ep:
                    show_id = int(m_ep.group(1))
                    self._show_cache[norm_key] = show_id
                    self._save_show_cache()
                    return show_id
        except ProviderError:
            pass

        # Dynamic Strategy 2: ajax_getShows.php
        try:
            body, _, _ = self._http_get(
                f"{ADDIC7ED_BASE_URL}/ajax_getShows.php",
                referer=f"{ADDIC7ED_BASE_URL}/",
            )
            html_text = body.decode("utf-8", errors="replace")
            exact_match: Optional[int] = None
            partial_match: Optional[int] = None
            for raw_id, raw_title in _OPTION_SHOW_RE.findall(html_text):
                sid = int(raw_id)
                if sid <= 0:
                    continue
                opt_key = _normalize_show_key(html_lib.unescape(raw_title))
                self._show_cache[opt_key] = sid
                if opt_key == norm_key or opt_key == without_the:
                    exact_match = sid
                elif norm_key in opt_key and partial_match is None:
                    partial_match = sid

            chosen = exact_match or partial_match
            if chosen is not None:
                self._show_cache[norm_key] = chosen
                self._save_show_cache()
                return chosen
        except ProviderError:
            pass

        return None

    def fetch_season_html(self, show_id: int, season: int) -> str:
        """
        Fetch the Addic7ed season subtitle table HTML via `ajax_loadShow.php`,
        reading from or writing to the local disk cache when `self.use_cache` is True.
        """
        cache_file = self.cache_dir / "addic7ed" / f"show_{show_id}_season_{season:02d}.html"
        if self.use_cache and cache_file.exists():
            try:
                cached_html = cache_file.read_text(encoding="utf-8")
                if "<tr" in cached_html.lower():
                    return cached_html
            except Exception:
                pass

        url = (
            f"{ADDIC7ED_BASE_URL}/ajax_loadShow.php"
            f"?show={int(show_id)}&season={int(season)}&langs=&hd=0&hi=0"
        )
        referer = f"{ADDIC7ED_BASE_URL}/show/{int(show_id)}"
        body, _, _ = self._http_get(
            url,
            referer=referer,
            extra_headers={"X-Requested-With": "XMLHttpRequest"},
        )
        html_text = body.decode("utf-8", errors="replace")

        if self.use_cache and "<tr" in html_text.lower():
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(html_text, encoding="utf-8")
            except Exception:
                pass

        return html_text

    @staticmethod
    def parse_season_html(
        html_text: str,
        show_id: int,
        series_name: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Parse the `#season tbody tr` rows returned by `ajax_loadShow.php`.

        Returns a list of row dicts with keys:
          `season`, `episode`, `title`, `language`, `version`,
          `completed`, `completed_state`, `hearing_impaired`, `download_href`, `is_updated`.
        """
        parsed_rows: List[Dict[str, Any]] = []
        if not html_text:
            return parsed_rows

        for tr_inner in _TR_RE.findall(html_text):
            cells = _TD_RE.findall(tr_inner)
            if len(cells) < 5:
                continue

            s_str = _strip_html_tags(cells[0])
            e_str = _strip_html_tags(cells[1])
            if not (s_str.isdigit() and e_str.isdigit()):
                continue

            season_num = int(s_str)
            episode_num = int(e_str)
            ep_title = _strip_html_tags(cells[2])
            lang_text = _strip_html_tags(cells[3])

            # Extract download links across the row (preferring /updated/ over /original/)
            dl_links = [
                lnk for lnk in _DL_HREF_RE.findall(tr_inner)
                if lnk not in _KNOWN_BAD_DOWNLOAD_HREFS
            ]
            if not dl_links:
                continue

            updated_links = [lnk for lnk in dl_links if "/updated/" in lnk.lower()]
            chosen_href = updated_links[0] if updated_links else dl_links[0]
            is_updated = "/updated/" in chosen_href.lower()

            # Standard Addic7ed ajax_loadShow.php rows have 10 or 11 <td> cells:
            # 0:Season, 1:Ep, 2:Title, 3:Lang, 4:Version, 5:Completed, 6:HI, 7:Corrected, 8:HD, 9:Download
            version = "Standard"
            completed_state = "Completed"
            hearing_impaired = False

            if len(cells) >= 6 and not _DL_HREF_RE.search(cells[4]):
                raw_ver = _strip_html_tags(cells[4])
                if raw_ver:
                    version = raw_ver
                raw_comp = _strip_html_tags(cells[5])
                if raw_comp:
                    completed_state = raw_comp

            if len(cells) >= 7 and not _DL_HREF_RE.search(cells[6]):
                hi_cell = cells[6].strip()
                hearing_impaired = bool(
                    "<img" in hi_cell.lower()
                    or "hearing" in hi_cell.lower()
                    or _strip_html_tags(hi_cell).lower() in {"hi", "yes", "true", "1"}
                )

            is_completed = (
                "completed" in completed_state.lower()
                and "%" not in completed_state
            )

            parsed_rows.append(
                {
                    "show_id": show_id,
                    "series_name": series_name,
                    "season": season_num,
                    "episode": episode_num,
                    "title": ep_title,
                    "language": lang_text,
                    "version": version,
                    "completed": is_completed,
                    "completed_state": completed_state,
                    "hearing_impaired": hearing_impaired,
                    "download_href": chosen_href,
                    "is_updated": is_updated,
                }
            )

        return parsed_rows

    def search_subtitles(
        self,
        meta: SeriesEpisodeMetadata,
        target_lang: str,
    ) -> List[SubtitleCandidate]:
        """
        Search Addic7ed for subtitles matching `meta` (series, season, episode)
        and `target_lang` (e.g. 'fr' -> 'French').
        """
        lang_spec = normalize_lang(target_lang)
        show_id = self.resolve_show_id(meta.series_name)
        if show_id is None:
            return []

        html_text = self.fetch_season_html(show_id=show_id, season=meta.season)
        rows = self.parse_season_html(
            html_text=html_text,
            show_id=show_id,
            series_name=meta.series_name,
        )

        referer = f"{ADDIC7ED_BASE_URL}/show/{show_id}"
        candidates: List[SubtitleCandidate] = []

        for row in rows:
            if row["season"] != meta.season or row["episode"] != meta.episode:
                continue
            if not matches_language(row["language"], target_lang):
                continue

            href = row["download_href"]
            full_url = href if href.startswith("http") else f"{ADDIC7ED_BASE_URL}{href}"

            score = score_subtitle_candidate(
                meta=meta,
                release_version=row["version"],
                completed=row["completed"],
                hearing_impaired=row["hearing_impaired"],
                prefer_hearing_impaired=self.prefer_hearing_impaired,
                is_updated=row["is_updated"],
            )

            candidates.append(
                SubtitleCandidate(
                    provider_name=self.name,
                    series_name=meta.series_name,
                    season=meta.season,
                    episode=meta.episode,
                    language=lang_spec.iso639_1,
                    release_version=row["version"],
                    download_url=full_url,
                    referer=referer,
                    hearing_impaired=row["hearing_impaired"],
                    completed=row["completed"],
                    score=score,
                    extra_metadata={
                        "show_id": show_id,
                        "episode_title": row["title"],
                        "addic7ed_language": row["language"],
                        "completed_state": row["completed_state"],
                        "download_href": href,
                        "is_updated": row["is_updated"],
                    },
                )
            )

        candidates.sort(
            key=lambda c: (
                c.score,
                c.completed,
                not c.hearing_impaired if not self.prefer_hearing_impaired else c.hearing_impaired,
            ),
            reverse=True,
        )
        return candidates

    def _candidate_cache_path(self, candidate: SubtitleCandidate) -> Path:
        url_hash = hashlib.sha256(candidate.download_url.encode("utf-8")).hexdigest()[:16]
        safe_series = re.sub(r"[^A-Za-z0-9_\-]+", "_", candidate.series_name).strip("_") or "series"
        safe_ver = re.sub(r"[^A-Za-z0-9_\-]+", "_", candidate.release_version).strip("_") or "std"
        filename = (
            f"{safe_series}_S{candidate.season:02d}E{candidate.episode:02d}_"
            f"{candidate.language}_{safe_ver}_{url_hash}.srt"
        )
        return self.cache_dir / "addic7ed" / "subtitles" / filename

    def download_subtitle(self, candidate: SubtitleCandidate) -> bytes:
        """
        Download a subtitle from Addic7ed using the required `Referer` header,
        validate that it is a real subtitle (not a daily-limit HTML page),
        and cache the normalized UTF-8 `.srt` on disk.
        """
        cache_path = self._candidate_cache_path(candidate)
        if self.use_cache and cache_path.exists():
            try:
                cached_bytes = cache_path.read_bytes()
                normalize_subtitle_bytes(cached_bytes)
                return cached_bytes
            except Exception:
                pass

        url = candidate.download_url
        if url.startswith("/"):
            url = f"{ADDIC7ED_BASE_URL}{url}"

        show_id = candidate.extra_metadata.get("show_id", 4008)
        referer = candidate.referer or f"{ADDIC7ED_BASE_URL}/show/{show_id}"

        raw_bytes, resp_headers, _ = self._http_get(url, referer=referer)

        content_type = resp_headers.get("content-type", "")
        declared_charset: Optional[str] = None
        m_cs = _CHARSET_RE.search(content_type)
        if m_cs:
            declared_charset = m_cs.group(1)

        content_disp = resp_headers.get("content-disposition", "")
        if content_disp:
            candidate.extra_metadata["content_disposition"] = content_disp

        try:
            normalized_text = normalize_subtitle_bytes(
                raw_bytes,
                declared_charset=declared_charset,
            )
        except SubtitleNormalizationError as exc:
            err_lower = str(exc).lower()
            if (
                "daily download limit" in err_lower
                or "html error page" in err_lower
                or "html page" in err_lower
            ):
                raise ProviderRateLimitError(str(exc)) from exc
            raise ProviderError(
                f"Addic7ed download from {url} did not return a valid subtitle: {exc}"
            ) from exc

        utf8_bytes = normalized_text.encode("utf-8")
        if self.use_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(utf8_bytes)
            except Exception:
                pass

        return utf8_bytes
