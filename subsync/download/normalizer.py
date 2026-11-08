"""Character encoding detection, normalization, and validation for downloaded subtitles."""

from __future__ import annotations

import gzip
import io
from pathlib import Path
import re
from typing import Optional, Union
import zipfile

try:
    import chardet
except ImportError:  # pragma: no cover
    chardet = None  # type: ignore[assignment]


class SubtitleNormalizationError(ValueError):
    """Raised when downloaded bytes cannot be normalized into a valid subtitle file."""


_SRT_ARROW_RE = re.compile(
    r"\d{1,2}:\d{2}:\d{2}[,\.]\d{2,3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{2,3}"
)
_ASS_EVENTS_RE = re.compile(r"\[Events\]|^\s*Dialogue\s*:", re.IGNORECASE | re.MULTILINE)
_MICRODVD_RE = re.compile(r"^\s*\{\d+\}\{\d+\}", re.MULTILINE)
_WEBVTT_RE = re.compile(r"^\s*WEBVTT\b", re.IGNORECASE | re.MULTILINE)

_HTML_REJECT_RE = re.compile(
    r"^\s*(?:<!DOCTYPE\s+html|<html\b|<head\b|<body\b)|"
    r"Daily download limit reached|"
    r"Attention Required!\s*\|\s*Cloudflare|"
    r"<title>[^<]*(?:403 Forbidden|404 Not Found|503 Service|Addic7ed\.com)[^<]*</title>",
    re.IGNORECASE | re.MULTILINE,
)

_SUBTITLE_ARCHIVE_EXTS = (".srt", ".ass", ".ssa", ".vtt", ".sub")


def _unwrap_compressed_payload(raw_bytes: bytes) -> bytes:
    """
    If `raw_bytes` is a ZIP archive or gzip stream, extract and return the first
    subtitle payload inside it. Otherwise return `raw_bytes` unchanged.
    """
    if len(raw_bytes) >= 4 and raw_bytes[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                names = [n for n in zf.namelist() if not n.endswith("/")]
                if not names:
                    raise SubtitleNormalizationError("Downloaded ZIP archive is empty")
                # Prefer .srt first, then other subtitle extensions
                for ext in _SUBTITLE_ARCHIVE_EXTS:
                    matching = [n for n in names if n.lower().endswith(ext)]
                    if matching:
                        return zf.read(matching[0])
                return zf.read(names[0])
        except zipfile.BadZipFile as exc:
            raise SubtitleNormalizationError(f"Corrupt subtitle ZIP archive: {exc}") from exc

    if len(raw_bytes) >= 2 and raw_bytes[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(raw_bytes)
        except Exception as exc:
            raise SubtitleNormalizationError(f"Corrupt gzip subtitle stream: {exc}") from exc

    return raw_bytes


def detect_subtitle_encoding(
    raw_bytes: bytes,
    declared_charset: Optional[str] = None,
) -> str:
    """
    Determine the character encoding of `raw_bytes` with special care for
    UTF-8, UTF-16 BOMs, and Western European / French encodings (Windows-1252 / ISO-8859-1).
    """
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw_bytes.startswith(b"\xff\xfe\x00\x00"):
        return "utf-32-le"
    if raw_bytes.startswith(b"\x00\x00\xfe\xff"):
        return "utf-32-be"
    if raw_bytes.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw_bytes.startswith(b"\xfe\xff"):
        return "utf-16-be"

    # Strict UTF-8 check: valid multi-byte UTF-8 sequences are statistically unambiguous
    try:
        raw_bytes.decode("utf-8", errors="strict")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # Check declared HTTP charset if provided
    if declared_charset:
        norm_decl = declared_charset.strip().lower().strip("\"'")
        if norm_decl in {"iso-8859-1", "iso8859-1", "latin-1", "latin1", "cp1252", "windows-1252"}:
            # Prefer cp1252 when possible because it maps 0x80-0x9F (e.g. French 'œ', '’', '…')
            # while remaining 100% identical to ISO-8859-1 on 0xA0-0xFF ('Ç', 'ê', 'é', 'à')
            try:
                raw_bytes.decode("cp1252", errors="strict")
                return "cp1252"
            except UnicodeDecodeError:
                return "iso-8859-1"
        try:
            raw_bytes.decode(norm_decl, errors="strict")
            return norm_decl
        except (LookupError, UnicodeDecodeError):
            pass

    # Run chardet if available
    if chardet is not None:
        try:
            detection = chardet.detect(raw_bytes[:65536])
            det_enc = (detection.get("encoding") or "").strip().lower()
            confidence = float(detection.get("confidence") or 0.0)
            if det_enc and confidence >= 0.5:
                if det_enc in {"iso-8859-1", "iso-8859-2", "latin-1", "windows-1252", "cp1252", "macroman"}:
                    try:
                        raw_bytes.decode("cp1252", errors="strict")
                        return "cp1252"
                    except UnicodeDecodeError:
                        return "iso-8859-1"
                try:
                    raw_bytes.decode(det_enc, errors="strict")
                    return det_enc
                except (LookupError, UnicodeDecodeError):
                    pass
        except Exception:
            pass

    # Fallback cascade for Western European / French subtitles
    for candidate_enc in ("cp1252", "iso-8859-15", "iso-8859-1"):
        try:
            raw_bytes.decode(candidate_enc, errors="strict")
            return candidate_enc
        except UnicodeDecodeError:
            continue

    return "utf-8"


def validate_subtitle_text(text: str) -> bool:
    """
    Return True if `text` has valid subtitle timing markers (SRT `-->`, ASS `[Events]`,
    WebVTT, or MicroDVD) and is not an HTML error or rate-limit page.
    """
    if not text or not text.strip():
        return False

    has_timing = bool(
        _SRT_ARROW_RE.search(text)
        or ("-->" in text and re.search(r"\d:\d{2}:\d{2}", text))
        or _ASS_EVENTS_RE.search(text)
        or _MICRODVD_RE.search(text)
        or _WEBVTT_RE.search(text)
    )
    if not has_timing:
        return False

    # Even if '-->' appeared in an HTML comment (`<!-- -->`), reject HTML documents
    head_sample = text.lstrip()[:1024]
    if re.match(r"^(?:<!DOCTYPE\s+html|<html\b|<head\b)", head_sample, re.IGNORECASE):
        return False
    if "Daily download limit reached" in head_sample:
        return False

    return True


def normalize_subtitle_bytes(
    raw_bytes: bytes,
    declared_charset: Optional[str] = None,
) -> str:
    """
    Decode raw subtitle bytes (handling UTF-8, UTF-8 BOM, UTF-16, Windows-1252,
    ISO-8859-1/Latin-1, and chardet detection), normalize line endings to `\\n`,
    and validate that the result is a genuine subtitle file (not an HTML error page).

    Raises `SubtitleNormalizationError` if the payload is empty, an HTML rate-limit/error
    page, or lacks valid subtitle timing structures.
    """
    if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
        raise SubtitleNormalizationError(f"Expected bytes-like object, got {type(raw_bytes).__name__}")

    payload = _unwrap_compressed_payload(bytes(raw_bytes))
    if not payload or not payload.strip():
        raise SubtitleNormalizationError("Subtitle payload is empty")

    encoding = detect_subtitle_encoding(payload, declared_charset=declared_charset)
    try:
        decoded = payload.decode(encoding, errors="strict")
    except UnicodeDecodeError:
        decoded = payload.decode("iso-8859-1", errors="replace")

    # Strip any residual Unicode BOM and normalize CRLF / CR line endings to LF
    if decoded.startswith("\ufeff"):
        decoded = decoded.lstrip("\ufeff")
    normalized = decoded.replace("\r\n", "\n").replace("\r", "\n")

    # Check for HTML error / rate-limit pages before generic validation for clearer error messages
    if _HTML_REJECT_RE.search(normalized[:2048]) and not _SRT_ARROW_RE.search(normalized):
        if "daily download limit" in normalized.lower():
            raise SubtitleNormalizationError("Provider returned HTML page: Daily download limit reached")
        raise SubtitleNormalizationError("Provider returned an HTML error page instead of a subtitle file")

    if not validate_subtitle_text(normalized):
        raise SubtitleNormalizationError(
            "Decoded payload is not a valid subtitle file (missing '-->' or '[Events]' timing markers)"
        )

    if not normalized.endswith("\n"):
        normalized += "\n"

    return normalized


def write_normalized_subtitle(
    raw_bytes: bytes,
    dest_path: Union[Path, str],
    declared_charset: Optional[str] = None,
) -> Path:
    """Normalize `raw_bytes` to clean UTF-8 with LF line endings and write to `dest_path`."""
    out_path = Path(dest_path)
    text = normalize_subtitle_bytes(raw_bytes, declared_charset=declared_charset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8", newline="\n")
    return out_path
