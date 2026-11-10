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

_CHAR_TO_WIN1252_BYTE = {}
for _b in range(0x80, 0x100):
    _CHAR_TO_WIN1252_BYTE[chr(_b)] = _b
    try:
        _CHAR_TO_WIN1252_BYTE[bytes([_b]).decode("cp1252")] = _b
    except UnicodeDecodeError:
        pass


def repair_mojibake_text(text: str) -> str:
    """
    Repair span-level double-encoded UTF-8-via-CP1252/Latin-1 mojibake (including
    mixed files where some accents like 'é' are clean UTF-8 while 'ç', 'à', 'è',
    'Ç', 'É', 'œ', '’', '…' were double-encoded as 'Ã§', 'Ã ', 'Ã¨', 'Ã‡', 'Ã‰', 'Å“', 'â€™')
    as well as raw C1 control bytes U+0080..U+009F from ISO-8859-1 misdecoding.
    """
    if not text:
        return text

    if "ï»¿" in text:
        text = text.replace("ï»¿", "")

    for _ in range(2):
        if not any(m in text for m in ("Ã", "Â", "Å", "â€", "â™", "â‚", "Ë")):
            break
        out = []
        i = 0
        n = len(text)
        changed = False
        while i < n:
            c0 = text[i]
            b0 = _CHAR_TO_WIN1252_BYTE.get(c0)
            # 3-byte UTF-8 mojibake starting with 0xE2 ('â', e.g. â€™ -> ’, â€¦ -> …, â™ª -> ♪)
            if b0 == 0xE2 and i + 2 < n:
                b1 = _CHAR_TO_WIN1252_BYTE.get(text[i + 1])
                b2 = _CHAR_TO_WIN1252_BYTE.get(text[i + 2])
                if b1 is not None and 0x80 <= b1 <= 0xBF and b2 is not None and 0x80 <= b2 <= 0xBF:
                    try:
                        out.append(bytes([b0, b1, b2]).decode("utf-8", errors="strict"))
                        i += 3
                        changed = True
                        continue
                    except UnicodeDecodeError:
                        pass
            # 2-byte UTF-8 mojibake starting with 0xC2..0xC5 (Â, Ã, Ä, Å) or 0xCB (Ë)
            if b0 in (0xC2, 0xC3, 0xC4, 0xC5, 0xCB) and i + 1 < n:
                b1 = _CHAR_TO_WIN1252_BYTE.get(text[i + 1])
                if b1 is not None and 0x80 <= b1 <= 0xBF:
                    try:
                        out.append(bytes([b0, b1]).decode("utf-8", errors="strict"))
                        i += 2
                        changed = True
                        continue
                    except UnicodeDecodeError:
                        pass
            out.append(c0)
            i += 1
        text = "".join(out)
        if not changed:
            break

    # Repair raw C1 control characters U+0080..U+009F (CP1252 bytes decoded as ISO-8859-1)
    if any(0x80 <= ord(ch) <= 0x9F for ch in text):
        fixed_c1 = []
        for ch in text:
            code = ord(ch)
            if 0x80 <= code <= 0x9F:
                try:
                    fixed_c1.append(bytes([code]).decode("cp1252"))
                    continue
                except UnicodeDecodeError:
                    pass
            fixed_c1.append(ch)
        text = "".join(fixed_c1)

    return text


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

    # Strip any residual Unicode BOM, repair mixed/double UTF-8 mojibake, and normalize line endings
    if decoded.startswith("\ufeff"):
        decoded = decoded.lstrip("\ufeff")
    decoded = repair_mojibake_text(decoded)
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
