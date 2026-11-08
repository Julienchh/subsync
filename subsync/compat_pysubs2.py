"""Pure-Python compatibility implementation of pysubs2 for subsync.

Supports loading, manipulating, and saving .srt, .ass/.ssa, .vtt, .microdvd (.sub),
and .tmp (.txt) subtitle formats with millisecond precision.
Automatically registers as `sys.modules['pysubs2']` if native `pysubs2` is absent.
"""

from collections.abc import MutableSequence
import copy
import os
import re
import sys
import types


class Pysubs2Error(Exception):
    """Base exception for pysubs2 errors."""


class UnknownFPSError(Pysubs2Error):
    """Raised when framerate is required but not provided."""


class UnknownFormatIdentifierError(Pysubs2Error):
    """Raised when subtitle format identifier is unknown."""


class FormatAutodetectionError(Pysubs2Error):
    """Raised when subtitle format cannot be autodetected."""


FILE_EXTENSION_TO_FORMAT_IDENTIFIER = {
    ".srt": "srt",
    ".ass": "ass",
    ".ssa": "ssa",
    ".sub": "microdvd",
    ".vtt": "vtt",
    ".txt": "tmp",
    ".json": "json",
}

FORMAT_IDENTIFIER_TO_FILE_EXTENSION = {
    "srt": ".srt",
    "ass": ".ass",
    "ssa": ".ssa",
    "microdvd": ".sub",
    "vtt": ".vtt",
    "tmp": ".txt",
    "json": ".json",
}


def time_to_ms(h=0, m=0, s=0.0, ms=0):
    """Convert hours, minutes, seconds, milliseconds to integer milliseconds."""
    total_s = (int(h) * 60 + int(m)) * 60 + float(s)
    return int(round(total_s * 1000.0 + float(ms)))


def ms_to_times(ms):
    """Convert milliseconds to (h, m, s, ms) tuple of non-negative integers."""
    ms = max(0, int(round(ms)))
    s, rem_ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), int(s), int(rem_ms)


def make_time(h=0, m=0, s=0.0, ms=0, frames=None, fps=None):
    """Convert time specification to integer milliseconds."""
    base_ms = time_to_ms(h=h, m=m, s=s, ms=ms)
    if frames is not None:
        if fps is None or fps <= 0:
            raise ValueError("fps must be specified and positive when frames is given")
        base_ms += int(round(float(frames) * 1000.0 / float(fps)))
    return base_ms


_SRT_TS_RE = re.compile(
    r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*$"
)
_ASS_TS_RE = re.compile(
    r"^\s*(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*$"
)


def parse_srt_timestamp(ts_str):
    """Parse SRT/VTT timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm or MM:SS.mmm) into ms."""
    m = _SRT_TS_RE.match(ts_str)
    if not m:
        raise ValueError(f"Invalid SRT/VTT timestamp: {ts_str!r}")
    h_s, m_s, s_s, frac_s = m.groups()
    h = int(h_s) if h_s is not None else 0
    mi = int(m_s)
    s = int(s_s)
    if len(frac_s) == 1:
        ms = int(frac_s) * 100
    elif len(frac_s) == 2:
        ms = int(frac_s) * 10
    else:
        ms = int(frac_s[:3])
    return time_to_ms(h=h, m=mi, s=s, ms=ms)


def format_srt_timestamp(ms):
    """Format milliseconds into SRT timestamp HH:MM:SS,mmm."""
    h, m, s, rem_ms = ms_to_times(ms)
    return f"{h:02d}:{m:02d}:{s:02d},{rem_ms:03d}"


def format_vtt_timestamp(ms):
    """Format milliseconds into WebVTT timestamp HH:MM:SS.mmm."""
    h, m, s, rem_ms = ms_to_times(ms)
    return f"{h:02d}:{m:02d}:{s:02d}.{rem_ms:03d}"


def parse_ass_timestamp(ts_str):
    """Parse ASS/SSA timestamp H:MM:SS.cc into ms."""
    m = _ASS_TS_RE.match(ts_str)
    if not m:
        raise ValueError(f"Invalid ASS timestamp: {ts_str!r}")
    h_s, m_s, s_s, frac_s = m.groups()
    h = int(h_s)
    mi = int(m_s)
    s = int(s_s)
    if len(frac_s) == 1:
        ms = int(frac_s) * 100
    elif len(frac_s) == 2:
        ms = int(frac_s) * 10
    else:
        ms = int(frac_s[:3])
    return time_to_ms(h=h, m=mi, s=s, ms=ms)


def format_ass_timestamp(ms):
    """Format milliseconds into ASS/SSA timestamp H:MM:SS.cc."""
    ms = max(0, int(round(ms)))
    cs_total = int(round(ms / 10.0))
    s_total, cs = divmod(cs_total, 100)
    m_total, s = divmod(s_total, 60)
    h, m = divmod(m_total, 60)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


class Color(object):
    """RGBA color representation compatible with pysubs2.Color."""

    def __init__(self, r=255, g=255, b=255, a=0):
        self.r = int(r)
        self.g = int(g)
        self.b = int(b)
        self.a = int(a)

    def __repr__(self):
        return f"Color(r={self.r}, g={self.g}, b={self.b}, a={self.a})"

    def __eq__(self, other):
        if isinstance(other, Color):
            return (self.r, self.g, self.b, self.a) == (other.r, other.g, other.b, other.a)
        return False


class SSAStyle(object):
    """ASS/SSA style definition compatible with pysubs2.SSAStyle."""

    DEFAULT_FORMAT = [
        "Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour",
        "OutlineColour", "BackColour", "Bold", "Italic", "Underline",
        "StrikeOut", "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle",
        "Outline", "Shadow", "Alignment", "MarginL", "MarginR", "MarginV",
        "Encoding",
    ]

    def __init__(
        self,
        fontname="Arial",
        fontsize=20.0,
        primarycolor=None,
        secondarycolor=None,
        tertiarycolor=None,
        outlinecolor=None,
        backcolor=None,
        bold=False,
        italic=False,
        underline=False,
        strikeout=False,
        scalex=100.0,
        scaley=100.0,
        spacing=0.0,
        angle=0.0,
        borderstyle=1,
        outline=2.0,
        shadow=2.0,
        alignment=2,
        marginl=10,
        marginr=10,
        marginv=10,
        alphalevel=0,
        encoding=1,
    ):
        self.fontname = fontname
        self.fontsize = fontsize
        self.primarycolor = primarycolor or Color(255, 255, 255, 0)
        self.secondarycolor = secondarycolor or Color(255, 0, 0, 0)
        self.tertiarycolor = tertiarycolor or Color(0, 0, 0, 0)
        self.outlinecolor = outlinecolor or Color(0, 0, 0, 0)
        self.backcolor = backcolor or Color(0, 0, 0, 0)
        self.bold = bold
        self.italic = italic
        self.underline = underline
        self.strikeout = strikeout
        self.scalex = scalex
        self.scaley = scaley
        self.spacing = spacing
        self.angle = angle
        self.borderstyle = borderstyle
        self.outline = outline
        self.shadow = shadow
        self.alignment = alignment
        self.marginl = marginl
        self.marginr = marginr
        self.marginv = marginv
        self.alphalevel = alphalevel
        self.encoding = encoding
        self._raw_line = None

    def copy(self):
        return copy.deepcopy(self)


_OVERRIDE_TAG_RE = re.compile(r"\{[^}]*\}")
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")


def _safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


class SSAEvent(object):
    """Single subtitle event compatible with pysubs2.SSAEvent."""

    def __init__(
        self,
        start=0,
        end=10000,
        text="",
        style="Default",
        type="Dialogue",
        effect="",
        name="",
        marginl=0,
        marginr=0,
        marginv=0,
        layer=0,
        marked=False,
    ):
        self.start = int(round(start))
        self.end = int(round(end))
        self.text = str(text)
        self.style = str(style)
        self.type = str(type)
        self.effect = str(effect)
        self.name = str(name)
        self.marginl = _safe_int(marginl, 0)
        self.marginr = _safe_int(marginr, 0)
        self.marginv = _safe_int(marginv, 0)
        self.layer = _safe_int(layer, 0)
        self.marked = bool(marked)

    @property
    def duration(self):
        return self.end - self.start

    @duration.setter
    def duration(self, val):
        if val < 0:
            raise ValueError("Subtitle duration cannot be negative")
        self.end = self.start + int(round(val))

    @property
    def is_comment(self):
        return self.type == "Comment"

    @is_comment.setter
    def is_comment(self, val):
        self.type = "Comment" if val else "Dialogue"

    @property
    def is_drawing(self):
        return bool(re.search(r"\{\\[^}]*\\p[1-9]", self.text))

    @property
    def plaintext(self):
        txt = _OVERRIDE_TAG_RE.sub("", self.text)
        txt = _HTML_TAG_RE.sub("", txt)
        txt = txt.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
        return txt

    @plaintext.setter
    def plaintext(self, val):
        self.text = str(val).replace("\n", r"\N")

    def shift(self, h=0, m=0, s=0.0, ms=0, frames=None, fps=None):
        delta = make_time(h=h, m=m, s=s, ms=ms, frames=frames, fps=fps)
        self.start += delta
        self.end += delta

    def copy(self):
        return SSAEvent(
            start=self.start,
            end=self.end,
            text=self.text,
            style=self.style,
            type=self.type,
            effect=self.effect,
            name=self.name,
            marginl=self.marginl,
            marginr=self.marginr,
            marginv=self.marginv,
            layer=self.layer,
            marked=self.marked,
        )

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        res = self.copy()
        memo[id(self)] = res
        return res

    def equals(self, other):
        if not isinstance(other, SSAEvent):
            return False
        return (
            self.start == other.start
            and self.end == other.end
            and self.text == other.text
            and self.style == other.style
            and self.type == other.type
            and self.effect == other.effect
            and self.name == other.name
            and self.marginl == other.marginl
            and self.marginr == other.marginr
            and self.marginv == other.marginv
            and self.layer == other.layer
            and self.marked == other.marked
        )

    def __eq__(self, other):
        if isinstance(other, SSAEvent):
            return self.equals(other)
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, SSAEvent):
            return not self.equals(other)
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, SSAEvent):
            return (self.start, self.end) < (other.start, other.end)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, SSAEvent):
            return (self.start, self.end) <= (other.start, other.end)
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, SSAEvent):
            return (self.start, self.end) > (other.start, other.end)
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, SSAEvent):
            return (self.start, self.end) >= (other.start, other.end)
        return NotImplemented

    def __repr__(self):
        return (
            f"<SSAEvent type={self.type} "
            f"start={format_srt_timestamp(self.start)} "
            f"end={format_srt_timestamp(self.end)} "
            f"text={self.text!r}>"
        )


class SSAFile(MutableSequence):
    """Subtitle file representation compatible with pysubs2.SSAFile."""

    DEFAULT_INFO = {
        "WrapStyle": "0",
        "ScaledBorderAndShadow": "yes",
        "Collisions": "Normal",
    }

    def __init__(self):
        self.events = []
        self.styles = {"Default": SSAStyle()}
        self.info = dict(self.DEFAULT_INFO)
        self.aegisub_project = {}
        self.fps = None
        self.format = None

    # MutableSequence / list-like interface
    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __getitem__(self, index):
        return self.events[index]

    def __setitem__(self, index, value):
        if isinstance(index, int):
            if not isinstance(value, SSAEvent):
                raise TypeError("SSAFile can only contain SSAEvent instances")
            self.events[index] = value
        else:
            vals = list(value)
            if not all(isinstance(v, SSAEvent) for v in vals):
                raise TypeError("SSAFile can only contain SSAEvent instances")
            self.events[index] = vals

    def __delitem__(self, index):
        del self.events[index]

    def insert(self, index, value):
        if not isinstance(value, SSAEvent):
            raise TypeError("SSAFile can only contain SSAEvent instances")
        self.events.insert(index, value)

    def append(self, value):
        if not isinstance(value, SSAEvent):
            raise TypeError("SSAFile can only contain SSAEvent instances")
        self.events.append(value)

    def sort(self):
        self.events.sort(key=lambda ev: (ev.start, ev.end))

    def copy(self):
        return copy.deepcopy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __copy__(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        cls = self.__class__
        res = cls.__new__(cls)
        memo[id(self)] = res
        res.events = [ev.copy() for ev in self.events]
        res.styles = copy.deepcopy(self.styles, memo)
        res.info = dict(self.info)
        res.aegisub_project = dict(self.aegisub_project)
        res.fps = self.fps
        res.format = self.format
        for k, v in self.__dict__.items():
            if k not in ("events", "styles", "info", "aegisub_project", "fps", "format"):
                setattr(res, k, copy.deepcopy(v, memo))
        return res

    def equals(self, other):
        if not isinstance(other, SSAFile):
            return False
        if len(self.events) != len(other.events):
            return False
        return all(a.equals(b) for a, b in zip(self.events, other.events))

    def shift(self, h=0, m=0, s=0.0, ms=0, frames=None, fps=None):
        """Shift all subtitle events by the specified time delta."""
        eff_fps = fps if fps is not None else self.fps
        delta = make_time(h=h, m=m, s=s, ms=ms, frames=frames, fps=eff_fps)
        for ev in self.events:
            ev.start += delta
            ev.end += delta

    def transform_framerate(self, in_fps, out_fps):
        """Scale all event timestamps by in_fps / out_fps."""
        in_fps = float(in_fps)
        out_fps = float(out_fps)
        if in_fps <= 0.0 or out_fps <= 0.0:
            raise ValueError(f"Framerates must be positive, got in_fps={in_fps}, out_fps={out_fps}")
        ratio = in_fps / out_fps
        for ev in self.events:
            ev.start = int(round(ev.start * ratio))
            ev.end = int(round(ev.end * ratio))
        self.fps = out_fps

    def remove_miscellaneous_events(self):
        """Remove comment and drawing events."""
        self.events = [ev for ev in self.events if not ev.is_comment and not ev.is_drawing]

    @classmethod
    def load(cls, path, encoding="utf-8", format_=None, fps=None, **kwargs):
        """Load subtitles from a file path."""
        text = None
        encodings_to_try = []
        if encoding:
            if encoding.lower().replace("-", "") == "utf8":
                encodings_to_try.append("utf-8-sig")
            encodings_to_try.append(encoding)
        for fallback in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            if fallback not in encodings_to_try:
                encodings_to_try.append(fallback)

        last_err = None
        for enc in encodings_to_try:
            try:
                with open(path, "r", encoding=enc) as fp:
                    text = fp.read()
                break
            except UnicodeDecodeError as err:
                last_err = err
                continue

        if text is None:
            with open(path, "r", encoding=encoding or "utf-8", errors="replace") as fp:
                text = fp.read()

        if format_ is None:
            ext = os.path.splitext(path)[1].lower()
            format_ = FILE_EXTENSION_TO_FORMAT_IDENTIFIER.get(ext)

        return cls.from_string(text, format_=format_, fps=fps, **kwargs)

    @classmethod
    def from_file(cls, fp, format_=None, fps=None, **kwargs):
        """Load subtitles from an open file-like object."""
        text = fp.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8-sig", errors="replace")
        return cls.from_string(text, format_=format_, fps=fps, **kwargs)

    @classmethod
    def from_string(cls, string, format_=None, fps=None, **kwargs):
        """Parse subtitles from a string."""
        if string.startswith("\ufeff"):
            string = string[1:]

        fmt = format_.lower() if format_ else _autodetect_format(string)
        subs = cls()
        subs.fps = fps
        subs.format = fmt

        if fmt == "srt":
            _parse_srt(subs, string)
        elif fmt in ("ass", "ssa"):
            _parse_ass(subs, string, fmt=fmt)
        elif fmt == "vtt":
            _parse_vtt(subs, string)
        elif fmt == "microdvd":
            _parse_microdvd(subs, string, fps=fps)
        elif fmt == "tmp":
            _parse_tmp(subs, string)
        else:
            # Try autodetection if unknown format identifier was passed or fallback to SRT
            detected = _autodetect_format(string)
            if detected == "ass":
                _parse_ass(subs, string, fmt="ass")
            elif detected == "vtt":
                _parse_vtt(subs, string)
            elif detected == "microdvd":
                _parse_microdvd(subs, string, fps=fps)
            elif detected == "tmp":
                _parse_tmp(subs, string)
            else:
                _parse_srt(subs, string)

        return subs

    def save(self, path, encoding="utf-8", format_=None, fps=None, **kwargs):
        """Save subtitles to a file path."""
        if format_ is None:
            ext = os.path.splitext(path)[1].lower()
            format_ = FILE_EXTENSION_TO_FORMAT_IDENTIFIER.get(ext, self.format or "srt")
        with open(path, "w", encoding=encoding, errors="replace") as fp:
            self.to_file(fp, format_=format_, fps=fps, **kwargs)

    def to_file(self, fp, format_="srt", fps=None, **kwargs):
        """Write formatted subtitles to an open file-like object."""
        content = self.to_string(format_=format_, fps=fps, **kwargs)
        fp.write(content)

    def to_string(self, format_="srt", fps=None, **kwargs):
        """Format subtitles as a string in the requested format."""
        fmt = (format_ or self.format or "srt").lower()
        eff_fps = fps if fps is not None else self.fps

        if fmt == "srt":
            return _format_srt(self)
        elif fmt in ("ass", "ssa"):
            return _format_ass(self, fmt=fmt)
        elif fmt == "vtt":
            return _format_vtt(self)
        elif fmt == "microdvd":
            return _format_microdvd(self, fps=eff_fps)
        elif fmt == "tmp":
            return _format_tmp(self)
        else:
            raise UnknownFormatIdentifierError(f"Unknown subtitle format: {format_!r}")

    def __repr__(self):
        if self.events:
            min_t = format_srt_timestamp(min(e.start for e in self.events))
            max_t = format_srt_timestamp(max(e.end for e in self.events))
            return f"<SSAFile with {len(self.events)} events ({min_t} to {max_t})>"
        return "<SSAFile with 0 events>"


def _autodetect_format(text):
    """Detect subtitle format from text content."""
    head = text[:4096].lstrip()
    if head.startswith("WEBVTT"):
        return "vtt"
    if "[Script Info]" in head or "[V4+ Styles]" in head or "[V4 Styles]" in head or "\nDialogue:" in head or head.startswith("Dialogue:"):
        if "[V4 Styles]" in head and "[V4+ Styles]" not in head:
            return "ssa"
        return "ass"
    if re.search(r"^\s*\{\d+\}\{\d+\}", head, re.MULTILINE):
        return "microdvd"
    if "-->" in head:
        return "srt"
    if re.search(r"^\s*\d{1,2}:\d{2}:\d{2}:", head, re.MULTILINE):
        return "tmp"
    return "srt"


_SRT_TIMING_LINE_RE = re.compile(
    r"^\s*((?:\d+:)?\d{1,2}:\d{1,2}[,.]\d{1,3})\s*-->\s*((?:\d+:)?\d{1,2}:\d{1,2}[,.]\d{1,3})(?:\s+.*)?$"
)


def _parse_srt(subs, text):
    """Parse SubRip (.srt) text into SSAFile."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        m = _SRT_TIMING_LINE_RE.match(line)
        if not m and i + 1 < n:
            # Usually line i is the numeric index and line i+1 is the timing line
            m_next = _SRT_TIMING_LINE_RE.match(lines[i + 1].strip())
            if m_next:
                i += 1
                m = m_next

        if m:
            try:
                start_ms = parse_srt_timestamp(m.group(1))
                end_ms = parse_srt_timestamp(m.group(2))
            except ValueError:
                i += 1
                continue

            i += 1
            text_lines = []
            while i < n and lines[i].strip() != "":
                # Check if a malformed SRT omitted a blank line before the next cue
                if (
                    lines[i].strip().isdigit()
                    and i + 1 < n
                    and _SRT_TIMING_LINE_RE.match(lines[i + 1].strip())
                ):
                    break
                if _SRT_TIMING_LINE_RE.match(lines[i].strip()):
                    break
                text_lines.append(lines[i])
                i += 1

            ev_text = r"\N".join(text_lines)
            subs.append(SSAEvent(start=start_ms, end=end_ms, text=ev_text))
        else:
            i += 1


def _format_srt(subs):
    """Serialize SSAFile to SubRip (.srt) string."""
    visible = [ev for ev in subs.events if not ev.is_comment and not ev.is_drawing]
    visible.sort(key=lambda e: (e.start, e.end))
    blocks = []
    for idx, ev in enumerate(visible, 1):
        start_str = format_srt_timestamp(max(0, ev.start))
        end_str = format_srt_timestamp(max(0, ev.end))
        # Convert ASS line breaks \N and \n to standard newlines and strip ASS override tags
        txt = _OVERRIDE_TAG_RE.sub("", ev.text)
        txt = txt.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
        blocks.append(f"{idx}\n{start_str} --> {end_str}\n{txt}\n")
    return "\n".join(blocks)


def _parse_vtt(subs, text):
    """Parse WebVTT (.vtt) text into SSAFile."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    n = len(lines)
    i = 0
    # Skip WEBVTT header
    if i < n and lines[i].strip().startswith("WEBVTT"):
        i += 1
        while i < n and lines[i].strip() != "":
            i += 1

    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("NOTE") or line.startswith("STYLE"):
            i += 1
            while i < n and lines[i].strip() != "":
                i += 1
            continue

        m = _SRT_TIMING_LINE_RE.match(line)
        if not m and i + 1 < n:
            m_next = _SRT_TIMING_LINE_RE.match(lines[i + 1].strip())
            if m_next:
                i += 1
                m = m_next

        if m:
            try:
                start_ms = parse_srt_timestamp(m.group(1))
                end_ms = parse_srt_timestamp(m.group(2))
            except ValueError:
                i += 1
                continue
            i += 1
            text_lines = []
            while i < n and lines[i].strip() != "":
                if _SRT_TIMING_LINE_RE.match(lines[i].strip()):
                    break
                text_lines.append(lines[i])
                i += 1
            ev_text = r"\N".join(text_lines)
            subs.append(SSAEvent(start=start_ms, end=end_ms, text=ev_text))
        else:
            i += 1


def _format_vtt(subs):
    """Serialize SSAFile to WebVTT (.vtt) string."""
    visible = [ev for ev in subs.events if not ev.is_comment and not ev.is_drawing]
    visible.sort(key=lambda e: (e.start, e.end))
    blocks = ["WEBVTT\n"]
    for idx, ev in enumerate(visible, 1):
        start_str = format_vtt_timestamp(max(0, ev.start))
        end_str = format_vtt_timestamp(max(0, ev.end))
        txt = _OVERRIDE_TAG_RE.sub("", ev.text)
        txt = txt.replace(r"\N", "\n").replace(r"\n", "\n").replace(r"\h", " ")
        blocks.append(f"{idx}\n{start_str} --> {end_str}\n{txt}\n")
    return "\n".join(blocks)


def _parse_ass(subs, text, fmt="ass"):
    """Parse ASS/SSA text into SSAFile."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    section = None
    event_format = [
        "Layer" if fmt == "ass" else "Marked",
        "Start", "End", "Style", "Name",
        "MarginL", "MarginR", "MarginV", "Effect", "Text",
    ]

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            continue

        if section == "script info":
            if ":" in line:
                k, v = line.split(":", 1)
                subs.info[k.strip()] = v.strip()
        elif section == "aegisub project garbage":
            if ":" in line:
                k, v = line.split(":", 1)
                subs.aegisub_project[k.strip()] = v.strip()
        elif section in ("v4+ styles", "v4 styles"):
            if line.lower().startswith("style:"):
                rest = line.split(":", 1)[1].strip()
                parts = [p.strip() for p in rest.split(",")]
                if parts:
                    st_name = parts[0]
                    st = SSAStyle()
                    if len(parts) > 1:
                        st.fontname = parts[1]
                    if len(parts) > 2:
                        try:
                            st.fontsize = float(parts[2])
                        except ValueError:
                            pass
                    st._raw_line = rest
                    subs.styles[st_name] = st
        elif section == "events" or section is None:
            low = line.lower()
            if low.startswith("format:"):
                rest = line.split(":", 1)[1]
                event_format = [f.strip() for f in rest.split(",")]
            elif low.startswith("dialogue:") or low.startswith("comment:"):
                ev_type = "Dialogue" if low.startswith("dialogue:") else "Comment"
                rest = raw_line.split(":", 1)[1].lstrip()
                num_fields = len(event_format)
                parts = rest.split(",", num_fields - 1)
                if len(parts) < num_fields:
                    continue
                field_map = {
                    event_format[idx].lower(): parts[idx].strip() if idx < num_fields - 1 else parts[idx]
                    for idx in range(num_fields)
                }
                try:
                    start_ms = parse_ass_timestamp(field_map.get("start", "0:00:00.00"))
                    end_ms = parse_ass_timestamp(field_map.get("end", "0:00:00.00"))
                except ValueError:
                    continue
                layer_val = _safe_int(field_map.get("layer", 0), 0)
                marked_str = field_map.get("marked", "0")
                marked_val = marked_str.endswith("1")
                ev = SSAEvent(
                    start=start_ms,
                    end=end_ms,
                    text=field_map.get("text", ""),
                    style=field_map.get("style", "Default") or "Default",
                    type=ev_type,
                    effect=field_map.get("effect", ""),
                    name=field_map.get("name", ""),
                    marginl=_safe_int(field_map.get("marginl", 0), 0),
                    marginr=_safe_int(field_map.get("marginr", 0), 0),
                    marginv=_safe_int(field_map.get("marginv", 0), 0),
                    layer=layer_val,
                    marked=marked_val,
                )
                subs.append(ev)


def _format_ass(subs, fmt="ass"):
    """Serialize SSAFile to ASS or SSA string."""
    is_ssa = fmt.lower() == "ssa"
    out = ["[Script Info]", "; Script generated by subsync"]
    info = dict(SSAFile.DEFAULT_INFO)
    info.update(subs.info)
    info["ScriptType"] = "v4.00" if is_ssa else "v4.00+"
    for k, v in info.items():
        out.append(f"{k}: {v}")
    out.append("")

    if subs.aegisub_project:
        out.append("[Aegisub Project Garbage]")
        for k, v in subs.aegisub_project.items():
            out.append(f"{k}: {v}")
        out.append("")

    if is_ssa:
        out.append("[V4 Styles]")
        out.append(
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "TertiaryColour, BackColour, Bold, Italic, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, AlphaLevel, Encoding"
        )
        for name, st in subs.styles.items():
            out.append(
                f"Style: {name},{st.fontname},{int(st.fontsize)},16777215,255,0,0,"
                f"{-1 if st.bold else 0},{-1 if st.italic else 0},{st.borderstyle},"
                f"{int(st.outline)},{int(st.shadow)},{st.alignment},"
                f"{st.marginl},{st.marginr},{st.marginv},{st.alphalevel},{st.encoding}"
            )
    else:
        out.append("[V4+ Styles]")
        out.append(
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding"
        )
        for name, st in subs.styles.items():
            if getattr(st, "_raw_line", None):
                out.append(f"Style: {st._raw_line}")
            else:
                out.append(
                    f"Style: {name},{st.fontname},{int(st.fontsize)},"
                    f"&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
                    f"{-1 if st.bold else 0},{-1 if st.italic else 0},"
                    f"{-1 if st.underline else 0},{-1 if st.strikeout else 0},"
                    f"{int(st.scalex)},{int(st.scaley)},{int(st.spacing)},{int(st.angle)},"
                    f"{st.borderstyle},{int(st.outline)},{int(st.shadow)},{st.alignment},"
                    f"{st.marginl},{st.marginr},{st.marginv},{st.encoding}"
                )
    out.append("")

    out.append("[Events]")
    if is_ssa:
        out.append("Format: Marked, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")
    else:
        out.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    for ev in subs.events:
        first_col = f"Marked={'1' if ev.marked else '0'}" if is_ssa else str(ev.layer)
        start_s = format_ass_timestamp(max(0, ev.start))
        end_s = format_ass_timestamp(max(0, ev.end))
        txt = ev.text.replace("\n", r"\N")
        out.append(
            f"{ev.type}: {first_col},{start_s},{end_s},{ev.style},{ev.name},"
            f"{ev.marginl},{ev.marginr},{ev.marginv},{ev.effect},{txt}"
        )
    out.append("")
    return "\n".join(out)


_MICRODVD_RE = re.compile(r"^\s*\{(\d+)\}\{(\d+)\}(.*)$")


def _parse_microdvd(subs, text, fps=None):
    """Parse MicroDVD (.sub) text into SSAFile."""
    eff_fps = float(fps) if fps and fps > 0 else 23.976
    subs.fps = eff_fps
    first_event = True
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = _MICRODVD_RE.match(raw_line)
        if not m:
            continue
        f_start = int(m.group(1))
        f_end = int(m.group(2))
        body = m.group(3).strip()
        if first_event and f_start <= 1 and f_end <= 1 and fps is None:
            try:
                cand_fps = float(body)
                if 10.0 <= cand_fps <= 120.0:
                    eff_fps = cand_fps
                    subs.fps = eff_fps
                    first_event = False
                    continue
            except ValueError:
                pass
        first_event = False
        start_ms = int(round(f_start * 1000.0 / eff_fps))
        end_ms = int(round(f_end * 1000.0 / eff_fps))
        ev_text = body.replace("|", r"\N")
        subs.append(SSAEvent(start=start_ms, end=end_ms, text=ev_text))


def _format_microdvd(subs, fps=None):
    """Serialize SSAFile to MicroDVD (.sub) string."""
    eff_fps = float(fps) if fps and fps > 0 else (subs.fps if subs.fps and subs.fps > 0 else 23.976)
    lines = []
    for ev in subs.events:
        if ev.is_comment or ev.is_drawing:
            continue
        f_start = max(0, int(round(ev.start * eff_fps / 1000.0)))
        f_end = max(0, int(round(ev.end * eff_fps / 1000.0)))
        txt = _OVERRIDE_TAG_RE.sub("", ev.text)
        txt = txt.replace(r"\N", "|").replace(r"\n", "|").replace("\n", "|")
        lines.append(f"{{{f_start}}}{{{f_end}}}{txt}")
    return "\n".join(lines) + ("\n" if lines else "")


_TMP_RE = re.compile(r"^\s*(\d{1,2}):(\d{2}):(\d{2}):(.*)$")


def _parse_tmp(subs, text):
    """Parse TMP (HH:MM:SS:text) subtitles into SSAFile."""
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        m = _TMP_RE.match(raw_line)
        if not m:
            continue
        h, mi, s, body = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        start_ms = time_to_ms(h=h, m=mi, s=s)
        end_ms = start_ms + 3000
        ev_text = body.replace("|", r"\N")
        subs.append(SSAEvent(start=start_ms, end=end_ms, text=ev_text))


def _format_tmp(subs):
    """Serialize SSAFile to TMP (HH:MM:SS:text) string."""
    lines = []
    for ev in subs.events:
        if ev.is_comment or ev.is_drawing:
            continue
        h, mi, s, _ = ms_to_times(max(0, ev.start))
        txt = _OVERRIDE_TAG_RE.sub("", ev.text)
        txt = txt.replace(r"\N", "|").replace(r"\n", "|").replace("\n", "|")
        lines.append(f"{h:02d}:{mi:02d}:{s:02d}:{txt}")
    return "\n".join(lines) + ("\n" if lines else "")


load = SSAFile.load
from_file = SSAFile.from_file
from_string = SSAFile.from_string

# Submodule namespaces expected by callers (e.g. pysubs2.formats, pysubs2.exceptions)
formats = types.ModuleType("pysubs2.formats")
formats.FILE_EXTENSION_TO_FORMAT_IDENTIFIER = FILE_EXTENSION_TO_FORMAT_IDENTIFIER
formats.FORMAT_IDENTIFIER_TO_FILE_EXTENSION = FORMAT_IDENTIFIER_TO_FILE_EXTENSION

exceptions = types.ModuleType("pysubs2.exceptions")
exceptions.Pysubs2Error = Pysubs2Error
exceptions.UnknownFPSError = UnknownFPSError
exceptions.UnknownFormatIdentifierError = UnknownFormatIdentifierError
exceptions.FormatAutodetectionError = FormatAutodetectionError

time = types.ModuleType("pysubs2.time")
time.make_time = make_time
time.time_to_ms = time_to_ms
time.ms_to_times = ms_to_times
time.TIMESTAMP = _SRT_TS_RE


def register_as_pysubs2():
    """Register this module as `pysubs2` in sys.modules if native pysubs2 is missing."""
    if "pysubs2" not in sys.modules:
        try:
            __import__("pysubs2")
        except ImportError:
            mod = sys.modules[__name__]
            sys.modules["pysubs2"] = mod
            sys.modules["pysubs2.formats"] = formats
            sys.modules["pysubs2.exceptions"] = exceptions
            sys.modules["pysubs2.time"] = time
    return sys.modules["pysubs2"]


register_as_pysubs2()
