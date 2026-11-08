"""Modern Python/FFmpeg synchronization engine and gizmo compatibility layer.

Provides:
- `SyncResult` and `synchronize_pair(sub_path, ref_path, out_path=None, ...)`
  for direct, high-precision (<= 0.1s) synchronization of subtitle and video/audio/subtitle pairs.
- A pure-Python `gizmo`-compatible shim registered in `sys.modules['gizmo']` whenever
  the legacy C++ `gizmo` extension is not compiled.
"""

from dataclasses import dataclass
import logging
import os
import sys
import types
from typing import List, Optional, Tuple

from subsync import compat_pysubs2 as pysubs2
from subsync.synchro import aligner, encdetect, vad
from subsync.synchro.aligner import AlignmentResult, FormulaLine

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Result returned by `synchronize_pair`."""

    success: bool
    correlated: bool
    offset_s: float
    speed_factor: float = 1.0
    factor: float = 0.0
    raw_factor: float = 0.0
    post_factor: float = 0.0
    raw_vocal_score: float = 0.0
    post_vocal_score: float = 0.0
    points: int = 0
    max_distance: float = 0.0
    max_change: float = 0.0
    formula: Optional[FormulaLine] = None
    window_offsets: Optional[List] = None
    refined_durations_count: int = 0
    sub_path: Optional[str] = None
    ref_path: Optional[str] = None
    out_path: Optional[str] = None
    subtitles: Optional[pysubs2.SSAFile] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.formula is None:
            self.formula = FormulaLine(self.speed_factor, self.offset_s)
        if self.post_factor <= 0.0 and self.factor > 0.0:
            self.post_factor = self.factor

    @property
    def correlation_score(self) -> float:
        return self.factor

    @property
    def a(self) -> float:
        return self.speed_factor

    @property
    def b(self) -> float:
        return self.offset_s

    @property
    def maxDistance(self) -> float:
        return self.max_distance

    @property
    def maxChange(self) -> float:
        return self.max_change


def load_subtitles_from_source(
    path: str,
    stream_index: Optional[int] = None,
    encoding: Optional[str] = None,
    lang: Optional[str] = None,
    fps: Optional[float] = None,
):
    """Load subtitles into a `subsync.subtitle.Subtitles` instance from a file or container stream."""
    from subsync import subtitle

    if not os.path.exists(path):
        raise FileNotFoundError(f"Subtitle source not found: {path!r}")

    if vad.is_subtitle_file_path(path):
        enc = encoding
        if not enc:
            try:
                enc = encdetect.detectEncoding(path, lang) or "utf-8"
            except Exception:
                enc = "utf-8"
        loaded = pysubs2.SSAFile.load(path, encoding=enc, fps=fps)
    else:
        loaded = vad.extract_embedded_subtitles(path, stream_index=stream_index)

    subs = subtitle.Subtitles()
    subs.events = list(loaded.events)
    subs.styles = dict(loaded.styles)
    subs.info = dict(loaded.info)
    subs.aegisub_project = dict(loaded.aegisub_project)
    subs.fps = loaded.fps or fps
    subs.format = loaded.format
    subs.sort()
    return subs


def compute_envelope_sync_metrics(
    intervals: List[Tuple[float, float]],
    ref_envelope: List[float],
    frame_hz: int = 100,
) -> Tuple[float, float]:
    """Compute `(ncc_score, vocal_hit_rate)` in `[0.0, 1.0]` for subtitle `intervals` against `ref_envelope`."""
    import numpy as np

    if not intervals or not ref_envelope:
        return 0.0, 0.0
    arr = np.asarray(ref_envelope, dtype=np.float32)
    n = len(arr)
    if n < 10:
        return 0.0, 0.0

    sub_env = np.zeros(n, dtype=np.float32)
    voice_hit = 0
    valid_subs = 0
    margin_frames = max(1, int(round(0.10 * frame_hz)))

    for s, e in intervals:
        if e <= 0.0 or e <= s:
            continue
        i0 = max(0, min(n, int(round(s * frame_hz))))
        i1 = max(0, min(n, int(round(e * frame_hz))))
        if i1 <= i0:
            continue
        sub_env[i0:i1] = 1.0
        valid_subs += 1
        seg = arr[i0:i1]
        m0 = max(0, i0 - margin_frames)
        m1 = min(n, i1 + margin_frames)
        seg_m = arr[m0:m1]
        if float(np.max(seg_m)) >= 0.35 and float(np.mean(seg)) >= 0.18:
            voice_hit += 1

    s_mean = sub_env - float(np.mean(sub_env))
    r_mean = arr - float(np.mean(arr))
    denom = float(np.linalg.norm(s_mean) * np.linalg.norm(r_mean))
    ncc = max(0.0, min(1.0, float(np.dot(s_mean, r_mean) / denom))) if denom > 1e-9 else 0.0
    hit_rate = (float(voice_hit) / float(valid_subs)) if valid_subs > 0 else 0.0
    return round(ncc, 4), round(hit_rate, 4)


def synchronize_pair(
    sub_path: str,
    ref_path: str,
    out_path: Optional[str] = None,
    sub_lang: Optional[str] = None,
    ref_lang: Optional[str] = None,
    max_offset_s: float = 600.0,
    allow_drift: bool = True,
    *,
    sub_stream: Optional[int] = None,
    ref_stream: Optional[int] = None,
    ref_stream_type: Optional[str] = None,
    encoding: Optional[str] = None,
    out_encoding: str = "utf-8",
    out_Time_offset: float = 0.0,
    overwrite: bool = True,
    fps: Optional[float] = None,
) -> SyncResult:
    """Synchronize `sub_path` against `ref_path` (video, audio, or reference subtitle).

    Parameters
    ----------
    sub_path : str
        Path to input subtitle file (or video container with subtitle stream).
    ref_path : str
        Path to reference video, audio, or subtitle file.
    out_path : str, optional
        If provided, saves the synchronized subtitles to `out_path`.
    sub_lang : str, optional
        Subtitle language code (optional; engine is language-agnostic).
    ref_lang : str, optional
        Reference language code (optional; engine is language-agnostic).
    max_offset_s : float, optional
        Maximum absolute offset search range in seconds (default 600.0).
    allow_drift : bool, optional
        Whether to evaluate framerate conversion / linear speed drift `a` (default True).

    Returns
    -------
    SyncResult
        Synchronization result including `formula`, `offset_s`, `speed_factor`,
        `subtitles` (synchronized `Subtitles` object), and saved `out_path`.
    """
    try:
        subs = load_subtitles_from_source(
            sub_path,
            stream_index=sub_stream,
            encoding=encoding,
            lang=sub_lang,
            fps=fps,
        )
        sub_intervals = vad.subtitles_to_intervals(subs)
        if not sub_intervals:
            return SyncResult(
                success=False,
                correlated=False,
                offset_s=0.0,
                speed_factor=1.0,
                sub_path=sub_path,
                ref_path=ref_path,
                out_path=out_path,
                subtitles=subs,
                error="No dialogue events found in input subtitles",
            )

        ref_envelope, ref_intervals = vad.extract_speech_envelope(
            ref_path,
            stream_index=ref_stream,
            stream_type=ref_stream_type,
            fps=fps,
            frame_hz=vad.FRAME_HZ,
        )
        if not ref_envelope:
            return SyncResult(
                success=False,
                correlated=False,
                offset_s=0.0,
                speed_factor=1.0,
                sub_path=sub_path,
                ref_path=ref_path,
                out_path=out_path,
                subtitles=subs,
                error="No speech or subtitle activity found in reference file",
            )

        align_res: AlignmentResult = aligner.align_intervals_and_envelopes(
            sub_intervals=sub_intervals,
            ref_envelope=ref_envelope,
            ref_intervals=ref_intervals,
            frame_hz=vad.FRAME_HZ,
            max_offset_s=max_offset_s,
            allow_drift=allow_drift,
        )

        synced_subs = subs.synchronize(align_res.formula)
        refined_count = 0
        if align_res.correlated and len(ref_envelope) >= int(300.0 * vad.FRAME_HZ):
            try:
                synced_subs.verify_and_heal_with_vad(
                    ref_envelope,
                    ref_intervals=ref_intervals,
                    frame_hz=vad.FRAME_HZ,
                )
            except Exception as ver_exc:
                logger.debug("Stage 4 verification & self-healing skipped: %s", ver_exc)
            try:
                refined_count = synced_subs.refine_durations_with_vad(
                    ref_envelope,
                    ref_intervals=ref_intervals,
                    frame_hz=vad.FRAME_HZ,
                )
            except Exception as ref_exc:
                logger.debug("Stage 3 duration refinement skipped: %s", ref_exc)
        if out_Time_offset:
            synced_subs.shift(s=out_Time_offset)

        raw_ncc, raw_vocal = compute_envelope_sync_metrics(
            sub_intervals, ref_envelope, frame_hz=vad.FRAME_HZ
        )
        synced_intervals = vad.subtitles_to_intervals(synced_subs)
        post_ncc, post_vocal = compute_envelope_sync_metrics(
            synced_intervals, ref_envelope, frame_hz=vad.FRAME_HZ
        )
        effective_post_factor = max(float(align_res.factor), float(post_ncc))

        saved_path = None
        if out_path and align_res.correlated:
            out_dir = os.path.dirname(os.path.abspath(out_path))
            if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            saved_path = synced_subs.save(
                out_path,
                encoding=out_encoding or "utf-8",
                fps=fps,
                overwrite=overwrite,
            )

        return SyncResult(
            success=bool(align_res.correlated),
            correlated=bool(align_res.correlated),
            offset_s=align_res.b,
            speed_factor=align_res.a,
            factor=effective_post_factor,
            raw_factor=raw_ncc,
            post_factor=effective_post_factor,
            raw_vocal_score=raw_vocal,
            post_vocal_score=post_vocal,
            points=align_res.points,
            max_distance=align_res.max_distance,
            max_change=align_res.max_change,
            formula=align_res.formula,
            window_offsets=align_res.window_offsets,
            refined_durations_count=refined_count,
            sub_path=sub_path,
            ref_path=ref_path,
            out_path=saved_path or out_path,
            subtitles=synced_subs,
        )

    except Exception as exc:
        logger.warning("synchronize_pair failed for sub=%r ref=%r: %s", sub_path, ref_path, exc, exc_info=True)
        return SyncResult(
            success=False,
            correlated=False,
            offset_s=0.0,
            speed_factor=1.0,
            sub_path=sub_path,
            ref_path=ref_path,
            out_path=out_path,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Pure-Python `gizmo` Compatibility Shim
# ---------------------------------------------------------------------------

_CHANNEL_DEFINITIONS = [
    (0x00000001, "FL", "front left"),
    (0x00000002, "FR", "front right"),
    (0x00000004, "FC", "front center"),
    (0x00000008, "LFE", "low frequency"),
    (0x00000010, "BL", "back left"),
    (0x00000020, "BR", "back right"),
    (0x00000040, "FLC", "front left-of-center"),
    (0x00000080, "FRC", "front right-of-center"),
    (0x00000100, "BC", "back center"),
    (0x00000200, "SL", "side left"),
    (0x00000400, "SR", "side right"),
    (0x00000800, "TC", "top center"),
    (0x00001000, "TFL", "top front left"),
    (0x00002000, "TFC", "top front center"),
    (0x00004000, "TFR", "top front right"),
    (0x00008000, "TBL", "top back left"),
    (0x00010000, "TBC", "top back center"),
    (0x00020000, "TBR", "top back right"),
]

_ID_TO_CHANNEL_NAME = {cid: name for cid, name, _ in _CHANNEL_DEFINITIONS}
_ID_TO_CHANNEL_DESC = {cid: desc for cid, _, desc in _CHANNEL_DEFINITIONS}
_NAME_TO_CHANNEL_ID = {name.upper(): cid for cid, name, _ in _CHANNEL_DEFINITIONS}


class GizmoError(Exception):
    """Exception compatible with `gizmo.Error`."""


class GizmoErrorTerminated(GizmoError):
    """Exception compatible with `gizmo.ErrorTerminated`."""


class AVSampleFormat(object):
    NONE = -1
    U8 = 0
    S16 = 1
    S32 = 2
    FLT = 3
    DBL = 4
    U8P = 5
    S16P = 6
    S32P = 7
    FLTP = 8
    DBLP = 9
    S64 = 10
    S64P = 11


class AudioFormat(object):
    """Audio stream format compatible with `gizmo.AudioFormat`."""

    def __init__(self, sampleFormat=AVSampleFormat.S16, sampleRate=16000, channelLayout=0, channelsNo=0):
        self.sampleFormat = sampleFormat
        self.sampleRate = int(sampleRate)
        self.channelLayout = int(channelLayout)
        if channelsNo:
            self.channelsNo = int(channelsNo)
        else:
            cnt = 0
            mask = self.channelLayout
            while mask > 0:
                cnt += mask & 1
                mask >>= 1
            self.channelsNo = max(1, cnt) if self.sampleRate > 0 else 0

    def getSampleSize(self):
        return 2

    def getLayoutString(self):
        names = [
            name for cid, name, _ in _CHANNEL_DEFINITIONS if self.channelLayout & cid
        ]
        return ",".join(names) if names else "EMPTY"

    def toString(self):
        return (
            f"<AudioFormat sampleFormat=\"S16\", sampleRate={self.sampleRate}, "
            f"channelsNo={self.channelsNo}, channelLayout={self.getLayoutString()}>"
        )

    def __repr__(self):
        return self.toString()

    @staticmethod
    def getChannelName(ch_id):
        try:
            return _ID_TO_CHANNEL_NAME.get(int(ch_id))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def getChannelDescription(ch_id):
        try:
            return _ID_TO_CHANNEL_DESC.get(int(ch_id))
        except (ValueError, TypeError):
            return None

    @staticmethod
    def getChannelIdByName(name):
        if not name:
            return 0
        return _NAME_TO_CHANNEL_ID.get(str(name).strip().upper(), 0)


class StreamFormat(object):
    """Media stream descriptor compatible with `gizmo.StreamFormat`."""

    def __init__(
        self,
        no=0,
        type="subtitle/text",
        codec="",
        lang="",
        title="",
        frameRate=0.0,
        audio=None,
    ):
        self.no = int(no)
        self.type = str(type)
        self.codec = str(codec or "")
        self.lang = str(lang or "")
        self.title = str(title or "")
        self.frameRate = float(frameRate or 0.0)
        self.audio = audio or AudioFormat(AVSampleFormat.NONE, 0, 0, 0)

    def toString(self):
        parts = [f"<StreamFormat no={self.no}", f'type="{self.type}"']
        if self.codec:
            parts.append(f'codec="{self.codec}"')
        if self.lang:
            parts.append(f'lang="{self.lang}"')
        if self.title:
            parts.append(f'title="{self.title}"')
        if self.frameRate > 0.0:
            parts.append(f"frameRate={self.frameRate}")
        if self.type == "audio":
            parts.append(f"audio={self.audio.toString()}")
        return ", ".join(parts) + ">"

    def __repr__(self):
        return self.toString()


class CorrelationStats(object):
    """Correlation statistics compatible with `gizmo.CorrelationStats`."""

    def __init__(
        self,
        correlated: bool = False,
        factor: float = 0.0,
        points: int = 0,
        maxDistance: float = 0.0,
        formula: Optional[FormulaLine] = None,
    ):
        self.correlated = bool(correlated)
        self.factor = float(factor)
        self.points = int(points)
        self.maxDistance = float(maxDistance)
        self.formula = formula if formula is not None else FormulaLine(1.0, 0.0)

    def toString(self, prefix="<CorrelationStats ", suffix=">") -> str:
        corr_str = "true" if self.correlated else "false"
        return (
            f"{prefix}correlated={corr_str}, factor={100.0 * self.factor:.3f}%, "
            f"points={self.points}, maxDist={self.maxDistance:.3f}, "
            f"formula={self.formula.toString()}{suffix}"
        )

    def __repr__(self) -> str:
        return self.toString()

    def __str__(self) -> str:
        return self.toString("", "")


class Demux(object):
    """Container demuxer compatible with `gizmo.Demux` backed by ffprobe."""

    def __init__(self, fileName: str, runCb=None):
        self.fileName = fileName
        self.runCb = runCb
        self._position = 0.0
        try:
            info = vad.probe_media_info(fileName)
        except Exception as exc:
            raise GizmoError(f"Cannot open media file {fileName!r}: {exc}") from exc

        self._duration = float(info.get("duration", 0.0) or 0.0)
        self._streams: List[StreamFormat] = []
        for st in info.get("streams", []):
            channels = int(st.get("channels") or 0)
            sample_rate = int(st.get("sample_rate") or 0)
            if channels == 1:
                layout = 0x4  # FC
            elif channels == 2:
                layout = 0x3  # FL | FR
            elif channels >= 6:
                layout = 0x3F  # 5.1 surround (includes FC=0x4)
            else:
                layout = (1 << channels) - 1 if channels > 0 else 0
            audio_fmt = AudioFormat(
                AVSampleFormat.S16 if st.get("type") == "audio" else AVSampleFormat.NONE,
                sample_rate,
                layout,
                channelsNo=channels,
            )
            self._streams.append(
                StreamFormat(
                    no=st.get("index", 0),
                    type=st.get("type", "other"),
                    codec=st.get("codec_name", ""),
                    lang=st.get("lang", ""),
                    title=st.get("title", ""),
                    frameRate=st.get("frame_rate", 0.0),
                    audio=audio_fmt,
                )
            )

    def getStreamsInfo(self) -> List[StreamFormat]:
        return list(self._streams)

    def getDuration(self) -> float:
        return self._duration

    def getPosition(self) -> float:
        return self._position

    def seek(self, pos: float):
        self._position = max(0.0, float(pos))

    def connectDec(self, dec, stream_no: int):
        pass

    def disconnectDec(self, stream_no: int):
        pass

    def disconnectAllDec(self):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def step(self) -> bool:
        return False

    def notifyDiscontinuity(self):
        pass


class Word(object):
    """Word token compatible with `gizmo.Word`."""

    def __init__(self, text="", time=0.0, duration=0.0, score=1.0):
        self.text = str(text)
        self.time = float(time)
        self.duration = float(duration)
        self.score = float(score)

    def __repr__(self):
        return f"<Word {self.text!r} @{self.time:.3f}s>"


class Dictionary(object):
    """In-memory bilingual dictionary compatible with `gizmo.Dictionary`."""

    def __init__(self):
        self._entries = {}

    def add(self, key: str, val: str):
        self._entries.setdefault(key, []).append(val)

    def size(self) -> int:
        return sum(len(v) for v in self._entries.values())


class Translator(object):
    """Translator compatible with `gizmo.Translator`."""

    def __init__(self, dictionary=None):
        self.dictionary = dictionary
        self._listeners = []
        self.minWordsSim = 0.6

    def setMinWordsSim(self, sim: float):
        self.minWordsSim = float(sim)

    def addWordsListener(self, listener):
        if listener not in self._listeners:
            self._listeners.append(listener)

    def removeWordsListener(self, listener=None):
        if listener is None:
            self._listeners.clear()
        elif listener in self._listeners:
            self._listeners.remove(listener)

    def pushWord(self, word):
        for cb in self._listeners:
            cb(word)


class _DummyNode(object):
    def __init__(self, *args, **kwargs):
        self._words_listeners = []
        self._subs_listeners = []

    def setMinWordLen(self, *args, **kwargs):
        pass

    def setMinWordProb(self, *args, **kwargs):
        pass

    def setEncoding(self, *args, **kwargs):
        pass

    def setRightToLeft(self, *args, **kwargs):
        pass

    def setParam(self, *args, **kwargs):
        pass

    def connectOutput(self, *args, **kwargs):
        pass

    def connectFormatChangeCallback(self, *args, **kwargs):
        pass

    def setChannelMap(self, *args, **kwargs):
        pass

    def addWordsListener(self, cb):
        self._words_listeners.append(cb)

    def removeWordsListener(self, cb=None):
        if cb is None:
            self._words_listeners.clear()

    def addSubsListener(self, cb):
        self._subs_listeners.append(cb)

    def removeSubsListener(self, cb=None):
        if cb is None:
            self._subs_listeners.clear()

    def pushWord(self, word):
        for cb in self._words_listeners:
            cb(word)


class Extractor(object):
    def __init__(self, demux):
        self.demux = demux
        self._running = False
        self._eos_cb = None
        self._err_cb = None

    def selectEndTime(self, end):
        pass

    def start(self, threadName=None):
        self._running = False
        if self._eos_cb:
            self._eos_cb()

    def stop(self):
        self._running = False

    def wait(self):
        pass

    def isRunning(self):
        return self._running

    def connectEosCallback(self, cb):
        self._eos_cb = cb

    def connectErrorCallback(self, cb):
        self._err_cb = cb


class Correlator(object):
    """Fallback correlator compatible with `gizmo.Correlator`."""

    def __init__(self, windowSize=1800.0, minCorrelation=0.9999, maxDist=2.0, minPointsNo=20, minWordsSim=0.6):
        self.windowSize = windowSize
        self.minCorrelation = minCorrelation
        self.maxDist = maxDist
        self.minPointsNo = minPointsNo
        self.minWordsSim = minWordsSim
        self._stats_cb = None
        self._running = False
        self._sub_intervals = []
        self._stats = CorrelationStats()

    def connectStatsCallback(self, cb):
        self._stats_cb = cb

    def start(self, threadName=""):
        self._running = False

    def stop(self, force=False):
        self._running = False

    def wait(self):
        pass

    def isRunning(self):
        return self._running

    def getProgress(self):
        return 1.0

    def pushSubWord(self, word):
        pass

    def pushRefWord(self, word):
        pass

    def pushSubtitle(self, start_s, end_s):
        self._sub_intervals.append((float(start_s), float(end_s)))

    def getSubs(self):
        return []

    def getRefs(self):
        return []

    def getAllPoints(self):
        return []

    def getUsedPoints(self):
        return []


def setDebugLevel(level: int):
    """No-op compatibility function for `gizmo.setDebugLevel`."""
    pass


def install_gizmo_shim():
    """Register a pure-Python `gizmo` module in `sys.modules` if native `gizmo` is absent."""
    existing = sys.modules.get("gizmo")
    if existing is not None and hasattr(existing, "AudioFormat") and hasattr(existing, "Demux"):
        return existing

    if existing is None:
        try:
            import gizmo as native_gizmo
            if hasattr(native_gizmo, "AudioFormat"):
                return native_gizmo
        except ImportError:
            pass
        mod = types.ModuleType("gizmo")
    else:
        mod = existing

    mod.__doc__ = "Pure-Python compatibility shim for gizmo"
    mod._IS_PYTHON_SHIM = True
    mod.Error = getattr(mod, "Error", GizmoError)
    mod.ErrorTerminated = getattr(mod, "ErrorTerminated", GizmoErrorTerminated)
    mod.setDebugLevel = setDebugLevel
    mod.Line = FormulaLine
    mod.CorrelationStats = CorrelationStats
    mod.AudioFormat = AudioFormat
    mod.AVSampleFormat = AVSampleFormat
    mod.StreamFormat = StreamFormat
    mod.Demux = Demux
    mod.Word = Word
    mod.Dictionary = Dictionary
    mod.Translator = Translator
    mod.SubtitleDec = _DummyNode
    mod.AudioDec = _DummyNode
    mod.Resampler = _DummyNode
    mod.SpeechRecognition = _DummyNode
    mod.NgramSplitter = _DummyNode
    mod.Extractor = Extractor
    mod.Correlator = Correlator

    sys.modules["gizmo"] = mod
    return mod
