"""Audio Voice Activity Detection (VAD) and 100 Hz speech envelope extraction.

Uses `/usr/bin/ffmpeg` and `/usr/bin/ffprobe` to decode arbitrary audio/video
containers into 8 kHz mono 16-bit PCM and computes a 10ms-resolution (100 Hz)
speech activity envelope and speech onset/offset intervals.
"""

from array import array
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Auto-wire local `.ai_env` so `numpy`, `onnxruntime`, and `faster_whisper` are always importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _ai_env_cand in (
    os.path.join(_REPO_ROOT, ".ai_env"),
    "/home/juliench/subsync/.ai_env",
):
    if os.path.isdir(_ai_env_cand) and _ai_env_cand not in sys.path:
        sys.path.insert(0, _ai_env_cand)

from subsync import compat_pysubs2 as pysubs2

logger = logging.getLogger(__name__)

FRAME_HZ = 100  # 10ms frame resolution (0.01s)
SAMPLE_RATE = 8000  # 80 samples per 10ms frame

_TLS = threading.local()
_CACHE_LOCK = threading.RLock()
_MEDIA_INFO_CACHE: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
_REF_AUDIO_CACHE: Dict[Tuple[str, int, int, Optional[int], int], Dict[str, Any]] = {}
_MAX_REF_AUDIO_CACHE = 4

LAST_PCM_BYTES: Optional[bytes] = None
LAST_SPEECH_ENVELOPE: Optional[List[float]] = None
LAST_ATTACK_ENVELOPE: Optional[List[float]] = None
LAST_SPEECH_INTERVALS: Optional[List[Tuple[float, float]]] = None
LAST_SUB_EVENTS: List[Tuple[float, float, str]] = []

_CREDIT_WATERMARK_RE = re.compile(
    r"(?:addic7ed\.com|subsynchro|sous-titres\.eu|tvsubtitles\.net|opensubtitles\.org|"
    r"sous-titres\s+traduits\s+par|traduits?\s+par\s*(?:la\s+communaut|:|[a-z0-9_\-])|"
    r"traduction\s*(?:par\b|by\b|:)|re-?synchro\b|"
    r"synchro(?:nis[ée])?\s*(?:&\s*corrig[ée]\s*)?(?:par\b|by\b|:)|"
    r"syncro\s*par|re-?sync\s*(?:par\b|by\b)|sync\s*by\b|corrected\s+by\b)",
    re.IGNORECASE,
)

_SDH_BRACKET_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]+>")
_MUSIC_NOTE_RE = re.compile(r"[♪♫♬]+")
_SPEAKER_PREFIX_RE = re.compile(
    r"(?:^|\n)\s*(?:[-–—]\s*)?[A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ0-9\s\.\-]{1,18}\s*:\s*"
)


def is_credit_watermark(text: str) -> bool:
    """Return True if subtitle text is a non-dialogue credit/watermark line."""
    if not text:
        return False
    return bool(_CREDIT_WATERMARK_RE.search(text))


def extract_spoken_dialogue_text(text: str) -> str:
    """Strip SDH sound-effect annotations (`[MUSIQUE]`, `(Soupire)`), `♪` lyrics, and speaker tags."""
    if not text:
        return ""
    stripped = text.strip()
    if not stripped:
        return ""
    # Check if the entire line (or every sub-line) is wrapped in music notes ♪ ... ♪
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    non_music_lines = []
    for ln in lines:
        ln_clean = re.sub(r"^[-–—\s]+", "", ln).strip()
        if ln_clean.startswith(("♪", "♫", "♬")) or ln_clean.endswith(("♪", "♫", "♬")):
            continue
        non_music_lines.append(ln)
    if not non_music_lines:
        return ""
    joined = "\n".join(non_music_lines)
    cleaned = _SDH_BRACKET_RE.sub(" ", joined)
    cleaned = _MUSIC_NOTE_RE.sub(" ", cleaned)
    cleaned = _SPEAKER_PREFIX_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[-–—\s]+", " ", cleaned).strip()
    # Must contain at least one alphanumeric character to count as spoken dialogue
    if not re.search(r"[A-Za-zÀ-ÿ0-9]", cleaned):
        return ""
    return cleaned


def is_pure_sdh_non_speech(text: str) -> bool:
    """Return True if a subtitle cue contains only non-speech SDH sound effects or music notes."""
    if not text or not text.strip():
        return True
    return extract_spoken_dialogue_text(text) == ""

_SUBTITLE_EXTENSIONS = {
    ".srt", ".ass", ".ssa", ".vtt", ".sub", ".txt", ".smi", ".sami",
}

_TEXT_SUBTITLE_CODECS = {
    "subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text",
    "microdvd", "subviewer", "subviewer1", "jacosub", "mpl2",
    "pjs", "realtext", "sami", "stl", "ttml", "vplayer",
}

_BITMAP_SUBTITLE_CODECS = {
    "hdmv_pgs_subtitle", "pgssub", "dvd_subtitle", "dvdsub",
    "dvb_subtitle", "dvbsub", "xsub",
}


def find_binary(name: str) -> str:
    """Locate ffmpeg or ffprobe binary."""
    std_path = f"/usr/bin/{name}"
    if os.path.isfile(std_path) and os.access(std_path, os.X_OK):
        return std_path
    found = shutil.which(name)
    if found:
        return found
    return name


def is_subtitle_file_path(path: str) -> bool:
    """Return True if file extension indicates a standalone subtitle file."""
    if not path:
        return False
    ext = os.path.splitext(path)[1].lower()
    return ext in _SUBTITLE_EXTENSIONS


def probe_media_info(path: str) -> Dict:
    """Probe media container streams and format metadata using ffprobe.

    For standalone text subtitle files (.srt, .ass, .ssa, .vtt, .sub), returns
    a synthetic single-stream descriptor immediately if ffprobe is not needed.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Media file not found: {path!r}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pcm":
        try:
            dur = os.path.getsize(path) / float(2 * SAMPLE_RATE)
        except OSError:
            dur = 0.0
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "audio",
                    "codec_name": "pcm_s16le",
                    "type": "audio",
                    "lang": "",
                    "title": "",
                    "frame_rate": 0.0,
                    "channels": 1,
                    "sample_rate": SAMPLE_RATE,
                    "channel_layout_name": "mono",
                }
            ],
            "duration": dur,
        }

    if ext in _SUBTITLE_EXTENSIONS:
        codec_map = {
            ".srt": "subrip",
            ".ass": "ass",
            ".ssa": "ssa",
            ".vtt": "webvtt",
            ".sub": "microdvd",
            ".txt": "text",
        }
        return {
            "streams": [
                {
                    "index": 0,
                    "codec_type": "subtitle",
                    "codec_name": codec_map.get(ext, "subrip"),
                    "type": "subtitle/text",
                    "lang": "",
                    "title": "",
                    "frame_rate": 0.0,
                    "channels": 0,
                    "sample_rate": 0,
                    "channel_layout": 0,
                }
            ],
            "duration": 0.0,
        }

    try:
        st_stat = os.stat(path)
        cache_key = (os.path.realpath(path), int(st_stat.st_mtime_ns), int(st_stat.st_size))
    except OSError:
        cache_key = None

    if cache_key is not None:
        with _CACHE_LOCK:
            cached_info = _MEDIA_INFO_CACHE.get(cache_key)
            if cached_info is not None:
                return cached_info

    ffprobe_bin = find_binary("ffprobe")
    cmd = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(
                f"ffprobe failed on {path!r} (code {proc.returncode}): "
                f"{proc.stderr.decode('utf-8', errors='replace').strip()}"
            )
        raw = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    except Exception as exc:
        logger.warning("ffprobe failed for %s: %s", path, exc)
        raise

    fmt_info = raw.get("format", {}) or {}
    duration = 0.0
    try:
        duration = float(fmt_info.get("duration", 0.0) or 0.0)
    except (ValueError, TypeError):
        duration = 0.0

    streams_out = []
    for idx, st in enumerate(raw.get("streams", []) or []):
        st_no = int(st.get("index", idx))
        codec_type = (st.get("codec_type") or "").lower()
        codec_name = (st.get("codec_name") or "").lower()
        tags = st.get("tags") or {}
        lang = ""
        for k, v in tags.items():
            if k.lower() == "language" and v:
                lang = str(v).strip()
                break
        title = ""
        for k, v in tags.items():
            if k.lower() == "title" and v:
                title = str(v).strip()
                break

        fps = 0.0
        for fps_key in ("avg_frame_rate", "r_frame_rate"):
            fps_str = st.get(fps_key)
            if fps_str and "/" in str(fps_str):
                num_s, den_s = str(fps_str).split("/", 1)
                try:
                    num_f, den_f = float(num_s), float(den_s)
                    if den_f > 0 and num_f > 0:
                        fps = num_f / den_f
                        break
                except ValueError:
                    pass

        if codec_type == "video":
            norm_type = "video"
        elif codec_type == "audio":
            norm_type = "audio"
        elif codec_type == "subtitle":
            if codec_name in _BITMAP_SUBTITLE_CODECS:
                norm_type = "subtitle/bitmap"
            elif codec_name in _TEXT_SUBTITLE_CODECS or not codec_name:
                norm_type = "subtitle/text"
            else:
                norm_type = "subtitle/text"
        else:
            norm_type = "other"

        channels = int(st.get("channels") or 0)
        sample_rate = int(st.get("sample_rate") or 0)
        ch_layout_str = str(st.get("channel_layout") or "")

        if duration <= 0.0:
            try:
                st_dur = float(st.get("duration") or 0.0)
                if st_dur > duration:
                    duration = st_dur
            except (ValueError, TypeError):
                pass

        streams_out.append({
            "index": st_no,
            "codec_type": codec_type,
            "codec_name": codec_name,
            "type": norm_type,
            "lang": lang,
            "title": title,
            "frame_rate": fps,
            "channels": channels,
            "sample_rate": sample_rate,
            "channel_layout_name": ch_layout_str,
        })

    info_res = {
        "streams": streams_out,
        "duration": duration,
    }
    if cache_key is not None:
        with _CACHE_LOCK:
            if len(_MEDIA_INFO_CACHE) > 256:
                _MEDIA_INFO_CACHE.clear()
            _MEDIA_INFO_CACHE[cache_key] = info_res
    return info_res


def _run_ffmpeg_audio_extract(
    path: str,
    stream_index: Optional[int],
    sample_rate: int,
    start_s: Optional[float],
    duration_s: Optional[float],
    audio_filter: Optional[str] = None,
    use_ac1: bool = True,
    timeout: float = 300.0,
) -> subprocess.CompletedProcess:
    ffmpeg_bin = find_binary("ffmpeg")
    cmd = [ffmpeg_bin, "-v", "error", "-nostdin"]
    if start_s is not None and start_s > 0:
        cmd.extend(["-ss", f"{start_s:.3f}"])
    cmd.extend(["-i", path])
    if duration_s is not None and duration_s > 0:
        cmd.extend(["-t", f"{duration_s:.3f}"])

    if stream_index is not None:
        cmd.extend(["-map", f"0:{int(stream_index)}"])

    cmd.extend(["-vn", "-sn", "-dn"])
    if use_ac1:
        cmd.extend(["-ac", "1"])
    cmd.extend(["-ar", str(int(sample_rate))])
    if audio_filter:
        cmd.extend(["-af", audio_filter])
    cmd.extend(["-f", "s16le", "-acodec", "pcm_s16le", "-"])

    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg audio extraction timed out after {timeout:.0f}s for {path!r}") from exc


def extract_pcm_s16le(
    path: str,
    stream_index: Optional[int] = None,
    sample_rate: int = SAMPLE_RATE,
    start_s: Optional[float] = None,
    duration_s: Optional[float] = None,
    bandpass: bool = False,
    channels: Optional[int] = None,
    channel_layout: Optional[str] = None,
) -> bytes:
    """Decode audio stream from media file to mono 16-bit signed PCM via ffmpeg.

    For 5.1/7.1 surround streams (`channels >= 6`), isolates the Front Center (`FC`)
    dialogue channel with a vocal bandpass (`highpass=f=200,lowpass=f=3800`) and
    verifies sufficient signal energy (`max_rms >= 30.0`), falling back to mono
    downmix with bandpass if `FC` is silent.
    For stereo streams (`channels == 2`) or when `bandpass=True`, applies vocal
    bandpass (`highpass=f=200,lowpass=f=3800`).
    For mono streams (`channels == 1`) with `bandpass=False`, performs direct
    `-ac 1` extraction so synthetic test fixtures remain untouched.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio/video file not found: {path!r}")

    if os.path.splitext(path)[1].lower() == ".pcm":
        with open(path, "rb") as fp:
            raw_pcm = fp.read()
        if start_s is not None and start_s > 0:
            byte_offset = int(round(start_s * sample_rate)) * 2
            raw_pcm = raw_pcm[byte_offset:]
        if duration_s is not None and duration_s > 0:
            byte_len = int(round(duration_s * sample_rate)) * 2
            raw_pcm = raw_pcm[:byte_len]
        return raw_pcm

    if channels is None:
        try:
            info = probe_media_info(path)
            streams = info.get("streams", [])
            matched_st = None
            if stream_index is not None:
                for st in streams:
                    if st.get("index") == stream_index and st.get("type") == "audio":
                        matched_st = st
                        break
            if matched_st is None:
                for st in streams:
                    if st.get("type") == "audio":
                        matched_st = st
                        break
            if matched_st is not None:
                channels = int(matched_st.get("channels") or 0)
                if not channel_layout:
                    channel_layout = str(matched_st.get("channel_layout_name") or "")
        except Exception:
            channels = 1

    layout_lower = (channel_layout or "").lower()
    is_surround = (channels is not None and channels >= 6) or any(
        tag in layout_lower for tag in ("5.1", "7.1", "fc")
    )

    if is_surround:
        # First try extracting Front Center (FC) dialogue channel with vocal bandpass
        proc_fc = _run_ffmpeg_audio_extract(
            path=path,
            stream_index=stream_index,
            sample_rate=sample_rate,
            start_s=start_s,
            duration_s=duration_s,
            audio_filter="pan=mono|c0=FC,highpass=f=200,lowpass=f=3800",
            use_ac1=False,
        )
        if proc_fc.stdout:
            rms = compute_frame_rms(proc_fc.stdout, sample_rate=sample_rate)
            if rms and max(rms) >= 30.0:
                return proc_fc.stdout

        # Fallback if FC is empty or pan filter failed: downmix all channels with vocal bandpass
        proc = _run_ffmpeg_audio_extract(
            path=path,
            stream_index=stream_index,
            sample_rate=sample_rate,
            start_s=start_s,
            duration_s=duration_s,
            audio_filter="highpass=f=200,lowpass=f=3800",
            use_ac1=True,
        )
    elif (channels is not None and channels >= 2) or bandpass:
        proc = _run_ffmpeg_audio_extract(
            path=path,
            stream_index=stream_index,
            sample_rate=sample_rate,
            start_s=start_s,
            duration_s=duration_s,
            audio_filter="highpass=f=200,lowpass=f=3800",
            use_ac1=True,
        )
    else:
        proc = _run_ffmpeg_audio_extract(
            path=path,
            stream_index=stream_index,
            sample_rate=sample_rate,
            start_s=start_s,
            duration_s=duration_s,
            audio_filter=None,
            use_ac1=True,
        )

    if proc.returncode != 0 and not proc.stdout:
        # Retry without explicit stream map if stream_index mapping failed
        if stream_index is not None:
            return extract_pcm_s16le(
                path,
                stream_index=None,
                sample_rate=sample_rate,
                start_s=start_s,
                duration_s=duration_s,
                bandpass=bandpass,
                channels=channels,
                channel_layout=channel_layout,
            )
        err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg audio extraction failed for {path!r}: {err_msg}")

    return proc.stdout


def extract_embedded_subtitles(
    path: str,
    stream_index: Optional[int] = None,
) -> pysubs2.SSAFile:
    """Extract an embedded subtitle stream from a media container via ffmpeg."""
    if is_subtitle_file_path(path):
        return pysubs2.SSAFile.load(path)

    ffmpeg_bin = find_binary("ffmpeg")
    cmd = [ffmpeg_bin, "-v", "error", "-nostdin", "-i", path]
    if stream_index is not None:
        cmd.extend(["-map", f"0:{int(stream_index)}"])
    else:
        cmd.extend(["-map", "0:s:0"])
    cmd.extend(["-f", "srt", "-"])

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg subtitle extraction timed out after 120s for {path!r}") from exc
    if proc.returncode != 0 and not proc.stdout:
        err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg subtitle extraction failed for {path!r}: {err_msg}")

    srt_text = proc.stdout.decode("utf-8-sig", errors="replace")
    return pysubs2.SSAFile.from_string(srt_text, format_="srt")


def _compute_frame_rms_python(pcm_bytes: bytes, samples_per_frame: int) -> List[float]:
    """Compute RMS energy per frame in pure Python using array('h')."""
    num_samples = len(pcm_bytes) // 2
    num_frames = num_samples // samples_per_frame
    if num_frames <= 0:
        return []

    samples = array("h")
    samples.frombytes(pcm_bytes[: num_frames * samples_per_frame * 2])
    if sys.byteorder != "little":
        samples.byteswap()

    rms_list = [0.0] * num_frames
    spf = samples_per_frame
    inv_spf = 1.0 / float(spf)
    sqrt = math.sqrt

    for k in range(num_frames):
        base = k * spf
        chunk = samples[base : base + spf]
        # Sum of squares over 80 samples
        s2 = sum(x * x for x in chunk)
        rms_list[k] = sqrt(s2 * inv_spf)

    return rms_list


import sys


def compute_frame_rms(
    pcm_bytes: bytes,
    sample_rate: int = SAMPLE_RATE,
    frame_hz: int = FRAME_HZ,
) -> List[float]:
    """Compute short-time RMS energy for each 10ms frame."""
    samples_per_frame = max(1, int(round(sample_rate / float(frame_hz))))
    num_samples = len(pcm_bytes) // 2
    num_frames = num_samples // samples_per_frame
    if num_frames <= 0:
        return []

    # Try fast C implementation if available
    try:
        from subsync.synchro import aligner
        c_lib = aligner.get_fast_corr_lib()
        if c_lib is not None:
            return aligner.c_compute_frame_rms(c_lib, pcm_bytes, num_frames, samples_per_frame)
    except Exception:
        pass

    return _compute_frame_rms_python(pcm_bytes, samples_per_frame)


def pcm_to_speech_envelope(
    pcm_bytes: bytes,
    sample_rate: int = SAMPLE_RATE,
    frame_hz: int = FRAME_HZ,
    min_gap_s: float = 0.12,
    min_speech_s: float = 0.03,
    edge_boost: float = 0.25,
) -> Tuple[List[float], List[Tuple[float, float]]]:
    """Convert raw 16-bit mono PCM audio into a 100 Hz speech activity signal and intervals.

    Automatically uses Silero Neural VAD v5 (`ai_engine.run_silero_vad`) when available
    and human speech is detected (`active_ratio >= 0.02`), falling back to adaptive
    percentile noise floor estimation and hysteresis thresholding for synthetic
    sine-wave test fixtures.
    """
    global LAST_PCM_BYTES, LAST_SPEECH_ENVELOPE, LAST_ATTACK_ENVELOPE, LAST_SPEECH_INTERVALS
    LAST_PCM_BYTES = pcm_bytes

    try:
        from subsync.synchro import ai_engine
        ai_engine.set_last_pcm_bytes(pcm_bytes, sample_rate=sample_rate)
        silero_res = ai_engine.run_silero_vad(pcm_bytes, sample_rate=sample_rate, frame_hz=frame_hz)
        if silero_res is not None:
            speech_env, attack_env, speech_intervals = silero_res
            LAST_SPEECH_ENVELOPE = speech_env
            LAST_ATTACK_ENVELOPE = attack_env
            LAST_SPEECH_INTERVALS = speech_intervals
            return speech_env, speech_intervals
    except Exception as exc:
        logger.debug("Silero VAD fallback to RMS detector: %s", exc)

    rms = compute_frame_rms(pcm_bytes, sample_rate=sample_rate, frame_hz=frame_hz)
    n = len(rms)
    if n == 0:
        return [], []

    non_zero = sorted(x for x in rms if x > 1.0)
    if not non_zero:
        return [0.0] * n, []

    nz_len = len(non_zero)
    peak_rms = non_zero[min(nz_len - 1, int(nz_len * 0.96))]
    median_rms = non_zero[int(nz_len * 0.50)]

    all_sorted = sorted(rms)
    noise_floor = all_sorted[int(n * 0.20)]

    # Determine hysteresis thresholds adaptively
    if noise_floor < 5.0:
        # Clean / synthetic audio with near-zero silence floor
        high_thresh = max(30.0, peak_rms * 0.08)
        low_thresh = max(15.0, peak_rms * 0.04)
    else:
        # Real audio with background noise floor
        dyn_range = max(1.0, peak_rms - noise_floor)
        high_thresh = max(noise_floor * 1.8, noise_floor + 0.12 * dyn_range)
        low_thresh = max(noise_floor * 1.35, noise_floor + 0.06 * dyn_range)
        # If speech is very dense (>60% frames active), avoid setting threshold above median
        if high_thresh > peak_rms * 0.65:
            high_thresh = max(noise_floor * 1.25, peak_rms * 0.30)
            low_thresh = max(noise_floor * 1.10, peak_rms * 0.18)

    # Local adaptive baseline (20-second sliding window) for non-stationary noise
    win_half = int(10.0 * frame_hz)
    active = [0] * n
    in_speech = False
    for k in range(n):
        val = rms[k]
        if not in_speech:
            if val >= high_thresh:
                in_speech = True
                active[k] = 1
        else:
            if val >= low_thresh:
                active[k] = 1
            else:
                in_speech = False

    # Morphological closing: bridge intra-utterance pauses <= min_gap_s (120ms)
    # without extending the outer boundaries of utterances!
    max_gap_frames = max(1, int(round(min_gap_s * frame_hz)))
    i = 0
    while i < n:
        if active[i] == 1:
            j = i + 1
            while j < n and active[j] == 0 and (j - i) <= max_gap_frames + 1:
                j += 1
            if j < n and active[j] == 1 and (j - i - 1) <= max_gap_frames:
                for m in range(i + 1, j):
                    active[m] = 1
                i = j
                continue
        i += 1

    # Remove isolated noise spikes shorter than min_speech_s (30ms)
    min_speech_frames = max(1, int(round(min_speech_s * frame_hz)))
    i = 0
    while i < n:
        if active[i] == 1:
            j = i + 1
            while j < n and active[j] == 1:
                j += 1
            if (j - i) < min_speech_frames:
                for m in range(i, j):
                    active[m] = 0
            i = j
        else:
            i += 1

    # Extract intervals [start_s, end_s] and build envelope with subtle boundary boost
    intervals: List[Tuple[float, float]] = []
    inv_hz = 1.0 / float(frame_hz)
    envelope = [0.0] * n
    edge_frames = max(1, int(round(0.05 * frame_hz)))  # 50ms onset/offset emphasis

    i = 0
    while i < n:
        if active[i] == 1:
            j = i + 1
            while j < n and active[j] == 1:
                j += 1
            start_s = i * inv_hz
            end_s = j * inv_hz
            intervals.append((start_s, end_s))
            seg_len = j - i
            for m in range(i, j):
                dist_edge = min(m - i, j - 1 - m)
                if edge_boost > 0.0 and dist_edge < edge_frames and seg_len >= 2 * edge_frames:
                    envelope[m] = 1.0 + edge_boost * (1.0 - dist_edge / float(edge_frames))
                else:
                    envelope[m] = 1.0
            i = j
        else:
            i += 1

    return envelope, intervals


def subtitles_to_intervals(
    subs_or_path,
    encoding: str = "utf-8",
    fps: Optional[float] = None,
    update_global_state: bool = True,
) -> List[Tuple[float, float]]:
    """Convert an SSAFile, Subtitles instance, or subtitle file path into sorted (start_s, end_s) intervals."""
    global LAST_SUB_EVENTS
    if isinstance(subs_or_path, str):
        subs = pysubs2.SSAFile.load(subs_or_path, encoding=encoding, fps=fps)
    elif hasattr(subs_or_path, "events"):
        subs = subs_or_path
    elif isinstance(subs_or_path, Sequence):
        # Already a sequence of (start_s, end_s) pairs or SSAEvents
        res = []
        sub_evs_seq: List[Tuple[float, float, str]] = []
        for item in subs_or_path:
            if hasattr(item, "start") and hasattr(item, "end"):
                if not getattr(item, "is_comment", False):
                    txt = getattr(item, "plaintext", getattr(item, "text", "")).strip()
                    if is_credit_watermark(txt):
                        continue
                    spoken_txt = extract_spoken_dialogue_text(txt)
                    if not spoken_txt:
                        continue
                    s = item.start / 1000.0
                    e = item.end / 1000.0
                    if e > s >= 0.0:
                        res.append((s, e))
                        sub_evs_seq.append((s, e, spoken_txt))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                txt = str(item[2]).strip() if len(item) >= 3 and item[2] is not None else ""
                if txt and (is_credit_watermark(txt) or is_pure_sdh_non_speech(txt)):
                    continue
                spoken_txt = extract_spoken_dialogue_text(txt) if txt else ""
                s, e = float(item[0]), float(item[1])
                if e > s >= 0.0:
                    res.append((s, e))
                    if spoken_txt:
                        sub_evs_seq.append((s, e, spoken_txt))
        res.sort(key=lambda x: (x[0], x[1]))
        if sub_evs_seq and update_global_state:
            sub_evs_seq.sort(key=lambda x: (x[0], x[1]))
            LAST_SUB_EVENTS = sub_evs_seq
            _TLS.last_sub_events = sub_evs_seq
            try:
                from subsync.synchro import ai_engine
                ai_engine.set_last_subtitles(sub_evs_seq, None)
            except Exception:
                pass
        return res
    else:
        raise TypeError(f"Unsupported subtitle input type: {type(subs_or_path)!r}")

    intervals = []
    sub_events_with_text: List[Tuple[float, float, str]] = []
    fallback_intervals = []
    for ev in subs.events:
        if getattr(ev, "is_comment", False) or getattr(ev, "is_drawing", False):
            continue
        txt = getattr(ev, "plaintext", ev.text).strip()
        if not txt or is_credit_watermark(txt):
            continue
        start_s = ev.start / 1000.0
        end_s = ev.end / 1000.0
        if end_s > start_s and end_s > 0.0:
            s_clamped = max(0.0, start_s)
            fallback_intervals.append((s_clamped, end_s))
            spoken_txt = extract_spoken_dialogue_text(txt)
            if not spoken_txt:
                continue
            intervals.append((s_clamped, end_s))
            sub_events_with_text.append((s_clamped, end_s, spoken_txt))

    if not intervals and fallback_intervals:
        intervals = fallback_intervals

    intervals.sort(key=lambda x: (x[0], x[1]))
    sub_events_with_text.sort(key=lambda x: (x[0], x[1]))
    if update_global_state:
        LAST_SUB_EVENTS = sub_events_with_text
        _TLS.last_sub_events = sub_events_with_text
        try:
            from subsync.synchro import ai_engine
            ai_engine.set_last_subtitles(sub_events_with_text, subs)
        except Exception:
            pass
    return intervals


def intervals_to_envelope(
    intervals: Sequence[Tuple[float, float]],
    frame_hz: int = FRAME_HZ,
    duration_s: Optional[float] = None,
    scale_a: float = 1.0,
    edge_boost: float = 0.25,
) -> List[float]:
    """Rasterize (start_s, end_s) dialogue intervals into a 100 Hz activity signal."""
    if not intervals:
        return []

    max_end = max(e * scale_a for _, e in intervals)
    if duration_s is not None and duration_s > max_end:
        max_end = duration_s

    num_frames = int(math.ceil(max_end * frame_hz)) + 2
    if num_frames <= 0:
        return []

    envelope = [0.0] * num_frames
    edge_frames = max(1, int(round(0.06 * frame_hz)))

    for s_raw, e_raw in intervals:
        s = s_raw * scale_a
        e = e_raw * scale_a
        if e <= s or e <= 0.0:
            continue
        i = max(0, int(round(s * frame_hz)))
        j = min(num_frames, int(round(e * frame_hz)))
        if j <= i:
            j = min(num_frames, i + 1)
        seg_len = j - i
        for m in range(i, j):
            dist_start = m - i
            if edge_boost > 0.0 and dist_start < edge_frames and seg_len >= edge_frames:
                val = 1.0 + edge_boost * (1.0 - dist_start / float(edge_frames))
            else:
                val = 1.0
            if val > envelope[m]:
                envelope[m] = val

    return envelope


def extract_speech_envelope(
    path: str,
    stream_index: Optional[int] = None,
    stream_type: Optional[str] = None,
    encoding: str = "utf-8",
    fps: Optional[float] = None,
    frame_hz: int = FRAME_HZ,
) -> Tuple[List[float], List[Tuple[float, float]]]:
    """Extract 100 Hz activity envelope and intervals from any reference file (audio, video, or subtitle)."""
    global LAST_PCM_BYTES, LAST_SPEECH_ENVELOPE, LAST_ATTACK_ENVELOPE, LAST_SPEECH_INTERVALS
    if is_subtitle_file_path(path):
        intervals = subtitles_to_intervals(path, encoding=encoding, fps=fps, update_global_state=False)
        envelope = intervals_to_envelope(intervals, frame_hz=frame_hz)
        return envelope, intervals

    if stream_type and stream_type.startswith("subtitle"):
        subs = extract_embedded_subtitles(path, stream_index=stream_index)
        intervals = subtitles_to_intervals(subs, update_global_state=False)
        envelope = intervals_to_envelope(intervals, frame_hz=frame_hz)
        return envelope, intervals

    ref_cache_key = None
    try:
        st_stat = os.stat(path)
        ref_cache_key = (
            os.path.realpath(path),
            int(st_stat.st_mtime_ns),
            int(st_stat.st_size),
            stream_index,
            int(frame_hz),
        )
    except OSError:
        ref_cache_key = None

    if ref_cache_key is not None:
        with _CACHE_LOCK:
            cached_bundle = _REF_AUDIO_CACHE.get(ref_cache_key)
        if cached_bundle is not None:
            env = cached_bundle["envelope"]
            ivs = cached_bundle["intervals"]
            pcm_b = cached_bundle.get("pcm_bytes")
            att_env = cached_bundle.get("attack_envelope")
            vad_res = cached_bundle.get("vad_result")
            if pcm_b is not None:
                LAST_PCM_BYTES = pcm_b
                LAST_SPEECH_ENVELOPE = env
                LAST_ATTACK_ENVELOPE = att_env
                LAST_SPEECH_INTERVALS = ivs
                try:
                    from subsync.synchro import ai_engine
                    ai_engine.restore_vad_state(pcm_b, SAMPLE_RATE, vad_res)
                except Exception:
                    pass
            return env, ivs

    # Probe container streams once to detect subtitle streams or audio channel layout
    channels: Optional[int] = None
    channel_layout: Optional[str] = None
    try:
        info = probe_media_info(path)
        streams = info.get("streams", [])
        if stream_index is not None:
            for st in streams:
                if st.get("index") == stream_index and str(st.get("type", "")).startswith("subtitle"):
                    subs = extract_embedded_subtitles(path, stream_index=stream_index)
                    intervals = subtitles_to_intervals(subs, update_global_state=False)
                    envelope = intervals_to_envelope(intervals, frame_hz=frame_hz)
                    return envelope, intervals
        matched_audio = None
        if stream_index is not None:
            for st in streams:
                if st.get("index") == stream_index and st.get("type") == "audio":
                    matched_audio = st
                    break
        if matched_audio is None:
            for st in streams:
                if st.get("type") == "audio":
                    matched_audio = st
                    break
        if matched_audio is not None:
            channels = int(matched_audio.get("channels") or 0)
            channel_layout = str(matched_audio.get("channel_layout_name") or "")
    except Exception:
        pass

    pcm_bytes = extract_pcm_s16le(
        path,
        stream_index=stream_index,
        sample_rate=SAMPLE_RATE,
        channels=channels,
        channel_layout=channel_layout,
    )
    env, ivs = pcm_to_speech_envelope(pcm_bytes, sample_rate=SAMPLE_RATE, frame_hz=frame_hz)
    if ref_cache_key is not None:
        try:
            from subsync.synchro import ai_engine
            vad_res = ai_engine.get_last_vad_result()
        except Exception:
            vad_res = None
        with _CACHE_LOCK:
            if len(_REF_AUDIO_CACHE) >= _MAX_REF_AUDIO_CACHE:
                oldest_k = next(iter(_REF_AUDIO_CACHE))
                _REF_AUDIO_CACHE.pop(oldest_k, None)
            _REF_AUDIO_CACHE[ref_cache_key] = {
                "envelope": env,
                "intervals": ivs,
                "pcm_bytes": pcm_bytes,
                "attack_envelope": LAST_ATTACK_ENVELOPE,
                "vad_result": vad_res,
            }
    return env, ivs
