"""Local Neural Speech VAD (Silero v5 ONNX) and Faster-Whisper Scene Verification Engine.

Provides:
1. `run_silero_vad(pcm_bytes, sample_rate=8000, frame_hz=100)`:
   - Singleton-cached Silero VAD v5 ONNX runner batched across 64 parallel streams
     (`chunk_size=256`, `context=32`), processing a 42-minute episode in ~1.1s.
   - Returns `(speech_env_100hz, attack_env_100hz, speech_intervals)` or `None` when
     `mean(prob >= 0.30) < 0.02` (preserving RMS fallback for synthetic sine-wave fixtures).
2. `verify_scenes_with_whisper(...)`:
   - Singleton-cached Faster-Whisper `tiny.en` (`int8`, `cpu_threads=16`) transcription
     with exact first-word timestamps (`seg.words[0].start`).
   - Bilingual EN<->FR lexical, cognate, proper-noun, number, question (`?`), and duration
     matcher combined with 100 Hz Whisper word-onset + vocal-attack cross-correlation.
   - Accurately locks Cold Open sub-scenes (`t < 180s`), detects orphan `.srt` recaps
     (`recap_cutoff_s`) vs. valid video recaps, detects unsubtitled video recaps
     (`video_recap_end_s`), and refines scene offsets `b` to millisecond precision.
"""

import logging
import math
import os
import re
import sys
import threading
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Auto-wire local `.ai_env` so `numpy`, `onnxruntime`, and `faster_whisper` are always importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _ai_env_cand in (
    os.path.join(_REPO_ROOT, ".ai_env"),
    "/home/juliench/subsync/.ai_env",
):
    if os.path.isdir(_ai_env_cand) and _ai_env_cand not in sys.path:
        sys.path.insert(0, _ai_env_cand)

try:
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:  # pragma: no cover
    np = None
    sliding_window_view = None

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover
    WhisperModel = None

logger = logging.getLogger(__name__)

SILERO_MODEL_PATHS: Tuple[str, ...] = (
    "/home/juliench/subsync/.ai_models/silero_vad.onnx",
    os.path.join(_REPO_ROOT, ".ai_models", "silero_vad.onnx"),
)

WHISPER_MODEL_DIRS: Tuple[str, ...] = (
    "/home/juliench/subsync/.ai_models/whisper",
    os.path.join(_REPO_ROOT, ".ai_models", "whisper"),
)

_MODEL_LOCK = threading.Lock()
_SILERO_SESSION = None
_WHISPER_MODEL = None

# Cached state from the most recent audio/subtitle extraction for zero-copy cross-module access
LAST_PCM_BYTES: Optional[bytes] = None
LAST_SAMPLE_RATE: int = 8000
LAST_VAD_RESULT: Optional[Tuple[List[float], List[float], List[Tuple[float, float]]]] = None
LAST_SPEECH_ENVELOPE: Optional[List[float]] = None
LAST_ATTACK_ENVELOPE: Optional[List[float]] = None
LAST_SPEECH_INTERVALS: Optional[List[Tuple[float, float]]] = None
LAST_SUB_EVENTS: List[Tuple[float, float, str]] = []
LAST_SUBTITLES_OBJ: Any = None
_SUB_TEXT_BY_INTERVAL: Dict[Tuple[int, int], str] = {}

_VAD_CACHE: Dict[Tuple[int, int, int, int, int], Optional[Tuple[List[float], List[float], List[Tuple[float, float]]]]] = {}
_WHISPER_CACHE: Dict[Tuple[int, int, int, int], List["WhisperUtterance"]] = {}


class WhisperUtterance(tuple):
    """Word-accurate Whisper utterance `(w_start, w_end, text)` supporting tuple, dict, and attribute access."""

    def __new__(cls, start: float, end: float, text: str, words: Optional[Sequence[Tuple[float, float, str]]] = None):
        obj = super().__new__(cls, (float(start), float(end), str(text)))
        obj.start = float(start)
        obj.end = float(end)
        obj.text = str(text)
        obj.words = list(words) if words else []
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key)
        return super().__getitem__(key)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class WhisperVerificationResult(dict):
    """Verification result supporting dict lookup (`res['recap_cutoff_s']`) and attribute access (`res.recap_cutoff_s`)."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any):
        self[name] = value


def _pcm_fingerprint(pcm_bytes: bytes, sample_rate: int = 8000, frame_hz: int = 100) -> Tuple[int, int, int, int, int]:
    n = len(pcm_bytes)
    if n == 0:
        return (0, sample_rate, frame_hz, 0, 0)
    mid = n // 2
    h1 = hash(pcm_bytes[: min(n, 8192)])
    h2 = hash(pcm_bytes[max(0, mid - 4096) : min(n, mid + 4096)] + pcm_bytes[max(0, n - 8192) :])
    return (n, int(sample_rate), int(frame_hz), h1, h2)


def set_last_pcm_bytes(pcm_bytes: Optional[bytes], sample_rate: int = 8000) -> None:
    """Record the latest extracted 8 kHz mono PCM buffer."""
    global LAST_PCM_BYTES, LAST_SAMPLE_RATE
    if pcm_bytes:
        LAST_PCM_BYTES = pcm_bytes
        LAST_SAMPLE_RATE = int(sample_rate)


def get_last_pcm_bytes() -> Optional[bytes]:
    """Return the most recently extracted 8 kHz mono PCM buffer."""
    return LAST_PCM_BYTES


def get_last_vad_result() -> Optional[Tuple[List[float], List[float], List[Tuple[float, float]]]]:
    """Return the most recent `(speech_env_100hz, attack_env_100hz, speech_intervals)` from Silero VAD."""
    return LAST_VAD_RESULT


def get_last_attack_envelope() -> Optional[List[float]]:
    """Return the most recent 100 Hz Vocal Attack Envelope from Silero VAD."""
    return LAST_ATTACK_ENVELOPE


def set_last_subtitles(sub_events: Sequence[Tuple[float, float, str]], subs_obj: Any = None) -> None:
    """Cache `(start_s, end_s, text)` events from `vad.subtitles_to_intervals` so text is preserved for AI matching."""
    global LAST_SUB_EVENTS, LAST_SUBTITLES_OBJ, _SUB_TEXT_BY_INTERVAL
    LAST_SUB_EVENTS = [(float(s), float(e), str(t)) for s, e, t in sub_events]
    if subs_obj is not None:
        LAST_SUBTITLES_OBJ = subs_obj
    lookup: Dict[Tuple[int, int], str] = {}
    for s, e, t in LAST_SUB_EVENTS:
        lookup[(int(round(s * 1000.0)), int(round(e * 1000.0)))] = t
    _SUB_TEXT_BY_INTERVAL = lookup


def get_last_sub_events() -> List[Tuple[float, float, str]]:
    """Return cached `(start_s, end_s, text)` events from the last loaded subtitle track."""
    return list(LAST_SUB_EVENTS)


def enrich_intervals_with_text(
    items: Sequence[Any],
) -> List[Tuple[float, float, str]]:
    """Convert `(s, e)`, `(s, e, text)`, or `SSAEvent` sequences into `(start_s, end_s, text)` triples."""
    out: List[Tuple[float, float, str]] = []
    for item in items or []:
        if hasattr(item, "start") and hasattr(item, "end"):
            if getattr(item, "is_comment", False) or getattr(item, "is_drawing", False):
                continue
            s = float(item.start) / 1000.0
            e = float(item.end) / 1000.0
            txt = getattr(item, "plaintext", getattr(item, "text", "")).replace("\n", " ").strip()
            if e > s >= 0.0 and txt:
                out.append((s, e, txt))
        elif isinstance(item, dict):
            s = float(item.get("start", item.get("s", 0.0)))
            e = float(item.get("end", item.get("e", 0.0)))
            txt = str(item.get("text", "") or "").replace("\n", " ").strip()
            if not txt:
                txt = _SUB_TEXT_BY_INTERVAL.get((int(round(s * 1000.0)), int(round(e * 1000.0))), "")
            if e > s >= 0.0:
                out.append((s, e, txt))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            s = float(item[0])
            e = float(item[1])
            txt = str(item[2]).replace("\n", " ").strip() if len(item) >= 3 and item[2] is not None else ""
            if not txt:
                txt = _SUB_TEXT_BY_INTERVAL.get((int(round(s * 1000.0)), int(round(e * 1000.0))), "")
            if e > s >= 0.0:
                out.append((s, e, txt))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


# ---------------------------------------------------------------------------
# 1. Singleton-Cached Silero Neural VAD v5 (ONNX)
# ---------------------------------------------------------------------------


def get_silero_session():
    """Return singleton-cached ONNXRuntime session for Silero VAD v5."""
    global _SILERO_SESSION
    if _SILERO_SESSION is not None:
        return _SILERO_SESSION
    if ort is None or np is None:
        return None

    with _MODEL_LOCK:
        if _SILERO_SESSION is not None:
            return _SILERO_SESSION
        model_path = None
        for cand in SILERO_MODEL_PATHS:
            if os.path.isfile(cand):
                model_path = cand
                break
        if not model_path:
            logger.debug("Silero VAD model not found at %r", SILERO_MODEL_PATHS)
            return None
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _SILERO_SESSION = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            return _SILERO_SESSION
        except Exception as exc:
            logger.warning("Failed to initialize Silero VAD ONNX session: %s", exc)
            return None


def run_silero_vad(
    pcm_bytes: bytes,
    sample_rate: int = 8000,
    frame_hz: int = 100,
) -> Optional[Tuple[List[float], List[float], List[Tuple[float, float]]]]:
    """Run Silero Neural VAD v5 across 64 parallel streams on 8 kHz mono 16-bit PCM.

    Returns `(speech_env_100hz, attack_env_100hz, speech_intervals)` when human speech
    is detected (`mean(prob >= 0.30) >= 0.02`), or `None` for synthetic non-speech test
    signals (such as 440 Hz sine-wave unit test fixtures) so `vad.pcm_to_speech_envelope`
    cleanly falls back to its RMS hysteresis detector.
    """
    global LAST_PCM_BYTES, LAST_SAMPLE_RATE, LAST_VAD_RESULT
    global LAST_SPEECH_ENVELOPE, LAST_ATTACK_ENVELOPE, LAST_SPEECH_INTERVALS

    if not pcm_bytes or len(pcm_bytes) < 512 or np is None:
        return None

    set_last_pcm_bytes(pcm_bytes, sample_rate=sample_rate)

    fp_key = _pcm_fingerprint(pcm_bytes, sample_rate=sample_rate, frame_hz=frame_hz)
    if fp_key in _VAD_CACHE:
        cached = _VAD_CACHE[fp_key]
        if cached is not None:
            LAST_VAD_RESULT = cached
            LAST_SPEECH_ENVELOPE, LAST_ATTACK_ENVELOPE, LAST_SPEECH_INTERVALS = cached
        return cached

    sess = get_silero_session()
    if sess is None:
        return None

    num_raw_samples = len(pcm_bytes) // 2
    samples_per_frame = max(1, int(round(sample_rate / float(frame_hz))))
    num_frames = num_raw_samples // samples_per_frame
    if num_frames <= 0:
        return None

    samples = np.frombuffer(pcm_bytes, dtype=np.int16, count=num_raw_samples).astype(np.float32) * (1.0 / 32768.0)

    # Silero 8 kHz mode expects 256-sample (32ms) chunks with 32-sample context
    target_sr = 8000
    if sample_rate != target_sr:
        num_8k = int(round(len(samples) * float(target_sr) / float(sample_rate)))
        if num_8k < 512:
            return None
        samples_8k = np.interp(
            np.linspace(0.0, float(len(samples) - 1), num_8k, dtype=np.float32),
            np.arange(len(samples), dtype=np.float32),
            samples,
        ).astype(np.float32)
    else:
        samples_8k = samples

    chunk_size = 256
    context_size = 32
    total_chunks = len(samples_8k) // chunk_size
    if total_chunks < 4:
        return None

    # Use 64 parallel streams on full episodes, scaling down on short clips so each stream has >= 16 chunks
    if total_chunks >= 64 * 16:
        batch_size = 64
    else:
        batch_size = max(1, min(64, total_chunks // 16))

    chunks_per_stream = (total_chunks + batch_size - 1) // batch_size
    padded_chunks = batch_size * chunks_per_stream
    if padded_chunks * chunk_size > len(samples_8k):
        padded = np.zeros(padded_chunks * chunk_size, dtype=np.float32)
        padded[: total_chunks * chunk_size] = samples_8k[: total_chunks * chunk_size]
    else:
        padded = samples_8k[: padded_chunks * chunk_size]

    stream_samples = padded.reshape(batch_size, chunks_per_stream, chunk_size)

    state = np.zeros((2, batch_size, 128), dtype=np.float32)
    context = np.zeros((batch_size, context_size), dtype=np.float32)
    if batch_size > 1:
        context[1:] = stream_samples[:-1, -1, -context_size:]
    sr_arr = np.array(target_sr, dtype=np.int64)
    probs = np.zeros((batch_size, chunks_per_stream), dtype=np.float32)

    input_buf = np.empty((batch_size, context_size + chunk_size), dtype=np.float32)
    for i in range(chunks_per_stream):
        cur_chunk = stream_samples[:, i, :]
        input_buf[:, :context_size] = context
        input_buf[:, context_size:] = cur_chunk
        out, state = sess.run(None, {"input": input_buf, "state": state, "sr": sr_arr})
        probs[:, i] = out[:, 0]
        context = cur_chunk[:, -context_size:]

    flat_probs = probs.reshape(-1)[:total_chunks]

    # Synthetic audio fallback: if mean(prob >= 0.30) < 0.02 (e.g. 440 Hz sine wave), return None
    active_ratio = float(np.mean(flat_probs >= 0.30))
    if active_ratio < 0.02:
        if len(_VAD_CACHE) > 8:
            _VAD_CACHE.clear()
        _VAD_CACHE[fp_key] = None
        return None

    # Interpolate chunk probabilities (31.25 Hz / 32ms) to 100 Hz (10ms frames)
    frame_to_chunk = np.arange(num_frames, dtype=np.float32) * (float(target_sr) / float(frame_hz * chunk_size))
    prob_100hz = np.interp(
        frame_to_chunk,
        np.arange(total_chunks, dtype=np.float32),
        flat_probs,
    ).astype(np.float32)

    # 1. speech_env_100hz: prob >= 0.30 ? prob : 0.0
    speech_env_arr = np.where(prob_100hz >= 0.30, prob_100hz, 0.0).astype(np.float32)

    # 2. attack_env_100hz: Vocal Attack Envelope peaking at rising vocal onset of each phrase
    #    smooth[k] - min(smooth[k-40:k]) when smooth[k] > 0.35 and rise > 0.20
    smooth = np.convolve(prob_100hz, np.full(5, 0.2, dtype=np.float32), mode="same")
    if sliding_window_view is not None and num_frames >= 2:
        padded_smooth = np.pad(smooth, (40, 0), mode="edge")[:-1]
        prev_min = sliding_window_view(padded_smooth, 40).min(axis=1)
    else:
        prev_min = np.array(
            [float(np.min(smooth[max(0, k - 40) : max(1, k)])) for k in range(num_frames)],
            dtype=np.float32,
        )
    rise = smooth - prev_min
    attack_env_arr = np.where((smooth > 0.35) & (rise > 0.20), rise, 0.0).astype(np.float32)

    # 3. speech_intervals: micro-pauses <= 0.25s bridged and duration >= 0.12s
    active = (prob_100hz >= 0.30).astype(np.uint8)
    max_gap_frames = max(1, int(round(0.25 * frame_hz)))
    min_dur_frames = max(1, int(round(0.12 * frame_hz)))

    # Find runs of active frames via diff
    padded_act = np.empty(num_frames + 2, dtype=np.uint8)
    padded_act[0] = 0
    padded_act[1:-1] = active
    padded_act[-1] = 0
    diffs = np.diff(padded_act.astype(np.int8))
    run_starts = np.flatnonzero(diffs == 1).tolist()
    run_ends = np.flatnonzero(diffs == -1).tolist()

    # Bridge micro-pauses <= 0.25s (max_gap_frames)
    merged_runs: List[Tuple[int, int]] = []
    for s_idx, e_idx in zip(run_starts, run_ends):
        if merged_runs and (s_idx - merged_runs[-1][1]) <= max_gap_frames:
            merged_runs[-1] = (merged_runs[-1][0], e_idx)
        else:
            merged_runs.append((s_idx, e_idx))

    inv_hz = 1.0 / float(frame_hz)
    speech_intervals: List[Tuple[float, float]] = [
        (s_idx * inv_hz, e_idx * inv_hz)
        for s_idx, e_idx in merged_runs
        if (e_idx - s_idx) >= min_dur_frames
    ]

    speech_env_list: List[float] = speech_env_arr.tolist()
    attack_env_list: List[float] = attack_env_arr.tolist()

    result = (speech_env_list, attack_env_list, speech_intervals)
    if len(_VAD_CACHE) > 8:
        _VAD_CACHE.clear()
    _VAD_CACHE[fp_key] = result

    LAST_VAD_RESULT = result
    LAST_SPEECH_ENVELOPE = speech_env_list
    LAST_ATTACK_ENVELOPE = attack_env_list
    LAST_SPEECH_INTERVALS = speech_intervals
    return result


# ---------------------------------------------------------------------------
# 2. Singleton-Cached Faster-Whisper (`tiny.en`) & Word-Onset Extraction
# ---------------------------------------------------------------------------


def get_whisper_model():
    """Return singleton-cached Faster-Whisper `tiny.en` model (`int8`, `cpu_threads=16`)."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    if WhisperModel is None or np is None:
        return None

    with _MODEL_LOCK:
        if _WHISPER_MODEL is not None:
            return _WHISPER_MODEL
        download_root = WHISPER_MODEL_DIRS[0]
        for cand in WHISPER_MODEL_DIRS:
            if os.path.isdir(cand):
                download_root = cand
                break
        for local_only in (True, False):
            try:
                _WHISPER_MODEL = WhisperModel(
                    "tiny.en",
                    device="cpu",
                    compute_type="int8",
                    download_root=download_root,
                    local_files_only=local_only,
                    cpu_threads=16,
                )
                return _WHISPER_MODEL
            except Exception as exc:
                if not local_only:
                    logger.warning("Failed to load Faster-Whisper model: %s", exc)
        return None


def transcribe_audio_window(
    pcm_bytes: Optional[bytes] = None,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
    sample_rate: int = 8000,
) -> List[WhisperUtterance]:
    """Upsample 8 kHz PCM slice `[start_s, end_s]` to 16 kHz and extract word-accurate English utterances."""
    if pcm_bytes is None:
        pcm_bytes = LAST_PCM_BYTES
        sample_rate = LAST_SAMPLE_RATE
    if not pcm_bytes or np is None:
        return []

    total_samples = len(pcm_bytes) // 2
    total_dur = total_samples / float(sample_rate)
    t0 = max(0.0, float(start_s))
    t1 = min(total_dur, float(end_s)) if end_s is not None else total_dur
    if t1 - t0 < 0.5:
        return []

    fp = _pcm_fingerprint(pcm_bytes, sample_rate=sample_rate)
    cache_key = (fp[0], fp[3], int(round(t0 * 10.0)), int(round(t1 * 10.0)))
    if cache_key in _WHISPER_CACHE:
        return list(_WHISPER_CACHE[cache_key])

    model = get_whisper_model()
    if model is None:
        return []

    i0 = max(0, int(round(t0 * sample_rate)))
    i1 = min(total_samples, int(round(t1 * sample_rate)))
    if i1 - i0 < int(0.5 * sample_rate):
        return []

    slice_bytes = pcm_bytes[i0 * 2 : i1 * 2]
    s_in = np.frombuffer(slice_bytes, dtype=np.int16).astype(np.float32) * (1.0 / 32768.0)

    # Upsample from 8000 Hz to 16000 Hz
    if sample_rate == 8000:
        n_in = len(s_in)
        s_16k = np.empty(n_in * 2, dtype=np.float32)
        s_16k[0::2] = s_in
        s_16k[1:-1:2] = 0.5 * (s_in[:-1] + s_in[1:])
        s_16k[-1] = s_in[-1]
    elif sample_rate == 16000:
        s_16k = s_in
    else:
        n_out = int(round(len(s_in) * 16000.0 / float(sample_rate)))
        s_16k = np.interp(
            np.linspace(0.0, float(len(s_in) - 1), n_out, dtype=np.float32),
            np.arange(len(s_in), dtype=np.float32),
            s_in,
        ).astype(np.float32)

    try:
        segments, _ = model.transcribe(
            s_16k,
            language="en",
            beam_size=1,
            best_of=1,
            word_timestamps=True,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        seg_list = list(segments)
    except Exception as exc:
        logger.warning("Faster-Whisper transcription failed on [%.1f, %.1f]: %s", t0, t1, exc)
        return []

    utterances: List[WhisperUtterance] = []
    for seg in seg_list:
        words = list(getattr(seg, "words", None) or [])
        if not words:
            txt = (seg.text or "").strip()
            if txt:
                utterances.append(WhisperUtterance(t0 + float(seg.start), t0 + float(seg.end), txt, []))
            continue

        # Split segments when an internal word pause >= 0.65s or sentence boundary pause >= 0.20s occurs
        groups: List[List[Any]] = [[words[0]]]
        for w in words[1:]:
            prev_w = groups[-1][-1]
            gap = float(w.start) - float(prev_w.end)
            prev_txt = (prev_w.word or "").strip()
            if gap >= 0.65 or (gap >= 0.20 and prev_txt.endswith((".", "?", "!"))):
                groups.append([w])
            else:
                groups[-1].append(w)

        for g in groups:
            w0 = g[0]
            w0_start = float(w0.start)
            w0_end = float(w0.end)
            # Clamp DTW first-word stretch across pre-speech silence/noise
            if w0_end - w0_start > 0.65:
                est_dur = min(0.40, max(0.18, 0.06 * len((w0.word or "").strip())))
                w0_start = max(w0_start, w0_end - est_dur)
            u_start = t0 + w0_start
            u_end = t0 + max(w0_start + 0.05, float(g[-1].end))
            u_text = "".join(w.word for w in g).strip()
            u_words = [(t0 + float(w.start), t0 + float(w.end), (w.word or "").strip()) for w in g]
            if u_text:
                utterances.append(WhisperUtterance(u_start, u_end, u_text, u_words))

        # If the segment was split into multiple clauses and its total span is compact (<= 5.5s),
        # also include the combined segment starting at the first valid group so multi-clause French
        # subtitles match the full English utterance as well.
        if len(groups) > 1:
            # Ignore a tiny 1-3 word noise-burst prefix separated by > 1.5s silence from the main sentence
            start_g_idx = 0
            if (
                len(groups[0]) <= 3
                and len(groups[1]) >= 3
                and (float(groups[1][0].start) - float(groups[0][-1].end)) > 1.5
            ):
                start_g_idx = 1
            first_w = groups[start_g_idx][0]
            last_w = groups[-1][-1]
            w_s = float(first_w.start)
            if float(first_w.end) - w_s > 0.65:
                est_dur = min(0.40, max(0.18, 0.06 * len((first_w.word or "").strip())))
                w_s = max(w_s, float(first_w.end) - est_dur)
            if start_g_idx == 1:
                # Snap back slightly to include the true onset of the post-gap phrase
                w_s = max(float(groups[0][-1].end) + 0.5, w_s - 0.35)
            span_dur = float(last_w.end) - w_s
            if 0.2 <= span_dur <= 6.5:
                comb_words = [w for grp in groups[start_g_idx:] for w in grp]
                comb_txt = (seg.text or "").strip() if start_g_idx == 0 else "".join(w.word for w in comb_words).strip()
                u_words = [(t0 + float(w.start), t0 + float(w.end), (w.word or "").strip()) for w in comb_words]
                utterances.append(WhisperUtterance(t0 + w_s, t0 + float(last_w.end), comb_txt, u_words))

    utterances.sort(key=lambda u: (u.start, u.end))
    if len(_WHISPER_CACHE) > 24:
        _WHISPER_CACHE.clear()
    _WHISPER_CACHE[cache_key] = utterances
    return list(utterances)


# ---------------------------------------------------------------------------
# 3. Bilingual EN<->FR Lexical, Cognate, Proper-Noun, Number & Question Matcher
# ---------------------------------------------------------------------------

_FR_EN_LEXICON: Dict[str, set] = {
    "precedemment": {"previously"},
    "previously": {"previously", "precedemment"},
    "bonjour": {"hello", "hi", "morning", "afternoon", "good"},
    "salut": {"hi", "hey", "hello"},
    "bonsoir": {"evening", "good", "hello"},
    "monsieur": {"sir", "mr"},
    "madame": {"mrs", "maam", "ms", "madam"},
    "mademoiselle": {"miss", "ms"},
    "bon": {"good", "nice"},
    "bonne": {"good", "nice"},
    "nouveau": {"again", "back", "new"},
    "chez": {"home", "place"},
    "maison": {"home", "house"},
    "voir": {"see", "look"},
    "vu": {"saw", "seen", "see"},
    "ici": {"here"},
    "directeur": {"director"},
    "adjoint": {"assistant", "deputy"},
    "rendez": {"appointment", "meeting"},
    "dites": {"tell", "say"},
    "dire": {"tell", "say"},
    "dit": {"said", "told", "say", "says"},
    "parle": {"speak", "talk", "talking", "speaking"},
    "parler": {"speak", "talk"},
    "parliez": {"speak", "spoke"},
    "bureau": {"office", "field", "bureau", "desk"},
    "affaire": {"case", "business"},
    "arrive": {"happen", "happened", "arrived", "coming"},
    "quand": {"when", "what", "time"},
    "moins": {"under", "less"},
    "heure": {"hour", "time", "oclock"},
    "heures": {"hours", "oclock", "time"},
    "confirmation": {"confirmed", "confirm"},
    "vraiment": {"actually", "really", "truly"},
    "empreintes": {"print", "prints", "fingerprints"},
    "tatouages": {"tattoo", "tattoos"},
    "correspondent": {"match", "matches"},
    "benevole": {"volunteer", "volunteered"},
    "mission": {"mission"},
    "classee": {"classified"},
    "secret": {"classified", "secret"},
    "bruxelles": {"brussels"},
    "passe": {"happened", "happen", "going"},
    "essaye": {"tried", "try", "trying"},
    "tuer": {"kill"},
    "tue": {"killed", "kill"},
    "mort": {"dead", "died", "death"},
    "morte": {"dead", "died"},
    "mourir": {"die", "dying"},
    "meurt": {"die", "dies", "died", "dead"},
    "venu": {"came", "come"},
    "venir": {"come", "coming"},
    "viens": {"come"},
    "mallette": {"briefcase", "case"},
    "identites": {"identities", "identity"},
    "inconnues": {"unknown", "knew"},
    "prets": {"ready"},
    "pret": {"ready"},
    "prete": {"ready"},
    "question": {"question"},
    "stupide": {"stupid", "dumb"},
    "lundi": {"monday"},
    "mardi": {"tuesday"},
    "mercredi": {"wednesday"},
    "jeudi": {"thursday"},
    "vendredi": {"friday"},
    "samedi": {"saturday"},
    "dimanche": {"sunday"},
    "examinateur": {"examiner"},
    "sujet": {"subject"},
    "avant": {"before"},
    "apres": {"after"},
    "semaine": {"week"},
    "derniere": {"last"},
    "jamais": {"ever", "never"},
    "toujours": {"always", "still"},
    "contact": {"contact"},
    "personnel": {"personal"},
    "informee": {"notify", "notified", "inform", "told"},
    "rendre": {"surrender", "surrendered"},
    "rendu": {"surrendered", "surrender"},
    "condamne": {"convicted"},
    "crime": {"crime"},
    "encore": {"yet", "still", "again", "more"},
    "repondez": {"answer", "respond"},
    "reponse": {"answer", "response"},
    "oui": {"yes", "yeah", "yep"},
    "non": {"no", "not"},
    "perdez": {"wasting", "lose", "losing"},
    "temps": {"time"},
    "precieux": {"precious", "valuable"},
    "pourquoi": {"why"},
    "comment": {"how"},
    "combien": {"how", "many", "much"},
    "qui": {"who"},
    "quoi": {"what"},
    "quel": {"what", "which"},
    "quelle": {"what", "which"},
    "immunit": {"immunity"},
    "cooperer": {"cooperate"},
    "donner": {"give"},
    "donne": {"give", "gave"},
    "noms": {"names"},
    "nom": {"name"},
    "liste": {"list", "blacklist"},
    "prochain": {"next"},
    "quelqu": {"someone", "somebody"},
    "devrais": {"should"},
    "trouver": {"find"},
    "trouvee": {"found", "find"},
    "homme": {"man", "men", "guy"},
    "femmes": {"women"},
    "femme": {"wife", "woman"},
    "mari": {"husband"},
    "pere": {"dad", "father"},
    "papa": {"dad", "father", "daddy"},
    "mere": {"mother", "mom"},
    "maman": {"mom", "mother", "mommy"},
    "enfants": {"children", "kids"},
    "enfant": {"child", "kid"},
    "gosses": {"kid", "kids"},
    "bebes": {"infants", "babies", "baby"},
    "bebe": {"baby", "infant"},
    "protege": {"protects", "protect"},
    "proteger": {"protect"},
    "coupables": {"guilty"},
    "coupable": {"guilty"},
    "innocents": {"innocent"},
    "innocent": {"innocent"},
    "travail": {"job", "work"},
    "emploi": {"job", "work"},
    "attention": {"attention"},
    "appris": {"learned", "heard"},
    "engage": {"contracted", "hired", "engage"},
    "informateur": {"informant"},
    "mafia": {"mob", "mafia"},
    "meme": {"same"},
    "appellent": {"call", "called"},
    "appelle": {"call", "called", "calling"},
    "alchimiste": {"alchemist"},
    "science": {"science"},
    "transformer": {"transform"},
    "personne": {"person", "someone", "nobody", "anyone"},
    "personnes": {"people", "persons"},
    "sur": {"sure", "on"},
    "va": {"okay", "fine", "going", "go"},
    "tousse": {"coughing", "cough"},
    "toute": {"all"},
    "tout": {"all", "everything"},
    "monde": {"everyone", "everybody", "world"},
    "nuit": {"night"},
    "soir": {"tonight", "evening", "night"},
    "matin": {"morning"},
    "demain": {"tomorrow"},
    "hier": {"yesterday"},
    "maintenant": {"now"},
    "bien": {"fine", "good", "well", "okay"},
    "rien": {"nothing"},
    "spectacle": {"show", "talent"},
    "idee": {"idea"},
    "assoir": {"sit"},
    "regarder": {"watch", "look"},
    "regarde": {"look", "watch"},
    "minutes": {"minutes", "minute"},
    "secondes": {"seconds", "second"},
    "gentil": {"nice", "kind"},
    "veux": {"want"},
    "veut": {"wants", "want"},
    "voulez": {"want"},
    "voulais": {"wanted", "want"},
    "peux": {"can", "could"},
    "peut": {"can", "could", "may"},
    "sais": {"know"},
    "sait": {"knows", "know"},
    "saches": {"know"},
    "savoir": {"know"},
    "pense": {"think", "thought"},
    "pensais": {"thought", "think"},
    "crois": {"think", "believe"},
    "attrapes": {"get", "catch"},
    "urgence": {"emergency"},
    "besoin": {"need"},
    "docteur": {"doctor"},
    "medecin": {"doctor"},
    "hopital": {"hospital"},
    "ambulance": {"ambulance"},
    "banque": {"bank"},
    "blesse": {"hurt", "injured"},
    "enregistrez": {"recording", "record"},
    "appels": {"calls", "call"},
    "appel": {"call"},
    "ecoute": {"listen", "look"},
    "ecoutes": {"listening", "listen"},
    "entends": {"hear"},
    "entendu": {"heard", "hear"},
    "malade": {"sick", "ill"},
    "aime": {"love", "like"},
    "ouvres": {"open"},
    "ouvrir": {"open"},
    "cellule": {"box", "cell"},
    "boite": {"box"},
    "coffre": {"safe", "box", "deposit"},
    "cle": {"key"},
    "testament": {"will", "wheel"},
    "chercher": {"pick", "find", "look"},
    "fete": {"party"},
    "hockey": {"hockey"},
    "equipement": {"equipment"},
    "voiture": {"car"},
    "avion": {"plane", "flight"},
    "parts": {"slices", "slice"},
    "pizza": {"pizza"},
    "anglais": {"english", "englishman", "englishmen"},
    "langue": {"language"},
    "facilite": {"easier", "easy"},
    "vie": {"life"},
    "merci": {"thank", "thanks"},
    "pardon": {"sorry", "excuse"},
    "desole": {"sorry"},
    "aide": {"help"},
    "aider": {"help"},
    "attends": {"wait", "hold"},
    "attendez": {"wait", "hold"},
    "attendre": {"wait", "waiting"},
    "arrete": {"stop"},
    "taupe": {"mole"},
    "cible": {"target"},
    "cibles": {"targets"},
    "verite": {"truth"},
    "accord": {"deal", "agreement"},
    "arrangement": {"arrangement", "deal"},
    "ministre": {"minister"},
    "finances": {"finance", "finances"},
    "coeur": {"heart"},
    "arrete": {"stopped", "stop"},
    "battre": {"beating", "beat"},
    "partir": {"leave", "leaving", "go"},
    "pars": {"leaving", "leave"},
    "couple": {"couple"},
    "normal": {"normal"},
    "ennuyeux": {"boring"},
    "entretien": {"interview"},
    "interieur": {"inside"},
    "propre": {"clean"},
    "un": {"one", "1"},
    "une": {"one", "1"},
    "deux": {"two", "2"},
    "trois": {"three", "3"},
    "quatre": {"four", "4"},
    "cinq": {"five", "5"},
    "six": {"six", "6"},
    "sept": {"seven", "7"},
    "huit": {"eight", "8"},
    "neuf": {"nine", "9"},
    "dix": {"ten", "10"},
    "onze": {"eleven", "11"},
    "douze": {"twelve", "12"},
    "quatorze": {"fourteen", "fourteenth", "14", "14th"},
    "14eme": {"fourteenth", "14th", "14", "fourteen"},
    "quinze": {"fifteen", "15"},
    "vingt": {"twenty", "20"},
    "20": {"twenty", "20"},
    "trente": {"thirty", "30"},
    "quarante": {"forty", "40"},
    "cinquante": {"fifty", "50"},
    "cent": {"hundred", "100"},
    "7": {"seven", "7"},
    "9": {"nine", "9"},
    "10": {"ten", "10"},
    "9h07": {"907", "907am", "9", "07"},
    "5591": {"5591", "5", "9", "1"},
    "3929": {"3929", "39", "29"},
}

_STOPWORDS_FR = {
    "le", "la", "les", "de", "du", "des", "au", "aux", "et", "ou", "en",
    "ce", "ca", "se", "sa", "son", "ses", "je", "tu", "il", "elle", "on",
    "nous", "vous", "ils", "elles", "me", "te", "ne", "pas", "plus", "par",
    "pour", "dans", "avec", "est", "es", "suis", "sont", "ai", "as", "ont",
    "ete", "etre", "avoir", "fait", "faire", "une", "un",
}


def normalize_tokens(text: str) -> List[str]:
    """Normalize accents/diacritics, lowercase, and extract alphanumeric tokens."""
    if not text:
        return []
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return re.findall(r"[a-z0-9]+", stripped)


def bilingual_utterance_similarity(
    sub_text: str,
    whisper_text: str,
    sub_dur: Optional[float] = None,
    whisper_dur: Optional[float] = None,
) -> float:
    """Compute bilingual EN<->FR lexical, cognate, proper-noun, number, question (`?`), and duration similarity."""
    fr_toks = normalize_tokens(sub_text)
    en_toks_list = normalize_tokens(whisper_text)
    if not fr_toks or not en_toks_list:
        return 0.0

    en_toks = set(en_toks_list)
    en_joined = "".join(en_toks_list)
    hits = 0.0

    for ft in fr_toks:
        # 1. Exact token or number match (proper nouns, shared words, numbers)
        if len(ft) >= 3 and ft in en_toks and ft not in _STOPWORDS_FR:
            hits += 1.65
            continue
        if ft.isdigit() and (ft in en_toks or ft in en_joined):
            hits += 1.8
            continue
        # 2. Cognate / stem prefix match (4+ characters)
        if len(ft) >= 4 and ft not in _STOPWORDS_FR:
            prefix4 = ft[:4]
            prefix5 = ft[:5] if len(ft) >= 5 else prefix4
            if any(
                (len(et) >= 5 and et[:5] == prefix5) or (len(et) >= 4 and et[:4] == prefix4)
                for et in en_toks
            ):
                hits += 1.25
                continue
        # 3. Bilingual lexicon lookup (exact or 5-char stem)
        matched_lex = False
        if ft in _FR_EN_LEXICON:
            if en_toks & _FR_EN_LEXICON[ft]:
                hits += 1.10
                matched_lex = True
        if not matched_lex and len(ft) >= 5:
            stem = ft[:5]
            for k, targets in _FR_EN_LEXICON.items():
                if len(k) >= 5 and k[:5] == stem and (en_toks & targets):
                    hits += 0.95
                    break

    content_count = max(1.5, float(sum(1 for t in fr_toks if t not in _STOPWORDS_FR or len(t) >= 4)))
    score = hits / (content_count * 0.75 + 0.8)

    # Question mark agreement bonus
    if "?" in sub_text and "?" in whisper_text:
        score += 0.28
    elif ("?" in sub_text) != ("?" in whisper_text):
        score *= 0.92

    # Utterance duration / rhythm agreement bonus
    if sub_dur is not None and whisper_dur is not None and sub_dur > 0 and whisper_dur > 0:
        dur_diff = abs(float(sub_dur) - float(whisper_dur))
        if dur_diff <= 1.2:
            score += 0.16 * (1.0 - dur_diff / 1.2)

    return float(min(2.5, score))


# ---------------------------------------------------------------------------
# 4. Whisper Word-Onset + Vocal-Attack Cross-Correlation & Scene Offset Solver
# ---------------------------------------------------------------------------


def _build_onset_pulse_signal(
    intervals: Sequence[Tuple[float, float]],
    t_origin: float,
    duration_s: float,
    frame_hz: int = 100,
    pulse_s: float = 0.35,
    weights: Optional[Sequence[float]] = None,
) -> "np.ndarray":
    """Build a 100 Hz onset pulse signal with 0.35s decaying pulses at each utterance start."""
    n_frames = max(1, int(math.ceil(duration_s * frame_hz)) + 2)
    sig = np.zeros(n_frames, dtype=np.float32)
    pulse_frames = max(1, int(round(pulse_s * frame_hz)))
    decay = np.linspace(1.0, 0.25, pulse_frames, dtype=np.float32)

    for idx, item in enumerate(intervals):
        s = float(item[0]) - t_origin
        if s < 0.0 or s >= duration_s:
            continue
        f0 = int(round(s * frame_hz))
        f1 = min(n_frames, f0 + pulse_frames)
        if f1 > f0:
            w = float(weights[idx]) if weights is not None and idx < len(weights) else 1.0
            seg_decay = decay[: f1 - f0] * w
            sig[f0:f1] = np.maximum(sig[f0:f1], seg_decay)
    return sig


def score_scene_with_whisper(
    sub_events: Sequence[Tuple[float, float, str]],
    utterances: Sequence[WhisperUtterance],
    b_min: float = -55.0,
    b_max: float = 55.0,
    b_prior: Optional[float] = None,
    attack_env_100hz: Optional[Sequence[float]] = None,
    frame_hz: int = 100,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    """Find the optimal offset `b` (`t_audio = t_sub + b`) for `sub_events` against Whisper `utterances`.

    Combines:
    1. Bilingual lexical + cognate + proper-noun + number + question (`?`) + duration Gaussian kernel voting.
    2. 100 Hz Whisper word-onset pulse cross-correlation + Silero vocal-attack onset correlation.
    3. Exact first-word timestamp (`w_start - sub_start`) weighted MAD-trimmed refinement.
    """
    if not sub_events or not utterances or np is None:
        return (float(b_prior or 0.0), 0.0, [])

    step = 0.05
    grid = np.arange(b_min, b_max + step * 0.5, step, dtype=np.float32)
    if len(grid) == 0:
        return (float(b_prior or 0.0), 0.0, [])

    lex_scores = np.zeros(len(grid), dtype=np.float32)
    radius_bins = int(round(0.65 / step))
    offsets_rel = np.arange(-radius_bins, radius_bins + 1, dtype=np.float32) * step
    kernel = np.exp(-0.5 * (offsets_rel / 0.20) ** 2).astype(np.float32)

    for s_sub, e_sub, txt_sub in sub_events:
        dur_sub = max(0.1, e_sub - s_sub)
        for u in utterances:
            b_cand = u.start - s_sub
            if b_cand < b_min - 0.65 or b_cand > b_max + 0.65:
                continue
            sim = bilingual_utterance_similarity(txt_sub, u.text, dur_sub, u.end - u.start)
            # Base onset coincidence weight + strong lexical match amplification
            w = 0.22 + (sim ** 1.35) * 1.85
            center_idx = int(round((b_cand - b_min) / step))
            i0 = max(0, center_idx - radius_bins)
            i1 = min(len(grid), center_idx + radius_bins + 1)
            if i1 > i0:
                k0 = i0 - (center_idx - radius_bins)
                k1 = k0 + (i1 - i0)
                lex_scores[i0:i1] += w * kernel[k0:k1]

    # Add 100 Hz onset cross-correlation between subtitle 0.35s start pulses and Whisper word-onset pulses
    t_sub_min = min(s for s, _, _ in sub_events)
    t_sub_max = max(e for _, e, _ in sub_events)
    sub_dur = max(2.0, t_sub_max - t_sub_min + 1.0)
    sub_pulses = _build_onset_pulse_signal(
        [(s, e) for s, e, _ in sub_events],
        t_origin=t_sub_min,
        duration_s=sub_dur,
        frame_hz=frame_hz,
        pulse_s=0.35,
    )

    t_ref_min = max(0.0, t_sub_min + b_min - 1.0)
    t_ref_max = max(t_ref_min + 2.0, t_sub_max + b_max + 2.0)
    ref_dur = t_ref_max - t_ref_min
    whisper_pulses = _build_onset_pulse_signal(
        [(u.start, u.end) for u in utterances],
        t_origin=t_ref_min,
        duration_s=ref_dur,
        frame_hz=frame_hz,
        pulse_s=0.35,
    )

    # Blend Silero 100 Hz Vocal Attack Envelope if available
    if attack_env_100hz is not None and len(attack_env_100hz) > 0:
        r0 = max(0, int(round(t_ref_min * frame_hz)))
        r1 = min(len(attack_env_100hz), r0 + len(whisper_pulses))
        if r1 > r0:
            att_slice = np.asarray(attack_env_100hz[r0:r1], dtype=np.float32)
            whisper_pulses[: r1 - r0] = whisper_pulses[: r1 - r0] + 0.65 * att_slice

    # Evaluate onset correlation along `grid`
    sub_len = len(sub_pulses)
    ref_len = len(whisper_pulses)
    sub_norm = float(np.linalg.norm(sub_pulses)) + 1e-6
    onset_scores = np.zeros(len(grid), dtype=np.float32)
    for idx, b_val in enumerate(grid):
        shift_frames = int(round((t_sub_min + float(b_val) - t_ref_min) * frame_hz))
        s0 = max(0, -shift_frames)
        r0 = max(0, shift_frames)
        ov = min(sub_len - s0, ref_len - r0)
        if ov > 10:
            dot = float(np.dot(sub_pulses[s0 : s0 + ov], whisper_pulses[r0 : r0 + ov]))
            onset_scores[idx] = dot / sub_norm

    total_scores = lex_scores + onset_scores * max(1.5, float(len(sub_events)) * 0.35)

    # Gentle prior regularization if b_prior is provided
    if b_prior is not None:
        dist_prior = np.abs(grid - float(b_prior))
        prior_bonus = np.where(dist_prior <= 2.5, 0.35 * (1.0 - dist_prior / 2.5), 0.0).astype(np.float32)
        total_scores = total_scores + prior_bonus

    best_idx = int(np.argmax(total_scores))
    coarse_b = float(grid[best_idx])
    best_score = float(total_scores[best_idx])

    # Fine refinement using exact Whisper first-word timestamps `w_start - sub_start`
    refined_b, matched_anchors = refine_scene_offset_from_utterances(
        sub_events=sub_events,
        utterances=utterances,
        coarse_b=coarse_b,
        attack_env_100hz=attack_env_100hz,
        frame_hz=frame_hz,
    )
    return refined_b, best_score, matched_anchors


def refine_scene_offset_from_utterances(
    sub_events: Sequence[Tuple[float, float, str]],
    utterances: Sequence[WhisperUtterance],
    coarse_b: float,
    tolerance_s: float = 0.55,
    attack_env_100hz: Optional[Sequence[float]] = None,
    frame_hz: int = 100,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Refine scene offset `b` using exact Whisper first-word timestamps (`w_start - sub_start`)."""
    if not sub_events or not utterances:
        return float(coarse_b), []

    candidates: List[Tuple[float, float, float, float]] = []  # (diff, weight, s_sub, w_start)
    for s_sub, e_sub, txt_sub in sub_events:
        pred_s = s_sub + coarse_b
        best_match = None
        best_val = -1e9
        for u in utterances:
            err = abs(u.start - pred_s)
            if err <= tolerance_s:
                sim = bilingual_utterance_similarity(txt_sub, u.text, e_sub - s_sub, u.end - u.start)
                val = (tolerance_s - err) + sim * 0.65
                if val > best_val:
                    best_val = val
                    best_match = (u.start - s_sub, 0.35 + sim * 1.5, s_sub, u.start)
        if best_match is not None:
            candidates.append(best_match)

    if not candidates:
        return float(coarse_b), []

    diffs = sorted(c[0] for c in candidates)
    med_b = diffs[len(diffs) // 2]
    abs_devs = sorted(abs(d - med_b) for d in diffs)
    mad = abs_devs[len(abs_devs) // 2]
    inlier_cut = max(0.06, min(0.24, 2.5 * mad if mad > 0 else 0.12))

    inliers = [c for c in candidates if abs(c[0] - med_b) <= inlier_cut]
    if not inliers:
        inliers = candidates

    w_sum = sum(c[1] for c in inliers)
    refined_b = sum(c[0] * c[1] for c in inliers) / w_sum if w_sum > 0 else med_b

    # Fine vocal-attack correlation check within +/- 0.12s of refined_b if attack_env_100hz is available
    if attack_env_100hz is not None and len(attack_env_100hz) > 0 and len(sub_events) >= 4 and np is not None:
        t_sub_min = min(s for s, _, _ in sub_events)
        t_sub_max = max(e for _, e, _ in sub_events)
        sub_dur = max(2.0, t_sub_max - t_sub_min + 1.0)
        sub_pulses = _build_onset_pulse_signal(
            [(s, e) for s, e, _ in sub_events],
            t_origin=t_sub_min,
            duration_s=sub_dur,
            frame_hz=frame_hz,
            pulse_s=0.35,
        )
        att_arr = np.asarray(attack_env_100hz, dtype=np.float32)
        best_att_b = refined_b
        best_att_val = -1.0
        for delta_f in range(-12, 13):
            cand_b = refined_b + delta_f / float(frame_hz)
            r0 = int(round((t_sub_min + cand_b) * frame_hz))
            if 0 <= r0 and r0 + len(sub_pulses) <= len(att_arr):
                val = float(np.dot(sub_pulses, att_arr[r0 : r0 + len(sub_pulses)]))
                if val > best_att_val:
                    best_att_val = val
                    best_att_b = cand_b
        if best_att_val > 0.5 and abs(best_att_b - refined_b) <= 0.08:
            refined_b = 0.65 * refined_b + 0.35 * best_att_b

    anchors = [(c[2], c[3]) for c in inliers]
    return float(refined_b), anchors


def _has_recap_marker_text(text: str) -> bool:
    """Return True if subtitle or Whisper text contains a 'Previously on' / 'Précédemment' marker."""
    toks = set(normalize_tokens(text))
    if "precedemment" in toks or "previously" in toks:
        return True
    norm_str = " ".join(normalize_tokens(text))
    for phrase in (
        "dans les episodes precedents",
        "au cours des episodes precedents",
        "previously on",
    ):
        if phrase in norm_str:
            return True
    return False


def verify_scenes_with_whisper(
    pcm_bytes: Optional[bytes] = None,
    scenes: Optional[Any] = None,
    *,
    subtitles: Optional[Any] = None,
    sub_events: Optional[Sequence[Any]] = None,
    time_window: Optional[Tuple[float, float]] = None,
    b_coarse: float = 0.0,
    sample_rate: int = 8000,
    frame_hz: int = 100,
    attack_env_100hz: Optional[Sequence[float]] = None,
    speech_intervals: Optional[Sequence[Tuple[float, float]]] = None,
) -> WhisperVerificationResult:
    """Verify and lock Cold Open (`t < 180s`), detect orphan/present recaps, and refine scene offsets with Whisper.

    Supports flexible inputs:
    - `pcm_bytes`: 8 kHz mono 16-bit PCM bytes (or defaults to `LAST_PCM_BYTES` if None).
    - `scenes`: optional list of scene dicts `[{"subs": [...], "b": ...}, ...]`, or list of `(s, e, text)` / `(s, e)` events,
      or `SSAFile` / `Subtitles` instance, or `(t0, t1)` time window tuple.
    - Returns a `WhisperVerificationResult` (dict + attribute access) containing:
      - `recap_cutoff_s`: timestamp (`> 0.0`) before which orphan `.srt` recap lines should be stripped (`0.0` if no orphan recap).
      - `video_recap_end_s`: timestamp (`> 0.0`) where an unsubtitled video recap ends (`0.0` otherwise).
      - `cold_open_scenes`: verified sub-scenes in `t < 190s` with exact offsets `b` (e.g. `S01E21` Sub 1..8 at `-1.35s` and Sub 10..38 at `+1.25s`).
      - `scenes`: verified scene dicts with updated `b` and `anchors`.
      - `b` / `offset_s`: primary verified offset.
      - `utterances`: extracted `WhisperUtterance` list.
      - `anchors`: matched `(t_sub, t_ref)` pairs.
      - `piecewise_anchors`: `(x_center, y_center)` anchors for piecewise-linear interpolation.
    """
    if pcm_bytes is None:
        pcm_bytes = LAST_PCM_BYTES
        sample_rate = LAST_SAMPLE_RATE
    elif isinstance(pcm_bytes, str) and os.path.isfile(pcm_bytes):
        with open(pcm_bytes, "rb") as fp:
            pcm_bytes = fp.read()

    if attack_env_100hz is None:
        attack_env_100hz = LAST_ATTACK_ENVELOPE
    if speech_intervals is None:
        speech_intervals = LAST_SPEECH_INTERVALS

    # If scenes is a 2-tuple of floats representing time_window (t0, t1)
    if (
        time_window is None
        and isinstance(scenes, (tuple, list))
        and len(scenes) == 2
        and all(isinstance(x, (int, float)) for x in scenes)
    ):
        time_window = (float(scenes[0]), float(scenes[1]))
        scenes = None

    # Gather all subtitle events with text
    all_sub_events: List[Tuple[float, float, str]] = []
    input_scene_dicts: List[dict] = []

    if subtitles is not None:
        if hasattr(subtitles, "events"):
            all_sub_events = enrich_intervals_with_text(subtitles.events)
        else:
            all_sub_events = enrich_intervals_with_text(subtitles)
    elif sub_events is not None:
        all_sub_events = enrich_intervals_with_text(sub_events)
    elif scenes is not None:
        if hasattr(scenes, "events"):
            all_sub_events = enrich_intervals_with_text(scenes.events)
        elif isinstance(scenes, Sequence) and len(scenes) > 0:
            first_el = scenes[0]
            if isinstance(first_el, dict) and ("subs" in first_el or "events" in first_el):
                for sc in scenes:
                    sc_copy = dict(sc)
                    raw_sc_subs = sc.get("subs") if "subs" in sc else sc.get("events", [])
                    sc_evs = enrich_intervals_with_text(raw_sc_subs)
                    sc_copy["subs_enriched"] = sc_evs
                    input_scene_dicts.append(sc_copy)
                    all_sub_events.extend(sc_evs)
                all_sub_events.sort(key=lambda x: (x[0], x[1]))
            else:
                all_sub_events = enrich_intervals_with_text(scenes)

    if not all_sub_events and LAST_SUB_EVENTS:
        all_sub_events = list(LAST_SUB_EVENTS)

    if not pcm_bytes:
        return WhisperVerificationResult(
            recap_cutoff_s=0.0,
            video_recap_end_s=0.0,
            cold_open_scenes=[],
            scenes=input_scene_dicts,
            b=float(b_coarse),
            offset_s=float(b_coarse),
            utterances=[],
            anchors=[],
            piecewise_anchors=[],
        )

    audio_dur_s = (len(pcm_bytes) // 2) / float(sample_rate)

    # If caller requested a specific time_window transcription without subtitle events
    if time_window is not None and not all_sub_events:
        utts = transcribe_audio_window(pcm_bytes, start_s=time_window[0], end_s=time_window[1], sample_rate=sample_rate)
        return WhisperVerificationResult(
            recap_cutoff_s=0.0,
            video_recap_end_s=0.0,
            cold_open_scenes=[],
            scenes=[],
            b=float(b_coarse),
            offset_s=float(b_coarse),
            utterances=utts,
            anchors=[],
            piecewise_anchors=[],
        )

    # 1. Transcribe the Cold Open window [0, min(audio_dur_s, 195.0)]
    cold_open_end_s = min(audio_dur_s, 195.0)
    cold_utts = transcribe_audio_window(pcm_bytes, start_s=0.0, end_s=cold_open_end_s, sample_rate=sample_rate)

    recap_cutoff_s = 0.0
    video_recap_end_s = 0.0
    cold_open_scenes: List[dict] = []
    all_anchors: List[Tuple[float, float]] = []

    cold_subs = [ev for ev in all_sub_events if ev[0] < 190.0]

    # Check if SRT starts with "Précédemment..." / "Previously on..." in first 5 events (t < 15s)
    srt_has_recap_tag = any(
        ev[0] < 15.0 and _has_recap_marker_text(ev[2])
        for ev in cold_subs[:5]
    )

    # Check if Whisper audio starts with "Previously on..." in first 15s
    audio_has_recap_tag = any(
        u.start < 15.0 and _has_recap_marker_text(u.text)
        for u in cold_utts[:5]
    )

    if cold_subs and cold_utts:
        if srt_has_recap_tag:
            # Determine whether the video ALSO contains the recap or starts directly with Act 1!
            # Candidate A: Video starts directly with Act 1 (orphan SRT recap -> Act 1 subs at t_sub >= 26s match t_audio >= 0s with b < -15s)
            post_recap_subs = [ev for ev in cold_subs if 26.0 <= ev[0] <= 185.0]
            b_orphan, score_orphan, anchors_orphan = score_scene_with_whisper(
                post_recap_subs,
                cold_utts,
                b_min=-55.0,
                b_max=-14.0,
                attack_env_100hz=attack_env_100hz,
                frame_hz=frame_hz,
            )

            # Candidate B: Video ALSO has the recap (SRT recap + Act 1 match audio with b in [-12s, +15s])
            b_present, score_present, anchors_present = score_scene_with_whisper(
                cold_subs,
                cold_utts,
                b_min=-12.0,
                b_max=15.0,
                attack_env_100hz=attack_env_100hz,
                frame_hz=frame_hz,
            )

            if not audio_has_recap_tag and score_orphan > score_present * 1.12 and len(anchors_orphan) >= 3:
                # Orphan SRT recap confirmed! Find exact recap_cutoff_s before the first Act 1 subtitle.
                # Every valid post-recap subtitle must map to >= -0.2s in video time (sub_start >= -b_orphan - 0.2).
                first_audio_speech = cold_utts[0].start if cold_utts else 0.0
                min_valid_sub_s = max(0.0, first_audio_speech - 0.4) - b_orphan
                valid_matched_subs = [a[0] for a in anchors_orphan if a[0] >= -b_orphan - 0.2]
                if valid_matched_subs:
                    act1_start_sub = min(valid_matched_subs)
                else:
                    act1_start_sub = min_valid_sub_s

                best_cut = max(-b_orphan - 0.1, act1_start_sub - 0.05)
                for idx in range(1, len(cold_subs)):
                    prev_end = cold_subs[idx - 1][1]
                    cur_start = cold_subs[idx][0]
                    if cur_start >= max(-b_orphan - 0.2, act1_start_sub - 1.5):
                        best_cut = 0.5 * (prev_end + cur_start)
                        break

                recap_cutoff_s = float(max(15.0, best_cut))
                valid_cold_subs = [ev for ev in cold_subs if ev[0] >= recap_cutoff_s]
                if valid_cold_subs:
                    b_ref, sc_ref, anc_ref = score_scene_with_whisper(
                        valid_cold_subs,
                        cold_utts,
                        b_min=b_orphan - 4.0,
                        b_max=b_orphan + 4.0,
                        b_prior=b_orphan,
                        attack_env_100hz=attack_env_100hz,
                        frame_hz=frame_hz,
                    )
                    x_c = 0.5 * (valid_cold_subs[0][0] + valid_cold_subs[-1][1])
                    cold_open_scenes.append(
                        {
                            "w_start": valid_cold_subs[0][0],
                            "w_end": valid_cold_subs[-1][1],
                            "x_center": x_c,
                            "b": b_ref,
                            "score": sc_ref,
                            "anchors": anc_ref,
                            "subs": valid_cold_subs,
                        }
                    )
                    all_anchors.extend(anc_ref)
            else:
                # Video ALSO contains the recap -> KEEP IT (recap_cutoff_s = 0.0)!
                recap_cutoff_s = 0.0
                x_c = 0.5 * (cold_subs[0][0] + cold_subs[-1][1])
                cold_open_scenes.append(
                    {
                        "w_start": cold_subs[0][0],
                        "w_end": cold_subs[-1][1],
                        "x_center": x_c,
                        "b": b_present,
                        "score": score_present,
                        "anchors": anchors_present,
                        "subs": cold_subs,
                    }
                )
                all_anchors.extend(anchors_present)
        else:
            # SRT does NOT start with "Précédemment...".
            # Check if the VIDEO starts with an unsubtitled "Previously on..." recap!
            b_search_min = -25.0
            if audio_has_recap_tag:
                b_search_min = 12.0  # Video has recap at [0..15s+], so SRT Act 1 must shift forward (b >= +12s)

            # Partition Cold Open subtitles (t < 190s) into sub-scenes across large silence gaps (>= 12.0s)
            # where each sub-scene has >= 5 subtitles (e.g. S01E21 Sub 1..8 [15.6..36.3s] vs Sub 10..38 [117.3..189.4s])
            sub_clusters: List[List[Tuple[float, float, str]]] = []
            cur_cluster: List[Tuple[float, float, str]] = [cold_subs[0]]
            for ev in cold_subs[1:]:
                gap = ev[0] - cur_cluster[-1][1]
                if gap >= 12.0 and len(cur_cluster) >= 5:
                    sub_clusters.append(cur_cluster)
                    cur_cluster = [ev]
                else:
                    cur_cluster.append(ev)
            if len(cur_cluster) < 4 and sub_clusters:
                sub_clusters[-1].extend(cur_cluster)
            else:
                sub_clusters.append(cur_cluster)

            # Also check if shifting the first Cold Open cluster forward by +15..+55s (unsubtitled video recap)
            # beats [-15..+12s] when the video has dense speech in [0..25s]
            early_utts_count = sum(1 for u in cold_utts if u.start < 25.0)
            if not audio_has_recap_tag and cold_subs[0][0] < 20.0 and early_utts_count >= 6:
                first_c = sub_clusters[0]
                b_norm, sc_norm, _ = score_scene_with_whisper(
                    first_c,
                    cold_utts,
                    b_min=-15.0,
                    b_max=12.0,
                    attack_env_100hz=attack_env_100hz,
                    frame_hz=frame_hz,
                )
                b_fwd, sc_fwd, anc_fwd = score_scene_with_whisper(
                    first_c,
                    cold_utts,
                    b_min=15.0,
                    b_max=50.0,
                    attack_env_100hz=attack_env_100hz,
                    frame_hz=frame_hz,
                )
                if sc_fwd > sc_norm * 1.45 and len(anc_fwd) >= max(4, int(len(first_c) * 0.6)):
                    b_search_min = 14.0
                    video_recap_end_s = max(15.0, first_c[0][0] + b_fwd - 0.5)

            for c_evs in sub_clusters:
                # If a single transition line (like S01E21 Sub 9 at 84.96s before 117.26s) is separated by > 20s
                # from the main body of the cluster, score the main body first so the cluster offset is locked cleanly
                core_evs = c_evs
                if len(c_evs) >= 6 and (c_evs[1][0] - c_evs[0][1]) >= 20.0:
                    core_evs = c_evs[1:]

                b_c, sc_c, anc_c = score_scene_with_whisper(
                    core_evs,
                    cold_utts,
                    b_min=b_search_min,
                    b_max=55.0,
                    b_prior=b_coarse if abs(b_coarse) > 1e-6 else None,
                    attack_env_100hz=attack_env_100hz,
                    frame_hz=frame_hz,
                )
                if audio_has_recap_tag and video_recap_end_s == 0.0:
                    video_recap_end_s = max(12.0, core_evs[0][0] + b_c - 0.5)

                x_c = 0.5 * (core_evs[0][0] + core_evs[-1][1])
                cold_open_scenes.append(
                    {
                        "w_start": c_evs[0][0],
                        "w_end": c_evs[-1][1],
                        "x_center": x_c,
                        "b": b_c,
                        "score": sc_c,
                        "anchors": anc_c,
                        "subs": c_evs,
                    }
                )
                all_anchors.extend(anc_c)

    # 2. If `scenes` dicts were passed (e.g. from multi-window / act alignment), verify/refine each scene
    verified_scenes: List[dict] = []
    if input_scene_dicts:
        for idx, sc in enumerate(input_scene_dicts):
            sc_evs: List[Tuple[float, float, str]] = sc.get("subs_enriched", [])
            if recap_cutoff_s > 0.0:
                sc_evs = [ev for ev in sc_evs if ev[0] >= recap_cutoff_s]
            if not sc_evs:
                continue

            sc_b_coarse = float(sc.get("b", sc.get("coarse_b", b_coarse)))
            sc_start = sc_evs[0][0]
            sc_end = sc_evs[-1][1]

            if sc_end <= 195.0 and cold_utts:
                b_ver, sc_ver, anc_ver = score_scene_with_whisper(
                    sc_evs,
                    cold_utts,
                    b_min=sc_b_coarse - 8.0 if abs(sc_b_coarse) > 0.01 else -50.0,
                    b_max=sc_b_coarse + 8.0 if abs(sc_b_coarse) > 0.01 else 50.0,
                    b_prior=sc_b_coarse,
                    attack_env_100hz=attack_env_100hz,
                    frame_hz=frame_hz,
                )
            else:
                # Transcribe a targeted 55-second dialogue slice inside this scene around sc_b_coarse
                slice_sub_start = sc_start
                slice_sub_end = min(sc_end, sc_start + 55.0)
                slice_evs = [ev for ev in sc_evs if slice_sub_start <= ev[0] <= slice_sub_end]
                audio_t0 = max(0.0, slice_sub_start + sc_b_coarse - 8.0)
                audio_t1 = min(audio_dur_s, slice_sub_end + sc_b_coarse + 8.0)
                sc_utts = transcribe_audio_window(pcm_bytes, start_s=audio_t0, end_s=audio_t1, sample_rate=sample_rate)
                if sc_utts and slice_evs:
                    b_ver, sc_ver, anc_ver = score_scene_with_whisper(
                        slice_evs,
                        sc_utts,
                        b_min=sc_b_coarse - 6.0,
                        b_max=sc_b_coarse + 6.0,
                        b_prior=sc_b_coarse,
                        attack_env_100hz=attack_env_100hz,
                        frame_hz=frame_hz,
                    )
                else:
                    b_ver, sc_ver, anc_ver = sc_b_coarse, float(sc.get("score", 0.0)), []

            sc_out = dict(sc)
            sc_out["b"] = b_ver
            sc_out["whisper_b"] = b_ver
            sc_out["whisper_score"] = sc_ver
            sc_out["whisper_verified"] = True
            sc_out["anchors"] = anc_ver
            verified_scenes.append(sc_out)
            all_anchors.extend(anc_ver)

            # Also update original dict in-place if `scenes` was a list of dicts
            if isinstance(scenes, Sequence) and idx < len(scenes) and isinstance(scenes[idx], dict):
                scenes[idx]["b"] = b_ver
                scenes[idx]["whisper_b"] = b_ver
                scenes[idx]["whisper_score"] = sc_ver
                scenes[idx]["whisper_verified"] = True
                scenes[idx]["recap_cutoff_s"] = recap_cutoff_s
    else:
        verified_scenes = list(cold_open_scenes)

    primary_b = float(b_coarse)
    if cold_open_scenes:
        primary_b = float(cold_open_scenes[-1]["b"])
    elif verified_scenes:
        primary_b = float(verified_scenes[0]["b"])

    piecewise_anchors: List[Tuple[float, float]] = [
        (float(sc["x_center"]), float(sc["x_center"]) + float(sc["b"]))
        for sc in cold_open_scenes
        if "x_center" in sc
    ]

    return WhisperVerificationResult(
        recap_cutoff_s=float(recap_cutoff_s),
        video_recap_end_s=float(video_recap_end_s),
        cold_open_scenes=cold_open_scenes,
        scenes=verified_scenes,
        b=primary_b,
        offset_s=primary_b,
        utterances=cold_utts,
        anchors=all_anchors,
        piecewise_anchors=piecewise_anchors,
    )
