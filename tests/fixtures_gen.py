"""Deterministic synthetic media and subtitle fixture generator for subsync tests.

Generates speech-like WAV and MKV media files (multi-tone vocal formant cadence
bursts at 250Hz + 500Hz + 1100Hz with syllable-rate amplitude modulation) at
known ground-truth timestamps [start_s, end_s] over durations of 120s to 600s,
along with ground-truth and distorted English/French SRT subtitle files, and
evaluates 0.1s alignment accuracy.
"""

import array
import math
import os
import random
import re
import struct
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


FFMPEG_BIN = "/usr/bin/ffmpeg"

# 5 canonical ground-truth distortion cases specified in R1 verification
GROUND_TRUTH_CASES: Dict[str, Dict[str, Any]] = {
    "case1_pos_offset": {
        "name": "case1_pos_offset",
        "description": "Pure positive offset +2.350s (a = 1.0)",
        "a": 1.0,
        "b": 2.350,
    },
    "case2_neg_offset": {
        "name": "case2_neg_offset",
        "description": "Pure negative offset -1.800s (a = 1.0)",
        "a": 1.0,
        "b": -1.800,
    },
    "case3_large_offset": {
        "name": "case3_large_offset",
        "description": "Large offset +8.420s (a = 1.0)",
        "a": 1.0,
        "b": 8.420,
    },
    "case4_subsec_offset": {
        "name": "case4_subsec_offset",
        "description": "Sub-second offset -0.650s (a = 1.0)",
        "a": 1.0,
        "b": -0.650,
    },
    "case5_framerate_drift": {
        "name": "case5_framerate_drift",
        "description": "Framerate drift 25.0 / 23.976 (a = 1.042709, b = +1.500s)",
        "a": 25.0 / 23.976,
        "b": 1.500,
    },
}

ENGLISH_LINES: List[str] = [
    "We need to review the classified dossier before the task force arrives.",
    "Raymond Reddington surrendered himself at headquarters this morning.",
    "Every name on the blacklist represents a high priority global target.",
    "Check the perimeter and secure all exits around the diplomatic compound.",
    "The encrypted transmission originated from a private server in Zurich.",
    "Agent Keen requested immediate backup near the harbor warehouse district.",
    "Do not underestimate how quickly he can vanish into the underground network.",
    "Forensic analysis confirmed the passport was forged in Vienna last week.",
    "They scheduled the clandestine exchange at the train station at midnight.",
    "Our surveillance team lost visual contact inside the crowded market square.",
    "The witness identified the courier carrying the diplomatic briefcase.",
    "Director Cooper authorized full tactical support for this operation.",
    "All flight manifests from Dulles Airport have been cross-referenced.",
    "He left a hidden message inside the vintage music box on the desk.",
    "Trace the wire transfer through the offshore banking accounts immediately.",
]

FRENCH_LINES: List[str] = [
    "Ça doit être bon, nous devons vérifier le dossier confidentiel immédiatement.",
    "Raymond Reddington s'est rendu de lui-même au quartier général ce matin.",
    "Chaque nom sur la liste noire représente une cible internationale très dangereuse.",
    "Vérifiez le périmètre et sécurisez toutes les issues autour du bâtiment.",
    "La transmission chiffrée provenait d'un serveur privé situé à Genève.",
    "L'agent Keen a demandé des renforts immédiats près de l'entrepôt du port.",
    "Ne sous-estimez jamais la rapidité avec laquelle son réseau peut disparaître.",
    "L'analyse médico-légale a confirmé que ce passeport était falsifié dès le début.",
    "Ils ont prévu l'échange clandestin à la gare centrale juste avant minuit.",
    "Notre équipe de surveillance a perdu le contact visuel sur la grande place.",
    "Le témoin a formellement identifié le messager portant la mallette diplomatique.",
    "Le directeur Cooper a autorisé un soutien tactique complet pour cette mission.",
    "Tous les manifestes de vol ont été minutieusement contrôlés par nos services.",
    "Il a laissé un message caché à l'intérieur de la boîte à musique ancienne.",
    "Suivez le virement bancaire à travers les comptes étrangers sans perdre une seconde.",
]


class AlignmentEvaluationResult(dict):
    """Dictionary result of alignment accuracy evaluation with attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def generate_dialogue_bursts(
    num_bursts: int = 35,
    duration_s: float = 180.0,
    seed: int = 42,
    start_margin_s: float = 14.0,
    end_margin_s: float = 14.0,
) -> List[Tuple[float, float]]:
    """Generate 25 to 60 distinct, non-periodic dialogue bursts [start_s, end_s].

    The bursts have varied durations and inter-burst silence gaps so that the
    resulting voice activity envelope has a unique, sharp cross-correlation peak
    and wide temporal baseline for slope/framerate estimation.
    """
    if not (25 <= num_bursts <= 60):
        raise ValueError(f"num_bursts must be between 25 and 60, got {num_bursts}")
    if not (120.0 <= duration_s <= 600.0):
        raise ValueError(f"duration_s must be between 120.0 and 600.0, got {duration_s}")

    rng = random.Random(seed)
    available_span = duration_s - start_margin_s - end_margin_s
    if available_span <= num_bursts * 1.8:
        start_margin_s = max(10.5, duration_s * 0.08)
        end_margin_s = max(10.5, duration_s * 0.08)
        available_span = duration_s - start_margin_s - end_margin_s

    raw_durations = [rng.uniform(1.35, 3.25) for _ in range(num_bursts)]
    raw_gaps = [rng.uniform(1.10, 4.60) for _ in range(num_bursts - 1)]

    total_raw = sum(raw_durations) + sum(raw_gaps)
    scale = available_span / total_raw

    bursts: List[Tuple[float, float]] = []
    cursor = start_margin_s
    for i in range(num_bursts):
        dur = max(0.95, round(raw_durations[i] * scale, 3))
        start_t = round(cursor, 3)
        end_t = round(start_t + dur, 3)
        bursts.append((start_t, end_t))
        if i < num_bursts - 1:
            gap = max(0.65, round(raw_gaps[i] * scale, 3))
            cursor = end_t + gap

    return bursts


def _build_formant_period_samples(
    sample_rate: int = 16000, peak_amplitude: int = 24000
) -> array.array:
    """Build a 0.5-second (half-sample-rate) multi-tone vocal formant template.

    Combines 250Hz + 500Hz + 1100Hz vocal formants modulated by a 4Hz syllable
    cadence envelope. All frequencies complete an exact integer number of cycles
    in 0.5 seconds, allowing fast slice tiling with zero phase discontinuity.
    """
    period_samples = sample_rate // 2
    buf = array.array("h", [0] * period_samples)
    two_pi = 2.0 * math.pi
    for i in range(period_samples):
        t = i / float(sample_rate)
        # 4.0 Hz syllable cadence modulation (stays between 0.45 and 1.0)
        envelope = 0.725 + 0.275 * math.sin(two_pi * 4.0 * t)
        # Vocal formants: F0/F1=250Hz, F1=500Hz, F2=1100Hz
        formants = (
            0.50 * math.sin(two_pi * 250.0 * t)
            + 0.35 * math.sin(two_pi * 500.0 * t)
            + 0.15 * math.sin(two_pi * 1100.0 * t)
        )
        sample_val = int(round(peak_amplitude * envelope * formants))
        if sample_val > 32767:
            sample_val = 32767
        elif sample_val < -32768:
            sample_val = -32768
        buf[i] = sample_val
    return buf


def generate_synthetic_wav(
    wav_path: Union[str, Path],
    bursts: Sequence[Tuple[float, float]],
    duration_s: float = 180.0,
    sample_rate: int = 16000,
) -> str:
    """Generate a synthetic mono 16-bit PCM WAV file with vocal formant bursts.

    Uses Python's standard `wave`, `struct`, and `math` modules.
    """
    wav_path = str(wav_path)
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)), exist_ok=True)

    total_samples = int(round(duration_s * sample_rate))
    pcm = array.array("h", [0]) * total_samples
    template = _build_formant_period_samples(sample_rate=sample_rate)
    period_len = len(template)
    fade_samples = min(int(0.005 * sample_rate), 80)  # 5ms attack/release ramp

    for start_s, end_s in bursts:
        i0 = max(0, min(total_samples, int(round(start_s * sample_rate))))
        i1 = max(i0, min(total_samples, int(round(end_s * sample_rate))))
        n = i1 - i0
        if n <= 0:
            continue
        full_periods, rem = divmod(n, period_len)
        burst_slice = template * full_periods + template[:rem]
        # Crisp 5ms attack/release to prevent click artifacts while keeping exact edges
        ramp = min(fade_samples, n // 4)
        for k in range(ramp):
            scale = (k + 1) / float(ramp)
            burst_slice[k] = int(struct.unpack("<h", struct.pack("<h", int(burst_slice[k] * scale)))[0])
            burst_slice[n - 1 - k] = int(
                struct.unpack("<h", struct.pack("<h", int(burst_slice[n - 1 - k] * scale)))[0]
            )
        pcm[i0:i1] = burst_slice

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return wav_path


def generate_synthetic_mkv(
    mkv_path: Union[str, Path],
    wav_path: Optional[Union[str, Path]] = None,
    bursts: Optional[Sequence[Tuple[float, float]]] = None,
    duration_s: float = 180.0,
    audio_lang: str = "eng",
    sample_rate: int = 16000,
) -> str:
    """Generate a synthetic MKV video file with black video + AAC audio via ffmpeg.

    Compensates for the 1024-sample AAC priming delay at 48kHz (0.021333s) so that
    decoded audio bursts align with ground-truth [start_s, end_s] to < 1ms.
    """
    mkv_path = str(mkv_path)
    os.makedirs(os.path.dirname(os.path.abspath(mkv_path)), exist_ok=True)

    temp_wav: Optional[str] = None
    if wav_path is None:
        if bursts is None:
            bursts = generate_dialogue_bursts(duration_s=duration_s)
        temp_wav = mkv_path + ".tmp.wav"
        generate_synthetic_wav(
            temp_wav, bursts=bursts, duration_s=duration_s, sample_rate=sample_rate
        )
        wav_source = temp_wav
    else:
        wav_source = str(wav_path)

    try:
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=160x90:r=5:d={duration_s:.3f}",
            "-i",
            wav_source,
            "-af",
            "aresample=48000,atrim=start=0.021333,asetpts=PTS-STARTPTS",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-b:a",
            "96k",
            "-metadata:s:a:0",
            f"language={audio_lang}",
            "-shortest",
            mkv_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except OSError:
                pass

    return mkv_path


def format_srt_timestamp(seconds: float) -> str:
    """Format a timestamp in seconds as SRT 'HH:MM:SS,mmm'."""
    total_ms = max(0, int(round(seconds * 1000.0)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


_SRT_TS_RE = re.compile(
    r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _hms_ms_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    ms_padded = ms.ljust(3, "0")[:3]
    return int(h) * 3600.0 + int(m) * 60.0 + int(s) + int(ms_padded) / 1000.0


def parse_srt_file(srt_path_or_content: Union[str, Path]) -> List[Dict[str, Any]]:
    """Parse an SRT file (or raw SRT content) into a list of subtitle event dicts."""
    if isinstance(srt_path_or_content, Path) or (
        isinstance(srt_path_or_content, str)
        and "\n" not in srt_path_or_content
        and os.path.exists(srt_path_or_content)
    ):
        raw_bytes = Path(srt_path_or_content).read_bytes()
        for enc in ("utf-8-sig", "utf-8", "cp1252", "iso-8859-1"):
            try:
                content = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            content = raw_bytes.decode("utf-8", errors="replace")
    else:
        content = str(srt_path_or_content)

    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        return []

    blocks = re.split(r"\n\s*\n", content)
    events: List[Dict[str, Any]] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        ts_line_idx = -1
        match = None
        for idx, line in enumerate(lines):
            m = _SRT_TS_RE.search(line)
            if m:
                ts_line_idx = idx
                match = m
                break
        if match is None:
            continue
        start_s = _hms_ms_to_seconds(*match.groups()[:4])
        end_s = _hms_ms_to_seconds(*match.groups()[4:8])
        text = "\n".join(lines[ts_line_idx + 1 :])
        events.append(
            {
                "index": len(events) + 1,
                "start_s": start_s,
                "end_s": end_s,
                "text": text,
            }
        )
    return events


def write_srt_file(
    srt_path: Union[str, Path],
    events: Sequence[Union[Tuple[float, float, str], Dict[str, Any]]],
    encoding: str = "utf-8",
) -> str:
    """Write subtitle events to an SRT file."""
    srt_path = str(srt_path)
    os.makedirs(os.path.dirname(os.path.abspath(srt_path)), exist_ok=True)
    blocks: List[str] = []
    for idx, ev in enumerate(events, start=1):
        if isinstance(ev, dict):
            start_s = float(ev["start_s"])
            end_s = float(ev["end_s"])
            text = str(ev["text"])
        else:
            start_s, end_s, text = float(ev[0]), float(ev[1]), str(ev[2])
        start_ts = format_srt_timestamp(start_s)
        end_ts = format_srt_timestamp(end_s)
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")

    with open(srt_path, "w", encoding=encoding) as fp:
        fp.write("\n".join(blocks))
    return srt_path


def get_dialogue_text(index: int, lang: str = "en") -> str:
    """Return deterministic English or French dialogue text for burst `index`."""
    lang_norm = (lang or "en").lower()
    if lang_norm.startswith("fr"):
        pool = FRENCH_LINES
    else:
        pool = ENGLISH_LINES
    return pool[index % len(pool)]


def generate_ground_truth_srt(
    srt_path: Union[str, Path],
    bursts: Sequence[Tuple[float, float]],
    lang: str = "en",
    encoding: str = "utf-8",
) -> str:
    """Generate a ground-truth .srt subtitle file matching `bursts` in English or French."""
    events = [
        (start_s, end_s, get_dialogue_text(i, lang=lang))
        for i, (start_s, end_s) in enumerate(bursts)
    ]
    return write_srt_file(srt_path, events, encoding=encoding)


def generate_distorted_srt(
    srt_path: Union[str, Path],
    bursts_or_gt: Union[Sequence[Tuple[float, float]], str, Path],
    offset_s: float = 0.0,
    scale_a: float = 1.0,
    lang: str = "en",
    encoding: str = "utf-8",
) -> str:
    """Generate a distorted .srt subtitle file with known ground-truth transformation.

    Applies the inverse transform:
        t_distorted = (t_gt - offset_s) / scale_a
    so that the ground-truth synchronization formula mapping `distorted.srt` back
    to `gt.srt` is:
        t_synced = scale_a * t_distorted + offset_s  (i.e. a = scale_a, b = offset_s).
    """
    if isinstance(bursts_or_gt, (str, Path)):
        parsed = parse_srt_file(bursts_or_gt)
        raw_events = [(ev["start_s"], ev["end_s"], ev["text"]) for ev in parsed]
    else:
        raw_events = [
            (
                float(b[0]),
                float(b[1]),
                b[2] if len(b) > 2 else get_dialogue_text(i, lang=lang),
            )
            for i, b in enumerate(bursts_or_gt)
        ]

    distorted_events: List[Tuple[float, float, str]] = []
    for start_s, end_s, text in raw_events:
        d_start = (start_s - offset_s) / scale_a
        d_end = (end_s - offset_s) / scale_a
        if d_start < 0.001:
            d_start = 0.001
        if d_end <= d_start:
            d_end = d_start + 0.500
        distorted_events.append((round(d_start, 3), round(d_end, 3), text))

    return write_srt_file(srt_path, distorted_events, encoding=encoding)


def evaluate_alignment_accuracy(
    synced_srt_path: Union[str, Path],
    ground_truth_srt_path: Union[str, Path],
    tolerance_s: float = 0.100,
    max_mae_s: float = 0.050,
    raise_on_failure: bool = True,
) -> AlignmentEvaluationResult:
    """Parse synced and ground-truth SRT files and verify <= 0.100s alignment accuracy.

    Computes per-line start and end timestamp errors in seconds, and asserts:
      - max_error_s <= tolerance_s (default 0.100s)
      - mean_abs_error_s <= max_mae_s (default 0.050s)
    """
    synced_events = parse_srt_file(synced_srt_path)
    gt_events = parse_srt_file(ground_truth_srt_path)

    if len(synced_events) != len(gt_events) or len(gt_events) == 0:
        msg = (
            f"Subtitle line count mismatch: synced has {len(synced_events)} lines, "
            f"ground truth has {len(gt_events)} lines."
        )
        if raise_on_failure:
            raise AssertionError(msg)
        return AlignmentEvaluationResult(
            passed=False,
            max_error_s=float("inf"),
            mean_abs_error_s=float("inf"),
            mae_s=float("inf"),
            tolerance_s=tolerance_s,
            num_lines=len(synced_events),
            errors=[],
            error_message=msg,
        )

    all_errors: List[float] = []
    per_line_errors: List[Dict[str, float]] = []
    for idx, (s_ev, gt_ev) in enumerate(zip(synced_events, gt_events), start=1):
        start_err = abs(s_ev["start_s"] - gt_ev["start_s"])
        end_err = abs(s_ev["end_s"] - gt_ev["end_s"])
        all_errors.extend([start_err, end_err])
        per_line_errors.append(
            {
                "index": idx,
                "start_error_s": start_err,
                "end_error_s": end_err,
                "synced_start_s": s_ev["start_s"],
                "gt_start_s": gt_ev["start_s"],
                "synced_end_s": s_ev["end_s"],
                "gt_end_s": gt_ev["end_s"],
            }
        )

    max_error_s = max(all_errors) if all_errors else 0.0
    mean_abs_error_s = sum(all_errors) / len(all_errors) if all_errors else 0.0
    effective_mae_limit = min(max_mae_s, tolerance_s)
    passed = (max_error_s <= tolerance_s) and (mean_abs_error_s <= effective_mae_limit)

    result = AlignmentEvaluationResult(
        passed=passed,
        max_error_s=max_error_s,
        mean_abs_error_s=mean_abs_error_s,
        mae_s=mean_abs_error_s,
        tolerance_s=tolerance_s,
        max_mae_s=effective_mae_limit,
        num_lines=len(gt_events),
        errors=all_errors,
        per_line_errors=per_line_errors,
    )

    if raise_on_failure:
        assert max_error_s <= tolerance_s, (
            f"Alignment max_error_s={max_error_s:.4f}s exceeds tolerance "
            f"{tolerance_s:.4f}s (mean_abs_error_s={mean_abs_error_s:.4f}s)"
        )
        assert mean_abs_error_s <= effective_mae_limit, (
            f"Alignment mean_abs_error_s={mean_abs_error_s:.4f}s exceeds limit "
            f"{effective_mae_limit:.4f}s (max_error_s={max_error_s:.4f}s)"
        )

    return result


def generate_fixture_bundle(
    output_dir: Union[str, Path],
    num_bursts: int = 35,
    duration_s: float = 180.0,
    seed: int = 42,
    include_mkv: bool = True,
) -> Dict[str, Any]:
    """Generate a complete synthetic test bundle in `output_dir`.

    Creates:
      - ref.wav (16kHz mono PCM with 250/500/1100Hz vocal formant bursts)
      - ref.mkv (H.264 black video + 48kHz AAC audio stream tagged 'eng')
      - gt.srt (English ground-truth subtitles)
      - gt_fr.srt (French ground-truth subtitles)
      - distorted_<case>.srt for all 5 ground-truth distortion cases (plus French)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bursts = generate_dialogue_bursts(
        num_bursts=num_bursts, duration_s=duration_s, seed=seed
    )
    wav_path = generate_synthetic_wav(out / "ref.wav", bursts, duration_s=duration_s)
    mkv_path = None
    if include_mkv:
        mkv_path = generate_synthetic_mkv(
            out / "ref.mkv",
            wav_path=wav_path,
            bursts=bursts,
            duration_s=duration_s,
            audio_lang="eng",
        )

    gt_en_path = generate_ground_truth_srt(out / "gt.srt", bursts, lang="en")
    gt_fr_path = generate_ground_truth_srt(out / "gt_fr.srt", bursts, lang="fr")

    cases_out: Dict[str, Dict[str, Any]] = {}
    for key, spec in GROUND_TRUTH_CASES.items():
        dist_en = generate_distorted_srt(
            out / f"distorted_{key}.srt",
            bursts,
            offset_s=spec["b"],
            scale_a=spec["a"],
            lang="en",
        )
        dist_fr = generate_distorted_srt(
            out / f"distorted_{key}_fr.srt",
            bursts,
            offset_s=spec["b"],
            scale_a=spec["a"],
            lang="fr",
        )
        cases_out[key] = {
            **spec,
            "distorted_srt": dist_en,
            "distorted_fr_srt": dist_fr,
        }

    return {
        "output_dir": str(out),
        "bursts": bursts,
        "duration_s": duration_s,
        "wav_path": wav_path,
        "mkv_path": mkv_path,
        "gt_srt": gt_en_path,
        "gt_en_srt": gt_en_path,
        "gt_fr_srt": gt_fr_path,
        "cases": cases_out,
    }
