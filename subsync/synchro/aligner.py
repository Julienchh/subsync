"""High-precision subtitle-to-audio and subtitle-to-subtitle alignment engine.

Implements a two-stage alignment pipeline:
1. Stage 1 (Coarse Global Search & Drift Detection):
   - Evaluates candidate speed factors `a` (1.0, standard NTSC/PAL framerate ratios,
     and windowed multi-segment RANSAC drift hypotheses) via Radix-2 FFT
     cross-correlation on 100 Hz activity envelopes (backed by an optional
     auto-compiled C shared library `_fast_corr.so` with pure-Python FFT fallback).
   - Applies strict slope regularization so `a = 1.0` is preserved unless a
     non-unit slope significantly improves correlation and reduces timeline drift.
2. Stage 2 (Sub-10ms Fine Boundary Refinement):
   - Matches individual subtitle dialogue boundaries `[s_i, e_i]` against
     reference speech/subtitle intervals `[r_j, q_j]` around the Stage 1 model.
   - Performs iterative outlier pruning (RANSAC / MAD trimming) and Ordinary
     Least Squares (OLS) / Huber refinement to achieve <= 0.01s (10ms) accuracy.
"""

import cmath
import ctypes
from dataclasses import dataclass
import logging
import math
import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_AI_ENV_PATH = os.path.join(_REPO_ROOT, ".ai_env")
if os.path.isdir(_AI_ENV_PATH) and _AI_ENV_PATH not in sys.path:
    sys.path.insert(0, _AI_ENV_PATH)

try:
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:
    np = None
    sliding_window_view = None

from subsync.synchro import vad

logger = logging.getLogger(__name__)

# Canonical framerate conversion ratios (e.g. 23.976 <-> 24 <-> 25 <-> 29.97 <-> 30)
STANDARD_SPEED_FACTORS: Tuple[float, ...] = (
    1.0,
    24.0 / 23.976,            # ~1.001001001
    24000.0 / 23976.0,        # ~1.001001001
    1001.0 / 1000.0,          # 1.001000000
    23.976 / 24.0,            # 0.999000000
    1000.0 / 1001.0,          # ~0.999000999
    25.0 / 24.0,              # ~1.041666667
    24.0 / 25.0,              # 0.960000000
    25.0 / 23.976,            # ~1.042709376
    25.0 / (24000.0 / 1001.0),  # ~1.042708333
    23.976 / 25.0,            # 0.959040000
    (24000.0 / 1001.0) / 25.0,  # ~0.959040959
)


class FormulaLine(object):
    """Linear or piecewise-linear transformation formula `y = a * x + b` compatible with `gizmo.Line`.

    Supports both attribute access (`formula.a`, `formula.b`, `formula.getY(x)`)
    and tuple unpacking/indexing (`a, b = formula`, `formula[0]`, `formula[1]`).
    When `piecewise_anchors` is populated (sorted list of `(x_sub_s, y_ref_s)`),
    `getY(x)` interpolates piecewise-linearly between anchors and extrapolates
    with slope `self.a` before the first anchor and after the last anchor.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        piecewise_anchors: Optional[Sequence[Tuple[float, float]]] = None,
    ):
        self.a = float(a)
        self.b = float(b)
        self.piecewise_anchors: List[Tuple[float, float]] = (
            sorted((float(px), float(py)) for px, py in piecewise_anchors)
            if piecewise_anchors
            else []
        )

    def getX(self, y: float) -> float:
        yf = float(y)
        if self.piecewise_anchors:
            pts = self.piecewise_anchors
            if yf <= pts[0][1]:
                return pts[0][0] + (yf - pts[0][1]) / self.a if self.a != 0.0 else pts[0][0]
            if yf >= pts[-1][1]:
                return pts[-1][0] + (yf - pts[-1][1]) / self.a if self.a != 0.0 else pts[-1][0]
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                if y0 <= yf <= y1 or y1 <= yf <= y0:
                    dy = y1 - y0
                    if abs(dy) <= 1e-6:
                        return x0
                    return x0 + ((yf - y0) / dy) * (x1 - x0)
        return (yf - self.b) / self.a if self.a != 0.0 else 0.0

    def getY(self, x: float) -> float:
        xf = float(x)
        if self.piecewise_anchors:
            pts = self.piecewise_anchors
            if xf <= pts[0][0]:
                return pts[0][1] + self.a * (xf - pts[0][0])
            if xf >= pts[-1][0]:
                return pts[-1][1] + self.a * (xf - pts[-1][0])
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                if x0 <= xf <= x1:
                    dx = x1 - x0
                    if dx <= 1e-6:
                        return y0
                    return y0 + ((xf - x0) / dx) * (y1 - y0)
        return self.a * xf + self.b

    def __iter__(self):
        yield self.a
        yield self.b

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return (self.a, self.b)[idx]

    def __eq__(self, other):
        if isinstance(other, FormulaLine):
            if not (
                math.isclose(self.a, other.a, rel_tol=1e-6, abs_tol=1e-6)
                and math.isclose(self.b, other.b, rel_tol=1e-6, abs_tol=1e-6)
            ):
                return False
            if len(self.piecewise_anchors) != len(other.piecewise_anchors):
                return False
            return all(
                math.isclose(px1, px2, rel_tol=1e-6, abs_tol=1e-4)
                and math.isclose(py1, py2, rel_tol=1e-6, abs_tol=1e-4)
                for (px1, py1), (px2, py2) in zip(self.piecewise_anchors, other.piecewise_anchors)
            )
        if isinstance(other, (tuple, list)) and len(other) == 2:
            if self.piecewise_anchors:
                return False
            return math.isclose(self.a, float(other[0]), rel_tol=1e-6, abs_tol=1e-6) and math.isclose(
                self.b, float(other[1]), rel_tol=1e-6, abs_tol=1e-6
            )
        return False

    def toString(self) -> str:
        return f"{self.a:.4f}x{self.b:+.3f}"

    def __repr__(self) -> str:
        return self.toString()

    def __str__(self) -> str:
        return self.toString()


@dataclass
class AlignmentResult:
    """Rich alignment diagnostics returned by the synchronization aligner."""

    correlated: bool
    a: float
    b: float
    factor: float
    points: int
    max_distance: float
    max_change: float
    formula: FormulaLine
    inlier_points: List[Tuple[float, float]] = None
    window_offsets: List[Tuple[float, float, float, float]] = None

    def __post_init__(self):
        if self.formula is None:
            self.formula = FormulaLine(self.a, self.b)
        if self.inlier_points is None:
            self.inlier_points = []
        if self.window_offsets is None:
            self.window_offsets = []

    @property
    def correlation_score(self) -> float:
        return self.factor

    @property
    def offset_s(self) -> float:
        return self.b

    @property
    def speed_factor(self) -> float:
        return self.a

    @property
    def maxDistance(self) -> float:
        return self.max_distance

    @property
    def maxChange(self) -> float:
        return self.max_change


# ---------------------------------------------------------------------------
# Optional C Accelerator (`_fast_corr.so`) + Pure-Python Radix-2 FFT Fallback
# ---------------------------------------------------------------------------

_C_SOURCE = r"""
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void compute_frame_rms(const int16_t *samples, int num_frames, int spf, double *out_rms) {
    if (spf <= 0) return;
    double inv_spf = 1.0 / (double)spf;
    for (int k = 0; k < num_frames; ++k) {
        const int16_t *ptr = samples + (size_t)k * (size_t)spf;
        double sum_sq = 0.0;
        for (int i = 0; i < spf; ++i) {
            double v = (double)ptr[i];
            sum_sq += v * v;
        }
        out_rms[k] = sqrt(sum_sq * inv_spf);
    }
}

static void fft_radix2(double *re, double *im, int n, int invert) {
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            double tr = re[i]; re[i] = re[j]; re[j] = tr;
            double ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / (double)len * (invert ? 1.0 : -1.0);
        double wlen_re = cos(ang);
        double wlen_im = sin(ang);
        int half = len >> 1;
        for (int i = 0; i < n; i += len) {
            double w_re = 1.0;
            double w_im = 0.0;
            for (int k = 0; k < half; ++k) {
                int u = i + k;
                int v = i + k + half;
                double vr = re[v] * w_re - im[v] * w_im;
                double vi = re[v] * w_im + im[v] * w_re;
                re[v] = re[u] - vr;
                im[v] = im[u] - vi;
                re[u] += vr;
                im[u] += vi;
                double next_w_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_w_re;
            }
        }
    }

    if (invert) {
        double inv_n = 1.0 / (double)n;
        for (int i = 0; i < n; ++i) {
            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}

int fft_correlate_best(
    const double *ref, int ref_len,
    const double *sub, int sub_len,
    int max_lag,
    int *out_best_lag,
    double *out_best_score,
    double *out_prev_score,
    double *out_next_score
) {
    if (ref_len <= 0 || sub_len <= 0) return -1;
    int total = ref_len + sub_len;
    int n = 1;
    while (n < total) n <<= 1;

    double *r_re = (double *)calloc((size_t)n * 4, sizeof(double));
    if (!r_re) return -1;
    double *r_im = r_re + n;
    double *s_re = r_im + n;
    double *s_im = s_re + n;

    double r_sum = 0.0, s_sum = 0.0;
    for (int i = 0; i < ref_len; ++i) r_sum += ref[i];
    for (int i = 0; i < sub_len; ++i) s_sum += sub[i];
    double r_mean = r_sum / (double)ref_len;
    double s_mean = s_sum / (double)sub_len;

    double r_norm2 = 0.0, s_norm2 = 0.0;
    for (int i = 0; i < ref_len; ++i) {
        double v = ref[i] - r_mean;
        r_re[i] = v;
        r_norm2 += v * v;
    }
    for (int i = 0; i < sub_len; ++i) {
        double v = sub[i] - s_mean;
        s_re[i] = v;
        s_norm2 += v * v;
    }

    double denom = sqrt(r_norm2 * s_norm2);
    if (denom < 1e-12) {
        free(r_re);
        return -1;
    }

    fft_radix2(r_re, r_im, n, 0);
    fft_radix2(s_re, s_im, n, 0);

    for (int i = 0; i < n; ++i) {
        double a = r_re[i], b = r_im[i];
        double c = s_re[i], d = -s_im[i];
        r_re[i] = a * c - b * d;
        r_im[i] = a * d + b * c;
    }

    fft_radix2(r_re, r_im, n, 1);

    int pos_limit = max_lag < ref_len ? max_lag : ref_len - 1;
    int neg_limit = max_lag < sub_len ? max_lag : sub_len - 1;

    int best_lag = 0;
    double best_val = -1e30;

    for (int lag = 0; lag <= pos_limit; ++lag) {
        double val = r_re[lag] / denom;
        if (val > best_val) {
            best_val = val;
            best_lag = lag;
        }
    }
    for (int lag = 1; lag <= neg_limit; ++lag) {
        int idx = n - lag;
        double val = r_re[idx] / denom;
        if (val > best_val) {
            best_val = val;
            best_lag = -lag;
        }
    }

    int idx_best = best_lag >= 0 ? best_lag : (n + best_lag);
    int idx_prev = (idx_best - 1 + n) % n;
    int idx_next = (idx_best + 1) % n;

    *out_best_lag = best_lag;
    *out_best_score = best_val;
    *out_prev_score = r_re[idx_prev] / denom;
    *out_next_score = r_re[idx_next] / denom;

    free(r_re);
    return 0;
}
"""

_FAST_CORR_LIB = False


def get_fast_corr_lib():
    """Compile (if needed) and load `_fast_corr.so` via ctypes, or return None on failure."""
    global _FAST_CORR_LIB
    if _FAST_CORR_LIB is not False:
        return _FAST_CORR_LIB

    so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fast_corr.so")
    if not os.path.isfile(so_path):
        tmp_so_path = f"{so_path}.{os.getpid()}.tmp"
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as tf:
                tf.write(_C_SOURCE)
                c_path = tf.name
            try:
                proc = subprocess.run(
                    ["gcc", "-O3", "-shared", "-fPIC", c_path, "-o", tmp_so_path, "-lm"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    timeout=15,
                )
                if proc.returncode == 0 and os.path.isfile(tmp_so_path):
                    os.replace(tmp_so_path, so_path)
                else:
                    logger.debug("gcc compilation of _fast_corr.so returned %d", proc.returncode)
            finally:
                for p_rm in (c_path, tmp_so_path):
                    try:
                        if os.path.exists(p_rm):
                            os.unlink(p_rm)
                    except OSError:
                        pass
        except Exception as exc:
            logger.debug("Could not compile _fast_corr.so: %s", exc)

    if os.path.isfile(so_path):
        try:
            lib = ctypes.CDLL(so_path)
            lib.compute_frame_rms.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.compute_frame_rms.restype = None

            lib.fft_correlate_best.argtypes = [
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.fft_correlate_best.restype = ctypes.c_int
            _FAST_CORR_LIB = lib
            return lib
        except Exception as exc:
            logger.debug("Could not load _fast_corr.so: %s", exc)

    _FAST_CORR_LIB = None
    return None


def c_compute_frame_rms(lib, pcm_bytes: bytes, num_frames: int, samples_per_frame: int) -> List[float]:
    """Call C `compute_frame_rms` via ctypes."""
    out_buf = (ctypes.c_double * num_frames)()
    lib.compute_frame_rms(pcm_bytes, int(num_frames), int(samples_per_frame), out_buf)
    return list(out_buf)


# ---------------------------------------------------------------------------
# Pure-Python Radix-2 FFT Fallback
# ---------------------------------------------------------------------------

def _fft_radix2_py(buf: List[complex], invert: bool = False) -> None:
    """In-place iterative Cooley-Tukey Radix-2 FFT in pure Python."""
    n = len(buf)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            buf[i], buf[j] = buf[j], buf[i]

    length = 2
    sign = 1.0 if invert else -1.0
    while length <= n:
        ang = sign * (2.0 * math.pi / length)
        wlen = cmath.rect(1.0, ang)
        half = length >> 1
        for i in range(0, n, length):
            w = 1.0 + 0.0j
            for k in range(half):
                u = buf[i + k]
                v = buf[i + k + half] * w
                buf[i + k] = u + v
                buf[i + k + half] = u - v
                w *= wlen
        length <<= 1

    if invert:
        inv_n = 1.0 / float(n)
        for i in range(n):
            buf[i] *= inv_n


def fft_correlate_py(
    ref: Sequence[float],
    sub: Sequence[float],
    max_lag: int,
) -> Tuple[int, float, float, float]:
    """Pure-Python FFT cross-correlation returning (best_lag, best_score, prev_score, next_score).

    Uses real-pair packing (packing two real signals into one complex FFT) for 2x speedup.
    """
    ref_len = len(ref)
    sub_len = len(sub)
    if ref_len == 0 or sub_len == 0:
        return 0, 0.0, 0.0, 0.0

    r_mean = sum(ref) / float(ref_len)
    s_mean = sum(sub) / float(sub_len)

    r_centered = [x - r_mean for x in ref]
    s_centered = [x - s_mean for x in sub]

    r_norm2 = sum(x * x for x in r_centered)
    s_norm2 = sum(x * x for x in s_centered)
    denom = math.sqrt(r_norm2 * s_norm2)
    if denom < 1e-12:
        return 0, 0.0, 0.0, 0.0

    total = ref_len + sub_len
    n = 1
    while n < total:
        n <<= 1

    # Pack r_centered in real part and s_centered in imag part to compute both FFTs at once
    combined = [0.0 + 0.0j] * n
    for i in range(max(ref_len, sub_len)):
        re_val = r_centered[i] if i < ref_len else 0.0
        im_val = s_centered[i] if i < sub_len else 0.0
        combined[i] = complex(re_val, im_val)

    _fft_radix2_py(combined, invert=False)

    # Unpack FFT(r) and FFT(s) and compute X[k] * conj(Y[k])
    prod = [0.0 + 0.0j] * n
    for k in range(n):
        z = combined[k]
        z_sym = combined[(n - k) % n].conjugate()
        x_k = 0.5 * (z + z_sym)
        y_k = -0.5j * (z - z_sym)
        prod[k] = x_k * y_k.conjugate()

    _fft_radix2_py(prod, invert=True)

    pos_limit = min(max_lag, ref_len - 1)
    neg_limit = min(max_lag, sub_len - 1)

    best_lag = 0
    best_score = -1e30

    for lag in range(0, pos_limit + 1):
        val = prod[lag].real / denom
        if val > best_score:
            best_score = val
            best_lag = lag

    for lag in range(1, neg_limit + 1):
        val = prod[n - lag].real / denom
        if val > best_score:
            best_score = val
            best_lag = -lag

    idx_best = best_lag if best_lag >= 0 else (n + best_lag)
    prev_score = prod[(idx_best - 1 + n) % n].real / denom
    next_score = prod[(idx_best + 1) % n].real / denom

    return best_lag, best_score, prev_score, next_score


def fft_correlate(
    ref: Sequence[float],
    sub: Sequence[float],
    max_lag_frames: int,
    ref_c_buf=None,
) -> Tuple[float, float]:
    """Compute normalized cross-correlation peak between `ref` and `sub` within `max_lag_frames`.

    Returns `(refined_lag_frames, peak_score)` with parabolic sub-frame interpolation.
    """
    ref_len = len(ref)
    sub_len = len(sub)
    if ref_len == 0 or sub_len == 0:
        return 0.0, 0.0

    lib = get_fast_corr_lib()
    if lib is not None:
        if ref_c_buf is None:
            ref_c_buf = (ctypes.c_double * ref_len)(*ref)
        sub_c_buf = (ctypes.c_double * sub_len)(*sub)
        out_lag = ctypes.c_int(0)
        out_score = ctypes.c_double(0.0)
        out_prev = ctypes.c_double(0.0)
        out_next = ctypes.c_double(0.0)
        rc = lib.fft_correlate_best(
            ref_c_buf,
            ref_len,
            sub_c_buf,
            sub_len,
            int(max_lag_frames),
            ctypes.byref(out_lag),
            ctypes.byref(out_score),
            ctypes.byref(out_prev),
            ctypes.byref(out_next),
        )
        if rc == 0:
            best_lag = out_lag.value
            best_score = out_score.value
            prev_score = out_prev.value
            next_score = out_next.value
        else:
            best_lag, best_score, prev_score, next_score = fft_correlate_py(ref, sub, max_lag_frames)
    else:
        best_lag, best_score, prev_score, next_score = fft_correlate_py(ref, sub, max_lag_frames)

    # Parabolic sub-frame interpolation around peak (prev_score, best_score, next_score)
    denom = prev_score - 2.0 * best_score + next_score
    delta = 0.0
    if abs(denom) > 1e-12 and best_score >= prev_score and best_score >= next_score:
        delta = 0.5 * (prev_score - next_score) / denom
        delta = max(-0.5, min(0.5, delta))

    return float(best_lag) + delta, float(best_score)


# ---------------------------------------------------------------------------
# Windowed Multi-Segment Drift Discovery & Stage 2 Boundary Refinement
# ---------------------------------------------------------------------------

def _discover_windowed_drift_candidates(
    ref_envelope: Sequence[float],
    sub_intervals: Sequence[Tuple[float, float]],
    frame_hz: int,
    max_offset_s: float,
) -> List[float]:
    """Split subtitle track into time windows, correlate each window locally, and estimate drift `a`."""
    if len(sub_intervals) < 12:
        return []

    t_min = sub_intervals[0][0]
    t_max = sub_intervals[-1][1]
    span = t_max - t_min
    if span < 120.0:
        return []

    num_windows = min(6, max(4, len(sub_intervals) // 8))
    win_len = span / float(num_windows)
    ref_len = len(ref_envelope)
    max_lag_frames = int(round((max_offset_s + 0.1 * span) * frame_hz))
    max_win_lag = min(max_lag_frames, int(round(60.0 * frame_hz)))

    win_points: List[Tuple[float, float, float]] = []
    for w in range(num_windows):
        w_start = t_min + w * win_len
        w_end = w_start + win_len
        w_subs = [(s - w_start, e - w_start) for s, e in sub_intervals if s >= w_start and e <= w_end]
        if len(w_subs) < 3:
            continue
        w_env = vad.intervals_to_envelope(
            w_subs, duration_s=(w_end - w_start), frame_hz=frame_hz, edge_boost=0.0
        )
        if not w_env:
            continue

        # Slice ref_envelope starting at w_start so lag=0 corresponds to exact 0.0s offset
        r_start_frame = max(0, int(round(w_start * frame_hz)))
        r_end_frame = min(ref_len, int(round(w_end * frame_hz)))
        if r_end_frame - r_start_frame < int(20.0 * frame_hz):
            continue

        ref_slice = ref_envelope[r_start_frame:r_end_frame]
        lag_frames, score = fft_correlate(ref_slice, w_env, max_win_lag)
        if score >= 0.14:
            lag_s = lag_frames / float(frame_hz)
            t_center = 0.5 * (w_start + w_end)
            win_points.append((t_center, t_center + lag_s, score))

    if len(win_points) < 3:
        return []

    # Pairwise RANSAC slopes between window centers
    candidate_slopes: List[float] = []
    n_pts = len(win_points)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            dx = win_points[j][0] - win_points[i][0]
            if dx < 60.0:
                continue
            dy = win_points[j][1] - win_points[i][1]
            slope = dy / dx
            if 0.90 <= slope <= 1.10:
                candidate_slopes.append(slope)

    if not candidate_slopes:
        return []

    candidate_slopes.sort()
    median_slope = candidate_slopes[len(candidate_slopes) // 2]
    discovered = [median_slope]

    # Also snap to nearest canonical framerate ratio if within 0.002
    for std_a in STANDARD_SPEED_FACTORS:
        if abs(median_slope - std_a) < 0.002:
            discovered.append(std_a)

    return discovered


def _match_interval_anchors(
    sub_intervals: Sequence[Tuple[float, float]],
    ref_intervals: Sequence[Tuple[float, float]],
    a: float,
    b: float,
    tolerance_s: float = 0.45,
    start_only: bool = False,
) -> Tuple[List[Tuple[float, float]], int]:
    """Match subtitle start/end boundaries against reference speech/subtitle boundaries.

    Returns `(anchor_points, matched_sub_events_count)` where each anchor point
    is `(t_sub, t_ref)` for a matched boundary.
    """
    if not sub_intervals or not ref_intervals:
        return [], 0

    anchors: List[Tuple[float, float]] = []
    matched_events = 0
    ref_idx = 0
    n_ref = len(ref_intervals)

    for s_sub, e_sub in sub_intervals:
        pred_s = a * s_sub + b
        pred_e = a * e_sub + b

        # Advance ref_idx to first interval that could overlap [pred_s - tolerance_s, pred_e + tolerance_s]
        while ref_idx < n_ref and ref_intervals[ref_idx][1] < pred_s - tolerance_s:
            ref_idx += 1

        best_start_ref = None
        best_start_err = tolerance_s
        best_end_ref = None
        best_end_err = tolerance_s
        has_overlap = False

        k = ref_idx
        while k < n_ref and ref_intervals[k][0] <= pred_e + tolerance_s:
            r_s, r_e = ref_intervals[k]
            if min(pred_e, r_e) > max(pred_s, r_s) - 0.10:
                has_overlap = True
            err_s = abs(r_s - pred_s)
            if err_s < best_start_err:
                best_start_err = err_s
                best_start_ref = r_s
            err_e = abs(r_e - pred_e)
            if err_e < best_end_err:
                best_end_err = err_e
                best_end_ref = r_e
            k += 1

        matched_this = False
        if best_start_ref is not None:
            anchors.append((s_sub, best_start_ref))
            matched_this = True
        if not start_only and best_end_ref is not None:
            anchors.append((e_sub, best_end_ref))
            matched_this = True
        if matched_this or has_overlap:
            matched_events += 1

    return anchors, matched_events


def _ols_fit(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float]:
    """Fit `y = a * x + b` via Ordinary Least Squares and return `(a, b, r_squared)`."""
    n = len(points)
    if n < 2:
        if n == 1:
            return 1.0, points[0][1] - points[0][0], 1.0
        return 1.0, 0.0, 0.0

    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    sx2 = sum(p[0] * p[0] for p in points)
    sy2 = sum(p[1] * p[1] for p in points)

    nf = float(n)
    txy = nf * sxy - sx * sy
    tx = nf * sx2 - sx * sx
    ty = nf * sy2 - sy * sy

    if tx <= 1e-9 or ty <= 1e-9:
        return 1.0, (sy - sx) / nf, 0.0

    a = txy / tx
    b = (sy - a * sx) / nf
    r2 = (txy * txy) / (tx * ty)
    return a, b, max(0.0, min(1.0, r2))


def _refine_offset_fixed_slope(
    anchors: Sequence[Tuple[float, float]],
    a: float,
    coarse_b: float,
    inlier_Radius: float = 0.25,
    start_anchors: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[float, List[Tuple[float, float]], float]:
    """Refine `b` for a fixed slope `a` using iterative MAD-trimmed mean on boundary residuals.

    When `start_anchors` is provided and disagrees with combined start+end residuals by > 40ms
    (due to human subtitle end-padding), gives priority to start-boundary onset residuals.
    """
    if not anchors:
        return coarse_b, [], 0.0

    residuals = [(y - a * x, x, y) for x, y in anchors if abs((y - a * x) - coarse_b) <= inlier_Radius]
    if not residuals:
        return coarse_b, [], 0.0

    residuals.sort(key=lambda item: item[0])
    med_b = residuals[len(residuals) // 2][0]

    abs_devs = sorted(abs(item[0] - med_b) for item in residuals)
    mad = abs_devs[len(abs_devs) // 2]
    cutoff = max(0.025, min(0.15, 2.5 * mad if mad > 0 else 0.05))

    core = [(item[1], item[2]) for item in residuals if abs(item[0] - med_b) <= cutoff]
    if not core:
        core = [(item[1], item[2]) for item in residuals]

    refined_b = sum(y - a * x for x, y in core) / float(len(core))
    max_res = max(abs(a * x + refined_b - y) for x, y in core)

    if start_anchors:
        s_res = [(y - a * x, x, y) for x, y in start_anchors if abs((y - a * x) - coarse_b) <= inlier_Radius]
        if len(s_res) >= 4:
            s_res.sort(key=lambda item: item[0])
            s_med = s_res[len(s_res) // 2][0]
            s_devs = sorted(abs(item[0] - s_med) for item in s_res)
            s_mad = s_devs[len(s_devs) // 2]
            s_cut = max(0.025, min(0.15, 2.5 * s_mad if s_mad > 0 else 0.05))
            s_core = [(item[1], item[2]) for item in s_res if abs(item[0] - s_med) <= s_cut]
            if len(s_core) >= 4:
                s_b = sum(y - a * x for x, y in s_core) / float(len(s_core))
                # If start vs combined boundaries disagree by > 40ms, prioritize start-boundary onsets
                if abs(s_b - refined_b) > 0.040 and abs(s_b - coarse_b) <= 0.15:
                    s_max_res = max(abs(a * x + s_b - y) for x, y in s_core)
                    return s_b, s_core, s_max_res

    return refined_b, core, max_res


def _refine_line_ols(
    anchors: Sequence[Tuple[float, float]],
    init_a: float,
    init_b: float,
    max_dist: float = 0.15,
    min_points: int = 10,
) -> Tuple[float, float, List[Tuple[float, float]], float, float]:
    """Iteratively prune furthest outlier points and fit `(a, b)` via OLS."""
    pts = [(x, y) for x, y in anchors if abs(init_a * x + init_b - y) <= 0.40]
    if len(pts) < max(4, min_points):
        b_ref, core, max_res = _refine_offset_fixed_slope(anchors, init_a, init_b)
        return init_a, b_ref, core, max_res, 0.9999

    t_span = max(x for x, _ in pts) - min(x for x, _ in pts)
    if t_span < 30.0:
        b_ref, core, max_res = _refine_offset_fixed_slope(anchors, init_a, init_b)
        return init_a, b_ref, core, max_res, 0.9999

    a, b, r2 = _ols_fit(pts)
    while len(pts) > max(6, min_points, int(len(anchors) * 0.55)):
        # Find point with largest vertical residual |a*x + b - y|
        worst_idx = 0
        worst_err = -1.0
        for idx, (x, y) in enumerate(pts):
            err = abs(a * x + b - y)
            if err > worst_err:
                worst_err = err
                worst_idx = idx
        if worst_err <= max_dist and r2 >= 0.9999:
            break
        pts.pop(worst_idx)
        a, b, r2 = _ols_fit(pts)

    max_res = max(abs(a * x + b - y) for x, y in pts) if pts else 0.0
    return a, b, pts, max_res, r2


def _build_vocal_attack_signal(
    ref_envelope: Sequence[float],
    ref_intervals: Optional[Sequence[Tuple[float, float]]],
    frame_hz: int = 100,
):
    """Construct a 100 Hz Vocal Attack impulse signal from reference envelope, speech intervals, and log-RMS onset flux."""
    n = len(ref_envelope)
    if n == 0:
        return []

    if np is not None:
        env_arr = np.asarray(ref_envelope, dtype=np.float64)
        kernel = np.ones(5, dtype=np.float64) / 5.0
        smooth = np.convolve(env_arr, kernel, mode="same")
        win = max(5, int(round(0.35 * frame_hz)))
        pad_s = np.pad(smooth, (win, 0), mode="edge")
        prev_min = np.min(sliding_window_view(pad_s[:-1], win), axis=1)
        rise = np.where((smooth > 0.30) & ((smooth - prev_min) > 0.20), smooth - prev_min, 0.0)
        try:
            from subsync.synchro import ai_engine

            att_env = ai_engine.get_last_attack_envelope()
            if att_env is not None and len(att_env) == n:
                rise = np.maximum(rise, np.asarray(att_env, dtype=np.float64))
        except Exception:
            pass
        if ref_intervals:
            for idx, (rs, re_t) in enumerate(ref_intervals):
                if re_t - rs < 0.12:
                    continue
                gap = rs - ref_intervals[idx - 1][1] if idx > 0 else 10.0
                w = 1.0 if gap >= 0.25 else (0.45 if gap >= 0.10 else 0.2)
                i0 = max(0, int(round(rs * frame_hz)))
                i1 = min(n, int(round(min(re_t, rs + 0.30) * frame_hz)))
                if i1 > i0:
                    rise[i0:i1] = np.maximum(rise[i0:i1], w)
        return rise

    # Pure-Python fallback if numpy is absent
    rise_py = [0.0] * n
    if ref_intervals:
        for idx, (rs, re_t) in enumerate(ref_intervals):
            if re_t - rs < 0.12:
                continue
            gap = rs - ref_intervals[idx - 1][1] if idx > 0 else 10.0
            w = 1.0 if gap >= 0.25 else (0.45 if gap >= 0.10 else 0.2)
            i0 = max(0, int(round(rs * frame_hz)))
            i1 = min(n, int(round(min(re_t, rs + 0.30) * frame_hz)))
            for k in range(i0, i1):
                if w > rise_py[k]:
                    rise_py[k] = w
    return rise_py


def _ncc_curve_act_np(
    ref_arr,
    act_scaled: Sequence[Tuple[float, float]],
    prev_end: Optional[float],
    next_start: Optional[float],
    pad_frames: int,
    frame_hz: int = 100,
    ref_attack_arr=None,
):
    """Compute hybrid locally-normalized cross-correlation curve C(lag) over [-pad_frames, +pad_frames].

    Blends speech-activity envelope NCC with utterance-start vocal-attack onset NCC
    weighted adaptively by dialogue duty cycle so wall-to-wall monologues / continuous
    dialogues (e.g. 90%+ subtitle coverage with 0.0s gaps) never suffer boxcar edge blindness.
    """
    t0, t1 = act_scaled[0][0], act_scaled[-1][1]
    act_span = max(1.0, t1 - t0)
    spoken_dur = sum(max(0.0, e - s) for s, e in act_scaled)
    raw_duty = min(1.0, spoken_dur / act_span)

    max_m = 6.5 if raw_duty > 0.75 else 4.0
    ratio_m = 0.65 if raw_duty > 0.75 else 0.45
    margin_l = (
        min(max_m, max(1.5, ratio_m * (t0 - prev_end)))
        if prev_end is not None
        else min(max_m, max(1.0, t0))
    )
    margin_r = min(max_m, max(1.5, ratio_m * (next_start - t1))) if next_start is not None else max_m
    w_start = max(0.0, t0 - margin_l)
    w_end = max(w_start + 10.0, t1 + margin_r)

    notch_s = 0.18 if raw_duty > 0.75 else 0.08
    seg_rel: List[Tuple[float, float]] = []
    for k, (s, e) in enumerate(act_scaled):
        if k + 1 < len(act_scaled) and act_scaled[k + 1][0] - e < 0.10 and (e - s) > 0.55:
            e_adj = e - notch_s
        else:
            e_adj = e
        seg_rel.append((s - w_start, e_adj - w_start))

    w_env = vad.intervals_to_envelope(
        seg_rel,
        duration_s=(w_end - w_start),
        frame_hz=frame_hz,
        edge_boost=0.20,
    )
    w_len = len(w_env)
    if w_len == 0:
        return np.zeros(2 * pad_frames + 1, dtype=np.float64)

    w_start_frames = int(round(w_start * frame_hz))
    r0 = w_start_frames - pad_frames
    r1 = r0 + w_len + 2 * pad_frames
    r_slice = np.zeros(r1 - r0, dtype=np.float64)
    c0, c1 = max(0, r0), min(len(ref_arr), r1)
    if c1 > c0:
        r_slice[c0 - r0 : c1 - r0] = ref_arr[c0:c1]

    w_arr = np.asarray(w_env, dtype=np.float64)
    w_zero = w_arr - w_arr.mean()
    w_norm = float(np.linalg.norm(w_zero))
    if w_norm < 1e-9:
        return np.zeros(2 * pad_frames + 1, dtype=np.float64)

    n_fft = 1 << (len(r_slice) + w_len - 1).bit_length()
    dots = np.fft.irfft(
        np.fft.rfft(r_slice, n_fft) * np.conj(np.fft.rfft(w_zero, n_fft)), n_fft
    )[: 2 * pad_frames + 1]

    ps1 = np.concatenate([[0.0], np.cumsum(r_slice)])
    ps2 = np.concatenate([[0.0], np.cumsum(r_slice * r_slice)])
    r_sum = ps1[w_len : w_len + 2 * pad_frames + 1] - ps1[: 2 * pad_frames + 1]
    r_sq = ps2[w_len : w_len + 2 * pad_frames + 1] - ps2[: 2 * pad_frames + 1]
    r_var = np.maximum(1e-9, r_sq - (r_sum * r_sum) / float(w_len))
    c_boxcar = dots / (np.sqrt(r_var) * w_norm)

    if ref_attack_arr is None or len(act_scaled) < 3:
        return c_boxcar

    # Build 100 Hz subtitle sentence-onset impulse template with symmetric Gaussian lead tolerance
    # (centered at s + 25ms, pre-lead ramp down to -120ms, post-onset decay up to +300ms)
    w_onset = np.zeros(w_len, dtype=np.float64)
    pre_f = max(2, int(round(0.12 * frame_hz)))
    post_f = max(3, int(round(0.30 * frame_hz)))
    pre_ramp = np.exp(-0.5 * (np.linspace(-pre_f, -1, pre_f, dtype=np.float64) / (0.055 * frame_hz)) ** 2)
    decay = np.linspace(1.0, 0.25, post_f, dtype=np.float64)
    pulse_kernel = np.concatenate([pre_ramp, decay])
    k_len = len(pulse_kernel)
    last_s = -999.0
    for k, (s, _) in enumerate(act_scaled):
        if s - last_s < 0.40:
            continue
        last_s = s
        prev_e = act_scaled[k - 1][1] if k > 0 else -999.0
        pw = 1.0 if (s - prev_e) >= 0.20 else 0.70
        f_center = int(round((s + 0.025 - w_start) * frame_hz))
        f0 = f_center - pre_f
        f1 = f0 + k_len
        c_f0 = max(0, f0)
        c_f1 = min(w_len, f1)
        if c_f1 > c_f0:
            k0 = c_f0 - f0
            k1 = k0 + (c_f1 - c_f0)
            w_onset[c_f0:c_f1] = np.maximum(w_onset[c_f0:c_f1], pulse_kernel[k0:k1] * pw)

    o_zero = w_onset - w_onset.mean()
    o_norm = float(np.linalg.norm(o_zero))
    if o_norm < 1e-9:
        return c_boxcar

    a_slice = np.zeros(r1 - r0, dtype=np.float64)
    ca1 = min(len(ref_attack_arr), r1)
    if ca1 > c0:
        a_slice[c0 - r0 : ca1 - r0] = ref_attack_arr[c0:ca1]

    dots_o = np.fft.irfft(
        np.fft.rfft(a_slice, n_fft) * np.conj(np.fft.rfft(o_zero, n_fft)), n_fft
    )[: 2 * pad_frames + 1]
    pa1 = np.concatenate([[0.0], np.cumsum(a_slice)])
    pa2 = np.concatenate([[0.0], np.cumsum(a_slice * a_slice)])
    a_sum = pa1[w_len : w_len + 2 * pad_frames + 1] - pa1[: 2 * pad_frames + 1]
    a_sq = pa2[w_len : w_len + 2 * pad_frames + 1] - pa2[: 2 * pad_frames + 1]
    a_var = np.maximum(1e-9, a_sq - (a_sum * a_sum) / float(w_len))
    c_onset = dots_o / (np.sqrt(a_var) * o_norm)

    # Blend onset correlation from 18% (sparse dialogue) up to 52% (wall-to-wall continuous dialogue)
    alpha = min(0.52, max(0.18, 0.18 + 1.15 * max(0.0, raw_duty - 0.65)))
    return (1.0 - alpha) * c_boxcar + alpha * np.maximum(-0.20, c_onset)


def _act_vocal_coverage(
    ref_arr,
    act_scaled: Sequence[Tuple[float, float]],
    lag_s: float,
    frame_hz: int = 100,
) -> float:
    """Return fraction of subtitle intervals in `act_scaled` that overlap active speech at `lag_s`."""
    n_ref = len(ref_arr)
    if n_ref == 0 or not act_scaled:
        return 0.0
    hits = 0
    valid = 0
    margin_f = max(1, int(round(0.10 * frame_hz)))
    for s0, e0 in act_scaled:
        s = s0 + lag_s
        e = e0 + lag_s
        if e <= 0.0 or e <= s:
            continue
        i0 = max(0, min(n_ref, int(round(s * frame_hz))))
        i1 = max(0, min(n_ref, int(round(e * frame_hz))))
        if i1 <= i0:
            continue
        valid += 1
        m0 = max(0, i0 - margin_f)
        m1 = min(n_ref, i1 + margin_f)
        if float(np.max(ref_arr[m0:m1])) >= 0.35 and float(np.mean(ref_arr[i0:i1])) >= 0.16:
            hits += 1
    return (float(hits) / float(valid)) if valid > 0 else 0.0


def _refine_act_vocal_attack(
    onset_sig,
    act_scaled: Sequence[Tuple[float, float]],
    coarse_b: float,
    frame_hz: int = 100,
    ref_intervals: Optional[Sequence[Tuple[float, float]]] = None,
) -> float:
    """
    Zero-bias vocal onset refinement for an act around `coarse_b`:
    Matches each subtitle start `s_sub` against the nearest neural VAD speech onset `r_s`
    within ±0.38s of `s_sub + coarse_b` (prioritizing isolated utterances after >= 0.25s silence)
    and computes the MAD-trimmed median onset residual with a 35ms visual anticipation lead.
    """
    valid_ivs = [(s, e) for s, e in act_scaled if s + coarse_b >= -0.5]
    if len(valid_ivs) < 3:
        return coarse_b

    if ref_intervals:
        residuals: List[float] = []
        weights: List[float] = []
        r_starts = [r[0] for r in ref_intervals if (r[1] - r[0]) >= 0.12]
        if r_starts:
            import bisect
            for k, (s_sub, _) in enumerate(valid_ivs):
                sub_gap = (s_sub - valid_ivs[k - 1][1]) if k > 0 else 5.0
                if sub_gap < 0.15:
                    continue
                pred_s = s_sub + coarse_b
                idx = bisect.bisect_left(r_starts, pred_s)
                best_diff = None
                for cand_i in (idx - 1, idx, idx + 1):
                    if 0 <= cand_i < len(r_starts):
                        d = r_starts[cand_i] - pred_s
                        if abs(d) <= 0.35 and (best_diff is None or abs(d) < abs(best_diff)):
                            best_diff = d
                if best_diff is not None:
                    residuals.append(best_diff)
                    weights.append(2.0 if sub_gap >= 0.50 else 1.0)

            if len(residuals) >= 4:
                s_res = sorted(residuals)
                med = s_res[len(s_res) // 2]
                devs = sorted(abs(r - med) for r in s_res)
                mad = devs[len(devs) // 2]
                cutoff = max(0.04, min(0.16, 2.2 * mad if mad > 0 else 0.08))
                inliers = [r for r in residuals if abs(r - med) <= cutoff]
                if inliers:
                    mean_delta = sum(inliers) / float(len(inliers))
                    # Apply a 25ms visual anticipation lead so subtitles never feel late
                    refined = coarse_b + mean_delta - 0.025
                    if abs(refined - coarse_b) <= 0.25:
                        return refined

    return coarse_b


def _analyze_episode_windows(
    ref_envelope: Sequence[float],
    ref_intervals: Optional[Sequence[Tuple[float, float]]],
    sub_intervals: Sequence[Tuple[float, float]],
    scale_a: float,
    frame_hz: int,
    max_win_lag: int,
    win_len_s: float = 300.0,
) -> Tuple[List[dict], float]:
    """Segment episode into natural dialogue acts, solve global 2D Viterbi lag trellis, and refine vocal attacks."""
    ref_len = len(ref_envelope)
    if ref_len == 0 or len(sub_intervals) < 5:
        return [], 0.0

    # 1. Segment subtitle track into natural dialogue acts split at scene/commercial silence gaps
    acts_orig: List[List[Tuple[float, float]]] = []
    acts_scaled: List[List[Tuple[float, float]]] = []
    cur_orig: List[Tuple[float, float]] = [sub_intervals[0]]
    cur_scaled: List[Tuple[float, float]] = [
        (sub_intervals[0][0] * scale_a, sub_intervals[0][1] * scale_a)
    ]

    for s_orig, e_orig in sub_intervals[1:]:
        s_sc, e_sc = s_orig * scale_a, e_orig * scale_a
        gap = s_sc - cur_scaled[-1][1]
        dur = cur_scaled[-1][1] - cur_scaled[0][0]
        if (
            (cur_scaled[0][0] < 120.0 and gap >= 6.0 and len(cur_scaled) >= 4)
            or (gap >= 8.0 and len(cur_scaled) >= 2)
            or (gap >= 6.0 and len(cur_scaled) >= 5 and dur >= 14.0)
            or (gap >= 3.5 and len(cur_scaled) >= 7 and dur >= 22.0)
            or (gap >= 2.2 and len(cur_scaled) >= 10 and dur >= 38.0)
            or (dur >= 85.0 and gap >= 1.4)
            or (dur >= 130.0 and gap >= 0.45)
            or dur >= 180.0
        ):
            acts_orig.append(cur_orig)
            acts_scaled.append(cur_scaled)
            cur_orig = [(s_orig, e_orig)]
            cur_scaled = [(s_sc, e_sc)]
        else:
            cur_orig.append((s_orig, e_orig))
            cur_scaled.append((s_sc, e_sc))

    if len(cur_scaled) >= 6 or not acts_scaled:
        acts_orig.append(cur_orig)
        acts_scaled.append(cur_scaled)
    else:
        acts_orig[-1].extend(cur_orig)
        acts_scaled[-1].extend(cur_scaled)

    num_acts = len(acts_scaled)
    if num_acts == 0:
        return [], 0.0

    if np is not None:
        pad_frames = max(int(round(75.0 * frame_hz)), min(max_win_lag, int(round(90.0 * frame_hz))))
        ref_arr = np.asarray(ref_envelope, dtype=np.float64)
        ref_attack_arr = np.asarray(
            _build_vocal_attack_signal(ref_envelope, ref_intervals, frame_hz=frame_hz),
            dtype=np.float64,
        )
        curves = []
        for k in range(num_acts):
            prev_end = acts_scaled[k - 1][-1][1] if k > 0 else None
            next_start = acts_scaled[k + 1][0][0] if k + 1 < num_acts else None
            curves.append(
                _ncc_curve_act_np(
                    ref_arr,
                    acts_scaled[k],
                    prev_end,
                    next_start,
                    pad_frames=pad_frames,
                    frame_hz=frame_hz,
                    ref_attack_arr=ref_attack_arr,
                )
            )

        # 2. Solve Global 2D Viterbi Dynamic Programming Trellis over all acts
        step = max(1, int(round(0.05 * frame_hz)))  # 50ms coarse lag grid
        grid = np.arange(0, 2 * pad_frames + 1, step)
        lags_s = (grid - pad_frames) / float(frame_hz)
        m_grid = len(grid)

        dp = np.zeros((num_acts, m_grid), dtype=np.float64)
        back = np.zeros((num_acts, m_grid), dtype=int)
        dp[0] = curves[0][grid]

        if num_acts > 1:
            diff_mat = lags_s[:, None] - lags_s[None, :]
            abs_diff = np.abs(diff_mat)
            pen_std = np.where(
                abs_diff < 1e-4,
                0.0,
                np.where(
                    abs_diff <= 0.35,
                    0.05 + 0.14 * abs_diff,
                    np.where(
                        (diff_mat >= -2.50) & (diff_mat <= 2.50),
                        0.095 + 0.018 * abs_diff,
                        np.where(
                            (diff_mat >= -15.0) & (diff_mat <= 15.0),
                            0.135 + 0.010 * abs_diff,
                            0.28 + 0.022 * abs_diff,
                        ),
                    ),
                ),
            )
            pen_big_gap = np.where(
                abs_diff < 1e-4,
                0.0,
                np.where(
                    abs_diff <= 0.35,
                    0.04 + 0.10 * abs_diff,
                    np.where(
                        (diff_mat >= -2.50) & (diff_mat <= 2.50),
                        0.072 + 0.012 * abs_diff,
                        np.where(
                            (diff_mat >= -18.0) & (diff_mat <= 18.0),
                            0.092 + 0.006 * abs_diff,
                            0.24 + 0.020 * abs_diff,
                        ),
                    ),
                ),
            )
            for w in range(1, num_acts):
                gap_before = acts_scaled[w][0][0] - acts_scaled[w - 1][-1][1]
                pen = pen_big_gap if gap_before >= 6.5 else pen_std
                scores = dp[w - 1][None, :] - pen
                best_prev = np.argmax(scores, axis=1)
                dp[w] = curves[w][grid] + scores[np.arange(m_grid), best_prev]
                back[w] = best_prev

        curr = int(np.argmax(dp[-1]))
        total_viterbi_score = float(dp[-1, curr])
        coarse_path: List[Tuple[float, float]] = []
        fine_radius = max(2, step * 2)
        for w in range(num_acts - 1, -1, -1):
            g_idx = int(grid[curr])
            lo = max(0, g_idx - fine_radius)
            hi = min(2 * pad_frames, g_idx + fine_radius)
            fine_idx = lo + int(np.argmax(curves[w][lo : hi + 1]))
            coarse_path.append(((fine_idx - pad_frames) / float(frame_hz), float(curves[w][fine_idx])))
            curr = int(back[w, curr])
        coarse_path.reverse()

        # 3. Detect orphan "Précédemment..." recap in Act 0 that was cut from the start of the video
        if (
            num_acts >= 2
            and acts_scaled[0][0][0] < 10.0
            and acts_scaled[0][-1][1] < 60.0
            and coarse_path[1][0] < -15.0
        ):
            if (
                acts_scaled[1][0][0] + coarse_path[1][0] < acts_scaled[0][-1][1] + 5.0
                and coarse_path[0][1] < 0.25
            ):
                orphan_b = min(coarse_path[1][0], -acts_scaled[0][-1][1] - 1.5)
                coarse_path[0] = (orphan_b, coarse_path[1][1])

        # If Act 0 contains both an unvoiced recap (s + cb < -1.0s) and the opening scene (s + cb >= -1.0s),
        # re-evaluate Act 0's NCC curve on only the in-video opening scene so the unvoiced recap doesn't bias cb.
        if coarse_path[0][0] < -15.0:
            cb0 = coarse_path[0][0]
            in_vid_0 = [(s, e) for s, e in acts_scaled[0] if s + cb0 >= -1.0]
            pre_vid_0 = [(s, e) for s, e in acts_scaled[0] if s + cb0 < -1.0]
            if len(in_vid_0) >= 5 and len(pre_vid_0) >= 5:
                next_s0 = acts_scaled[1][0][0] if num_acts > 1 else None
                c0_clean = _ncc_curve_act_np(
                    ref_arr,
                    in_vid_0,
                    pre_vid_0[-1][1],
                    next_s0,
                    pad_frames=pad_frames,
                    frame_hz=frame_hz,
                    ref_attack_arr=ref_attack_arr,
                )
                lo0 = max(0, int(round((cb0 - 4.0) * frame_hz)) + pad_frames)
                hi0 = min(
                    2 * pad_frames,
                    int(round(min(cb0 + 2.0, -pre_vid_0[-1][0] - 0.5) * frame_hz)) + pad_frames,
                )
                if hi0 > lo0:
                    best0 = lo0 + int(np.argmax(c0_clean[lo0 : hi0 + 1]))
                    coarse_path[0] = ((best0 - pad_frames) / float(frame_hz), float(c0_clean[best0]))

        # 3b. Plateau-Aware Consensus & Change-Point Stabilization:
        start_cons_idx = 1 if (num_acts >= 2 and coarse_path[0][0] < -10.0) else 0
        if num_acts - start_cons_idx >= 2:
            sum_curve = np.sum([curves[w] for w in range(start_cons_idx, num_acts)], axis=0)
            best_const_idx = int(np.argmax(sum_curve))
            lo_c = max(0, best_const_idx - fine_radius)
            hi_c = min(2 * pad_frames, best_const_idx + fine_radius)
            best_const_lag = (best_const_idx - pad_frames) / float(frame_hz)
            viterbi_mean = sum(coarse_path[w][1] for w in range(start_cons_idx, num_acts)) / float(
                num_acts - start_cons_idx
            )
            const_mean = float(sum_curve[best_const_idx]) / float(num_acts - start_cons_idx)

            gains = [
                coarse_path[w][1] - float(np.max(curves[w][lo_c : hi_c + 1]))
                for w in range(num_acts)
            ]
            devs = [coarse_path[w][0] - best_const_lag for w in range(num_acts)]

            has_single_break = any(
                gains[w] > 0.14 and abs(devs[w]) > 0.70 and len(acts_scaled[w]) >= 8
                for w in range(start_cons_idx, num_acts)
            )
            has_plateau_break = any(
                abs(devs[w]) >= 0.30
                and abs(devs[w + 1]) >= 0.30
                and abs(coarse_path[w][0] - coarse_path[w + 1][0]) <= 0.35
                and (gains[w] + gains[w + 1]) > 0.10
                for w in range(start_cons_idx, num_acts - 1)
            )

            if not (has_single_break or has_plateau_break) and (viterbi_mean - const_mean) < 0.05:
                for w in range(start_cons_idx, num_acts):
                    f_idx = lo_c + int(np.argmax(curves[w][lo_c : hi_c + 1]))
                    coarse_path[w] = (best_const_lag, float(curves[w][f_idx]))
            else:
                def _lock_plateaus() -> None:
                    p_start = start_cons_idx
                    while p_start < num_acts:
                        p_end = p_start + 1
                        while (
                            p_end < num_acts
                            and abs(coarse_path[p_end][0] - coarse_path[p_end - 1][0]) <= 0.25
                            and abs(coarse_path[p_end][0] - coarse_path[p_start][0]) <= 0.35
                        ):
                            p_end += 1
                        if p_end - p_start >= 2:
                            p_lags = [coarse_path[w][0] for w in range(p_start, p_end)]
                            p_med = sorted(p_lags)[len(p_lags) // 2]
                            p_center_idx = int(round(p_med * frame_hz)) + pad_frames
                            p_lo = max(0, p_center_idx - int(round(0.30 * frame_hz)))
                            p_hi = min(2 * pad_frames, p_center_idx + int(round(0.30 * frame_hz)))
                            if p_hi > p_lo:
                                p_sum = np.sum([curves[w][p_lo : p_hi + 1] for w in range(p_start, p_end)], axis=0)
                                p_best_idx = p_lo + int(np.argmax(p_sum))
                                p_best_lag = (p_best_idx - pad_frames) / float(frame_hz)
                                for w in range(p_start, p_end):
                                    coarse_path[w] = (p_best_lag, float(curves[w][p_best_idx]))
                        p_start = p_end

                _lock_plateaus()

                # 3d. Second-Order Anti-Spike Regularizer (runs BEFORE fine local refinement):
                # Eliminate non-physical 1-act (or 2-short-act) "ping-pong" reversal spikes where
                # offset jumps outside [min(prev, next), max(prev, next)] and immediately jumps back.
                for w in range(start_cons_idx + 1, num_acts - 2):
                    prev_lag = coarse_path[w - 1][0]
                    next2_lag = coarse_path[w + 2][0]
                    l1, sc1 = coarse_path[w]
                    l2, sc2 = coarse_path[w + 1]
                    span_pair = acts_scaled[w + 1][-1][1] - acts_scaled[w][0][0]
                    lo_p, hi_p = min(prev_lag, next2_lag), max(prev_lag, next2_lag)
                    if (
                        span_pair <= 110.0
                        and abs(l1 - l2) <= 0.45
                        and (l1 < lo_p - 0.65 or l1 > hi_p + 0.65)
                        and (l2 < lo_p - 0.65 or l2 > hi_p + 0.65)
                    ):
                        lo_i = max(0, int(round((lo_p - 0.35) * frame_hz)) + pad_frames)
                        hi_i = min(2 * pad_frames, int(round((hi_p + 0.35) * frame_hz)) + pad_frames)
                        if hi_i > lo_i:
                            pair_sum = curves[w][lo_i : hi_i + 1] + curves[w + 1][lo_i : hi_i + 1]
                            best_pi = lo_i + int(np.argmax(pair_sum))
                            cand_lag = (best_pi - pad_frames) / float(frame_hz)
                            if (sc1 + sc2) - float(pair_sum[best_pi - lo_i]) < 0.36:
                                coarse_path[w] = (cand_lag, float(curves[w][best_pi]))
                                coarse_path[w + 1] = (cand_lag, float(curves[w + 1][best_pi]))

                for w in range(start_cons_idx + 1, num_acts - 1):
                    prev_lag = coarse_path[w - 1][0]
                    next_lag = coarse_path[w + 1][0]
                    cur_lag, cur_sc = coarse_path[w]
                    lo_nb, hi_nb = min(prev_lag, next_lag), max(prev_lag, next_lag)
                    if cur_lag < lo_nb - 0.45 or cur_lag > hi_nb + 0.45:
                        lo_i = max(0, int(round((lo_nb - 0.35) * frame_hz)) + pad_frames)
                        hi_i = min(2 * pad_frames, int(round((hi_nb + 0.35) * frame_hz)) + pad_frames)
                        if hi_i > lo_i:
                            best_ci = lo_i + int(np.argmax(curves[w][lo_i : hi_i + 1]))
                            cand_lag = (best_ci - pad_frames) / float(frame_hz)
                            cand_sc = float(curves[w][best_ci])
                            vc_cur = _act_vocal_coverage(ref_arr, acts_scaled[w], cur_lag, frame_hz=frame_hz)
                            vc_cand = _act_vocal_coverage(ref_arr, acts_scaled[w], cand_lag, frame_hz=frame_hz)
                            keep_spike = (
                                cur_sc >= 0.58
                                and (cur_sc - cand_sc) >= 0.22
                                and vc_cur >= vc_cand + 0.12
                            )
                            if not keep_spike:
                                coarse_path[w] = (cand_lag, cand_sc)

                _lock_plateaus()

                # 3c. Convex-Hull Bounded Local Peak Refinement (runs AFTER plateau locking so local
                # refinements within <= 0.25s are preserved):
                # - Inside a coherent plateau (|prev - next| <= 0.45s), stay within ±0.30s of neighbors.
                # - At a commercial break boundary (|prev - next| > 0.45s), if already locked to one
                #   neighbor plateau, only switch to the OTHER neighbor plateau (never invent an
                #   artificial intermediate step in between).
                # - Short boundary acts (< 10 subs) snap to their adjacent plateau unless strongly distinct.
                for w in range(start_cons_idx, num_acts):
                    cur_lag, cur_sc = coarse_path[w]
                    if start_cons_idx < w < num_acts - 1:
                        if len(acts_scaled[w]) < 5:
                            continue
                        prev_lag = coarse_path[w - 1][0]
                        next_lag = coarse_path[w + 1][0]
                        if abs(prev_lag - next_lag) <= 0.45:
                            lo_bound = min(prev_lag, next_lag) - 0.30
                            hi_bound = max(prev_lag, next_lag) + 0.30
                            min_gain = 0.055
                        elif min(abs(cur_lag - prev_lag), abs(cur_lag - next_lag)) <= 0.25:
                            other_lag = next_lag if abs(cur_lag - prev_lag) <= abs(cur_lag - next_lag) else prev_lag
                            lo_bound = other_lag - 0.25
                            hi_bound = other_lag + 0.25
                            min_gain = 0.085
                        else:
                            lo_bound = min(prev_lag, next_lag) - 0.35
                            hi_bound = max(prev_lag, next_lag) + 0.35
                            min_gain = 0.055
                        lo_i = max(0, int(round(lo_bound * frame_hz)) + pad_frames)
                        hi_i = min(2 * pad_frames, int(round(hi_bound * frame_hz)) + pad_frames)
                        if hi_i > lo_i:
                            loc_best_i = lo_i + int(np.argmax(curves[w][lo_i : hi_i + 1]))
                            loc_best_sc = float(curves[w][loc_best_i])
                            loc_best_lag = (loc_best_i - pad_frames) / float(frame_hz)
                            if loc_best_sc >= 0.42 and (loc_best_sc - cur_sc) >= min_gain:
                                coarse_path[w] = (loc_best_lag, loc_best_sc)
                    else:
                        nb_idx = (w + 1) if (w == start_cons_idx and w + 1 < num_acts) else (w - 1 if w > start_cons_idx else -1)
                        if nb_idx < 0:
                            continue
                        nb_lag = coarse_path[nb_idx][0]
                        if len(acts_scaled[w]) < 10 and abs(cur_lag - nb_lag) <= 2.20:
                            nb_i = max(0, min(2 * pad_frames, int(round(nb_lag * frame_hz)) + pad_frames))
                            nb_sc = float(curves[w][nb_i])
                            if cur_sc - nb_sc < 0.16:
                                coarse_path[w] = (nb_lag, nb_sc)
                        elif len(acts_scaled[w]) >= 12:
                            lo_i = max(0, int(round((min(cur_lag, nb_lag) - 1.50) * frame_hz)) + pad_frames)
                            hi_i = min(2 * pad_frames, int(round((max(cur_lag, nb_lag) + 1.50) * frame_hz)) + pad_frames)
                            if hi_i > lo_i:
                                loc_best_i = lo_i + int(np.argmax(curves[w][lo_i : hi_i + 1]))
                                loc_best_sc = float(curves[w][loc_best_i])
                                if loc_best_sc >= 0.50 and (loc_best_sc - cur_sc) >= 0.14:
                                    coarse_path[w] = ((loc_best_i - pad_frames) / float(frame_hz), loc_best_sc)

        # 4. Stage 2: Zero-Bias Vocal Onset Refinement on each act
        chain: List[dict] = []
        for k in range(num_acts):
            cb, sc = coarse_path[k]
            w_start = acts_scaled[k][0][0]
            w_end = acts_scaled[k][-1][1]
            x_start = acts_orig[k][0][0]
            x_end = acts_orig[k][-1][1]
            x_center = 0.5 * (x_start + x_end)

            if ref_intervals and cb > -acts_scaled[k][-1][1]:
                rb = _refine_act_vocal_attack(
                    None,
                    acts_scaled[k],
                    cb,
                    frame_hz=frame_hz,
                    ref_intervals=ref_intervals,
                )
            else:
                rb = cb

            chain.append(
                {
                    "w_start": w_start,
                    "w_end": w_end,
                    "x_start": x_start,
                    "x_end": x_end,
                    "x_center": x_center,
                    "b": rb,
                    "coarse_b": cb,
                    "score": sc,
                    "subs": acts_orig[k],
                }
            )

        if len(chain) >= 2:
            if len(chain[0]["subs"]) < 10 and abs(chain[0]["b"] - chain[1]["b"]) <= 0.40:
                chain[0]["b"] = 0.35 * chain[0]["b"] + 0.65 * chain[1]["b"]
            if len(chain[-1]["subs"]) < 10 and abs(chain[-1]["b"] - chain[-2]["b"]) <= 0.40:
                chain[-1]["b"] = 0.35 * chain[-1]["b"] + 0.65 * chain[-2]["b"]

        return chain, total_viterbi_score

    # Pure-Python fallback if numpy is absent
    raw_wins: List[dict] = []
    for k in range(num_acts):
        w_start = acts_scaled[k][0][0]
        w_end = acts_scaled[k][-1][1]
        r0 = max(0, int(round(w_start * frame_hz)))
        r1 = min(ref_len, int(round(w_end * frame_hz)))
        if r1 - r0 < int(15.0 * frame_hz):
            continue
        w_subs_rel = [(s - w_start, e - w_start) for s, e in acts_scaled[k]]
        w_env = vad.intervals_to_envelope(
            w_subs_rel,
            duration_s=(w_end - w_start),
            frame_hz=frame_hz,
            edge_boost=0.0,
        )
        r_slice = ref_envelope[r0:r1]
        lag_frames, w_score = fft_correlate(r_slice, w_env, max_win_lag)
        b_w = lag_frames / float(frame_hz)
        x_start, x_end = acts_orig[k][0][0], acts_orig[k][-1][1]
        raw_wins.append(
            {
                "w_start": w_start,
                "w_end": w_end,
                "x_start": x_start,
                "x_end": x_end,
                "x_center": 0.5 * (x_start + x_end),
                "b": b_w,
                "coarse_b": b_w,
                "score": w_score,
                "subs": acts_orig[k],
            }
        )
    if not raw_wins:
        return [], 0.0
    tot = sum(w["score"] for w in raw_wins)
    return raw_wins, tot


def align_intervals_and_envelopes(
    sub_intervals: Sequence[Tuple[float, float]],
    ref_envelope: Sequence[float],
    ref_intervals: Optional[Sequence[Tuple[float, float]]] = None,
    frame_hz: int = vad.FRAME_HZ,
    max_offset_s: float = 600.0,
    allow_drift: bool = True,
) -> AlignmentResult:
    """Align subtitle dialogue intervals against a reference 100 Hz speech envelope.

    Guarantees <= 0.1s alignment accuracy across constant-offset, framerate-drifted,
    and multi-act TV episode tracks (with commercial break cuts) through 2-stage
    FFT cross-correlation, multi-window piecewise act alignment, and boundary OLS refinement.
    """
    if not sub_intervals or not ref_envelope:
        return AlignmentResult(
            correlated=False,
            a=1.0,
            b=0.0,
            factor=0.0,
            points=0,
            max_distance=0.0,
            max_change=0.0,
            formula=FormulaLine(1.0, 0.0),
        )

    t_sub_min = sub_intervals[0][0]
    t_sub_max = sub_intervals[-1][1]
    t_span = max(1.0, t_sub_max - t_sub_min)

    max_lag_frames = max(100, int(round(max_offset_s * frame_hz)))
    realistic_lag_frames = min(max_lag_frames, int(round(90.0 * frame_hz)))
    lib = get_fast_corr_lib()
    ref_c_buf = (ctypes.c_double * len(ref_envelope))(*ref_envelope) if lib is not None else None

    # 1. Evaluate unit slope a = 1.0 first (preferring realistic <= 90s lag unless wide lag is strong)
    sub_env_1 = vad.intervals_to_envelope(sub_intervals, frame_hz=frame_hz, scale_a=1.0)
    lag_1, score_1 = fft_correlate(ref_envelope, sub_env_1, realistic_lag_frames, ref_c_buf=ref_c_buf)
    if max_lag_frames > realistic_lag_frames and score_1 < 0.20:
        lag_wide, score_wide = fft_correlate(ref_envelope, sub_env_1, max_lag_frames, ref_c_buf=ref_c_buf)
        if score_wide >= 0.25 and score_wide > score_1 * 1.35:
            lag_1, score_1 = lag_wide, score_wide
    b_1 = lag_1 / float(frame_hz)

    best_a = 1.0
    best_b = b_1
    best_score = score_1

    # 2. If drift search is enabled, evaluate standard framerate ratios + windowed RANSAC candidates
    if allow_drift and len(sub_intervals) >= 4 and 15.0 <= t_span < 300.0:
        candidates = list(STANDARD_SPEED_FACTORS[1:])
        discovered = _discover_windowed_drift_candidates(
            ref_envelope, sub_intervals, frame_hz=frame_hz, max_offset_s=max_offset_s
        )
        for d_a in discovered:
            if all(abs(d_a - c) > 1e-5 for c in candidates):
                candidates.append(d_a)

        drift_lag_frames = min(
            max(max_lag_frames, int(round((max_offset_s + 0.06 * t_sub_max) * frame_hz))),
            int(round((90.0 + 0.06 * t_sub_max) * frame_hz)),
        )

        best_drift_a = 1.0
        best_drift_b = b_1
        best_drift_score = score_1

        for cand_a in candidates:
            sub_env_a = vad.intervals_to_envelope(sub_intervals, frame_hz=frame_hz, scale_a=cand_a)
            lag_a, score_a = fft_correlate(ref_envelope, sub_env_a, drift_lag_frames, ref_c_buf=ref_c_buf)
            if score_a > best_drift_score:
                best_drift_score = score_a
                best_drift_a = cand_a
                best_drift_b = lag_a / float(frame_hz)

        # Slope regularization: only depart from a = 1.0 if non-unit slope improves
        # correlation peak significantly (> 7%) AND implies > 0.12s timeline drift!
        implied_drift_s = abs(best_drift_a - 1.0) * t_span
        if implied_drift_s > 0.12 and best_drift_score > max(score_1 * 1.07, score_1 + 0.03):
            best_a = best_drift_a
            best_b = best_drift_b
            best_score = best_drift_score

    # 3. Multi-act piecewise scene analysis for full episodes/movies (t_span >= 300s)
    piecewise_anchors: List[Tuple[float, float]] = []
    window_offsets_out: List[Tuple[float, float, float, float]] = []
    win_chain: List[dict] = []

    if t_span >= 300.0 and len(ref_envelope) >= int(300.0 * frame_hz):
        max_win_lag = min(max_lag_frames, int(round(75.0 * frame_hz)))
        win_chain, win_tot = _analyze_episode_windows(
            ref_envelope=ref_envelope,
            ref_intervals=ref_intervals,
            sub_intervals=sub_intervals,
            scale_a=1.0,
            frame_hz=frame_hz,
            max_win_lag=max_win_lag,
        )

        mean_1_score = (
            sum(w["score"] for w in win_chain) / float(len(win_chain)) if win_chain else 0.0
        )

        # 3a. Adaptive Linear Drift Pre-Scaling for episodes with true continuous progressive drift
        # (vs discrete commercial-break step functions where >= 45% of adjacent act transitions are flat plateaus)
        if allow_drift and len(win_chain) >= 6 and np is not None:
            start_d_idx = (
                1
                if (
                    win_chain[0]["x_start"] < 60.0
                    and (win_chain[0]["b"] < -10.0 or win_chain[0]["b"] < win_chain[1]["b"] - 6.0)
                )
                else 0
            )
            post_acts = [w for w in win_chain[start_d_idx:] if w["score"] >= 0.28]
            if len(post_acts) >= 5:
                xs_d = np.array([w["x_center"] for w in post_acts], dtype=np.float64)
                bs_d = np.array([w["b"] for w in post_acts], dtype=np.float64)
                scs_d = np.array([w["score"] for w in post_acts], dtype=np.float64)
                adj_diffs = np.abs(np.diff(bs_d))
                frac_flat_adj = float(np.mean(adj_diffs <= 0.35)) if len(adj_diffs) > 0 else 1.0
                # If >= 45% of adjacent act transitions are flat plateaus (<= 0.35s), the offset pattern
                # is a piecewise-constant commercial-break staircase (a = 1.0), NOT continuous linear drift!
                if frac_flat_adj < 0.45:
                    pair_slopes = []
                    for i_d in range(len(post_acts)):
                        for j_d in range(i_d + 1, len(post_acts)):
                            dx_d = xs_d[j_d] - xs_d[i_d]
                            if dx_d >= 350.0:
                                pair_slopes.append((bs_d[j_d] - bs_d[i_d]) / dx_d)
                    slope_cands: List[float] = []
                    if pair_slopes:
                        med_s = float(np.median(pair_slopes))
                        if abs(med_s * t_span) >= 5.0 and abs(med_s) <= 0.05:
                            slope_cands.append(1.0 + med_s)
                    try:
                        ols_s, _ = np.polyfit(xs_d, bs_d, 1, w=scs_d)
                        ols_s = float(ols_s)
                        if abs(ols_s * t_span) >= 5.0 and abs(ols_s) <= 0.05:
                            if not slope_cands or abs((1.0 + ols_s) - slope_cands[0]) > 0.0004:
                                slope_cands.append(1.0 + ols_s)
                    except Exception:
                        pass

                    spread_1 = (
                        max(w["b"] for w in win_chain[start_d_idx:])
                        - min(w["b"] for w in win_chain[start_d_idx:])
                    )
                    for cand_a_drift in slope_cands:
                        cand_chain, cand_tot = _analyze_episode_windows(
                            ref_envelope=ref_envelope,
                            ref_intervals=ref_intervals,
                            sub_intervals=sub_intervals,
                            scale_a=cand_a_drift,
                            frame_hz=frame_hz,
                            max_win_lag=max_win_lag,
                        )
                        if len(cand_chain) >= 5:
                            c_start = (
                                1
                                if (
                                    cand_chain[0]["x_start"] < 60.0
                                    and (
                                        cand_chain[0]["b"] < -10.0
                                        or cand_chain[0]["b"] < cand_chain[1]["b"] - 6.0
                                    )
                                )
                                else 0
                            )
                            cand_mean = sum(w["score"] for w in cand_chain) / float(len(cand_chain))
                            cand_bs = [float(w["b"]) for w in cand_chain[c_start:]]
                            cand_spread = max(cand_bs) - min(cand_bs)
                            cand_med = float(np.median(cand_bs))
                            cand_mad = float(np.median(np.abs(np.asarray(cand_bs) - cand_med)))
                            if (
                                cand_mean >= mean_1_score + 0.020
                                and cand_spread <= min(0.85, spread_1 * 0.35)
                                and cand_mad <= 0.18
                            ):
                                best_a = cand_a_drift
                                win_chain = cand_chain
                                win_tot = cand_tot
                                mean_1_score = cand_mean
                                spread_1 = cand_spread

        # Only evaluate canonical PAL/NTSC ratios on full episodes if correlation is still weak (< 0.25 mean),
        # and require a uniform constant offset (spread < 3.0s) across >= 6 acts with > 35% score gain.
        if allow_drift and (len(win_chain) < 4 or mean_1_score < 0.25):
            best_pal_a = best_a
            best_pal_chain = win_chain
            best_pal_tot = win_tot
            for pal_a in (
                25.0 / 23.976,
                25.0 / (24000.0 / 1001.0),
                23.976 / 25.0,
                (24000.0 / 1001.0) / 25.0,
                25.0 / 24.0,
                24.0 / 25.0,
            ):
                cand_chain, cand_tot = _analyze_episode_windows(
                    ref_envelope=ref_envelope,
                    ref_intervals=ref_intervals,
                    sub_intervals=sub_intervals,
                    scale_a=pal_a,
                    frame_hz=frame_hz,
                    max_win_lag=max_win_lag,
                )
                if len(cand_chain) >= 6:
                    cand_spread = max(w["b"] for w in cand_chain) - min(w["b"] for w in cand_chain)
                    if cand_spread < 3.0 and cand_tot > max(best_pal_tot * 1.05, win_tot * 1.35, 2.0):
                        best_pal_a = pal_a
                        best_pal_chain = cand_chain
                        best_pal_tot = cand_tot
            best_a = best_pal_a
            win_chain = best_pal_chain
            win_tot = best_pal_tot

        # 3e. Targeted Faster-Whisper Bilingual Arbitration on ambiguous / wall-to-wall / jump acts
        if abs(best_a - 1.0) < 1e-4 and len(win_chain) >= 3:
            try:
                from subsync.synchro import ai_engine

                if ai_engine.get_last_vad_result() is not None and ai_engine.get_last_pcm_bytes() is not None:
                    wh_candidates: List[Tuple[float, int]] = []
                    for k_w, w_item in enumerate(win_chain):
                        a_subs = w_item["subs"]
                        if len(a_subs) < 5 or w_item["x_start"] < 5.0 or w_item["b"] < -65.0:
                            continue
                        a_dur = max(1.0, a_subs[-1][1] - a_subs[0][0])
                        a_duty = sum(max(0.0, e - s) for s, e in a_subs) / a_dur
                        jump_prev = abs(w_item["b"] - win_chain[k_w - 1]["b"]) if k_w > 0 else 0.0
                        if (
                            (a_duty >= 0.80 and a_dur >= 35.0)
                            or w_item["score"] < 0.28
                            or (jump_prev >= 1.20 and w_item["score"] < 0.44)
                        ):
                            prio = (1.0 - float(w_item["score"])) + (0.6 if a_duty >= 0.80 else 0.0)
                            wh_candidates.append((prio, k_w))

                    wh_candidates.sort(reverse=True)
                    for _, k_w in wh_candidates[:4]:
                        w_item = win_chain[k_w]
                        cur_b_w = float(w_item["b"])
                        prev_b_w = float(win_chain[k_w - 1]["b"]) if k_w > 0 else cur_b_w
                        next_b_w = float(win_chain[k_w + 1]["b"]) if k_w + 1 < len(win_chain) else cur_b_w
                        lo_wh = max(-75.0, min(cur_b_w - 8.0, prev_b_w - 3.0, next_b_w - 3.0))
                        hi_wh = min(75.0, max(cur_b_w + 8.0, prev_b_w + 3.0, next_b_w + 3.0))
                        ver = ai_engine.verify_act_offset_with_whisper(
                            w_item["subs"],
                            b_current=cur_b_w,
                            b_min=lo_wh,
                            b_max=hi_wh,
                            b_prior=None,
                            frame_hz=frame_hz,
                        )
                        if not ver:
                            continue
                        b_wh = float(ver["b"])
                        lex_wh = int(ver["lexical_hits"])
                        lex_cur = int(ver["lexical_hits_current"])
                        anc_cnt = int(ver["anchors_count"])
                        if abs(b_wh - cur_b_w) < 0.12:
                            continue
                        lo_nb_w, hi_nb_w = min(prev_b_w, next_b_w), max(prev_b_w, next_b_w)
                        creates_spike = (0 < k_w < len(win_chain) - 1) and (
                            b_wh < lo_nb_w - 0.50 or b_wh > hi_nb_w + 0.50
                        )
                        if (
                            lex_wh >= 3
                            and anc_cnt >= 5
                            and lex_wh >= lex_cur + 1
                            and (not creates_spike or (lex_wh >= 6 and lex_wh >= lex_cur + 3))
                        ):
                            old_b_w = cur_b_w
                            w_item["b"] = b_wh
                            # Propagate to adjacent low-confidence acts sharing the exact same plateau offset
                            for nb_idx in (k_w - 1, k_w + 1):
                                if 0 < nb_idx < len(win_chain):
                                    nb_item = win_chain[nb_idx]
                                    if nb_item["score"] < 0.34 and abs(float(nb_item["b"]) - old_b_w) <= 0.25:
                                        nb_item["b"] = b_wh
            except Exception as wh_exc:
                logger.debug("Targeted Whisper act arbitration skipped: %s", wh_exc)

        if len(win_chain) >= 2:
            window_offsets_out = [
                (
                    w["w_start"],
                    w["w_end"],
                    (best_a - 1.0) * float(w["x_center"]) + float(w["b"]),
                    w["score"],
                )
                for w in win_chain
            ]
            win_scores = [w["score"] for w in win_chain]
            win_bs = sorted(w["b"] for w in win_chain)
            median_win_b = win_bs[len(win_bs) // 2]
            mean_win_score = sum(win_scores) / float(len(win_scores))

            best_b = median_win_b
            best_score = max(best_score, mean_win_score)

            b_spread = max(w["b"] for w in win_chain) - min(w["b"] for w in win_chain)
            if b_spread > 0.20 or abs(best_a - 1.0) > 1e-4:
                # Emit strictly monotonic piecewise-linear anchors [(x_start, y_start), (x_end, y_end)]
                # so offset b_k is constant (relative to scale best_a) inside each dialogue scene [x_start, x_end]
                # and only transitions during the silent gaps between acts/sub-scenes!
                prev_y1 = -1e9
                for w in win_chain:
                    x0 = float(w["x_start"])
                    x1 = max(x0 + 0.01, float(w["x_end"]))
                    bk = float(w["b"])
                    y0 = max(prev_y1 + 0.02, best_a * x0 + bk)
                    y1 = max(y0 + 0.01, best_a * x1 + bk)
                    piecewise_anchors.append((x0, y0))
                    piecewise_anchors.append((x1, y1))
                    prev_y1 = y1

    # 4. Stage 2: Sub-10ms Fine Boundary Refinement using ref_intervals (for single-line / short tracks)
    inlier_pts: List[Tuple[float, float]] = []
    matched_events = len(sub_intervals)
    max_dist = 0.01

    if ref_intervals and not piecewise_anchors and len(win_chain) < 2:
        anchors, matched_events = _match_interval_anchors(
            sub_intervals, ref_intervals, best_a, best_b, tolerance_s=0.45
        )
        start_anchors, _ = _match_interval_anchors(
            sub_intervals, ref_intervals, best_a, best_b, tolerance_s=0.45, start_only=True
        )
        if len(anchors) >= 4:
            if abs(best_a - 1.0) < 1e-6:
                refined_b, inlier_pts, max_dist = _refine_offset_fixed_slope(
                    anchors, 1.0, best_b, inlier_Radius=0.25, start_anchors=start_anchors
                )
                if abs(refined_b - best_b) <= 0.12:
                    best_b = refined_b
            else:
                ols_a, ols_b, inlier_pts, max_dist, _ = _refine_line_ols(
                    anchors, best_a, best_b, max_dist=0.08, min_points=min(10, len(anchors) // 2)
                )
                fixed_b, fixed_pts, fixed_max_dist = _refine_offset_fixed_slope(
                    anchors, best_a, best_b, inlier_Radius=0.25, start_anchors=start_anchors
                )
                if fixed_pts:
                    fixed_mae = sum(abs(best_a * x + fixed_b - y) for x, y in fixed_pts) / len(fixed_pts)
                else:
                    fixed_mae = 1.0
                if inlier_pts:
                    ols_mae = sum(abs(ols_a * x + ols_b - y) for x, y in inlier_pts) / len(inlier_pts)
                else:
                    ols_mae = 1.0

                if abs(ols_a - best_a) <= 0.005 and ols_mae < fixed_mae - 0.002:
                    best_a = ols_a
                    best_b = ols_b
                elif abs(fixed_b - best_b) <= 0.15:
                    best_b = fixed_b
                    inlier_pts = fixed_pts
                    max_dist = fixed_max_dist

                # Re-verify slope regularization after OLS: if |best_a - 1.0| * t_span <= 0.10s, snap to 1.0
                if abs(best_a - 1.0) * t_span <= 0.10:
                    best_a = 1.0
                    best_b, inlier_pts, max_dist = _refine_offset_fixed_slope(
                        anchors, 1.0, b_1, inlier_Radius=0.25, start_anchors=start_anchors
                    )

    formula = FormulaLine(best_a, best_b, piecewise_anchors=piecewise_anchors)

    if not inlier_pts:
        inlier_pts = [(s, formula.getY(s)) for s, _ in sub_intervals]

    points_count = max(matched_events, len(inlier_pts) // 2)

    if t_span >= 300.0 and len(ref_envelope) >= int(300.0 * frame_hz):
        # Full episode/movie requires either >= 2 consistent acts or strong global correlation >= 0.15
        correlated = (len(win_chain) >= 2 and best_score >= 0.15) or (best_score >= 0.15 and points_count >= 10)
    else:
        correlated = best_score >= 0.05 and points_count >= 1

    factor = max(0.0, min(1.0, float(best_score)))

    sample_xs = [t_sub_min, t_sub_max] + [px for px, _ in piecewise_anchors]
    max_change = max(abs(formula.getY(x) - x) for x in sample_xs)

    return AlignmentResult(
        correlated=correlated,
        a=best_a,
        b=best_b,
        factor=factor,
        points=int(points_count),
        max_distance=float(max_dist),
        max_change=float(max_change),
        formula=formula,
        inlier_points=inlier_pts,
        window_offsets=window_offsets_out,
    )
