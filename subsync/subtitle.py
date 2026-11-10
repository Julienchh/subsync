from subsync import compat_pysubs2
compat_pysubs2.register_as_pysubs2()

from subsync import error
from subsync import utils
from subsync.data import filetypes
from subsync.translations import _
import bisect
import pysubs2
import copy
import re
import threading
import os

import logging
logger = logging.getLogger(__name__)

_WATERMARK_RE = re.compile(
    r"(?:addic7ed\.com|subsynchro|sous-titres\.eu|tvsubtitles\.net|opensubtitles\.org|"
    r"sous-titres\s+traduits\s+par|traduits?\s+par\s*(?:la\s+communaut|:|[a-z0-9_\-])|"
    r"traduction\s*(?:par\b|by\b|:)|re-?synchro\b|"
    r"synchro(?:nis[ée])?\s*(?:&\s*corrig[ée]\s*)?(?:par\b|by\b|:)|"
    r"syncro\s*par|re-?sync\s*(?:par\b|by\b)|sync\s*by\b|corrected\s+by\b)",
    re.IGNORECASE,
)


def is_credit_watermark(text: str) -> bool:
    """Return True if subtitle text is a non-dialogue credit/watermark line."""
    if not text:
        return False
    return bool(_WATERMARK_RE.search(text))


def _destack_zero_duration_clusters(events, ref_intervals=None):
    """Unstack pathological 0ms / micro-duration cue clusters (e.g. S09E02 Mandarin cues) across active speech."""
    if not events or len(events) < 2:
        return 0
    n = len(events)
    fixed = 0
    i = 0
    while i < n:
        dur_i = events[i].end - events[i].start
        if dur_i <= 250:
            j = i + 1
            while (
                j < n
                and abs(events[j].start - events[i].start) <= 30
                and (events[j].end - events[j].start) <= 250
            ):
                j += 1
            cluster_len = j - i
            if cluster_len >= 2:
                cluster_t_ms = events[i].start
                prev_end_ms = events[i - 1].end if i > 0 else 0
                next_start_ms = events[j].start if j < n else (cluster_t_ms + cluster_len * 2200)
                gap_before_ms = cluster_t_ms - prev_end_ms
                gap_after_ms = next_start_ms - cluster_t_ms

                if gap_before_ms >= 8000 and gap_after_ms < gap_before_ms * 0.5:
                    win_s_ms = prev_end_ms + 250
                    win_e_ms = cluster_t_ms - 120
                elif gap_after_ms >= 4000:
                    win_s_ms = max(prev_end_ms + 80, cluster_t_ms)
                    win_e_ms = next_start_ms - 120
                elif gap_before_ms >= 4000:
                    win_s_ms = prev_end_ms + 150
                    win_e_ms = max(win_s_ms + cluster_len * 450, next_start_ms - 80)
                else:
                    win_s_ms = max(prev_end_ms + 70, cluster_t_ms)
                    win_e_ms = max(win_s_ms + cluster_len * 450, next_start_ms - 70)

                # Narrow [win_s_ms, win_e_ms] to actual speech span from ref_intervals if available
                if ref_intervals:
                    win_s_sec = win_s_ms / 1000.0
                    win_e_sec = win_e_ms / 1000.0
                    active_ivs = [
                        (max(win_s_sec, rs), min(win_e_sec, re_t))
                        for rs, re_t in ref_intervals
                        if min(win_e_sec, re_t) - max(win_s_sec, rs) >= 0.25
                    ]
                    if active_ivs:
                        sp_s_ms = int(round(active_ivs[0][0] * 1000.0))
                        sp_e_ms = int(round(active_ivs[-1][1] * 1000.0))
                        if sp_e_ms - sp_s_ms >= cluster_len * 450:
                            win_s_ms = max(win_s_ms, sp_s_ms)
                            win_e_ms = min(win_e_ms, sp_e_ms)

                span_ms = max(cluster_len * 420, win_e_ms - win_s_ms)
                weights = []
                for k in range(i, j):
                    txt = re.sub(r"<[^>]+>", "", getattr(events[k], "plaintext", events[k].text) or "").strip()
                    weights.append(max(6.0, float(len(txt))))
                w_sum = sum(weights)
                gap_ms = 70
                usable_ms = max(cluster_len * 350, span_ms - gap_ms * (cluster_len - 1))
                cur_ms = win_s_ms
                for idx_c, k in enumerate(range(i, j)):
                    alloc_ms = int(round(usable_ms * (weights[idx_c] / w_sum)))
                    alloc_ms = max(350, min(4200, alloc_ms))
                    events[k].start = cur_ms
                    events[k].end = cur_ms + alloc_ms
                    cur_ms = events[k].end + gap_ms
                    fixed += 1
                i = j
                continue
            elif dur_i < 300:
                next_s_ms = events[i + 1].start if (i + 1 < n) else (events[i].start + 1500)
                prev_e_ms = events[i - 1].end if i > 0 else 0
                if events[i].start < prev_e_ms + 40:
                    events[i].start = prev_e_ms + 40
                target_end = min(next_s_ms - 60, events[i].start + 900)
                if target_end >= events[i].start + 300:
                    events[i].end = target_end
                else:
                    events[i].end = events[i].start + 300
                fixed += 1
        i += 1
    return fixed


class Subtitles(pysubs2.SSAFile):
    def __init__(self):
        super().__init__()

    def cleanWatermarks(self):
        """Remove non-dialogue credit/watermark events in-place."""
        if not self.events:
            return
        self.events = [
            ev
            for ev in self.events
            if not is_credit_watermark(getattr(ev, "plaintext", ev.text))
            and (getattr(ev, "plaintext", ev.text) or "").strip() != ""
        ]

    def sort(self):
        self.cleanWatermarks()
        super().sort()

    def setHeader(self, header):
        s = super().from_string(header)
        self.aegisub_project = s.aegisub_project
        self.info = s.info
        self.styles = s.styles

    def add(self, begin, end, text):
        if text.startswith('[Script Info]'):
            self.setHeader(text)
        else:
            entry = parseLine(text)
            if is_credit_watermark(entry.get('text', '')):
                return
            event = pysubs2.SSAEvent(
                    type = 'Dialogue',
                    start = int(round(begin * 1000.0)),
                    end = int(round(end * 1000.0)),
                    **entry)
            self.insert(bisect.bisect_left(self, event), event)

    def synchronize(self, formula):
        res = copy.deepcopy(self)
        res.cleanWatermarks()
        piecewise = getattr(formula, 'piecewise_anchors', None)
        if piecewise and hasattr(formula, 'getY'):
            for ev in res.events:
                orig_dur = max(200, ev.end - ev.start)
                ev.start = int(round(formula.getY(ev.start / 1000.0) * 1000.0))
                ev.end = int(round(formula.getY(ev.end / 1000.0) * 1000.0))
                if ev.end <= ev.start:
                    ev.end = ev.start + orig_dur
        else:
            if hasattr(formula, 'a') and hasattr(formula, 'b'):
                a, b = float(formula.a), float(formula.b)
            else:
                a, b = float(formula[0]), float(formula[1])

            # Map each timestamp t_ms -> round(a * t_ms + b * 1000.0) in a single step
            # for 1ms accuracy without intermediate rounding loss.
            b_ms = b * 1000.0
            for ev in res.events:
                ev.start = int(round(a * ev.start + b_ms))
                ev.end = int(round(a * ev.end + b_ms))

        res.sort()
        # Drop any event whose synchronized end <= 0 or start < -300ms (-0.30s),
        # which cleanly removes orphan "Précédemment..." recap lines that precede 00:00:00 of the video.
        filtered_events = []
        for ev in res.events:
            if ev.end <= 0 or ev.start < -300:
                logger.debug('removing subtitle line with negative time: %r', ev)
                continue
            if ev.start < 0:
                ev.start = 0
            filtered_events.append(ev)
        res.events = filtered_events
        _destack_zero_duration_clusters(res.events)
        return res

    def verify_and_heal_with_vad(self, ref_envelope, ref_intervals=None, frame_hz=100):
        """
        Stage 4 (Closed-Loop Audit & Self-Healing):
        Partitions synchronized subtitle events into dialogue verification sub-windows (15s-65s),
        evaluates each sub-window's local NCC correlation around delta_b = 0.0s against the
        100Hz neural VAD envelope, and self-heals any sub-window that exhibits a clear residual
        lag (|delta_b| >= 0.16s with >= +6.5% NCC gain and non-decreasing vocal hit-rate) while
        strictly preserving monotonicity between adjacent windows.
        Returns `(healed_events_count, verified_windows_count, total_windows_count)`.
        """
        if not self.events or not ref_envelope or frame_hz <= 0:
            return 0, 0, 0

        _destack_zero_duration_clusters(self.events, ref_intervals=ref_intervals)

        env_len = len(ref_envelope)
        if env_len < frame_hz * 30:
            return 0, 0, 0

        try:
            import numpy as np
            from subsync.synchro import aligner, vad
        except Exception:
            return 0, 0, 0

        ref_arr = np.asarray(ref_envelope, dtype=np.float64)
        ref_attack_arr = np.asarray(
            aligner._build_vocal_attack_signal(ref_envelope, ref_intervals, frame_hz=frame_hz),
            dtype=np.float64,
        )
        n_ev = len(self.events)
        if n_ev < 5:
            return 0, 1, 1

        # 1. Partition events into verification sub-windows
        raw_windows = []
        cur_indices = [0]
        for i in range(1, n_ev):
            prev_e = self.events[i - 1].end / 1000.0
            cur_s = self.events[i].start / 1000.0
            gap = cur_s - prev_e
            win_dur = (self.events[cur_indices[-1]].end - self.events[cur_indices[0]].start) / 1000.0
            if (
                (gap >= 3.0 and len(cur_indices) >= 5 and win_dur >= 14.0)
                or (gap >= 1.8 and len(cur_indices) >= 9 and win_dur >= 32.0)
                or (win_dur >= 65.0 and gap >= 1.2)
                or win_dur >= 95.0
            ):
                raw_windows.append(cur_indices)
                cur_indices = [i]
            else:
                cur_indices.append(i)
        if cur_indices:
            if raw_windows and len(cur_indices) < 4:
                raw_windows[-1].extend(cur_indices)
            else:
                raw_windows.append(cur_indices)

        healed_events = 0
        verified_windows = 0
        total_windows = len(raw_windows)
        search_rad_s = 6.0
        rad_frames = int(round(search_rad_s * frame_hz))

        def _win_vocal_hit_rate(ivs, shift_s):
            hits = 0
            valid = 0
            margin_f = max(1, int(round(0.10 * frame_hz)))
            for s0, e0 in ivs:
                s = s0 + shift_s
                e = e0 + shift_s
                if e <= 0.0 or e <= s:
                    continue
                i0 = max(0, min(env_len, int(round(s * frame_hz))))
                i1 = max(0, min(env_len, int(round(e * frame_hz))))
                if i1 <= i0:
                    continue
                valid += 1
                seg = ref_arr[i0:i1]
                m0 = max(0, i0 - margin_f)
                m1 = min(env_len, i1 + margin_f)
                seg_m = ref_arr[m0:m1]
                if float(np.max(seg_m)) >= 0.35 and float(np.mean(seg)) >= 0.18:
                    hits += 1
            return (float(hits) / float(valid)) if valid > 0 else 0.0

        # Phase A: Evaluate hybrid (envelope + vocal-attack) NCC curves across all sub-windows
        win_evals = []
        for w_idx, ev_idxs in enumerate(raw_windows):
            w_ivs = [
                (self.events[idx].start / 1000.0, self.events[idx].end / 1000.0)
                for idx in ev_idxs
                if self.events[idx].end > self.events[idx].start
                and not vad.is_pure_sdh_non_speech(getattr(self.events[idx], "plaintext", self.events[idx].text))
            ]
            w_dur = (w_ivs[-1][1] - w_ivs[0][0]) if w_ivs else 0.0
            if len(w_ivs) < 5 or w_dur < 12.0:
                win_evals.append(None)
                continue

            prev_iv_end = (self.events[raw_windows[w_idx - 1][-1]].end / 1000.0) if w_idx > 0 else None
            next_iv_start = (
                (self.events[raw_windows[w_idx + 1][0]].start / 1000.0)
                if (w_idx + 1 < total_windows)
                else None
            )
            curve = aligner._ncc_curve_act_np(
                ref_arr,
                w_ivs,
                prev_iv_end,
                next_iv_start,
                pad_frames=rad_frames,
                frame_hz=frame_hz,
                ref_attack_arr=ref_attack_arr,
            )
            if curve is None or len(curve) != 2 * rad_frames + 1:
                win_evals.append(None)
                continue

            tol_f = max(1, int(round(0.10 * frame_hz)))
            c0 = max(0, rad_frames - tol_f)
            c1 = min(len(curve), rad_frames + tol_f + 1)
            zero_ncc = float(np.max(curve[c0:c1]))

            prev_bound_s = (prev_iv_end + 0.08) if prev_iv_end is not None else 0.0
            next_bound_s = (
                (next_iv_start - 0.08)
                if next_iv_start is not None
                else (w_ivs[-1][1] + search_rad_s + 10.0)
            )
            min_delta = max(-search_rad_s, prev_bound_s - w_ivs[0][0])
            max_delta = min(search_rad_s, next_bound_s - w_ivs[-1][1])
            if max_delta <= min_delta + 0.05:
                win_evals.append(None)
                continue

            k_lo = max(0, min(len(curve) - 1, int(round((min_delta + search_rad_s) * frame_hz))))
            k_hi = max(k_lo, min(len(curve) - 1, int(round((max_delta + search_rad_s) * frame_hz))))
            sub_curve = curve[k_lo : k_hi + 1]
            if len(sub_curve) == 0:
                win_evals.append(None)
                continue

            best_k = k_lo + int(np.argmax(sub_curve))
            best_ncc = float(curve[best_k])
            best_delta = float(best_k - rad_frames) / float(frame_hz)
            win_evals.append(
                {
                    "w_ivs": w_ivs,
                    "zero_ncc": zero_ncc,
                    "best_ncc": best_ncc,
                    "best_delta": best_delta,
                    "min_delta": min_delta,
                    "max_delta": max_delta,
                }
            )

        # Phase B: Neighbor-coherent anti-spike audit & self-healing
        for w_idx, ev_idxs in enumerate(raw_windows):
            ev_info = win_evals[w_idx]
            if ev_info is None:
                verified_windows += 1
                continue

            w_ivs = ev_info["w_ivs"]
            zero_ncc = ev_info["zero_ncc"]
            best_ncc = ev_info["best_ncc"]
            best_delta = ev_info["best_delta"]
            min_delta = ev_info["min_delta"]
            max_delta = ev_info["max_delta"]

            # Recompute tight neighbor collision bounds using any already-healed previous window
            if w_idx > 0:
                live_prev_end = (self.events[raw_windows[w_idx - 1][-1]].end / 1000.0) + 0.08
                min_delta = max(min_delta, live_prev_end - w_ivs[0][0])

            prev_delta = win_evals[w_idx - 1]["best_delta"] if (w_idx > 0 and win_evals[w_idx - 1]) else 0.0
            next_delta = (
                win_evals[w_idx + 1]["best_delta"]
                if (w_idx + 1 < total_windows and win_evals[w_idx + 1])
                else 0.0
            )
            isolated_spike = (
                0 < w_idx < total_windows - 1
                and abs(best_delta) >= 0.22
                and abs(prev_delta) <= 0.14
                and abs(next_delta) <= 0.14
            )

            vh_before = _win_vocal_hit_rate(w_ivs, 0.0)
            eligible = False
            if isolated_spike:
                if (
                    zero_ncc < 0.22
                    and vh_before < 0.76
                    and best_ncc >= 0.50
                    and (best_ncc - zero_ncc) >= 0.16
                ):
                    eligible = True
            elif (
                abs(best_delta) >= 0.16
                and best_ncc >= 0.46
                and (best_ncc - zero_ncc) >= 0.065
            ):
                eligible = True

            if eligible and max_delta > min_delta + 0.05:
                refined_delta = aligner._refine_act_vocal_attack(
                    None,
                    w_ivs,
                    best_delta,
                    frame_hz=frame_hz,
                    ref_intervals=ref_intervals,
                )
                refined_delta = max(min_delta, min(max_delta, refined_delta))
                vh_after = _win_vocal_hit_rate(w_ivs, refined_delta)
                vh_ok = (
                    (vh_after >= vh_before + 0.10)
                    if isolated_spike
                    else (vh_after >= vh_before - 0.02 and (vh_after >= 0.80 or vh_after > vh_before))
                )
                if vh_ok:
                    shift_ms = int(round(refined_delta * 1000.0))
                    if abs(shift_ms) >= 120:
                        for idx in ev_idxs:
                            self.events[idx].start = max(0, self.events[idx].start + shift_ms)
                            self.events[idx].end = max(
                                self.events[idx].start + 250,
                                self.events[idx].end + shift_ms,
                            )
                        healed_events += len(ev_idxs)
                        verified_windows += 1
                        continue

            if abs(best_delta) <= 0.20 or zero_ncc >= best_ncc - 0.045 or zero_ncc >= 0.35:
                verified_windows += 1

        return healed_events, verified_windows, total_windows

    def refine_durations_with_vad(self, ref_envelope, ref_intervals=None, frame_hz=100):
        """
        Stage 3 (Micro): Per-subtitle vocal boundary & elastic duration refinement.
        Uses the 100Hz neural VAD speech probability envelope to:
          1. Pull back late subtitle starts (`new_s`, up to -650ms) when the actor's sentence
             already started earlier after a clean silence gap (tolerant of brief consonant dips).
          2. Delay early subtitle starts (`new_s`, up to +800ms) when the subtitle appears during
             dead silence before the actor actually speaks, and micro-snap starts within ±220ms
             (with a 35ms visual anticipation lead).
          3. Extend `end` (up to +500ms) if the actor is still continuously speaking past `end`
             and before the next subtitle starts.
          4. Trim `end` (uncapped, down to `voice_stop + 100ms` / `min_read_dur`) via a Forward
             Speech-Stop Scanner + Tail Silence verifier so subtitles never linger on screen
             after a sentence ends.
          5. Enforce a clean >= 70ms (~2 frames) visual gap between consecutive subtitles.
        """
        if not self.events or not ref_envelope or frame_hz <= 0:
            return 0

        env_len = len(ref_envelope)
        if env_len < frame_hz * 10:
            return 0

        try:
            from subsync.synchro import vad
        except Exception:
            vad = None

        modified_count = 0
        n_ev = len(self.events)
        orig_starts = [ev.start / 1000.0 for ev in self.events]
        orig_ends = [ev.end / 1000.0 for ev in self.events]

        for i, ev in enumerate(self.events):
            s_sec = orig_starts[i]
            e_sec = orig_ends[i]
            dur = e_sec - s_sec
            if dur <= 0.15 or dur > 12.0:
                continue

            raw_txt = getattr(ev, "plaintext", ev.text) or ""
            if vad is not None and vad.is_pure_sdh_non_speech(raw_txt):
                continue

            prev_end_s = max(self.events[i - 1].end / 1000.0, orig_ends[i - 1]) if i > 0 else 0.0
            next_start_s = orig_starts[i + 1] if (i + 1 < n_ev) else (e_sec + 10.0)

            clean_txt = re.sub(r"<[^>]+>", "", raw_txt).replace("\n", " ").strip()
            char_count = max(1, len(clean_txt))
            min_read_dur = max(0.65, min(2.20, char_count / 21.0))

            new_s = s_sec
            new_e = e_sec
            snapped_start = False

            k_s = int(round(s_sec * frame_hz))
            k_e = int(round(e_sec * frame_hz))

            # 1A. Anti-Early Start (Dead Silence Stripping up to +0.80s):
            # If the subtitle appears while nobody is speaking yet (VAD < 0.20 at s_sec),
            # delay `new_s` right onto the actual speech onset when the actor starts speaking.
            if 5 <= k_s < env_len - 25:
                initial_slice = ref_envelope[k_s : min(env_len, k_s + int(round(0.08 * frame_hz)))]
                if initial_slice and (sum(initial_slice) / len(initial_slice)) < 0.20 and max(initial_slice) < 0.35:
                    max_wait_s = min(e_sec - 0.35, s_sec + 0.80)
                    max_wait_k = min(env_len - 6, int(round(max_wait_s * frame_hz)))
                    for k in range(k_s + 2, max_wait_k + 1):
                        pre_sil = ref_envelope[k_s:k]
                        if pre_sil and (sum(pre_sil) / len(pre_sil)) < 0.20 and max(pre_sil) < 0.35:
                            post_sp = ref_envelope[k : k + 5]
                            if len(post_sp) == 5 and (sum(post_sp) / 5.0) >= 0.45:
                                cand_s = max(prev_end_s + 0.070 if i > 0 else 0.0, (k / float(frame_hz)) - 0.035)
                                if cand_s > s_sec + 0.035 and (e_sec - cand_s) >= 0.45:
                                    shift_fwd = cand_s - s_sec
                                    new_s = cand_s
                                    snapped_start = True
                                    if (new_e - new_s) < min(dur, min_read_dur):
                                        new_e = min(next_start_s - 0.075, new_e + shift_fwd)
                                break

            # 1B. Anti-Late Start (Unclaimed Continuous Speech Capture capped at -0.25s):
            # If the actor is already speaking at s_sec (or within [-30ms, +60ms]), walk backward
            # up to 0.25s to find the true vocal attack immediately preceded by >= 150ms of clean silence.
            if not snapped_start and 15 <= k_s < env_len - 10:
                at_start_slice = ref_envelope[max(0, k_s - 3) : min(env_len, k_s + 6)]
                if at_start_slice and (
                    (sum(at_start_slice) / len(at_start_slice)) >= 0.38 or max(at_start_slice) >= 0.52
                ):
                    earliest_allowed_s = max((prev_end_s + 0.16) if i > 0 else 0.0, s_sec - 0.25)
                    earliest_k = max(15, int(round(earliest_allowed_s * frame_hz)))
                    if k_s > earliest_k + 3:
                        onset_k = None
                        low_run = 0
                        for k in range(k_s - 1, earliest_k - 1, -1):
                            v = ref_envelope[k]
                            if v < 0.20:
                                low_run += 1
                                if low_run >= 8:
                                    break
                            else:
                                low_run = 0
                                # Check if [k - 15 : k] is a clean >= 150ms silence gap before frame k
                                sil_before = ref_envelope[max(0, k - 15) : k]
                                if (
                                    len(sil_before) >= 12
                                    and (sum(sil_before) / len(sil_before)) < 0.15
                                    and max(sil_before) < 0.28
                                ):
                                    onset_k = k
                        if onset_k is not None:
                            cand_s = max(prev_end_s + 0.070 if i > 0 else 0.0, (onset_k / float(frame_hz)) - 0.035)
                            if cand_s < s_sec - 0.035:
                                new_s = cand_s
                                snapped_start = True

            # 1C. Fine Start Micro-Snap: look for a clean silence -> speech transition within [-0.22s, +0.22s]
            win_rad = int(round(0.22 * frame_hz))
            if not snapped_start and k_s - win_rad >= 6 and k_s + win_rad < env_len - 6:
                best_onset_k = None
                best_jump = 0.32
                for k in range(k_s - win_rad, k_s + win_rad + 1):
                    pre_silence = sum(ref_envelope[k - 5 : k]) / 5.0
                    post_speech = sum(ref_envelope[k : k + 5]) / 5.0
                    jump = post_speech - pre_silence
                    if pre_silence < 0.22 and post_speech >= 0.48 and jump > best_jump:
                        best_jump = jump
                        best_onset_k = k
                if best_onset_k is not None:
                    cand_s = (best_onset_k / float(frame_hz)) - 0.035
                    cand_s = max(prev_end_s + 0.070 if i > 0 else 0.0, cand_s)
                    if abs(cand_s - s_sec) <= 0.22 and (e_sec - cand_s) >= 0.45:
                        new_s = cand_s

            # 2. End elastic adaptation:
            k_new_s = max(0, min(env_len - 1, int(round(new_s * frame_hz))))
            k_e = int(round(e_sec * frame_hz))
            if 10 <= k_e < env_len - 10 and k_e > k_new_s + 15:
                # Case A: Actor is still speaking right across e_sec -> extend to end of utterance
                active_at_end = (sum(ref_envelope[max(0, k_e - 3) : min(env_len, k_e + 3)]) / 6.0) >= 0.45
                if active_at_end:
                    max_ext_s = min(next_start_s - 0.075, e_sec + 0.50)
                    max_ext_k = min(env_len - 1, int(round(max_ext_s * frame_hz)))
                    if max_ext_k > k_e:
                        voice_end_k = k_e
                        for k in range(k_e, max_ext_k + 1):
                            if ref_envelope[k] >= 0.35:
                                voice_end_k = k
                            else:
                                lookahead = ref_envelope[k : min(env_len, k + 4)]
                                if sum(lookahead) / max(1, len(lookahead)) < 0.22:
                                    break
                        cand_e = min(max_ext_s, (voice_end_k / float(frame_hz)) + 0.10)
                        if cand_e > new_e + 0.03:
                            new_e = cand_e
                else:
                    # Case B: Backward Last-Speech Finder + Tail Silence Verification
                    # Scan backward from k_e - 1 down to scan_start_k to locate the end of the LAST
                    # speech burst inside [k_new_s, k_e], so multi-clause subtitles with internal
                    # pauses are never truncated at the first clause.
                    first_sp_k = None
                    for k in range(k_new_s, k_e):
                        if ref_envelope[k] >= 0.35:
                            first_sp_k = k
                            break
                    if first_sp_k is not None:
                        scan_start_k = max(first_sp_k + 8, int(round((new_s + 0.20) * frame_hz)))
                        stop_k = None
                        for k in range(k_e - 1, scan_start_k - 1, -1):
                            if ref_envelope[k] >= 0.32:
                                burst = ref_envelope[max(k_new_s, k - 3) : k + 1]
                                if burst and (sum(burst) / len(burst)) >= 0.26:
                                    stop_k = k + 1
                                    break
                        if stop_k is not None and stop_k < k_e - 8:
                            rem_tail = ref_envelope[stop_k:k_e]
                            if rem_tail and (sum(rem_tail) / len(rem_tail)) < 0.18:
                                voice_stop_s = (stop_k / float(frame_hz)) + 0.10
                                if voice_stop_s < e_sec - 0.06:
                                    cand_e = max(new_s + min_read_dur, voice_stop_s)
                                    if cand_e < new_e - 0.035:
                                        new_e = cand_e

            # 3. Minimum comfortable reading duration extension if space permits and speech was active
            if (new_e - new_s) < min_read_dur and (next_start_s - new_e) > 0.15:
                cand_e = min(next_start_s - 0.075, new_s + min_read_dur, e_sec + 0.25)
                if cand_e > new_e:
                    new_e = cand_e

            # 4. Anti-flicker 2-frame (70ms) gap guard before next subtitle
            if i + 1 < n_ev and (next_start_s - new_e) < 0.070 and (next_start_s - new_s) > 0.55:
                new_e = max(new_s + 0.40, next_start_s - 0.070)

            new_start_ms = int(round(new_s * 1000.0))
            new_end_ms = int(round(new_e * 1000.0))
            if abs(new_start_ms - ev.start) >= 15 or abs(new_end_ms - ev.end) >= 20:
                ev.start = new_start_ms
                ev.end = max(new_start_ms + 300, new_end_ms)
                modified_count += 1

        return modified_count

    def save(self, path, encoding=u'utf-8', fmt=None, fps=None, overwrite=False):
        if not overwrite and os.path.exists(path):
            logger.info('file "%s" exists, generating new path', path)
            path = genUniquePath(path)

        logger.info('saving subtitles to "%s", %s', path,
                utils.fmtstr(enc=encoding, fmt=fmt, fps=fps))

        try:
            if fmt is None:
                ext = os.path.splitext(path)[1].lower()
                fmts = [ x['type'] for x in filetypes.subtitleTypes if x['ext'] == ext ]
                if len(fmts):
                    fmt = fmts[0]
                elif ext == '.vtt':
                    fmt = 'vtt'
            if fmt is None:
                raise Exception(_('Unknown file extension'))

            with open(path, 'w', encoding=encoding, errors='replace') as fp:
                super().to_file(fp, format_=fmt, fps=fps)

        except Exception as e:
            raise error.Error(_('Can\'t save subtitle file') + '. ' + str(e)) \
                    .add('path', path) \
                    .add('encoding', encoding) \
                    .addn('format', fmt) \
                    .addn('fps', fps)

        return path

    def getMaxChange(self, formula):
        if len(self.events) > 0:
            if hasattr(formula, 'getY'):
                get_y = formula.getY
            elif hasattr(formula, 'a') and hasattr(formula, 'b'):
                get_y = lambda x: formula.a * x + formula.b
            else:
                get_y = lambda x: formula[0] * x + formula[1]
            return max(abs(get_y(x) - x) for x in
                    (self.events[0].start/1000.0, self.events[-1].end/1000.0))
        else:
            return 0.0


def genUniquePath(path):
    prefix, ext = os.path.splitext(path)
    if not prefix.endswith('.'):
        prefix += '.'

    i = 0
    while os.path.exists(path):
        i += 1
        path = '{}{}{}'.format(prefix, i, ext)

    return path


def isFpsBased(path):
    _, ext = os.path.splitext(path)
    fmt = pysubs2.formats.FILE_EXTENSION_TO_FORMAT_IDENTIFIER.get(ext.lower())
    return fmt == 'microdvd'


class SubtitlesCollector(object):
    def __init__(self):
        self.subtitles = Subtitles()
        self.subtitlesLock = threading.Lock()

    def __len__(self):
        with self.subtitlesLock:
            return len(self.subtitles)

    def addSubtitle(self, begin, end, text):
        with self.subtitlesLock:
            self.subtitles.add(begin, end, text)

    def setSubtitles(self, subs):
        with self.subtitlesLock:
            if isinstance(subs, Subtitles):
                self.subtitles = subs
            else:
                s = Subtitles()
                s.events = list(subs.events)
                s.styles = dict(getattr(subs, 'styles', {}))
                s.info = dict(getattr(subs, 'info', {}))
                s.aegisub_project = dict(getattr(subs, 'aegisub_project', {}))
                s.fps = getattr(subs, 'fps', None)
                s.format = getattr(subs, 'format', None)
                self.subtitles = s
            self.subtitles.cleanWatermarks()

    def getMaxSubtitleDiff(self, formula):
        with self.subtitlesLock:
            return self.subtitles.getMaxChange(formula)

    def getSubtitles(self):
        return self.subtitles

    def getSynchronizedSubtitles(self, formula):
        logger.info('subtitles synchronized with %s', str(formula))
        with self.subtitlesLock:
            return self.subtitles.synchronize(formula)


def parseLine(text):
    fields = text.split(',', 8)
    if len(fields) == 9:
        entry = {
            'style':   fields[2],
            'name':    fields[3],
            'marginl': fields[4],
            'marginr': fields[5],
            'marginv': fields[6],
            'effect':  fields[7],
            'text':    fields[8] }
    else:
        entry = {
            'style': 'Default',
            'text':  text }
    return entry
