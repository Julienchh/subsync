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
    r"(?:addic7ed\.com|synchro\s*par|syncro\s*par|re-?sync\s*par|"
    r"sync\s*by|traduit\s*par\s*la\s*communaut|subsynchro|sous-titres\.eu|"
    r"tvsubtitles\.net|opensubtitles\.org)",
    re.IGNORECASE,
)


def is_credit_watermark(text: str) -> bool:
    """Return True if subtitle text is a non-dialogue credit/watermark line."""
    if not text:
        return False
    return bool(_WATERMARK_RE.search(text))


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

        env_len = len(ref_envelope)
        if env_len < frame_hz * 30:
            return 0, 0, 0

        try:
            import numpy as np
            from subsync.synchro import aligner
        except Exception:
            return 0, 0, 0

        ref_arr = np.asarray(ref_envelope, dtype=np.float32)
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

        for w_idx, ev_idxs in enumerate(raw_windows):
            w_ivs = [
                (self.events[idx].start / 1000.0, self.events[idx].end / 1000.0)
                for idx in ev_idxs
                if self.events[idx].end > self.events[idx].start
            ]
            w_dur = (w_ivs[-1][1] - w_ivs[0][0]) if w_ivs else 0.0
            if len(w_ivs) < 5 or w_dur < 12.0:
                verified_windows += 1
                continue

            curve = aligner._correlate_window_curve(
                w_ivs,
                ref_arr,
                frame_hz=frame_hz,
                search_center_b=0.0,
                search_radius_s=search_rad_s,
            )
            if curve is None or len(curve) != 2 * rad_frames + 1:
                verified_windows += 1
                continue

            tol_f = max(1, int(round(0.10 * frame_hz)))
            c0 = max(0, rad_frames - tol_f)
            c1 = min(len(curve), rad_frames + tol_f + 1)
            zero_ncc = float(np.max(curve[c0:c1]))

            prev_end_s = (
                (self.events[raw_windows[w_idx - 1][-1]].end / 1000.0) + 0.08
                if w_idx > 0
                else 0.0
            )
            next_start_s = (
                (self.events[raw_windows[w_idx + 1][0]].start / 1000.0) - 0.08
                if (w_idx + 1 < total_windows)
                else (w_ivs[-1][1] + search_rad_s + 10.0)
            )
            min_delta = max(-search_rad_s, prev_end_s - w_ivs[0][0])
            max_delta = min(search_rad_s, next_start_s - w_ivs[-1][1])
            if max_delta <= min_delta + 0.05:
                verified_windows += 1
                continue

            k_lo = max(0, min(len(curve) - 1, int(round((min_delta + search_rad_s) * frame_hz))))
            k_hi = max(k_lo, min(len(curve) - 1, int(round((max_delta + search_rad_s) * frame_hz))))
            sub_curve = curve[k_lo : k_hi + 1]
            if len(sub_curve) == 0:
                verified_windows += 1
                continue

            best_rel_k = int(np.argmax(sub_curve))
            best_k = k_lo + best_rel_k
            best_ncc = float(curve[best_k])
            best_delta = (float(best_k - rad_frames)) / float(frame_hz)

            if (
                abs(best_delta) >= 0.16
                and best_ncc >= 0.46
                and (best_ncc - zero_ncc) >= 0.065
            ):
                refined_delta = aligner._refine_act_vocal_attack(
                    w_ivs, 1.0, best_delta, ref_arr, frame_hz
                )
                refined_delta = max(min_delta, min(max_delta, refined_delta))
                vh_before = _win_vocal_hit_rate(w_ivs, 0.0)
                vh_after = _win_vocal_hit_rate(w_ivs, refined_delta)
                if vh_after >= vh_before - 0.02 and (vh_after >= 0.80 or vh_after > vh_before):
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

        modified_count = 0
        n_ev = len(self.events)

        for i, ev in enumerate(self.events):
            s_sec = ev.start / 1000.0
            e_sec = ev.end / 1000.0
            dur = e_sec - s_sec
            if dur <= 0.15 or dur > 12.0:
                continue

            prev_end_s = (self.events[i - 1].end / 1000.0) if i > 0 else 0.0
            next_start_s = (self.events[i + 1].start / 1000.0) if (i + 1 < n_ev) else (e_sec + 10.0)

            raw_txt = getattr(ev, "plaintext", ev.text) or ""
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

            # 1B. Anti-Late Start (Unclaimed Continuous Speech Capture up to -0.65s):
            # If the actor is already speaking at s_sec (or within [-30ms, +60ms]), walk backward
            # up to 0.65s (tolerant of brief <80ms consonant dips) to find the true vocal attack
            # immediately preceded by >= 120ms of clean silence after prev_end_s.
            if not snapped_start and 15 <= k_s < env_len - 10:
                at_start_slice = ref_envelope[max(0, k_s - 3) : min(env_len, k_s + 6)]
                if at_start_slice and (
                    (sum(at_start_slice) / len(at_start_slice)) >= 0.38 or max(at_start_slice) >= 0.52
                ):
                    earliest_allowed_s = max((prev_end_s + 0.14) if i > 0 else 0.0, s_sec - 0.65)
                    earliest_k = max(12, int(round(earliest_allowed_s * frame_hz)))
                    if k_s > earliest_k + 3:
                        onset_k = None
                        low_run = 0
                        for k in range(k_s - 1, earliest_k - 1, -1):
                            v = ref_envelope[k]
                            if v < 0.20:
                                low_run += 1
                                # Only stop backward search if we hit >= 80ms of continuous silence
                                if low_run >= 8:
                                    break
                            else:
                                low_run = 0
                                # Check if [k - 12 : k] is a clean >= 120ms silence gap before frame k
                                sil_before = ref_envelope[max(0, k - 12) : k]
                                if (
                                    len(sil_before) >= 10
                                    and (sum(sil_before) / len(sil_before)) < 0.18
                                    and max(sil_before) < 0.32
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
                    # Case B: Forward Speech-Stop Scanner + Tail Silence Check
                    # Find where speech starts inside [k_new_s, k_e] and where the final utterance ends,
                    # trimming any dead-silence overhang right down to max(new_s + min_read_dur, t_stop + 0.10s).
                    first_sp_k = None
                    for k in range(k_new_s, k_e):
                        if ref_envelope[k] >= 0.35:
                            first_sp_k = k
                            break
                    if first_sp_k is not None:
                        scan_start_k = max(first_sp_k + 8, int(round((new_s + 0.20) * frame_hz)))
                        stop_k = None
                        for k in range(scan_start_k, k_e - 12):
                            if ref_envelope[k - 1] >= 0.25 and ref_envelope[k] < 0.25:
                                sil_win = ref_envelope[k : min(k_e, k + 22)]
                                if sil_win and (sum(sil_win) / len(sil_win)) < 0.18:
                                    rem_tail = ref_envelope[k:k_e]
                                    rem_mean = sum(rem_tail) / len(rem_tail)
                                    silent_frac = sum(1 for x in rem_tail if x < 0.25) / float(len(rem_tail))
                                    if rem_mean < 0.16 and silent_frac >= 0.85:
                                        stop_k = k
                                        break
                        if stop_k is None:
                            # Fallback: check last frame with speech >= 0.32 before k_e
                            for k in range(k_e - 1, scan_start_k - 1, -1):
                                if ref_envelope[k] >= 0.32:
                                    stop_k = k + 1
                                    break
                        if stop_k is not None:
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
