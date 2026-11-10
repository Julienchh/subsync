import threading
import logging

from subsync import subtitle
from subsync.synchro import aligner, controller, encdetect, modern_engine, vad

gizmo = modern_engine.install_gizmo_shim()

logger = logging.getLogger(__name__)


class Synchronizer(object):
    """Subtitle-to-reference synchronizer powered by the modern VAD + FFT/OLS alignment engine."""

    def __init__(self, sub, ref):
        self.sub = sub
        self.ref = ref

        self.onUpdate = lambda: None
        self.onError = lambda src, err: None

        self.stats = gizmo.CorrelationStats()
        self.effortBegin = None
        self.statsLock = threading.Lock()

        self.subtitlesCollector = subtitle.SubtitlesCollector()
        self.options = {}
        self._runCb = None
        self._worker = None
        self._running = False
        self._progress = 0.0
        self._stop_requested = False

        # Legacy attributes kept for API compatibility
        self.correlator = None
        self.translator = None
        self.subPipeline = None
        self.refPipelines = []
        self.pipelines = []
        self.wordsDumpers = []

    def destroy(self):
        logger.info('releasing synchronizer resources')
        self.stop(force=True)
        self.subPipeline = None
        self.refPipelines = []
        self.pipelines = []
        self.translator = None

    def init(self, options=None, runCb=None):
        logger.info('initializing synchronization jobs')
        self.options = dict(options or {})
        self._runCb = runCb
        self._stop_requested = False
        self._progress = 0.0

        for stream in (self.sub, self.ref):
            if stream and stream.type == 'subtitle/text' and not stream.enc and len(stream.streams) <= 1:
                try:
                    stream.enc = encdetect.detectEncoding(stream.path, stream.lang)
                except Exception as err:
                    logger.debug('encoding detection fallback for %s: %r', stream.path, err)
                    stream.enc = 'UTF-8'

        try:
            loaded_subs = modern_engine.load_subtitles_from_source(
                self.sub.path,
                stream_index=self.sub.no,
                encoding=self.sub.enc,
                lang=self.sub.lang,
                fps=self.sub.fps,
            )
            self.subtitlesCollector.setSubtitles(loaded_subs)
        except Exception as err:
            self.onSubError(err)
            raise

    def start(self):
        logger.info('starting synchronization')
        self._running = True
        self._worker = threading.Thread(
            target=self._runAlignmentWorker,
            name='ModernSynchronizer',
            daemon=True,
        )
        self._worker.start()

    def _runAlignmentWorker(self):
        try:
            if self._stop_requested or (self._runCb and not self._runCb()):
                return

            with self.statsLock:
                self._progress = 0.2

            sub_obj = self.subtitlesCollector.getSubtitles()
            sub_intervals = vad.subtitles_to_intervals(sub_obj)
            if not sub_intervals:
                raise RuntimeError(f"No subtitle events found in {self.sub.path!r}")

            if self._stop_requested or (self._runCb and not self._runCb()):
                return

            try:
                ref_envelope, ref_intervals = vad.extract_speech_envelope(
                    self.ref.path,
                    stream_index=self.ref.no,
                    stream_type=self.ref.type,
                    encoding=self.ref.enc or 'utf-8',
                    fps=self.ref.fps,
                    frame_hz=vad.FRAME_HZ,
                )
            except Exception as err:
                self.onRefError(err)
                return

            if self._stop_requested or (self._runCb and not self._runCb()):
                return

            with self.statsLock:
                self._progress = 0.7

            window_size = float(self.options.get('windowSize', 1800.0) or 1800.0)
            allow_drift = bool(self.options.get('allowDrift', True))

            align_res = aligner.align_intervals_and_envelopes(
                sub_intervals=sub_intervals,
                ref_envelope=ref_envelope,
                ref_intervals=ref_intervals,
                frame_hz=vad.FRAME_HZ,
                max_offset_s=window_size,
                allow_drift=allow_drift,
            )

            new_stats = gizmo.CorrelationStats(
                correlated=align_res.correlated,
                factor=align_res.factor,
                points=align_res.points,
                maxDistance=align_res.max_distance,
                formula=align_res.formula,
            )

            with self.statsLock:
                self._ref_envelope = ref_envelope
                self._ref_intervals = ref_intervals
                self.stats = new_stats
                self.effortBegin = 0.0
                self._progress = 1.0

        except Exception as err:
            logger.warning('alignment worker error: %r', err, exc_info=True)
            self.onError('core', err)
        finally:
            with self.statsLock:
                self._running = False
                self._progress = 1.0
            try:
                self.onUpdate()
            except Exception:
                pass

    def stop(self, force=True):
        logger.info('stopping synchronizer')
        self._stop_requested = True
        worker = self._worker
        if worker and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=5.0)
        with self.statsLock:
            self._running = False

    def isRunning(self):
        with self.statsLock:
            return self._running

    def getProgress(self):
        with self.statsLock:
            return max(0.0, min(1.0, self._progress))

    def getStatus(self):
        with self.statsLock:
            stats = self.stats
            progress = max(0.0, min(1.0, self._progress))
            begin = self.effortBegin
            running = self._running

        effort = -1.0
        if begin is not None:
            if progress >= 1.0 or not running:
                effort = 1.0
            elif progress <= begin:
                effort = 0.0
            else:
                effort = (progress - begin) / max(1e-6, 1.0 - begin)

        return controller.SyncStatus(
            correlated=bool(stats.correlated and not running),
            maxChange=self.subtitlesCollector.getMaxSubtitleDiff(stats.formula),
            progress=progress,
            factor=stats.factor,
            points=stats.points,
            formula=stats.formula,
            effort=effort,
            maxDistance=getattr(stats, 'maxDistance', 0.0),
        )

    def getSynchronizedSubtitles(self):
        with self.statsLock:
            formula = self.stats.formula
            ref_env = getattr(self, '_ref_envelope', None)
            ref_ivs = getattr(self, '_ref_intervals', None)
        synced = self.subtitlesCollector.getSynchronizedSubtitles(formula)
        if ref_env and len(ref_env) >= int(300.0 * vad.FRAME_HZ):
            try:
                synced.verify_and_heal_with_vad(
                    ref_env,
                    ref_intervals=ref_ivs,
                    frame_hz=vad.FRAME_HZ,
                )
            except Exception as err:
                logger.warning('verification & self-healing failed: %r', err, exc_info=True)
            try:
                synced.refine_durations_with_vad(
                    ref_env,
                    ref_intervals=ref_ivs,
                    frame_hz=vad.FRAME_HZ,
                )
            except Exception as err:
                logger.debug('duration refinement skipped: %r', err)
        return synced

    def onStatsUpdate(self, stats):
        with self.statsLock:
            self.stats = stats
        self.onUpdate()

    def onSubError(self, err):
        logger.warning('SUB: %r', str(err).replace('\n', '; '))
        self.onError('sub', err)

    def onRefError(self, err):
        logger.warning('REF: %r', str(err).replace('\n', '; '))
        self.onError('ref', err)
