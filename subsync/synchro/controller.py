from collections import namedtuple
from typing import Iterable
import time
import threading
from subsync.settings import settings
from subsync.error import Error

import logging
logger = logging.getLogger(__name__)


__pdoc__ = {
        'SyncJobResult.success': 'Whether synchronization succeeded.',
        'SyncJobResult.terminated': 'Whether synchronization was terminated with `SyncController.terminate`.',
        'SyncJobResult.path': 'Where output subtitles was saved. `None` if subtites was not saved automatically.',

        'SyncStatus.correlated': 'Whether subtitles are correlated and ready to be saved.',
        'SyncStatus.maxChange': 'Maximum time delta that was applied for a single subtitle line, in seconds.',
        'SyncStatus.progress': 'Progress estimation, between 0 and 1.',
        'SyncStatus.factor': 'Correlation factor, between 0 and 1.',
        'SyncStatus.points': 'Number of synchronization points.',
        'SyncStatus.formula': 'Mathematical formula used for synchronization.',
        'SyncStatus.effort': 'Current effort spent for synchronization, between 0 and 1 or -1 if not yet correlated.',
        }


SyncJobResult = namedtuple('SyncJobResult', [
    'success',
    'terminated',
    'path',
    ])
SyncJobResult.__doc__ = """Result of single synchronization task."""


_SyncStatusTuple = namedtuple('SyncStatus', [
    'correlated',
    'maxChange',
    'progress',
    'factor',
    'points',
    'formula',
    'effort',
])


class SyncStatus(_SyncStatusTuple):
    """Synchronization status."""

    def __new__(cls, correlated=False, maxChange=0.0, progress=0.0, factor=0.0,
                points=0, formula=None, effort=-1.0, maxDistance=0.0):
        if formula is None:
            from subsync.synchro.aligner import FormulaLine
            formula = FormulaLine(1.0, 0.0)
        obj = super().__new__(
            cls,
            bool(correlated),
            float(maxChange),
            float(progress),
            float(factor),
            int(points),
            formula,
            float(effort),
        )
        obj.maxDistance = float(maxDistance)
        return obj

    @property
    def offset_s(self):
        return float(getattr(self.formula, 'b', 0.0))

    @property
    def speed_factor(self):
        return float(getattr(self.formula, 'a', 1.0))


class SyncController(object):
    """Synchronization controller.

    Controls subtitle synchronization process. Synchronization is performed
    asynchronously in separate threads controlled internally. Status is reported
    by optional callbacks.
    """

    def __init__(self, listener=None, **kw):
        self._onJobStart = kw.get('onJobStart', getattr(listener, 'onJobStart', lambda task: None))
        self._onJobInit = kw.get('onJobInit', getattr(listener, 'onJobInit', lambda task: None))
        self._onJobEnd = kw.get('onJobEnd', getattr(listener, 'onJobEnd', lambda task, status, result: None))
        self._onJobUpdate = kw.get('onJobUpdate', getattr(listener, 'onJobUpdate', lambda task, status: None))
        self._onFinish = kw.get('onFinish', getattr(listener, 'onFinish', lambda terminated: None))
        self._onError = kw.get('onError', getattr(listener, 'onError', lambda task, source, error: None))

        self._options = settings().getSynchronizationOptions()
        self._thread = None
        self._semaphore = threading.Semaphore()
        self._sync = None
        self._terminated = False

    def configure(self, **kw):
        """Override default synchronization options."""
        for key, val in kw.items():
            if key not in self._options and key not in ('allowDrift',):
                raise TypeError("Unexpected keyword argument '{}'".format(key))
            self._options[key] = val

    def synchronize(self, tasks, *, timeout=None, interactive=False):
        """Start task[s] synchronization."""
        if self.isRunning():
            raise RuntimeError('Another synchronization in progress')

        logger.debug('synchronization options: %s', self._options)

        self._terminated = False
        if isinstance(tasks, Iterable) and not isinstance(tasks, (str, bytes, dict)):
            tasks = list(tasks)
            for task in tasks:
                self.validateTask(task, interactive=interactive)

            self._thread = threading.Thread(
                    target=self._run,
                    args=(tasks, timeout, interactive),
                    name='Synchronizer')
        else:
            self.validateTask(tasks, interactive=interactive)
            self._thread = threading.Thread(
                    target=self._runTask,
                    args=(tasks, timeout, interactive),
                    name='Synchronizer')
        self._thread.start()

    def terminate(self):
        """Terminate running synchronization."""
        if self.isRunning():
            self._terminated = True
            self._semaphore.release()

    def isRunning(self):
        """Check if synchronization is running."""
        return bool(self._thread and self._thread.is_alive())

    def wait(self):
        """Block until synchronization ends."""
        if self._thread:
            self._thread.join()
        return not self._terminated

    def getStatus(self):
        """Return current synchronization status."""
        return self._sync and self._sync.getStatus()

    def getProgress(self):
        """Return synchronization progress."""
        return self._sync and self._sync.getProgress()

    def getSynchronizedSubtitles(self):
        if self._sync is None:
            raise RuntimeError('Subtitles not synchronized')
        return self._sync.getSynchronizedSubtitles()

    def saveSynchronizedSubtitles(self, path=None, task=None):
        """Save synchronized subtitles."""
        if not path and not task:
            raise RuntimeError('At least one of the following arguments must be set: path or task')

        subs = self.getSynchronizedSubtitles()
        offset = self._options.get('outTimeOffset')
        if offset:
            logger.info('adjusting timestamps by offset %.3f', offset)
            subs.shift(s=offset)

        enc = (task and task.out and task.out.enc) \
                or self._options.get('outputCharEnc') \
                or (task and task.sub and task.sub.enc) or 'UTF-8'

        return subs.save(path=path or task.getOutputPath(),
                encoding=enc,
                fps=task and task.out and task.out.fps,
                overwrite=self._options.get('overwrite'))

    def validateTask(self, task, *, interactive=False):
        """Check if task is properly defined."""
        sub, ref, out = task.sub, task.ref, task.out
        if sub is None or not sub.path or sub.no is None:
            raise Error('subtitles not set', task=task)
        if ref is None or not ref.path or ref.no is None:
            raise Error('reference file not set', task=task)
        if not interactive and (out is None or not out.path):
            raise Error('output path not set', task=task)
        if not interactive:
            out.validateOutputPattern()

    def _run(self, tasks, timeout, interactive):
        try:
            for no, task in enumerate(tasks):
                if not self._terminated:
                    logger.info('running task %i/%i: %r', no, len(tasks), task)
                    self._runTask(task, timeout, interactive)
                else:
                    break

        except Exception as err:
            logger.warning('%r', err, exc_info=True)
            self._onError(None, 'core', err)

        finally:
            logger.info('synchronization finished')
            self._onFinish(self._terminated)

    def _runTask(self, task, timeout, interactive):
        sync = None
        try:
            from .synchronizer import Synchronizer
            self._onJobStart(task)
            self._sync = sync = Synchronizer(task.sub, task.ref)
            sync.onUpdate = self._semaphore.release
            sync.onError = lambda src, err: self._onError(task, src, err)

            sync.init(self._options, runCb=lambda: not self._terminated)
            if not self._terminated:
                sync.start()

            self._onJobInit(task)

            status = sync.getStatus()
            minEffort = self._options.get('minEffort', 1.0)
            if timeout is not None:
                lastTime = time.monotonic() - timeout

            while not self._terminated and sync.isRunning() \
                    and (interactive or minEffort >= 1.0 or status.effort < minEffort):
                self._semaphore.acquire(timeout=timeout)
                status = sync.getStatus()

                if timeout is not None:
                    now = time.monotonic()
                    if now - lastTime >= timeout:
                        lastTime = now
                        self._onJobUpdate(task, status)

        except Exception as err:
            logger.warning('%r', err, exc_info=True)
            self._onError(task, 'core', err)

        if sync is None:
            return

        try:
            sync.stop(force=True)
            status = sync.getStatus()
            if timeout is not None and status is not None:
                self._onJobUpdate(task, status)
            logger.info('result: %r', status)
            succeeded = bool(not self._terminated and status and status.correlated)
            path = None

            if not interactive and succeeded and task.out:
                try:
                    path = self.saveSynchronizedSubtitles(task=task)

                except Exception as err:
                    logger.warning('subtitle save failed: %r', err, exc_info=True)
                    self._onError(task, 'core', err)
                    succeeded = False

            res = SyncJobResult(succeeded, self._terminated, path)
            self._onJobEnd(task, status, res)

        except Exception as err:
            logger.warning('%r', err, exc_info=True)
            self._onError(task, 'core', err)

        finally:
            sync.destroy()
            logger.info('task finished %r', task)
