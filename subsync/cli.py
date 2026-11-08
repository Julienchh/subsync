from subsync.synchro import SyncController
from subsync.assets import assetManager
from subsync.settings import settings
from subsync import validator, utils, error
import time
import threading


class Printer(object):
    def __init__(self, verbosity=1):
        self.lock = threading.Lock()
        self.verbosity = verbosity
        self.lineLen = 0

    def println(self, v, *args, **kwargs):
        if self.verbosity >= v:
            with self.lock:
                if self.lineLen:
                    print('', **kwargs)
                    self.lineLen = 0
                print(*args, **kwargs)

    def reprint(self, v, msg, **kwargs):
        if self.verbosity >= v:
            with self.lock:
                padding = ' ' * (len(msg) - self.lineLen)
                print('{}{}\r'.format(msg, padding), end='', flush=True, **kwargs)
                self.lineLen = len(msg)

    def printException(self, v, exc, msg=None):
        res = error.getExceptionMessage(exc)
        if msg:
            res = '{}: {}'.format(msg, res)
        self.println(v, '[!] ' + res.replace('\n', '\n[-] '))
        if isinstance(exc, error.Error):
            for key, val in exc.fields.items():
                self.println(v+1, '[-] {}: {}'.format(key, val))


pr = Printer()


def _is_native_gizmo_active():
    try:
        import gizmo
        return not getattr(gizmo, '_IS_MODERN_FALLBACK', False)
    except Exception:
        return False


class AssetsVerifier(object):
    def run(self, tasks):
        if not _is_native_gizmo_active():
            return True
        assets = assetManager().getAssetsForTasks(tasks)

        if assets.notInstalled():
            self.printMissingAssets(assets.notInstalled())
            return False
        else:
            return True

    def printMissingAssets(self, assets):
        pr.println(0, '[!] there is no assets needed to perform synchronization')
        for asset in assets:
            pr.println(1, '[-] asset {} is missing'.format(asset.getPrettyName()))


class AssetsDownloader(AssetsVerifier):
    def run(self, tasks):
        if not _is_native_gizmo_active():
            return True
        self.updateAssetList()
        assets = assetManager().getAssetsForTasks(tasks)

        if assets.missing():
            self.printMissingAssets(assets.missing())
            return False

        return self.installAssets(assets.hasUpdate())

    def updateAssetList(self):
        listUpdater = assetManager().getAssetListUpdater()
        if not listUpdater.isRunning() and not listUpdater.isUpdated():
            pr.println(2, '[+] updating assets list')
            listUpdater.run()
            listUpdater.wait()
        return listUpdater.isUpdated()

    def installAssets(self, assets):
        for asset in assets:
            downloader = asset.downloader()
            try:
                downloader.registerCallbacks(self)
                downloader.run(timeout=0.5)
                downloader.wait(reraise=True)

            except KeyboardInterrupt:
                downloader.terminate()
                raise

            except error.Error as err:
                pr.printException(0, err)
                return False

            finally:
                downloader and downloader.unregisterCallbacks(self)

        return True

    def onUpdate(self, asset, pos=None, size=None, start=False):
        msg = [ '[+] downloading ', asset.getPrettyName() ]
        if pos is not None and size:
            msg += [ ' {:3.0f}%'.format(100 * pos / size) ]
        if pos is not None:
            msg += [ ': ', utils.fileSizeFmt(pos) ]
        if size:
            msg += [ ' / ', utils.fileSizeFmt(size) ]

        if start or pr.verbosity >= 3:
            pr.println(1, ''.join(msg))
        else:
            pr.reprint(1, ''.join(msg))


def runBatchCommand(batch_args, verbosity=1):
    from subsync.batch import BatchConfig, BatchProgressReporter, run_batch, print_summary_table
    pr.verbosity = verbosity
    cfg = BatchConfig(
        ref_dir=batch_args.get('ref_dir'),
        sub_dir=batch_args.get('sub_dir'),
        ref_pattern=batch_args.get('ref_pattern'),
        sub_pattern=batch_args.get('sub_pattern'),
        out_dir=batch_args.get('out_dir'),
        out_pattern=batch_args.get('out_pattern') or '{ref_dir}/{ref_name}.synced.srt',
        in_place=bool(batch_args.get('in_place', False)),
        backup=bool(batch_args.get('backup', False)),
        skip_if_exists=bool(batch_args.get('skip_if_exists', False)),
        recursive=bool(batch_args.get('recursive', False)),
        matcher_mode=batch_args.get('matcher_mode') or 'auto',
        lang=batch_args.get('lang'),
        ref_lang=batch_args.get('ref_lang'),
        task_timeout=float(batch_args.get('task_timeout', 300.0)),
        continue_on_error=bool(batch_args.get('continue_on_error', True)),
        dry_run=bool(batch_args.get('dry_run', False)),
        jobs=int(batch_args.get('jobs', 1)),
        summary_json=batch_args.get('summary_json'),
        verbose=verbosity,
    )
    reporter = BatchProgressReporter(verbose=verbosity)
    summary = run_batch(cfg, progress_callback=reporter)
    if verbosity >= 1:
        print_summary_table(summary)
    return summary.exit_code


def runSeriesCommand(series_args, verbosity=1):
    from subsync.download import run_series_sync
    pr.verbosity = verbosity
    series_dir = series_args.get('series_dir')
    if not series_dir:
        pr.println(0, '[!] missing required series directory (usage: subsync series <SERIES_DIR> --lang fr)')
        return 1
    summary = run_series_sync(
        series_dir=series_dir,
        target_lang=series_args.get('target_lang') or 'fr',
        ref_lang=series_args.get('ref_lang'),
        provider=series_args.get('provider') or 'auto',
        opensubtitles_api_key=series_args.get('opensubtitles_api_key'),
        out_pattern=series_args.get('out_pattern') or '{ref_dir}/{ref_name}.{lang}.synced.srt',
        cache_dir=series_args.get('cache_dir'),
        max_episodes=series_args.get('max_episodes'),
        episodes_filter=series_args.get('episodes_filter'),
        season_filter=series_args.get('season_filter'),
        recursive=bool(series_args.get('recursive', False)),
        dry_run=bool(series_args.get('dry_run', False)),
        skip_if_exists=bool(series_args.get('skip_if_exists', False)),
        task_timeout=float(series_args.get('task_timeout', 300.0)),
        workers=int(series_args.get('workers', 0)),
        summary_json=series_args.get('summary_json'),
        quiet=(verbosity <= 0),
    )
    return summary.exit_code


class App(object):
    def __init__(self, verbosity=1, offline=False):
        pr.verbosity = verbosity
        if offline:
            self.assetsDownloader = AssetsVerifier()
        else:
            self.assetsDownloader = AssetsDownloader()

    def runTasks(self, tasks):
        if not tasks:
            pr.println(1, '[-] nothing to do')
            return 0

        self.startTime = time.monotonic()
        self.lastTime = -1000
        self.succeeded = 0
        sync = None

        valid_tasks = []
        validation_failures = 0
        for task in tasks:
            try:
                validator.validateTask(task, outputRequired=True)
                valid_tasks.append(task)
            except Exception as err:
                pr.printException(0, err)
                validation_failures += 1

        if not valid_tasks:
            return 1

        try:
            if not self.assetsDownloader.run(valid_tasks):
                return 2

            sync = SyncController(listener=self)
            sync.synchronize(valid_tasks, timeout=1)
            sync.wait()

            if self.succeeded == len(tasks) and validation_failures == 0:
                return 0
            elif self.succeeded > 0:
                return 2
            else:
                return 1 if validation_failures == len(tasks) else 2

        except KeyboardInterrupt:
            pr.println(1, '[-] interrupted by user')
            sync and sync.terminate()
            return 1

        except error.Error as err:
            pr.printException(0, err)
            raise

    def onJobStart(self, task):
        pr.println(1, '[*] starting synchronization {}'.format(task.sub.path))
        pr.println(2, '[+] sub: {}'.format(task.sub))
        pr.println(2, '[+] ref: {}'.format(task.ref))
        pr.println(2, '[+] out: {}'.format(task.out))

    def onJobUpdate(self, task, status):
        self.printStats(status)

    def onJobEnd(self, task, status, result):
        if status and not result.terminated:
            self.printStats(status, finished=True)
        if result.success:
            pr.println(1, '[+] done, saved to {}'.format(result.path))
            self.succeeded += 1
        elif not result.terminated:
            pr.println(0, '[-] couldn\'t synchronize!')
        pr.println(1, '')

    def printStats(self, status, finished=False):
        if pr.verbosity >= 1:
            if finished:
                progress = 1.0
            else:
                progress = status.progress
                effort = settings().minEffort
                if effort:
                    progress = min(max(progress, status.effort / effort, 0), 1)

            msg = '[+] {}: progress {:3.0f}%, {} points'.format(
                    utils.timeStampFmt(time.monotonic() - self.startTime),
                    100 * progress,
                    status.points)
            if pr.verbosity >= 2:
                msg += ', correlation={:.2f}, formula={}, maxChange={}'.format(
                        100 * status.factor,
                        str(status.formula),
                        utils.timeStampFractionFmt(status.maxChange))

            if pr.verbosity >= 3:
                pr.println(1, msg)
            else:
                pr.reprint(1, msg)

    def onError(self, task, source, err):
        pr.printException(1, err, source)
