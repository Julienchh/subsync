"""Subtitle Speech Synchronizer https://subsync.online

This is an automatic movie subtitle synchronization library.
"""

import os
import shutil
import sys
import types

# 1. Ensure subsync/config.py exists on clean git checkouts and auto-wire .ai_env
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_pkg_dir)
for _ai_env_cand in (os.path.join(_repo_root, '.ai_env'), '/home/juliench/subsync/.ai_env'):
    if os.path.isdir(_ai_env_cand) and _ai_env_cand not in sys.path:
        sys.path.insert(0, _ai_env_cand)

_cfg_path = os.path.join(_pkg_dir, 'config.py')
_cfg_tmpl = os.path.join(_pkg_dir, 'config.py.template')
if not os.path.isfile(_cfg_path) and os.path.isfile(_cfg_tmpl):
    try:
        shutil.copyfile(_cfg_tmpl, _cfg_path)
    except OSError:
        pass

# 2. Ensure pysubs2 is available (falls back to pure-Python compat_pysubs2)
from . import compat_pysubs2
compat_pysubs2.register_as_pysubs2()

# 3. Bootstrap gizmo in sys.modules before importing submodules to prevent circular imports
if 'gizmo' not in sys.modules:
    try:
        import gizmo
    except ImportError:
        _shim = types.ModuleType('gizmo')
        _shim._IS_PYTHON_SHIM = True
        _shim.Error = type('Error', (Exception,), {})
        _shim.ErrorTerminated = type('ErrorTerminated', (_shim.Error,), {})
        _shim.setDebugLevel = lambda level: None
        sys.modules['gizmo'] = _shim
        import gizmo

from .synchro import modern_engine
gizmo = modern_engine.install_gizmo_shim()

from .synchro import SyncController, SyncTask, InputFile, SubFile, RefFile, OutputFile
from .synchro.modern_engine import synchronize_pair, SyncResult
from .assets import assetManager, AssetList

__all__ = [
        'synchronize', 'synchronize_pair', 'SyncResult',
        'version', 'subsync',
        'SyncController', 'SyncTask',
        'InputFile', 'SubFile', 'RefFile', 'OutputFile',
        'assetManager', 'AssetList',
        ]

__pdoc__ = { key: False for key in [
        'cli', 'cmdargs', 'config', 'data', 'error', 'gui', 'img', 'loggercfg',
        'pubkey', 'settings', 'subtitle', 'thread', 'translations', 'utils',
        'validator',
        ]}


def synchronize(sub, ref, out, *, onError=None, options={}, offline=False, updateAssets=True):
    """Synchronize single subtitle file.

    This is simplified high level synchronization API. For finer-grained
    control use `SyncController` object.
    """
    task = SyncTask(sub, ref, out)
    assets = assetManager().getAssetsForTasks([ task ])

    if not offline and (assets.missing() or updateAssets):
        listUpdater = assetManager().getAssetListUpdater()
        if not listUpdater.isRunning() and not listUpdater.isUpdated():
            listUpdater.run()
            listUpdater.wait()

    assets.validate()

    if not offline:
        if updateAssets:
            downloadAssets = assets.hasUpdate()
        else:
            downloadAssets = assets.notInstalled()

        for asset in downloadAssets:
            downloader = asset.downloader()
            try:
                downloader.run()
                downloader.wait(reraise=True)
            except:
                downloader.terminate()
                raise

    assets.validate(localOnly=True)

    res = (None, None)
    def onJobEnd(task, status, result):
        nonlocal res
        res = (result, status)

    sync = SyncController(onJobEnd=onJobEnd, onError=onError)

    try:
        sync.configure(**options)
        sync.synchronize(task)
        sync.wait()
    except:
        sync.terminate()
        raise

    return res


def version():
    """Get subsync version."""
    try:
        from .version import version, version_short
        return version_short, version
    except:
        return '0.17.0', '0.17.0'


def subsync(argv=None):
    """Run subsync with list of command line arguments."""
    from .__main__ import subsync
    return subsync(argv)
