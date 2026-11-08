from subsync.assets import item, listupdater, assetlist
from subsync import config, utils
import threading, json
import os

import logging
logger = logging.getLogger(__name__)


class AssetManager(object):
    """Manages remote and local assets.

    Assets are used for synchronization (speech recognition models,
    dictionaries) and application updates.
    Note: The modernized VAD + FFT/OLS synchronization engine is
    language-agnostic and does not require remote PocketSphinx models
    or bilingual dictionaries.
    """

    _instance = None

    def instance():
        """Get singleton instance of `AssetManager`."""
        if AssetManager._instance is None:
            AssetManager._instance = AssetManager()
        return AssetManager._instance

    def __init__(self):
        if AssetManager._instance is not None:
            return AssetManager._instance

        self._assets = {}
        self._listUpdater = None
        self._assetsLock = threading.RLock()

        try:
            if os.path.isfile(config.assetspath):
                logger.info('reading remote asset list from %s', config.assetspath)
                with open(config.assetspath, encoding='utf8') as fp:
                    self._updateAssetsRemoteData(json.load(fp))
        except:
            logger.error('cannot read asset list from %s',
                    config.assetspath, exc_info=True)

        updAsset = self.getSelfUpdaterAsset()
        if updAsset:
            local = updAsset.localVersion()
            installer = updAsset.installerVersion()
            if local and installer and local >= installer:
                logger.info('removing old installer (%s) files', installer)
                updAsset._removeLocalData()

    def getAsset(self, assetId, params=None):
        """Get asset of given ID or type/params."""
        if params:
            typ = assetId
            par = params
            id = item.mkId(typ, par)
        elif isinstance(assetId, str):
            id = assetId
            typ, par = item.parseId(assetId)
        else:
            typ = assetId[0]
            par = assetId[1]
            id = item.mkId(typ, par)

        with self._assetsLock:
            if id not in self._assets:
                self._assets[id] = item.createAsset(typ, par)
            return self._assets[id]

    def getAssetsForTasks(self, tasks):
        """Get assets list for requested tasks.

        With the modern language-agnostic VAD synchronization engine, external
        PocketSphinx models and translation dictionaries are not required.
        """
        return assetlist.AssetList()

    def getAssetListUpdater(self, autoUpdate=False):
        """Get remote assets list updater."""

        def onUpdate(assets):
            if assets:
                self._updateAssetsRemoteData(assets)
            if autoUpdate:
                self.runSelfUpdater()

        if not self._listUpdater:
            self._listUpdater = listupdater.ListUpdater(onUpdate=onUpdate)
        return self._listUpdater

    def runSelfUpdater(self):
        """Start self upgrade if available."""
        asset = self.getSelfUpdaterAsset()
        if asset and asset.hasUpdate() and not asset.hasInstaller():
            logger.info('new version available to download, %s -> %s',
                    asset.localVersion(), asset.remoteVersion())
            downloader = asset.downloader()
            downloader and downloader.run(timeout=0.5)
            return downloader

    def isListUpToDate(self):
        """Check if asset list is synchronized with remote server."""
        return self._listUpdater and self._listUpdater.isUpdated()

    def getSelfUpdaterAsset(self):
        """Get asset corresponding with application upgrade (if exist)."""
        if config.assetupd and utils.getCurrentVersion():
            return self.getAsset(config.assetupd)

    def _updateAssetsRemoteData(self, assets):
        logger.info('updating remote asset list, got %i assets', len(assets))
        for id, remote in assets.items():
            if item.validateRemoteData(remote):
                self.getAsset(id)._setRemoteData(remote)
            else:
                logger.warning('invalid asset remote data: %r', remote)
                self.getAsset(id)._setRemoteData({})

        with self._assetsLock:
            for id in self._assets.keys() - assets.keys():
                logger.debug('asset %s removed from server', id)
                self.getAsset(id)._setRemoteData({})
