from subsync import utils
from subsync.translations import _
from subsync.data import languages
from subsync.error import Error
import os

import logging
logger = logging.getLogger(__name__)

__pdoc__ = {
        'PathFormatter': False,
        'ConditionalFormatter': False,
        }


class OutputFile(object):
    """Subtitle target description - specifies where output should be saved."""

    def __init__(self, path=None, *, enc=None, fps=None):
        self.path = path
        self.enc  = enc or 'UTF-8'
        self.fps  = fps
        self.pathFormatter = None

    def getPath(self, sub, ref):
        """Compile path pattern for given `sub` and `ref`."""
        if self.pathFormatter is None:
            self.pathFormatter = PathFormatter()
        return self.pathFormatter.format(self.path, sub, ref)

    def validateOutputPattern(self):
        """Raise exception for invalid path pattern."""
        validatePattern(self.path)

    def serialize(self):
        res = {}
        if self.path: res['path'] = self.path
        if self.enc:  res['enc'] = self.enc
        if self.fps:  res['fps'] = self.fps
        return res

    def __repr__(self):
        return utils.fmtobj(self.__class__.__name__,
                path = self.path,
                enc = self.enc,
                fps = self.fps)

    def __str__(self):
        return utils.fmtstr(self.path,
                enc = self.enc,
                fps = self.fps)


class PathFormatter(object):
    def __init__(self):
        self.clearCache()

    def clearCache(self):
        self.d = {}
        self.cache = (None, None, None)

    def format(self, pattern, sub, ref):
        if pattern is None or sub is None or ref is None:
            return None

        cacheKey = (sub.path, sub.no, sub.lang, ref.path, ref.no, ref.lang)

        if self.cache[0] == cacheKey:
            if self.cache[1] == pattern:
                return self.cache[2]

        else:
            self.d = {}
            for prefix, item in [ ('sub_', sub), ('ref_', ref) ]:
                no_val = item.no if item.no is not None else 0
                self.d[ prefix + 'path' ] = item.path
                self.d[ prefix + 'no'   ] = str(no_val + 1)
                self.d[ prefix + 'lang' ] = item.lang or ''
                self.d[ prefix + 'lang2' ] = languages.get(code3=item.lang).code2 or ''
                self.d[ prefix + 'name' ] = os.path.splitext(os.path.basename(item.path))[0]
                self.d[ prefix + 'dir'  ] = os.path.dirname(item.path)

        path = _formatPattern(pattern, self.d)
        self.cache = (cacheKey, pattern, path)
        return path


def validatePattern(pattern):
    """Raise exception for invalid path pattern."""
    d = {}
    for prefix in [ 'sub_', 'ref_' ]:
        for name in [ 'path', 'no', 'lang', 'lang2', 'name', 'dir' ]:
            d[ prefix + name ] = ''
    _formatPattern(pattern, d)


def _formatPattern(pattern, formatter):
    try:
        return pattern.format(**formatter, **{
            'if': ConditionalFormatter(formatter),
            'if_not': ConditionalFormatter(formatter, inverted=True)
            })

    except KeyError as e:
        raise Error(_('Invalid output pattern, invalid keyword: {}').format(e),
                pattern=pattern)

    except Exception as e:
        raise Error(_('Invalid output pattern, {}').format(e), pattern=pattern)


class ConditionalFormatter(object):
    def __init__(self, items, inverted=False):
        self.items = items
        self.inverted = inverted

    def __format__(self, fmt):
        key, val = fmt.split(':', 1)
        if bool(self.items[key]) ^ self.inverted:
            return val
        else:
            return ''
