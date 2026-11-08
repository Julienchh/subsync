from subsync.data import descriptions
from subsync.translations import _
import argparse
import textwrap
import sys


def _parse_episode_list(val):
    if not val:
        return None
    episodes = []
    for part in str(val).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            episodes.extend(range(int(a), int(b) + 1))
        else:
            episodes.append(int(part))
    return sorted(set(episodes))


def buildBatchParser():
    parser = argparse.ArgumentParser(
        prog='subsync batch',
        description='Unattended batch resynchronization across multiple video and subtitle files.',
        add_help=False,
    )
    parser.add_argument('pos_ref_dir', nargs='?', default=None, help='Directory containing reference video files')
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and exit')
    parser.add_argument('--ref-dir', '--video-dir', dest='ref_dir', default=None, help='Directory containing reference video files')
    parser.add_argument('--sub-dir', dest='sub_dir', default=None, help='Directory containing subtitle files (default: same as ref-dir)')
    parser.add_argument('--ref', '--ref-pattern', dest='ref_pattern', default=None, help='Glob pattern for reference video files')
    parser.add_argument('--sub', '--sub-pattern', dest='sub_pattern', default=None, help='Glob pattern for subtitle files')
    parser.add_argument('--out-dir', dest='out_dir', default=None, help='Directory to save synchronized subtitles')
    parser.add_argument('--out-pattern', dest='out_pattern', default=None, help='Output filename template (default: {ref_dir}/{ref_name}.synced.srt)')
    parser.add_argument('--jellyfin', dest='jellyfin', action='store_true', help='Use Jellyfin/Plex naming convention ({ref_dir}/{ref_name}.{lang}.srt)')
    parser.add_argument('--jellyfin-default', dest='jellyfin_default', action='store_true', help='Use Jellyfin default subtitle naming ({ref_dir}/{ref_name}.{lang}.default.srt)')
    parser.add_argument('--in-place', dest='in_place', action='store_true', help='Safely replace original subtitle file in-place')
    parser.add_argument('--backup', dest='backup', action='store_true', help='Create backup (.bak) when using --in-place')
    parser.add_argument('--skip-if-exists', dest='skip_if_exists', action='store_true', help='Skip task if output file already exists')
    parser.add_argument('--force', '--overwrite', dest='skip_if_exists', action='store_false', help='Force overwrite of existing synchronized subtitles')
    parser.add_argument('--matcher', dest='matcher_mode', choices=['auto', 'episode', 'name', 'fuzzy'], default='auto', help='File pairing algorithm')
    parser.add_argument('--lang', '-l', '--sub-lang', dest='lang', default=None, help='Subtitle language code (e.g. fr, fra, en)')
    parser.add_argument('--ref-lang', dest='ref_lang', default=None, help='Reference audio language code (e.g. eng, en)')
    parser.add_argument('--recursive', '-R', dest='recursive', action='store_true', help='Scan directories recursively')
    parser.add_argument('--task-timeout', dest='task_timeout', type=float, default=300.0, help='Timeout per task in seconds')
    parser.add_argument('--continue-on-error', dest='continue_on_error', action='store_true', default=True, help='Continue remaining tasks if one fails')
    parser.add_argument('--fail-fast', dest='continue_on_error', action='store_false', help='Stop batch on first task failure')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', help='Show matched pairs without synchronizing')
    parser.add_argument('--jobs', '-j', dest='jobs', type=int, default=1, help='Number of concurrent synchronization jobs')
    parser.add_argument('--summary-json', '--report-json', dest='summary_json', default=None, help='Write batch execution summary to a JSON file')
    parser.add_argument('--verbose', '-v', dest='verbose', type=int, default=1, help='Verbosity level (0..3)')
    parser.add_argument('--loglevel', dest='logLevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default=None)
    parser.add_argument('--logfile', dest='logFile', default=None)
    return parser


def buildSeriesParser():
    parser = argparse.ArgumentParser(
        prog='subsync series',
        description='Automated subtitle download and resynchronization for TV series episodes.',
        add_help=False,
    )
    parser.add_argument('pos_series_dir', nargs='?', default=None, help='Directory containing series episode video files')
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and exit')
    parser.add_argument('--series-dir', '--dir', dest='series_dir', default=None, help='Directory containing series episode video files')
    parser.add_argument('--lang', '-l', dest='lang', default='fr', help='Target subtitle language to download (default: fr)')
    parser.add_argument('--ref-lang', dest='ref_lang', default=None, help='Reference audio language (default: auto-detected via ffprobe)')
    parser.add_argument('--provider', dest='provider', choices=['auto', 'addic7ed', 'opensubtitles', 'podnapisi', 'subdl', 'mock'], default='auto', help='Online subtitle provider')
    parser.add_argument('--opensubtitles-api-key', dest='opensubtitles_api_key', default=None, help='API key for OpenSubtitles REST API v2')
    parser.add_argument('--out-pattern', dest='out_pattern', default=None, help='Output filename template (default: {ref_dir}/{ref_name}.{lang}.synced.srt)')
    parser.add_argument('--jellyfin', dest='jellyfin', action='store_true', help='Use Jellyfin/Plex naming convention ({ref_dir}/{ref_name}.{lang}.srt)')
    parser.add_argument('--jellyfin-default', dest='jellyfin_default', action='store_true', help='Use Jellyfin default subtitle naming ({ref_dir}/{ref_name}.{lang}.default.srt)')
    parser.add_argument('--cache-dir', dest='cache_dir', default=None, help='Local cache directory for downloaded subtitles')
    parser.add_argument('--max-episodes', dest='max_episodes', type=int, default=None, help='Maximum number of episodes to process')
    parser.add_argument('--episodes', dest='episodes_filter', type=_parse_episode_list, default=None, help='Episode numbers to process (e.g. 1,2,5-8)')
    parser.add_argument('--season', dest='season_filter', type=int, default=None, help='Filter by season number')
    parser.add_argument('--recursive', '-R', dest='recursive', action='store_true', help='Scan series directory recursively')
    parser.add_argument('--skip-if-exists', dest='skip_if_exists', action='store_true', help='Skip download/sync if synchronized subtitle already exists')
    parser.add_argument('--force', '--overwrite', dest='skip_if_exists', action='store_false', help='Force overwrite of existing synchronized subtitles')
    parser.add_argument('--task-timeout', dest='task_timeout', type=float, default=300.0, help='Timeout per episode synchronization in seconds')
    parser.add_argument('--workers', '--jobs', '-j', dest='workers', type=int, default=0, help='Number of parallel episode worker processes (0 = auto multi-core)')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true', help='Display planned episode downloads and sync actions without executing')
    parser.add_argument('--summary-json', '--report-json', dest='summary_json', default=None, help='Write series execution summary to a JSON file')
    parser.add_argument('--verbose', '-v', dest='verbose', type=int, default=1, help='Verbosity level (0..3)')
    parser.add_argument('--loglevel', dest='logLevel', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default=None)
    parser.add_argument('--logfile', dest='logFile', default=None)
    return parser


def buildGuiParser():
    parser = argparse.ArgumentParser(
        prog='subsync gui',
        description='Launch the standalone SubSync Studio GUI (Hybrid Math + Local AI).',
        add_help=False,
    )
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message and exit')
    parser.add_argument('--host', dest='host', default='127.0.0.1', help='Host interface to bind (default: 127.0.0.1)')
    parser.add_argument('--port', '-p', dest='port', type=int, default=8765, help='Port to bind (default: 8765)')
    parser.add_argument('--no-browser', dest='no_browser', action='store_true', help='Do not automatically open browser window')
    return parser


def parseCmdArgs(argv=None):
    raw_argv = list(argv if argv is not None else sys.argv)
    tokens = raw_argv[1:]

    # Detect 'batch', 'series', or 'gui' subcommand as first non-flag token (or after --cli)
    filtered_tokens = []
    subcommand = None
    for idx, tok in enumerate(tokens):
        if subcommand is None and tok in ('batch', 'series', 'gui'):
            subcommand = tok
            continue
        if tok in ('--cli', '-c') and subcommand in ('batch', 'series', 'gui'):
            continue
        filtered_tokens.append(tok)

    if subcommand == 'gui':
        parser = buildGuiParser()
        try:
            ns = parser.parse_args(filtered_tokens)
        except SystemExit:
            return None
        if ns.help:
            parser.print_help()
            return {'help_printed': True}
        return {
            'cli': False,
            'subcommand': 'gui',
            'gui_args': {
                'host': ns.host,
                'port': ns.port,
                'open_browser': not ns.no_browser,
            },
            'options': {},
        }

    if subcommand == 'batch':
        # Strip leading --cli/-c if placed before 'batch'
        clean_tokens = [t for t in filtered_tokens if t not in ('--cli', '-c')]
        parser = buildBatchParser()
        try:
            ns = parser.parse_args(clean_tokens)
        except SystemExit:
            return None
        if ns.help:
            parser.print_help()
            return {'help_printed': True}
        ref_dir = ns.ref_dir or ns.pos_ref_dir
        out_pattern = ns.out_pattern
        if not out_pattern:
            if getattr(ns, 'jellyfin_default', False):
                out_pattern = '{ref_dir}/{ref_name}.{lang}.default.srt' if ns.lang else '{ref_dir}/{ref_name}.default.srt'
            elif getattr(ns, 'jellyfin', False):
                out_pattern = '{ref_dir}/{ref_name}.{lang}.srt' if ns.lang else '{ref_dir}/{ref_name}.srt'
            else:
                out_pattern = '{ref_dir}/{ref_name}.synced.srt'
        opts = {}
        if ns.logLevel:
            opts['logLevel'] = ns.logLevel
        if ns.logFile:
            opts['logFile'] = ns.logFile
        return {
            'cli': True,
            'subcommand': 'batch',
            'batch_args': {
                'ref_dir': ref_dir,
                'sub_dir': ns.sub_dir,
                'ref_pattern': ns.ref_pattern,
                'sub_pattern': ns.sub_pattern,
                'out_dir': ns.out_dir,
                'out_pattern': out_pattern,
                'in_place': ns.in_place,
                'backup': ns.backup,
                'skip_if_exists': ns.skip_if_exists,
                'matcher_mode': ns.matcher_mode,
                'lang': ns.lang,
                'ref_lang': ns.ref_lang,
                'recursive': ns.recursive,
                'task_timeout': ns.task_timeout,
                'continue_on_error': ns.continue_on_error,
                'dry_run': ns.dry_run,
                'jobs': ns.jobs,
                'summary_json': ns.summary_json,
                'verbose': ns.verbose,
            },
            'verbose': ns.verbose,
            'options': opts,
        }

    if subcommand == 'series':
        clean_tokens = [t for t in filtered_tokens if t not in ('--cli', '-c')]
        parser = buildSeriesParser()
        try:
            ns = parser.parse_args(clean_tokens)
        except SystemExit:
            return None
        if ns.help:
            parser.print_help()
            return {'help_printed': True}
        series_dir = ns.series_dir or ns.pos_series_dir
        out_pattern = ns.out_pattern
        if not out_pattern:
            if getattr(ns, 'jellyfin_default', False):
                out_pattern = '{ref_dir}/{ref_name}.{lang}.default.srt'
            elif getattr(ns, 'jellyfin', False):
                out_pattern = '{ref_dir}/{ref_name}.{lang}.srt'
            else:
                out_pattern = '{ref_dir}/{ref_name}.{lang}.synced.srt'
        opts = {}
        if ns.logLevel:
            opts['logLevel'] = ns.logLevel
        if ns.logFile:
            opts['logFile'] = ns.logFile
        return {
            'cli': True,
            'subcommand': 'series',
            'series_args': {
                'series_dir': series_dir,
                'target_lang': ns.lang,
                'ref_lang': ns.ref_lang,
                'provider': ns.provider,
                'opensubtitles_api_key': ns.opensubtitles_api_key,
                'out_pattern': out_pattern,
                'cache_dir': ns.cache_dir,
                'max_episodes': ns.max_episodes,
                'episodes_filter': ns.episodes_filter,
                'season_filter': ns.season_filter,
                'recursive': ns.recursive,
                'skip_if_exists': ns.skip_if_exists,
                'task_timeout': ns.task_timeout,
                'workers': ns.workers,
                'dry_run': ns.dry_run,
                'summary_json': ns.summary_json,
                'verbose': ns.verbose,
            },
            'verbose': ns.verbose,
            'options': opts,
        }

    argvReader = ArgvReader(raw_argv)
    try:
        opts = parse(argvReader, options())
        if len(argvReader) > 0:
            raise Exception("unrecognized option '{}'".format(argvReader.peekKey()))
        return opts
    except Exception as e:
        print(str(e), file=sys.stderr)


def printHelp(argv=None):
    app = (argv or sys.argv or ['subsync'])[0]
    print(_('Usage: {} [OPTIONS] [COMMAND]').format(app))
    print('')
    print('Commands:')
    print('  gui                 Launch standalone SubSync Studio GUI (see: subsync gui --help)')
    print('  sync                Synchronize individual subtitle and reference files')
    print('  batch [DIR]         Unattended batch synchronization across multiple files (see: subsync batch --help)')
    print('  series [DIR]        Automated series subtitle download & resynchronization (see: subsync series --help)')

    def printHelpFor(opts, prefix=''):
        for opt in opts:
            if type(opt) is dict:
                name = getOptionName(opt, withGroup=True)
                descr = descriptions.cmdopts.get(prefix + name)
                if descr:
                    msg = '  ' + formatOptArg(opt) + '  '
                    msg += ' ' * (30 - len(msg))
                    msg += descr
                    print(textwrap.fill(msg, width=80, subsequent_indent=' ' * 30))
                if 'opts' in opt:
                    printHelpFor(opt['opts'], prefix=name + '.')
            else:
                print('')
                print(opt)

    printHelpFor(options())


class ArgvReader(object):
    def __init__(self, argv):
        self.argv = argv[1:]
        self.offset = 0

    def __len__(self):
        return len(self.argv)

    def peekKey(self):
        if len(self.argv) > 0:
            return self.argv[0].split('=', 1)[0]

    def popKey(self):
        if len(self.argv) > 0:
            if self.offset:
                raise Exception("option '{}' doesn't allow an argument".format(self.argv[0]))
            arg = self.argv[0].split('=', 1)
            if len(arg) == 1:
                self.argv.pop(0)
            else:
                self.offset = len(arg[0]) + 1
            return arg[0]

    def popValue(self):
        if len(self.argv) == 0:
            raise Exception('last option requires an argument')
        res = self.argv.pop(0)[self.offset:]
        self.offset = 0
        return res


### Helper functions ###

def parse(argv, opts):
    res = {}
    while len(argv) > 0:
        key = argv.peekKey()
        opt = findOption(opts, key)
        if opt is None:
            break
        argv.popKey()
        parserFn = opt.get('parser', parseVar)
        readArgs = parserFn(argv, res, key, **opt)

    for opt in opts:
        if type(opt) is dict and opt.get('required'):
            name = getOptionName(opt)
            group = opt.get('group')
            r = res
            if group:
                r = res.get(group, {})
            if name not in r:
                raise Exception("missing required option '{}'".format(name))
    return res

def findOption(options, name):
    for opt in options:
        if type(opt) is dict:
            if 'alias' in opt:
                if name == opt['alias']:
                    return opt
            elif 'aliases' in opt:
                if name in opt['aliases']:
                    return opt
            elif name == '--' + opt['name']:
                return opt

def getOptionName(opt, withGroup=False):
    name = opt['name'].split('-')
    name = name[0] + ''.join([ n.capitalize() for n in name[1:] ])
    if withGroup and 'group' in opt:
        name = '{}.{}'.format(opt['group'], name)
    return name

def addOptionVal(res, opt, value):
    name = getOptionName(opt)
    group = opt.get('group')
    if group:
        res[group] = res.get(group, {})
        res = res[group]

    if opt.get('multiple'):
        if name not in res:
            res[name] = []
        res[name].append(value)
    else:
        if name in res:
            raise Exception("duplicated option '{}'".format(opt[name]))
        res[name] = value

def formatOptArg(opt):
    name = opt['name']
    descr = descriptions.cmdopts.get(name)
    args = opt.get('aliases') or [ opt.get('alias') or '--' + name ]

    arglong =  [ arg for arg in args if arg.startswith('--') or not arg.startswith('-') ]
    argshort = [ arg for arg in args if arg not in arglong ]
    res = ', '.join(argshort + arglong)

    if 'values' in opt:
        res += '={{{}}}'.format(','.join(opt['values']))

    elif opt.get('parser') in [ None, parseVar ]:
        val = opt.get('metavar') or args[0].split('-')[-1].upper()
        res += '=' + val

    return res


### Specialized parsers used in options ###

def parseConst(argv, res, key, value=True, **opt):
    addOptionVal(res, opt, value)
    return 0

def parseVar(argv, res, key, type=str, **opt):
    try:
        addOptionVal(res, opt, type(argv.popValue()))
    except ValueError:
        raise Exception("invalid argument for option '{}', expected {}"
                .format(key, getattr(type, '__name__', str(type))))
    return 1

def parseEnum(argv, res, key, values, **opt):
    val = argv.popValue()
    if val not in values:
        raise Exception(
                "illegal value for option '{}', valid values are: {}".format(key,
                    ', '.join([ "'{}'".format(v) for v in values ])))
    addOptionVal(res, opt, val)
    return 1

def parseCmd(argv, res, key, opts, **opt):
    no = len(argv)
    addOptionVal(res, opt, parse(argv, opts))
    return no - len(argv)

def parseWordsDump(argv, res, key, **opt):
    vals = argv.popValue().split(':', 1)
    src = vals[0]
    path = None
    if len(vals) == 2:
        path = vals[1]
    addOptionVal(res, opt, (src, path))


### Command line options ###

def options():
    return [
        _('General options:'),
        { 'name': 'help', 'parser': parseConst, 'aliases': ['--help', '-h'] },
        { 'name': 'version', 'parser': parseConst, 'aliases': ['--version', '-v'] },
        { 'group': 'options', 'name': 'language', 'alias': '--lang' },

        _('Headless options:'),
        { 'name': 'cli', 'parser': parseConst, 'aliases': ['--cli', '-c'] },
        { 'name': 'verbose', 'type': int },
        { 'name': 'offline', 'parser': parseConst },

        _('GUI options:'),
        { 'name': 'batch', 'parser': parseConst },

        _('Synchronization job:'),
        { 'name': 'sync', 'parser': parseCmd, 'alias': 'sync', 'multiple': True, 'opts': [
            { 'group': 'sub', 'name': 'path', 'aliases': ['--sub', '--sub-file', '-s'], 'metavar': 'PATH', 'required': True },
            { 'group': 'sub', 'name': 'stream', 'alias': '--sub-stream', 'type': int, 'metavar': 'NO' },
            { 'group': 'sub', 'name': 'streamByLang', 'alias': '--sub-stream-by-lang' },
            { 'group': 'sub', 'name': 'lang', 'alias': '--sub-lang' },
            { 'group': 'sub', 'name': 'enc', 'alias': '--sub-enc' },
            { 'group': 'sub', 'name': 'fps', 'alias': '--sub-fps', 'type': float },

            { 'group': 'ref', 'name': 'path', 'aliases': ['--ref', '--ref-file', '-r'], 'metavar': 'PATH', 'required': True },
            { 'group': 'ref', 'name': 'stream', 'alias': '--ref-stream', 'type': int, 'metavar': 'NO' },
            { 'group': 'ref', 'name': 'streamByType', 'alias': '--ref-stream-by-type', 'parser': parseEnum,
                'values': ['sub', 'audio'] },
            { 'group': 'ref', 'name': 'streamByLang', 'alias': '--ref-stream-by-lang' },
            { 'group': 'ref', 'name': 'lang', 'alias': '--ref-lang' },
            { 'group': 'ref', 'name': 'enc', 'alias': '--ref-enc' },
            { 'group': 'ref', 'name': 'fps', 'alias': '--ref-fps', 'type': float },
            { 'group': 'ref', 'name': 'channels', 'alias': '--ref-channels' },

            { 'group': 'out', 'name': 'path', 'alias': '--out', 'aliases': ['--out', '--out-file', '-o'] },
            { 'group': 'out', 'name': 'enc', 'alias': '--out-enc' },
            { 'group': 'out', 'name': 'fps', 'alias': '--out-fps', 'type': float },
        ]},

        _('Synchronization options:'),
        { 'name': 'fromFile', 'alias': '--import', 'metavar': 'PATH' },
        { 'group': 'options', 'name': 'minEffort', 'alias': '--effort', 'type': float },
        { 'group': 'options', 'name': 'overwrite', 'parser': parseConst },
        { 'group': 'options', 'name': 'jobsNo',          'type': int, 'aliases': ['--jobs', '-j'], 'metavar': 'NO' },
        { 'group': 'options', 'name': 'window-size',     'type': float },
        { 'group': 'options', 'name': 'max-point-dist',  'type': float },
        { 'group': 'options', 'name': 'min-points-no',   'type': int },
        { 'group': 'options', 'name': 'min-word-prob',   'type': float },
        { 'group': 'options', 'name': 'min-word-len',    'type': int },
        { 'group': 'options', 'name': 'min-correlation', 'type': float, 'metavar': 'COR' },
        { 'group': 'options', 'name': 'min-words-sim',   'type': float },
        { 'group': 'options', 'name': 'out-time-offset', 'type': float },

        _('Debug options:'),
        { 'group': 'options', 'name': 'logLevel', 'alias': '--loglevel', 'parser': parseEnum,
                'values': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] },
        { 'group': 'options', 'name': 'logFile', 'alias': '--logfile' },
        { 'group': 'options', 'name': 'dump-words', 'metavar': 'SRC[:PATH]', 'parser': parseWordsDump, 'multiple': True },
        { 'group': 'options', 'name': 'test', 'parser': parseConst },
]
