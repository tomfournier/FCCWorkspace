from logger import get_logger
LOGGER = get_logger(__name__)

# Lazy import of ROOT - only load when colors are accessed
_ROOT = None


def get_root():
    """Lazily import ROOT.

    Defers ROOT import until it is actually accessed to avoid unnecessary overhead.

    Returns:
        The ROOT module.
    """
    global _ROOT
    if _ROOT is None:
        LOGGER.info('Loading ROOT...')
        import ROOT
        _ROOT = ROOT
    return _ROOT


class LazyColorDict(dict):
    """Dictionary proxy that initializes ROOT colors on first access."""

    def __init__(self, builder):
        super().__init__()
        self._builder = builder

    def _ensure(self):
        if not self:
            self.update(self._builder())

    def __getitem__(self, key):
        self._ensure()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure()
        return super().__contains__(key)

    def get(self, key, default=None):
        self._ensure()
        return super().get(key, default)

    def __iter__(self):
        self._ensure()
        return super().__iter__()

    def items(self):
        self._ensure()
        return super().items()

    def keys(self):
        self._ensure()
        return super().keys()

    def values(self):
        self._ensure()
        return super().values()


def label_parts(label: str) -> tuple[str, str]:
    from re import search
    match = search(r'\s*\[([^]]+)\]\s*$', label)
    return (label[:match.start()], match.group(1)) if match else (label, '')


def has_events(procs: list[str], path: str = '', end: str = 'root') -> list[str]:
    '''Filter processes that contain valid ROOT event trees.

    Validates that all files for each process contain the 'events' TTree.
    Supports both single files and directories with multiple files.

    Args:
        procs: List of process names to validate
        path: Base path where process files are located
        end: File extension to search for (default: '.root')

    Returns:
        List of process names where all associated files contain 'events' TTree
    '''
    import glob, uproot
    from pathlib import Path

    newprocs = []
    for proc in procs:
        indir, infile = f'{path}/{proc}', Path(path) / f'{proc}.{end}'
        
        # Check for single file or directory with multiple files
        fNames = [infile] if infile.exists() else glob.glob(f'{indir}/*')

        # Verify all files contain 'events' TTree
        isTTree = [i for i, f in enumerate(fNames) if 'events' in uproot.open(f)]
        if len(isTTree)==len(fNames): newprocs.append(proc)
    return newprocs
