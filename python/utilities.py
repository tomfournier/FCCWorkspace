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


def timer(t: float) -> None:
    '''Log formatted elapsed time since provided timestamp.

    Calculates and logs elapsed time in human-readable format (hours, minutes,
    seconds, milliseconds) with formatted header and footer separators.

    Args:
        t: Starting timestamp from time.time().
    '''
    import time
    dt = time.time() - t

    # Split time into components
    h, m = int(dt // 3600), int(dt // 60 % 60),
    s, ms = int(dt % 60), int((dt % 1) * 1000)

    # Build time string with non-zero components
    time_parts = []
    if h  > 0: time_parts.append(f'{h} h')
    if m  > 0: time_parts.append(f'{m} min')
    if s  > 0: time_parts.append(f'{s} s')
    if ms > 0: time_parts.append(f'{ms} ms')
    if not time_parts:
        time_parts.append('0 ms')

    elapsed = f"Elapsed time: {' '.join(time_parts)}"
    lenght = len(elapsed) + 4

    LOGGER.info(f'\n{" CODE ENDED ":=^{lenght}}\n{elapsed:^{lenght}}\n{"="*lenght}\n')
    return None
