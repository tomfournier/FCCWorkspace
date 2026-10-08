"""Path-like strings and templates used by the analysis configurations."""

from __future__ import annotations

import os

from typing import Type
from pathlib import Path

# Select the appropriate Path base class for the OS
_PathBase = type(Path())



###########################
### STR-LIKE PATH CLASS ###
###########################

class LocPath(str):
    """String subclass for templates and expanded paths.

    Extends str with methods to:
    - `get(...)`: expand placeholders in a template
    - `astype(type)`: convert between LocPath (str) and PathObj (Path)
    - `mkdir(...)`: create the represented directory with Path-like options
    """

    def astype(
        self,
        type: Type[str | Path]
    ) -> 'LocPath' | 'PathObj':
        """Convert to str (LocPath) or Path (PathObj).

        Args:
            type: Either str or Path class

        Returns:
            LocPath or PathObj with an `astype()` method for roundtrip conversion
        """
        # str → LocPath; Path → PathObj; keep interface symmetric
        if (type is str) or (type is LocPath):
            return LocPath(self)
        elif (type is Path) or (type is PathObj):
            return PathObj(str(self))
        raise TypeError("Only 'str', 'LocPath', 'PathObj' or 'Path' supported")

    def get(
        self,
        name: str | None = None,
        cat: str | None = None,
        ecm: int | None = None,
        sel: str | None = None,
        type: Type[str | Path] = str
    ) -> 'LocPath' | 'PathObj':
        """Expand placeholders in this template or a named template.

        Args:
            name: Template name ('EVENTS', etc.). If None, self is the template.
            cat, ecm, sel: Placeholder values
            type: Return type - str (default) or Path

        Returns:
            Expanded LocPath or PathObj (both with `astype()`)
        """
        # If name is None, expand self; otherwise fetch `loc.<name>`
        template = self if name is None else getattr(loc, name, name)
        expanded = loc.expand(template, cat=cat, ecm=ecm, sel=sel)
        return LocPath(expanded).astype(type)

    def mkdir(
        self,
        exist_ok: bool = True,
        parents: bool = True,
        mode: int = 0o777,
    ) -> None:
        """Create this path as a directory.

        The arguments match :meth:`pathlib.Path.mkdir`: by default only the
        final directory is created, ``parents=True`` creates missing parents,
        and ``exist_ok=True`` suppresses the error when the directory exists.

        Args:
            exist_ok: Do not raise if the directory already exists.
            parents: Create missing parent directories when true.
            mode: Permission bits for the created directory.
        """
        if parents:
            os.makedirs(self, mode=mode, exist_ok=exist_ok)
        else:
            try:
                os.mkdir(self, mode=mode)
            except FileExistsError:
                if not exist_ok or not os.path.isdir(self):
                    raise



####################################
### PATHLIB.PATH-LIKE PATH CLASS ###
####################################

class PathObj(_PathBase):
    """Path subclass with get and astype methods, inheriting all Path functionality.

    Extends pathlib.Path with methods to:
    - `get(...)`: expand a named template from `loc`
    - `astype(type)`: convert between PathObj (Path) and LocPath (str)
    """

    def astype(
        self,
        type: Type[str | Path | LocPath | 'PathObj']
    ) -> LocPath | 'PathObj':
        """Convert to str (LocPath) or Path (PathObj).

        Args:
            type: Either str or Path class

        Returns:
            LocPath or PathObj with an `astype()` method for roundtrip conversion
        """
        # Path → PathObj; str → LocPath; keep interface symmetric
        if (type is Path) or (type is PathObj):
            return PathObj(str(self))
        elif (type is str) or (type is LocPath):
            return LocPath(str(self))
        raise TypeError("Only 'str', 'LocPath', 'PathObj' or 'Path' supported")

    def get(
        self, name: str,
        cat: str | None = None,
        ecm: int | None = None,
        sel: str | None = None,
        type: Type[str | Path] = str
    ) -> LocPath | 'PathObj':
        """Fetch named template from `loc`, expand, and return.

        Args:
            name: Template name ('EVENTS', etc.)
            cat, ecm, sel: Placeholder values
            type: Return type - str (default) or Path

        Returns:
            Expanded LocPath or PathObj (both with `astype()`)
        """
        template = getattr(loc, name, name)
        expanded = loc.expand(template, cat=cat, ecm=ecm, sel=sel)
        return LocPath(expanded).astype(type)



######################################
### META CLASS FOR PATH DEFINITION ###
######################################

class locMeta(type):
    """Metaclass for loc to handle type conversion based on default type setting."""

    def __init__(cls, name, bases, dct) -> None:
        super().__init__(name, bases, dct)
        cls._default_type = str  # Default returns LocPath (str type)

    def __getattribute__(cls, name: str):
        """Intercept attribute access to convert templates based on default type."""
        obj = super().__getattribute__(name)

        # Convert templates (LocPath) based on default type
        # Only convert uppercase attributes (templates), not methods or private attrs
        if (isinstance(obj, LocPath) and not callable(obj) and
                name.isupper() and not name.startswith('_')):
            default_type = super().__getattribute__('_default_type')
            # Convert to PathObj if default type is Path or PathObj
            if default_type is Path or default_type is PathObj:
                return PathObj(str(obj))

        return obj

    def set_default_type(cls, type_: Type[str | Path] | str) -> None:
        """Set the default type for loc.ATTRIBUTE access and loc.get() calls.

        Args:
            type_: Either a type object (str, LocPath, Path, PathObj) or a string
                   name ('str', 'LocPath', 'Path', 'PathObj')

        Raises:
            ValueError: If type_ is not one of the supported types or names
        """
        # Map of string names to type objects
        type_map = {'str':  str,  'LocPath': LocPath,
                    'Path': Path, 'PathObj': PathObj}

        # If type_ is a string, convert it to the actual type
        if isinstance(type_, str):
            if type_ not in type_map:
                raise ValueError(f"type_ must be one of {list(type_map.keys())}")
            type_ = type_map[type_]

        # Validate that the resulting type is one of the supported types
        if type_ not in (str, LocPath, Path, PathObj):
            raise ValueError("type_ must be str, LocPath, Path, or PathObj")

        cls._default_type = type_



####################################
### PATH CLASS USED FOR ANALYSIS ###
####################################

class loc(metaclass=locMeta):
    """Path template registry with placeholders for category, energy, and selection.

    Templates use placeholders: 'cat' (channel), 'ecm' (energy), 'sel' (selection).
    All templates are LocPath instances that can be expanded via get() or astype().
    """

    base = str(Path(__file__).parent.parent.resolve())
    WORKSPACE   = LocPath(base)                                   # Workspace directory
    FCCANALYSIS = LocPath(f'{base}/FCCAnalyses')                  # FCCAnalyses directory
    COMBLIMIT   = LocPath(f'{base}/HiggsAnalysis/CombinedLimit')  # CombinedLimit directory


    @staticmethod
    def expand(
        template: str | Path,
        cat: str | None = None,
        ecm: int | None = None,
        sel: str | None = None
    ) -> str:
        """Replace placeholders in a template string.

        Args:
            template: Template string containing 'cat', 'ecm', 'sel' placeholders
            cat, ecm, sel: Values to substitute. Raises ValueError if required but None.

        Returns:
            str: Template with placeholders replaced
        """
        tpl = str(template)
        needs_cat = 'cat' in tpl
        needs_ecm = 'ecm' in tpl
        needs_sel = 'sel' in tpl

        if needs_cat and cat is None:
            raise ValueError("'cat' is required to expand this path")
        if needs_ecm and ecm is None:
            raise ValueError("'ecm' is required to expand this path")
        if needs_sel and sel is None:
            raise ValueError("'sel' is required to expand this path")

        tpl = tpl.replace('cat', cat or '')
        tpl = tpl.replace('ecm', str(ecm) if ecm is not None else '')
        tpl = tpl.replace('sel', sel or '')
        return tpl


    @classmethod
    def get(
        cls,
        name: str,
        cat: str | None = None,
        ecm: int | None = None,
        sel: str | None = None,
        type: Type[str | Path] | None = None
    ) -> LocPath | PathObj:
        """Fetch template by name, expand placeholders, and return as LocPath or PathObj.

        Args:
            name: Template name (e.g., 'EVENTS')
            cat, ecm, sel: Placeholder values
            type: Return type - str (default) returns LocPath, Path returns PathObj.
                  If None, uses the default type set via set_default_type()

        Returns:
            LocPath or PathObj (both with astype() and get() methods)
        """
        template = getattr(cls, name)
        expanded = cls.expand(template, cat, ecm, sel)

        # Use default type if not explicitly provided
        if type is None:
            type = cls._default_type

        return LocPath(expanded).astype(type)
