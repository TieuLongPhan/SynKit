"""Compatibility alias for :mod:`synkit.Chem.Mapper`.

Atom-to-atom mapping is now a top-level chemistry concern. New code should
import from :mod:`synkit.Chem.Mapper`; this package preserves the former
``synkit.Chem.Reaction.Mapper`` imports, including imports of its submodules.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys

from synkit.Chem import Mapper as _mapper
from synkit.Chem.Mapper import *  # noqa: F401,F403

__all__ = _mapper.__all__
__path__ = _mapper.__path__

# Bind every canonical mapper module to its historical name. Besides keeping
# deep imports working, this ensures classes imported through old and new
# paths have the same identity.
for _module_info in pkgutil.walk_packages(
    _mapper.__path__, prefix=f"{_mapper.__name__}."
):
    _canonical_name = _module_info.name
    _module = importlib.import_module(_canonical_name)
    _suffix = _canonical_name.removeprefix(_mapper.__name__)
    sys.modules[f"{__name__}{_suffix}"] = _module
    if _suffix.count(".") == 1:
        globals()[_suffix[1:]] = _module

del _canonical_name, _mapper, _module, _module_info, _suffix
