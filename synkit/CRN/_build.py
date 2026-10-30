"""Build metadata and Boost discovery shared by native and wheel builders."""

from pathlib import Path
import sys

CAPABILITIES = (
    "closure",
    "structural_sets",
    "structural_unlimited",
    "exhaustive_siphons",
    "exact_semiflows",
    "exact_algebra",
    "crnt",
    "pathway",
    "symmetry",
)


def boost_include(explicit=None):
    """Find installed headers, including an existing Conda package cache."""
    import os

    candidates = [Path(explicit)] if explicit is not None else []
    if explicit is None:
        if os.environ.get("BOOST_INCLUDE_DIR"):
            candidates.append(Path(os.environ["BOOST_INCLUDE_DIR"]))
        candidates += [
            Path(sys.prefix) / "include",
            Path(sys.prefix) / "Library/include",
            Path("/usr/include"),
            Path("/usr/local/include"),
        ]
        candidates += [Path("/opt/homebrew/include")]
        for base in (sys.prefix, sys.base_prefix):
            prefix = Path(base)
            candidates.append(prefix / "include")
            candidates.append(prefix / "Library/include")
            if prefix.parent.name == "envs":
                prefix = prefix.parent.parent
            candidates += sorted(
                (prefix / "pkgs").glob("libboost-headers-*/include"), reverse=True
            )
            candidates += sorted(
                (prefix / "pkgs").glob("libboost-headers-*/Library/include"),
                reverse=True,
            )
    return next(
        (p for p in candidates if (p / "boost/multiprecision/cpp_int.hpp").is_file()),
        None,
    )
