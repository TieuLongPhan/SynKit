# synkit/examples.py

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files, as_file

from synkit.IO import load_database  # adjust if the import path differs


def _sanitize_slug(slug: str) -> str:
    if ".." in slug or slug.startswith(("/", "\\")):
        raise ValueError(f"Invalid example name: {slug!r}")
    return slug


def list_examples() -> list[str]:
    """
    Returns all available example slugs (without extensions), e.g. ['paracetamol', ...]
    """
    data_dir = files("synkit").joinpath("Data")
    slugs: set[str] = set()
    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        if p.name.endswith(".json.gz"):
            slugs.add(p.name[: -len(".json.gz")])
        elif p.name.endswith(".json"):
            slugs.add(p.name[: -len(".json")])
    return sorted(slugs)


@lru_cache(maxsize=32)
def load_example(slug: str):
    """
    Load an example by slug, preferring compressed (.json.gz) over plain (.json).
    Delegates actual parsing to synkit.IO.load_database.
    """
    slug = _sanitize_slug(slug)
    data_dir = files("synkit").joinpath("Data")
    candidates = [f"{slug}.json.gz", f"{slug}.json"]
    for name in candidates:
        resource = data_dir.joinpath(name)
        if resource.is_file():
            # importlib.resources may keep resources inside archives; as_file gives a real path
            with as_file(resource) as real_path:
                return load_database(real_path)
    raise FileNotFoundError(
        f"Example '{slug}' not found (looked for {', '.join(candidates)})"
    )
