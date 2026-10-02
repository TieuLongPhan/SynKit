"""Capture the active dependency closure and repository sources for C1.

The generated platform lock still needs a clean-install validation; installed
metadata alone is not proof that a pinned version is publicly obtainable.
"""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from Experiment.Synister.development import ROOT, digest, save


ROOTS = ("rxnmapper", "rdkit", "numpy", "scipy", "networkx", "pandas",
         "scikit-learn", "seaborn", "requests", "regex", "sympy", "pytest", "pip")


def closure(roots):
    pending = [(name, frozenset()) for name in roots]
    states, packages = set(), {}
    while pending:
        name, extras = pending.pop()
        name = canonicalize_name(name)
        state = (name, extras)
        if state in states:
            continue
        states.add(state)
        distribution = metadata.distribution(name)
        packages[name] = distribution.version
        for spec in distribution.requires or ():
            requirement = Requirement(spec)
            if requirement.marker is not None and not any(
                    requirement.marker.evaluate({"extra": extra}) for extra in (set(extras) | {""})):
                continue
            installed = metadata.version(requirement.name)
            if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
                raise ValueError(f"Installed dependency mismatch: {name}: {spec}, found {installed}")
            pending.append((requirement.name, frozenset(requirement.extras)))
    return dict(sorted(packages.items()))


def freeze(output):
    packages = closure(ROOTS)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "requirements.lock").open("x") as stream:
        stream.write("# Platform-specific installed dependency closure; clean-install validation required.\n")
        stream.writelines(f"{name}=={version}\n" for name, version in packages.items())
    # Mapper imports WLHash and other shared helpers: freeze their source too,
    # without editing anything outside the authorized development directories.
    paths = sorted((ROOT / "synkit").rglob("*.py"))
    paths += sorted((ROOT / "Experiment/Synister").glob("*.py"))
    paths += sorted((ROOT / "Experiment/Synister/tests").glob("*.py"))
    save(output / "repository_sources.json", {str(p.relative_to(ROOT)): p.read_text() for p in paths})
    manifest = dict(scope="platform dependency closure and source capture; clean install unverified",
                    python=sys.version, platform=platform.platform(), machine=platform.machine(),
                    roots=ROOTS, packages=packages,
                    requirements_sha256=digest((output / "requirements.lock").read_bytes()),
                    sources_sha256=digest((output / "repository_sources.json").read_bytes()))
    save(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    freeze(parser.parse_args().output)
