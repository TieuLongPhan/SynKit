# pepkit/version.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("synkit")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
