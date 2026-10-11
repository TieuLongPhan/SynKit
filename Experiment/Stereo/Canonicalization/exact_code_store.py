"""Disk-backed certificate interning with full-byte equality, including collisions."""

import hashlib
import sqlite3
from tempfile import TemporaryDirectory
from pathlib import Path


class ExactCodeStore:
    """Store complete codes once; a digest only narrows the candidate lookup.

    Equal IDs imply equal full certificate bytes. Different full codes receive
    different IDs even if the indexing function returns the same digest.
    """

    def __init__(self, digest=None):
        self.digest = digest or (lambda value: hashlib.sha256(value).hexdigest())
        self.directory = None
        self.connection = None

    def __enter__(self):
        self.directory = TemporaryDirectory(prefix="synkit-exact-certificates-")
        self.connection = sqlite3.connect(Path(self.directory.name) / "codes.sqlite")
        self.connection.execute("PRAGMA cache_size=-2048")
        self.connection.execute("CREATE TABLE codes (id INTEGER PRIMARY KEY, digest TEXT, code BLOB)")
        self.connection.execute("CREATE INDEX digest_candidates ON codes(digest)")
        return self

    def __exit__(self, *_error):
        self.connection.close()
        self.directory.cleanup()

    def intern(self, code):
        """Return an exact equality ID after comparing all colliding full codes."""
        value = code.encode("utf-8")
        digest = self.digest(value)
        candidates = self.connection.execute("SELECT id, code FROM codes WHERE digest=?", (digest,))
        for identifier, candidate in candidates:
            if candidate == value:
                return identifier
        cursor = self.connection.execute("INSERT INTO codes(digest,code) VALUES (?,?)", (digest, value))
        return cursor.lastrowid

    def index_digest(self, identifier):
        """Return the lookup digest for reporting, never for identity decisions."""
        return self.connection.execute("SELECT digest FROM codes WHERE id=?", (identifier,)).fetchone()[0]

    @property
    def bytes_on_disk(self):
        return sum(path.stat().st_size for path in Path(self.directory.name).iterdir())
