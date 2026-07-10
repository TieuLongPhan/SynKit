import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

loaded = sys.modules.get("synkit")
if loaded is not None and not str(getattr(loaded, "__file__", "")).startswith(
    str(ROOT)
):
    for name in list(sys.modules):
        if name == "synkit" or name.startswith("synkit."):
            del sys.modules[name]
