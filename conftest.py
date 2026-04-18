from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

collect_ignore: list[str] = ["__init__.py"]

HERMES_ROOT: Path = Path(
    os.environ.get("HERMES_ROOT")
    or Path.home() / ".hermes" / "hermes-agent"
)
TEST_HERMES_HOME: Path = Path(
    os.environ.get("HERMES_HOME")
    or tempfile.mkdtemp(prefix="operational-checkpoint-hermes-home-")
)
os.environ["HERMES_HOME"] = str(TEST_HERMES_HOME)

if HERMES_ROOT.exists():
    hermes_root_text: str = str(HERMES_ROOT)
    if hermes_root_text not in sys.path:
        sys.path.insert(0, hermes_root_text)
