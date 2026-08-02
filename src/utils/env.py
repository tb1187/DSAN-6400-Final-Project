"""Load credentials from the repo's ``.env`` file.

Scripts call :func:`load_env` at start-up so that ``ANTHROPIC_API_KEY`` in
``.env`` behaves the same as one exported in the shell. Real environment
variables always win, so a key exported for a single run overrides the file
without editing it.

``.env`` is gitignored and must stay that way — it holds live credentials.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"


def load_env(path: Path | None = None) -> bool:
    """Load ``.env`` into the process environment. Returns whether a file was read."""
    target = Path(path) if path else ENV_PATH
    if not target.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Fall back to a minimal parser so a missing dependency is not a blocker.
        for line in target.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return True
    # override=False: an exported variable beats the file.
    load_dotenv(target, override=False)
    return True


def require(name: str) -> str:
    """Fetch a credential, loading ``.env`` first, with an actionable error."""
    load_env()
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to {ENV_PATH} as `{name}=...`, "
            f"or export it in your shell."
        )
    return value


__all__ = ["load_env", "require", "ENV_PATH"]
