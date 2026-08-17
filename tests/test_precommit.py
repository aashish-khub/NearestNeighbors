"""Check that the pre-commit hooks (ruff lint, ruff format, pyright) pass.

Linting is enforced on Linux only. The `mixed-line-ending` hook conflicts with
Windows' CRLF checkouts, and pyright needs a Node toolchain that is not
guaranteed on every runner, so the cross-platform jobs run the functional tests
without re-checking style. See `.github/workflows/ci.yml`.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.skipif(
        shutil.which("pre-commit") is None,
        reason="pre-commit is not installed; run `pip install -e '.[dev]'`",
    ),
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="lint checks are enforced on Linux only",
    ),
    pytest.mark.skipif(
        not Path(__file__).resolve().parent.parent.joinpath(".venv").exists(),
        reason=(
            "no .venv to point pyright at; [tool.pyright] pins venv = '.venv', "
            "so the hook cannot resolve imports without one. CI covers this in "
            "its dedicated lint job."
        ),
    ),
]


def test_pre_commit() -> None:
    """Test that pre-commit hooks run successfully."""
    result = subprocess.run(
        ["pre-commit", "run", "--all-files"], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"Pre-commit hooks failed:\n{result.stdout}\n{result.stderr}"
    )


if __name__ == "__main__":
    pytest.main()
