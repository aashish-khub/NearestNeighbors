import subprocess
import pytest


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
