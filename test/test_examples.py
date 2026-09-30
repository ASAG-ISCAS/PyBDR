"""
Execute the example notebooks (the outputs are not saved, use scripts/run_checks.py for that).
"""

from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.slow
@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.ipynb")), ids=lambda p: p.stem)
def test_notebook(path):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")

    nb = nbformat.read(path, as_version=4)
    nbclient.NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}}).execute()
