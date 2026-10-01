"""
Run all checks before a commit and write a report to reports/summary.md:

1. the test suite (without the slow tests),
2. every example notebook in examples/; the outputs of the notebooks that run through are saved
   in place, so that GitHub shows them (without the interactive plotly figures, which GitHub does not
   display), failed notebooks are saved to reports/failed/ instead.

    python scripts/run_checks.py                      # everything (about 30 minutes)
    python scripts/run_checks.py --skip-notebooks     # only the tests
    python scripts/run_checks.py examples/gira2005hscc.ipynb   # only the tests and this notebook

Needs the dev dependencies: pip install -e ".[dev]"
"""

import argparse
import datetime
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PACKAGES = ["pybdr", "numpy", "scipy", "sympy", "cvxpy", "codac", "matplotlib", "plotly"]


def run_tests() -> dict:
    junit = REPORTS / "pytest.xml"
    start = time.perf_counter()
    subprocess.run([sys.executable, "-m", "pytest", "-m", "not slow", "-q", "-p", "no:cacheprovider",
                    f"--junitxml={junit}"], cwd=ROOT)
    elapsed = time.perf_counter() - start
    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    failed = [f"{case.get('classname')}::{case.get('name')}" for case in suite.iter("testcase")
              if case.find("failure") is not None or case.find("error") is not None]
    return {"total": int(suite.get("tests")), "failed": failed, "skipped": int(suite.get("skipped")), "time": elapsed}


def run_notebook(path: Path) -> dict:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    nb = nbformat.read(path, as_version=4)
    # no timestamps in the saved notebooks, re-running a notebook only changes its outputs
    client = NotebookClient(nb, timeout=3600, kernel_name="python3", record_timing=False,
                            resources={"metadata": {"path": str(path.parent)}})
    start = time.perf_counter()
    error = None
    try:
        client.execute()
    except CellExecutionError as err:
        error = str(err).strip().splitlines()[-1]
    elapsed = time.perf_counter() - start

    if error is None:
        nbformat.write(_without_plotly(nb), path)
    else:
        (REPORTS / "failed").mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, REPORTS / "failed" / path.name)
    figures = sum("image/png" in out.get("data", {}) for cell in nb.cells for out in cell.get("outputs", []))
    return {"name": path.name, "error": error, "time": elapsed, "figures": figures}


def _without_plotly(nb):
    """drop interactive plotly outputs: GitHub does not display them and they can be very large"""
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = [out for out in cell.outputs if "application/vnd.plotly.v1+json" not in out.get("data", {})]
    return nb


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def write_report(tests, notebooks) -> Path:
    lines = [
        "# PyBDR checks",
        "",
        f"- date: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"- commit: `{git('rev-parse', '--short', 'HEAD')}` on `{git('branch', '--show-current')}`"
        + (" (with uncommitted changes)" if git("status", "--porcelain", "--untracked-files=no") else ""),
        f"- platform: {platform.platform()}, Python {platform.python_version()}",
        "- packages: " + ", ".join(f"{p} {version(p)}" for p in PACKAGES),
        "",
    ]
    if tests is not None:
        passed = tests["total"] - len(tests["failed"]) - tests["skipped"]
        lines += ["## Tests (`pytest -m \"not slow\"`)", "",
                  f"{'✅' if not tests['failed'] else '❌'} {passed} passed, {len(tests['failed'])} failed, "
                  f"{tests['skipped']} skipped in {tests['time']:.0f}s", ""]
        lines += [f"- ❌ `{name}`" for name in tests["failed"]]
        lines += [""]
    if notebooks:
        lines += ["## Example notebooks", "", "| notebook | result | time | figures |", "|---|---|---|---|"]
        for nb in notebooks:
            result = "✅" if nb["error"] is None else f"❌ `{nb['error'][:120]}` (see `reports/failed/{nb['name']}`)"
            lines.append(f"| {nb['name']} | {result} | {nb['time']:.0f}s | {nb['figures']} |")
        lines += [""]
    path = REPORTS / "summary.md"
    path.write_text("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebooks", nargs="*", type=Path, help="notebooks to run, default: all in examples/")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-notebooks", action="store_true")
    args = parser.parse_args()

    REPORTS.mkdir(exist_ok=True)
    # the notebooks show their figures inline, a non-interactive backend from the environment would hide them
    os.environ.pop("MPLBACKEND", None)

    tests = None if args.skip_tests else run_tests()
    notebooks = []
    if not args.skip_notebooks:
        paths = [p.resolve() for p in args.notebooks] or sorted((ROOT / "examples").glob("*.ipynb"))
        for path in paths:
            print(f"running {path.name} ...", flush=True)
            notebooks.append(run_notebook(path))
            print(f"  {'ok' if notebooks[-1]['error'] is None else 'FAILED'} in {notebooks[-1]['time']:.0f}s", flush=True)

    report = write_report(tests, notebooks)
    print(report.read_text())
    failed = (tests is not None and tests["failed"]) or any(nb["error"] for nb in notebooks)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
