"""
Entry point: Launch the Streamlit frontend.

Usage: python run.py [--port 8501]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _project_python(project_root: Path) -> str:
    """Prefer the project's virtual environment on any platform."""
    candidates = [
        project_root / "venv" / "Scripts" / "python.exe",  # Windows
        project_root / "venv" / "bin" / "python",          # Linux / macOS
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def main():
    """Start the Streamlit application."""
    parser = argparse.ArgumentParser(description="Launch the IELTS Multi-Agent Grader UI.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8501")),
        help="Port for the Streamlit server (default: $PORT or 8501)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cmd = [
        _project_python(project_root), "-m", "streamlit", "run",
        "app/streamlit_app/main.py",
        "--server.port", str(args.port),
        "--theme.base", "dark",
    ]
    print("Starting IELTS Multi-Agent Grader...")
    print(f"   Open: http://localhost:{args.port}")
    return subprocess.run(cmd, cwd=project_root, env=env, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
