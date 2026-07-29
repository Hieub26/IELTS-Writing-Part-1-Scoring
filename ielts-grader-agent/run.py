"""
Entry point: Launch the Streamlit frontend.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Start the Streamlit application."""
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / "venv" / "Scripts" / "python.exe"
    python_executable = str(venv_python) if venv_python.exists() else sys.executable

    import os
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    cmd = [
        python_executable, "-m", "streamlit", "run",
        "app/streamlit_app/main.py",
        "--server.port", "8501",
        "--theme.base", "dark",
    ]
    print("Starting IELTS Multi-Agent Grader...")
    print(f"   Open: http://localhost:8501")
    subprocess.run(cmd, cwd=project_root, env=env, check=False)


if __name__ == "__main__":
    main()
