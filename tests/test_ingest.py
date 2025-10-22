import os
import shutil
from pathlib import Path

def test_repo_layout():
    expected = [
        "app.py",
        "ingest.py",
        "config.py",
        "requirements.txt",
        "README.md",
        "data/docs",
        "data/vectorstore",
    ]
    for p in expected:
        assert Path(p).exists() or Path(p).is_dir()
