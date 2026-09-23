"""Project paths independent of the shell's current working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
CACHE_DIR = PROJECT_ROOT / ".cache"
ENV_FILE = PROJECT_ROOT / ".env"
