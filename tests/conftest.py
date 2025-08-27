
import sys
from pathlib import Path

# Add the project's src directory to the Python path so tests can import the code
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
