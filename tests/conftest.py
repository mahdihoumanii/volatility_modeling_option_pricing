import sys
from pathlib import Path

# Ensure project root (parent of src/) is on sys.path for imports during pytest
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
