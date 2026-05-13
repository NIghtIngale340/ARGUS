import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows-only stability guard: pyarrow/pandas can interfere with torch DLL
# initialization when they are imported first in subprocess-heavy tests.
try:
    import torch  # noqa: F401
except Exception:
    pass
