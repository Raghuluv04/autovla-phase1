import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import model  # noqa: E402,F401  (puts data_pipeline/ on sys.path for `import autovla`)
