"""AutoVLA Phase II: Qwen2.5-VL backbone, SFT, uncertainty head, gating, GRPO.

Phase I lives in ``data_pipeline/autovla``; it is put on ``sys.path`` here so
that Phase II modules can ``import autovla`` without installing it.
"""

import sys
from pathlib import Path

_PIPELINE = Path(__file__).resolve().parents[1] / "data_pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
