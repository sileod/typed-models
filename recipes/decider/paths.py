"""Where the decider recipes read and write: the Decision Index kit checkout (https://github.com/apolinario/decision-index,
with its raw benchmark artifacts built) and a data directory for downloads and manifests."""
import os
import sys
from pathlib import Path

KIT = Path(os.environ.get("DECISION_INDEX_KIT", "decision-index")).resolve()
DATA = Path(os.environ.get("TYPED_MODELS_DATA", "data")).resolve()
HUB, SUITE, RAW = DATA / "hub", KIT / "suite-0.3", KIT / "work/artifacts/benchmark-suite/raw"
sys.path.insert(0, str(KIT))
