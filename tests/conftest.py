import os
import tempfile
from pathlib import Path

import pytest

# Tests must never depend on an API key or a downloaded speech model, and
# must never touch the real call-history database.
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ["ASSIST_DISABLE_STT"] = "1"
os.environ["ASSIST_DB"] = str(Path(tempfile.mkdtemp()) / "test-history.db")

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "kb"


@pytest.fixture(scope="session")
def retriever():
    from assist.retriever import TfidfRetriever

    return TfidfRetriever.from_kb(KB_DIR)


@pytest.fixture(scope="session")
def agent(retriever):
    from assist.agent import AssistAgent
    from assist.suggest import ExtractiveSuggester

    return AssistAgent(retriever, ExtractiveSuggester())
