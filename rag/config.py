import os
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CHROMA_DB_DIR = os.path.join(BASE_DIR, 'chroma_db')

COLLECTION_NAME = "mu_academic_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5


def resolve_collection_name(explicit: str | None = None) -> str:
    """Production collection by default; isolated test collection under pytest."""
    if explicit:
        return explicit
    import sys
    if (
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("WORKSPACE_DB_TEST_MODE") == "1"
        or "pytest" in sys.modules
    ):
        return os.environ.get("PYQRAG_TEST_COLLECTION", "pyqrag_pytest_collection")
    return COLLECTION_NAME


LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "").strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "").strip()
# LLM_API_KEY read only via rag.llm_client — never hardcode keys here


def current_academic_year() -> int:
    return datetime.now().year

