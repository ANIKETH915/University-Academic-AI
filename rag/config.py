import os
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Tests may redirect the vector store to a private directory via
# PYQRAG_CHROMA_DB_DIR so pytest never contends with a running backend
# process for the same SQLite/HNSW files. Unset => unchanged production path.
CHROMA_DB_DIR = os.environ.get("PYQRAG_CHROMA_DB_DIR") or os.path.join(BASE_DIR, 'chroma_db')

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





def current_academic_year() -> int:
    return datetime.now().year




