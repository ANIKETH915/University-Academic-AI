"""Isolate pytest from production workspaces.json and Chroma collection."""

import os
import tempfile

# Must run before VectorStore / WorkspaceDB construction in any test module.
os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_pytest_collection")
# Never share the persistent Chroma directory with a live backend process:
# concurrent access to the same SQLite/HNSW files hangs or crashes natively.
os.environ.setdefault(
    "PYQRAG_CHROMA_DB_DIR",
    os.path.join(tempfile.gettempdir(), "pyqrag_pytest_chroma"),
)
