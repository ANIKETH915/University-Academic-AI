"""Isolate pytest from production workspaces.json and Chroma collection."""

import os

# Must run before VectorStore / WorkspaceDB construction in any test module.
os.environ.setdefault("WORKSPACE_DB_TEST_MODE", "1")
os.environ.setdefault("PYQRAG_TEST_COLLECTION", "pyqrag_pytest_collection")
