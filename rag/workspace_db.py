import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import sys
import tempfile
import shutil
import os
import json

WORKSPACES_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspaces.json")

DEFAULT_WORKSPACES = []

class WorkspaceDB:
    def __init__(self, json_path: Optional[str] = None):
        self._explicit_json_path = json_path
        self._ensure_exists()

    @property
    def json_path(self) -> str:
        if self._explicit_json_path:
            return self._explicit_json_path
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("WORKSPACE_DB_TEST_MODE") == "1" or "pytest" in sys.modules:
            if not hasattr(sys, "_test_workspaces_json_path") or not sys._test_workspaces_json_path or not os.path.exists(sys._test_workspaces_json_path):
                temp_file = tempfile.NamedTemporaryFile(suffix="_workspaces.json", delete=False)
                temp_file.close()
                if os.path.exists(WORKSPACES_JSON):
                    shutil.copyfile(WORKSPACES_JSON, temp_file.name)
                else:
                    with open(temp_file.name, 'w', encoding='utf-8') as f:
                        json.dump(DEFAULT_WORKSPACES, f, indent=2)
                sys._test_workspaces_json_path = temp_file.name
            return sys._test_workspaces_json_path
        return WORKSPACES_JSON

    def _ensure_exists(self):
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_WORKSPACES, f, indent=2)

    def _read_all(self) -> List[Dict[str, Any]]:
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except Exception:
            raw_data = DEFAULT_WORKSPACES

        # Deduplicate workspaces by ID preserving first appearance
        seen_ids = set()
        deduped = []
        for ws in raw_data:
            if not isinstance(ws, dict) or not ws.get("id"):
                continue
            if ws["id"] not in seen_ids:
                seen_ids.add(ws["id"])
                # Also deduplicate file lists inside each workspace
                syl_seen = set()
                deduped_syl = []
                for sf in ws.get("syllabus_files", []):
                    sf_id = sf.get("id") or sf.get("name")
                    if sf_id and sf_id not in syl_seen:
                        syl_seen.add(sf_id)
                        deduped_syl.append(sf)
                ws["syllabus_files"] = deduped_syl

                pyq_seen = set()
                deduped_pyq = []
                for pf in ws.get("pyq_files", []):
                    pf_id = pf.get("id") or pf.get("name")
                    if pf_id and pf_id not in pyq_seen:
                        pyq_seen.add(pf_id)
                        deduped_pyq.append(pf)
                ws["pyq_files"] = deduped_pyq

                deduped.append(ws)
        return deduped

    def _write_all(self, data: List[Dict[str, Any]]):
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_all(self) -> List[Dict[str, Any]]:
        return self._read_all()

    def get_by_id(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        for ws in self._read_all():
            if ws["id"] == workspace_id:
                return ws
        return None

    def get_or_create(self, workspace_id: str, university: str = "Academic Institution", branch: str = "General Program", semester: str = "Semester 1", subject: str = "Academic Subject", subject_code: str = "COURSE") -> Dict[str, Any]:
        ws = self.get_by_id(workspace_id)
        if ws:
            return ws

        data = self._read_all()
        now_iso = datetime.now(timezone.utc).isoformat()

        new_ws = {
            "id": workspace_id,
            "university": university,
            "branch": branch,
            "program": branch,
            "semester": semester,
            "subject": subject,
            "subject_code": subject_code,
            "is_demo": False,
            "created_at": now_iso,
            "syllabus_files": [],
            "pyq_files": []
        }
        data.insert(0, new_ws)
        self._write_all(data)
        return new_ws

    def create(self, university: str, branch: str, semester: str, subject: str, subject_code: str = "") -> Dict[str, Any]:
        data = self._read_all()
        # Unique canonical ID — never collide across subjects or recreate "ws-N-subject"
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in (subject or "subject"))[:16].strip("-") or "subject"
        ws_id = f"ws-{slug}-{uuid.uuid4().hex[:8]}"
        now_iso = datetime.now(timezone.utc).isoformat()

        new_ws = {
            "id": ws_id,
            "university": university,
            "branch": branch,
            "program": branch,
            "semester": semester,
            "subject": subject,
            "subject_code": subject_code or "COURSE",
            "is_demo": False,
            "created_at": now_iso,
            "syllabus_files": [],
            "pyq_files": []
        }
        data.insert(0, new_ws)
        self._write_all(data)
        return new_ws

    def add_file(self, workspace_id: str, file_info: Dict[str, Any], doc_type: str) -> Optional[Dict[str, Any]]:
        data = self._read_all()
        target = None
        target_id = file_info.get("id") or file_info.get("name")
        for ws in data:
            if ws["id"] == workspace_id:
                if doc_type == "syllabus":
                    # Remove any existing item with same ID/name before prepending
                    ws["syllabus_files"] = [f for f in ws.get("syllabus_files", []) if (f.get("id") or f.get("name")) != target_id]
                    ws["syllabus_files"].insert(0, file_info)
                else:
                    ws["pyq_files"] = [f for f in ws.get("pyq_files", []) if (f.get("id") or f.get("name")) != target_id]
                    ws["pyq_files"].append(file_info)
                target = ws
                break
        if target:
            self._write_all(data)
        return target

    def remove_file(self, workspace_id: str, file_id: str, doc_type: str, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        data = self._read_all()
        target = None
        target_keys = {k for k in [file_id, filename] if k}
        for ws in data:
            if ws["id"] == workspace_id:
                if doc_type == "syllabus":
                    ws["syllabus_files"] = [
                        f for f in ws.get("syllabus_files", [])
                        if not ({f.get("id"), f.get("name"), f.get("filename")} & target_keys)
                    ]
                else:
                    ws["pyq_files"] = [
                        f for f in ws.get("pyq_files", [])
                        if not ({f.get("id"), f.get("name"), f.get("filename")} & target_keys)
                    ]
                target = ws
                break
        if target:
            self._write_all(data)
        return target

    def delete_workspace(self, workspace_id: str) -> bool:
        data = self._read_all()
        filtered = [ws for ws in data if ws["id"] != workspace_id]
        if len(filtered) < len(data):
            self._write_all(filtered)
            return True
        return False

workspace_db = WorkspaceDB()
