@echo off
cd /d "d:\pyqrag"
"C:\Program Files\Python313\python.exe" -u -m uvicorn rag.api:app --host 127.0.0.1 --port 8000
