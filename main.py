import uvicorn

if __name__ == "__main__":
    # reload=False ensures ingest/API code changes load reliably on Windows
    uvicorn.run("rag.api:app", host="0.0.0.0", port=8000, reload=False)
