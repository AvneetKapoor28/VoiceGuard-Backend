from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from inference import run_inference


app = FastAPI(title="ASVspoof LCNN API", version="1.0.0")


# CORS (adjust origins for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(..., description="Audio file (wav/flac/mp3/etc.)"),
    return_array: bool = Query(False, description="Include raw log-mel matrix"),
):
    # Basic validation
    if file.size and file.size > 25 * 1024 * 1024:  # 25 MB guard
        raise HTTPException(status_code=413, detail="File too large")
    # Use a NamedTemporaryFile to support any backend that needs a file path
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = run_inference(tmp_path, return_array=return_array)
        return result
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass