# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import shutil
import uuid

from optimizer import run_optimizer

app = FastAPI()

# ----- CORS: allow your Vercel domain in production
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://YOUR-VERCEL-APP.vercel.app",   # <- change this later
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "service": "casper-backend"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/jobs")
async def create_job(
    courses: UploadFile = File(...),
    professors: UploadFile = File(...),      # if unused in optimizer, we still store it
    prof_prefs: UploadFile = File(...),
    rooms: UploadFile = File(...),
    student_prefs: UploadFile = File(...),
    timeblocks: UploadFile = File(...),
    timeblock_dict: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())
    input_dir  = Path(f"jobs/{job_id}/inputs")
    output_dir = Path(f"jobs/{job_id}/outputs")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    uploads = {
        "course_offered.csv": courses,
        "professor_list.csv": professors,              # optional for your pipeline
        "professor_preference.csv": prof_prefs,
        "rooms.csv": rooms,
        "student_preference.csv": student_prefs,
        "time_blocks.csv": timeblocks,
        "time_block_dict.csv": timeblock_dict,
    }

    try:
        for filename, file in uploads.items():
            with open(input_dir / filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        # Synchronous run (simple). Later: move to a worker if needed.
        result_file = run_optimizer(input_dir, output_dir, seed=int(job_id.split('-')[0], 16) % (2**32))

    except Exception as e:
        # Clean up on failure (optional)
        return JSONResponse(status_code=500, content={"error": repr(e)})

    return {
        "job_id": job_id,
        "status": "done",
        "results": {
            "best_json": f"/jobs/{job_id}/result?name=best_assignment.json",
            "schedule_time_csv": f"/jobs/{job_id}/result?name=schedule_time.csv",
            "schedule_room_csv": f"/jobs/{job_id}/result?name=schedule_room.csv",
        },
    }

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    result_path = Path(f"jobs/{job_id}/outputs/best_assignment.json")
    if result_path.exists():
        # Return a tiny status; frontend can fetch full file from /result
        return {"status": "done"}
    raise HTTPException(status_code=404, detail="job not found")

@app.get("/jobs/{job_id}/result")
def download_job_file(job_id: str, name: str = "best_assignment.json"):
    path = Path(f"jobs/{job_id}/outputs") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    # Streams file (JSON or CSV)
    media = "application/json" if path.suffix == ".json" else "text/csv"
    return FileResponse(path, media_type=media, filename=path.name)
