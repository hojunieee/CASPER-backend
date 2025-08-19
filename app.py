from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import uuid

from optimizer import run_optimizer

app = FastAPI()

@app.post("/jobs")
async def create_job(
    courses: UploadFile = File(...),
    professors: UploadFile = File(...),
    prof_prefs: UploadFile = File(...),
    rooms: UploadFile = File(...),
    student_prefs: UploadFile = File(...),
    timeblocks: UploadFile = File(...),
    timeblock_dict: UploadFile = File(...),
):
    # Unique job ID
    job_id = str(uuid.uuid4())
    input_dir = Path(f"jobs/{job_id}/inputs")
    output_dir = Path(f"jobs/{job_id}/outputs")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files to input_dir
    uploads = {
        "course_offered.csv": courses,
        "professor_list.csv": professors,  # optional if you use it
        "professor_preference.csv": prof_prefs,
        "rooms.csv": rooms,
        "student_preference.csv": student_prefs,
        "time_blocks.csv": timeblocks,
        "time_block_dict.csv": timeblock_dict,
    }

    for filename, file in uploads.items():
        with open(input_dir / filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Run optimizer synchronously (later: background worker)
    result_file = run_optimizer(input_dir, output_dir)

    return {"job_id": job_id, "status": "done", "result_file": str(result_file)}

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    result_path = Path(f"jobs/{job_id}/outputs/best_assignment.json")
    if result_path.exists():
        return {"status": "done", "result": result_path.read_text()}
    return {"status": "not_found"}
