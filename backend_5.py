from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import subprocess

app = FastAPI()

# Enable CORS to allow access from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/run-forecast")
def run_forecast(date: str):
    try:
        # Run the server with the selected date passed as CLI arg
        subprocess.Popen(["python", "server_5.py", date])
        return {"status": "started", "date": date}
    except Exception as e:
        return {"status": "error", "message": str(e)}
