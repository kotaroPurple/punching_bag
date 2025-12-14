from pathlib import Path
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

BASE_DIR = Path(__file__).parent
app = FastAPI(title="HTMX + FastAPI mini demo")


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    """Serve the static demo page that lives next to this file."""
    return FileResponse(BASE_DIR / "index.html")


@app.get("/time", response_class=HTMLResponse)
async def time_fragment() -> str:
    """Return the current server time as a tiny HTML fragment."""
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"<div class='card'>Server time: {now}</div>"


@app.post("/echo", response_class=HTMLResponse)
async def echo_fragment(message: str = Form(...)) -> str:
    """Echo back the submitted message as a fragment for htmx swap."""
    return f"<p class='card'>You said: {message}</p>"


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
