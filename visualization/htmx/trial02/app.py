
"""How to Run

>>> uvicorn visualization.htmx.trial02.app:app --reload --port 8001
"""

from pathlib import Path
from io import BytesIO
import base64

import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

BASE_DIR = Path(__file__).parent
app = FastAPI(title="HTMX + FastAPI Plot demo")


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


def _build_plot() -> str:
    """Return a data-URI png of a simple sine plot with noise."""
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x) + np.random.normal(scale=0.15, size=x.shape)

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(x, y, color="#0ea5e9", linewidth=2)
    ax.set_title("noisy sin(x)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"


@app.get("/plot", response_class=HTMLResponse)
async def plot_fragment() -> str:
    data_uri = _build_plot()
    return (
        "<div class='card'><div class='plot-wrap'>"
        f"<img src='{data_uri}' alt='plot' />"
        "</div></div>"
    )


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
