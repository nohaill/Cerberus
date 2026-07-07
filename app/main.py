"""
Cerberus API entrypoint.

    uvicorn app.main:app --reload

In real (non-mock) mode, all HF models are loaded at startup via the lifespan
handler. This avoids two problems: the PyTorch meta tensor bug (device_map='auto'
on CPU-only machines) and per-request cold-start latency.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import get_settings
from app.gateway import Gateway
from app.observability import metrics, stats
from app.observability.logging_config import log_decision
from app.schemas import CheckRequest, CheckResponse

_gateway = Gateway()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.mock_mode:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _gateway.ensemble.preload)
    yield


app = FastAPI(
    title="Cerberus",
    description="Multi-layer LLM security & trust gateway",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    settings = get_settings()
    return {"status": "ok", "mock_mode": settings.mock_mode}


@app.post("/v1/check", response_model=CheckResponse)
async def check(request: CheckRequest) -> CheckResponse:
    response = await _gateway.check(request)
    log_decision(request, response)
    metrics.record_response(response)
    stats.stats.record(response)
    return response


@app.get("/metrics")
def prometheus_metrics():
    body, content_type = metrics.render_metrics()
    return Response(content=body, media_type=content_type)


@app.get("/v1/stats")
def get_stats():
    return stats.stats.snapshot()


@app.get("/dashboard")
def dashboard():
    return FileResponse("app/static/dashboard.html")
