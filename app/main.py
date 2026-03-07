import asyncio
import base64
import traceback
from contextlib import asynccontextmanager
from functools import lru_cache
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

import httpx

from app.ai import AgentOrchestrator, PromptBuilder, SessionStore, sse_event, sse_response
from app.mace import MACEService
from app.tools import build_registry


# ─── Config ───────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    tavily_api_key: str = ""
    host: str = "0.0.0.0"
    port: int = 10000
    workers: int = 1
    max_sessions: int = 500
    session_ttl: int = 3600
    mace_device: str = ""
    mace_compile_mode: str = ""  # "default", "reduce-overhead", or "max-autotune"
    cors_origins: list[str] = ["*"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ─── Pydantic Models ─────────────────────────────────────────────────────────


class ToolResult(BaseModel):
    tool_call_id: str
    content: str


class ToolResultsPayload(BaseModel):
    results: list[ToolResult]
    assistant_message: Optional[dict] = Field(None, alias="assistantMessage")


class ChatRequest(BaseModel):
    session_id: str = Field("default", alias="sessionId")
    message: str = ""
    state: dict = {}
    tool_results: Optional[ToolResultsPayload] = Field(None, alias="toolResults")
    model: str = "claude-sonnet-4-6"
    thinking_budget: int = Field(4096, alias="thinkingBudget")

    model_config = {"populate_by_name": True}


class AtomData(BaseModel):
    element: str
    x: float
    y: float
    z: float


class EnergyRequest(BaseModel):
    atoms: list[AtomData]
    model: str = "mace-mp-0a"
    include_forces: bool = Field(True, alias="includeForces")

    model_config = {"populate_by_name": True}


class OptimizeRequest(BaseModel):
    atoms: list[AtomData]
    model: str = "medium"
    fmax: float = 0.05
    max_steps: int = Field(100, alias="maxSteps")
    include_forces: bool = Field(True, alias="includeForces")

    model_config = {"populate_by_name": True}


class MDRequest(BaseModel):
    atoms: list[AtomData]
    model: str = "medium"
    temperature: float = 300.0
    timestep: float = 1.0
    friction: float = 0.01
    frames: Optional[int] = None
    steps: int = 500
    save_interval: int = Field(10, alias="saveInterval")
    include_forces: bool = Field(True, alias="includeForces")

    model_config = {"populate_by_name": True}


class BatchEnergyRequest(BaseModel):
    frames: list[list[AtomData]]
    model: str = "mace-mp-0a"
    include_forces: bool = Field(True, alias="includeForces")

    model_config = {"populate_by_name": True}


class WebSearchRequest(BaseModel):
    query: str
    search_depth: str = "basic"
    max_results: int = 5
    topic: str = "general"


class PythonExecRequest(BaseModel):
    code: str
    description: str = ""
    atoms: list[AtomData] = []
    frames: list[list[AtomData]] = []
    energies: list[float] = []
    frame_infos: list[dict] = Field(default_factory=list, alias="frameInfos")

    model_config = {"populate_by_name": True}


# ─── Chart Service ────────────────────────────────────────────────────────────


class ChartService:
    async def generate_chart(self, data: dict) -> dict:
        return await asyncio.to_thread(self._generate_sync, data)

    def _generate_sync(self, data: dict) -> dict:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_type = data.get("type", "line")
        title = data.get("title", "")
        x_label = data.get("xLabel", "")
        y_label = data.get("yLabel", "")
        x_values = data.get("x", [])
        y_values = data.get("y", [])
        labels = data.get("labels", None)

        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#444")

        colors = ["#667eea", "#f093fb", "#4fd1c5", "#f6ad55", "#fc8181"]

        if chart_type == "line":
            if y_values and isinstance(y_values[0], list):
                for i, series in enumerate(y_values):
                    label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
                    ax.plot(
                        x_values,
                        series,
                        marker="o",
                        color=colors[i % len(colors)],
                        label=label,
                        linewidth=2,
                        markersize=4,
                    )
                ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
            else:
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    color=colors[0],
                    linewidth=2,
                    markersize=4,
                )
        elif chart_type == "bar":
            ax.bar(
                x_values, y_values, color=colors[0], edgecolor="white", linewidth=0.5
            )
        elif chart_type == "scatter":
            ax.scatter(
                x_values, y_values, c=colors[0], s=50, edgecolor="white", linewidth=0.5
            )

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        if x_label:
            ax.set_xlabel(x_label, fontsize=11)
        if y_label:
            ax.set_ylabel(y_label, fontsize=11)

        ax.grid(True, alpha=0.2, color="white")
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {"success": True, "image": img_base64}




# ─── Routes ───────────────────────────────────────────────────────────────────

health_router = APIRouter(tags=["health"])
chat_router = APIRouter(prefix="/ai", tags=["chat"])
mace_router = APIRouter(prefix="/ai/mace", tags=["mace"])
chart_router = APIRouter(prefix="/ai", tags=["chart"])
search_router = APIRouter(prefix="/ai", tags=["search"])
python_router = APIRouter(prefix="/ai", tags=["python"])

_chart_service = ChartService()


# Health
@health_router.get("/health")
async def health():
    return {"status": "ok"}


# Chat
@chat_router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    orchestrator = AgentOrchestrator(
        session_store=request.app.state.session_store,
        prompt_builder=request.app.state.prompt_builder,
        tool_registry=request.app.state.tool_registry,
        settings=request.app.state.settings,
    )

    tool_results = None
    if body.tool_results:
        tool_results = {
            "results": [r.model_dump() for r in body.tool_results.results],
            "assistantMessage": body.tool_results.assistant_message,
        }

    event_stream = orchestrator.run_turn(
        session_id=body.session_id,
        user_message=body.message,
        tool_results=tool_results,
        state=body.state,
        model=body.model,
        thinking_budget=body.thinking_budget,
    )

    return sse_response(event_stream)


@chat_router.post("/clear")
async def clear_history(request: Request, body: dict = {}):
    session_store: SessionStore = request.app.state.session_store
    session_id = body.get("sessionId", "default")
    await session_store.clear(session_id)
    return {"success": True}


# MACE
@mace_router.post("/energy")
async def calculate_energy(request: Request, body: EnergyRequest):
    if not body.atoms:
        raise HTTPException(status_code=400, detail="No atoms provided")
    try:
        service = request.app.state.mace_service
        return await service.calculate_energy(
            atoms_data=[a.model_dump() for a in body.atoms],
            model_id=body.model,
            include_forces=body.include_forces,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mace_router.get("/test")
async def test_mace(request: Request):
    try:
        service = request.app.state.mace_service
        return await service.test()
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@mace_router.get("/device-info")
async def mace_device_info(request: Request):
    try:
        service = request.app.state.mace_service
        return service.device_info()
    except Exception as e:
        return {"error": str(e)}


@mace_router.post("/optimize")
async def optimize_geometry(request: Request, body: OptimizeRequest):
    if not body.atoms:
        raise HTTPException(status_code=400, detail="No atoms provided")
    try:
        service = request.app.state.mace_service
        return await service.optimize_geometry(
            atoms_data=[a.model_dump() for a in body.atoms],
            model_name=body.model,
            fmax=body.fmax,
            max_steps=body.max_steps,
            include_forces=body.include_forces,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mace_router.post("/energy-batch")
async def calculate_energy_batch(request: Request, body: BatchEnergyRequest):
    if not body.frames:
        raise HTTPException(status_code=400, detail="No frames provided")
    try:
        service = request.app.state.mace_service
        frames_data = [[a.model_dump() for a in frame] for frame in body.frames]
        return await service.calculate_energy_batch(
            frames_data=frames_data,
            model_id=body.model,
            include_forces=body.include_forces,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mace_router.post("/md")
async def run_md(request: Request, body: MDRequest):
    if not body.atoms:
        raise HTTPException(status_code=400, detail="No atoms provided")
    try:
        service = request.app.state.mace_service
        return await service.run_md(
            atoms_data=[a.model_dump() for a in body.atoms],
            model_name=body.model,
            temperature=body.temperature,
            timestep=body.timestep,
            friction=body.friction,
            frames=body.frames,
            steps=body.steps,
            save_interval=body.save_interval,
            include_forces=body.include_forces,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@mace_router.post("/optimize/stream")
async def optimize_stream(request: Request, body: OptimizeRequest):
    if not body.atoms:
        raise HTTPException(status_code=400, detail="No atoms provided")
    service = request.app.state.mace_service

    async def event_generator():
        async for item in service.optimize_geometry_stream(
            atoms_data=[a.model_dump() for a in body.atoms],
            model_name=body.model,
            fmax=body.fmax,
            max_steps=body.max_steps,
            include_forces=body.include_forces,
        ):
            yield sse_event(item)

    return sse_response(event_generator())


@mace_router.post("/md/stream")
async def md_stream(request: Request, body: MDRequest):
    if not body.atoms:
        raise HTTPException(status_code=400, detail="No atoms provided")
    service = request.app.state.mace_service

    async def event_generator():
        async for item in service.run_md_stream(
            atoms_data=[a.model_dump() for a in body.atoms],
            model_name=body.model,
            temperature=body.temperature,
            timestep=body.timestep,
            friction=body.friction,
            frames=body.frames,
            steps=body.steps,
            save_interval=body.save_interval,
            include_forces=body.include_forces,
        ):
            yield sse_event(item)

    return sse_response(event_generator())


# Chart
@chart_router.post("/chart")
async def generate_chart(body: dict):
    try:
        return await _chart_service.generate_chart(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Web Search ──────────────────────────────────────────────────────────────


@search_router.post("/web-search")
async def web_search(request: Request, body: WebSearchRequest):
    settings: Settings = request.app.state.settings
    api_key = settings.tavily_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="Web search not configured (TAVILY_API_KEY missing)")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": body.query,
                    "search_depth": body.search_depth,
                    "max_results": min(body.max_results, 10),
                    "topic": body.topic,
                    "include_answer": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return {
            "answer": data.get("answer", ""),
            "results": [
                {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
                for r in data.get("results", [])
            ],
        }
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Search API error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Python Execution ────────────────────────────────────────────────────────


@python_router.post("/python")
async def execute_python(request: Request, body: PythonExecRequest):
    try:
        result = await asyncio.to_thread(_run_python_sync, body)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _run_python_sync(body: PythonExecRequest) -> dict:
    import io
    import contextlib

    import numpy as np

    # Build injected variables
    atoms = [{"element": a.element, "x": a.x, "y": a.y, "z": a.z} for a in body.atoms]

    # Build positions array (n_frames, n_atoms, 3)
    if body.frames:
        positions = np.array([
            [[a.x, a.y, a.z] for a in frame]
            for frame in body.frames
        ], dtype=np.float64)
    elif atoms:
        positions = np.array([[[a["x"], a["y"], a["z"]] for a in atoms]], dtype=np.float64)
    else:
        positions = np.empty((0, 0, 3), dtype=np.float64)

    energies = np.array(body.energies, dtype=np.float64) if body.energies else np.array([], dtype=np.float64)

    # Build frames list with metadata
    frames_list = []
    for i, frame_atoms in enumerate(body.frames or []):
        frame_data = {
            "index": i,
            "atoms": [{"element": a.element, "x": a.x, "y": a.y, "z": a.z} for a in frame_atoms],
        }
        if i < len(body.frame_infos):
            frame_data.update(body.frame_infos[i])
        frames_list.append(frame_data)

    # Extract MD arrays from frame_infos
    steps = np.array([fi.get("step", i) for i, fi in enumerate(body.frame_infos)], dtype=np.float64) if body.frame_infos else np.array([], dtype=np.float64)
    temperatures = np.array([fi.get("temperature", 0) for fi in body.frame_infos], dtype=np.float64) if body.frame_infos else np.array([], dtype=np.float64)

    # Namespace for exec
    import math
    namespace = {
        "atoms": atoms,
        "positions": positions,
        "energies": energies,
        "frames": frames_list,
        "steps": steps,
        "temperatures": temperatures,
        "np": np,
        "math": math,
        "__builtins__": __builtins__,
    }

    # Try importing optional libs
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        namespace["plt"] = plt
    except ImportError:
        pass

    try:
        import scipy
        namespace["scipy"] = scipy
    except ImportError:
        pass

    # Capture stdout
    stdout_buf = io.StringIO()
    figures = []

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(body.code, namespace)

        # Capture any matplotlib figures
        try:
            import matplotlib.pyplot as plt
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", facecolor=fig.get_facecolor() or "#1a1a2e", edgecolor="none", dpi=100, bbox_inches="tight")
                buf.seek(0)
                figures.append(base64.b64encode(buf.read()).decode("utf-8"))
            plt.close("all")
        except Exception:
            pass

        stdout_text = stdout_buf.getvalue()
        # Truncate if too long
        if len(stdout_text) > 50000:
            stdout_text = stdout_text[:50000] + "\n... [output truncated]"

        return {"stdout": stdout_text, "figures": figures}

    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "stdout": stdout_buf.getvalue(), "figures": figures}


# ─── App Factory ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.session_store = SessionStore(
        max_sessions=settings.max_sessions,
        ttl_seconds=settings.session_ttl,
    )
    app.state.tool_registry = build_registry()
    app.state.prompt_builder = PromptBuilder()
    app.state.settings = settings

    # Eagerly create and warm up MACE so first request is fast
    mace_service = MACEService(
        device=settings.mace_device, compile_mode=settings.mace_compile_mode
    )
    app.state.mace_service = mace_service
    await asyncio.to_thread(mace_service.warmup, "mace-mp-0a")

    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ChopChopMol AI Backend",
        version="3.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(mace_router)
    app.include_router(chart_router)
    app.include_router(search_router)
    app.include_router(python_router)

    return app


app = create_app()
