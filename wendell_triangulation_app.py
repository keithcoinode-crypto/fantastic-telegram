from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

APP_NAME = "Wendell Triangulation Engine"
APP_VERSION = "1.3.0-owner-auth"

import os
OWNER_USERNAME = os.getenv("OWNER_USERNAME", "keith")
OWNER_PASSWORD = os.getenv("OWNER_PASSWORD", "change-this-now")

START_TIME = time.time()
SESSIONS: Dict[str, Dict[str, Any]] = {}
CASE_STORE: Dict[str, "TriangulationResult"] = {}

SUSPICIOUS_TAGS = {"mixer", "peel_chain", "bridge", "privacy_pool", "obfuscation"}
STRUCTURING_RANGE = (9000, 10000)
RISK_THRESHOLDS = {
    "low": 0.30,
    "monitor": 0.50,
    "elevated": 0.70,
    "high": 0.85,
}

from security_patch import secure_app
app = secure_app(APP_NAME, APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class TransactionEvent(BaseModel):
    tx_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    wallet_address: str
    amount: float = 0.0
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    counterparty: Optional[str] = None
    asset: str = "BTC"
    direction: str = "out"
    tags: List[str] = Field(default_factory=list)


class TriangulationRequest(BaseModel):
    wallet_address: str
    events: List[TransactionEvent] = Field(default_factory=list)
    watchlist_hits: int = 0
    sanctions_hits: int = 0
    edgar_entity_hits: int = 0
    finra_entity_hits: int = 0
    geo_variance_score: float = 0.0
    device_variance_score: float = 0.0
    manual_signals: Dict[str, float] = Field(default_factory=dict)
    case_id: Optional[str] = None


@dataclass
class TriangulationResult:
    case_id: str
    wallet_address: str
    risk_score: float
    confidence: float
    trust_score: float
    risk_class: str
    flags: List[str] = field(default_factory=list)
    components: Dict[str, float] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    proof_hash: str = ""
    created_at: float = field(default_factory=time.time)
    recognized_owner: bool = False


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def require_session(x_session_token: Optional[str]) -> Dict[str, Any]:
    if not x_session_token:
        raise HTTPException(status_code=401, detail="Missing session token")
    session = SESSIONS.get(x_session_token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")
    return session


def build_proof_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.blake2b(raw, digest_size=32).hexdigest()


def velocity_score(events: List[TransactionEvent]) -> float:
    if not events:
        return 0.0
    timestamps = sorted(e.timestamp for e in events)
    if len(timestamps) == 1:
        return 0.1
    deltas = [max(1, timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]
    avg_delta = sum(deltas) / len(deltas)
    intensity = min(len(events) / 25.0, 1.0)
    burstiness = 1.0 - min(avg_delta / 86400.0, 1.0)
    return clamp((0.55 * intensity) + (0.45 * burstiness))


def amount_anomaly_score(events: List[TransactionEvent]) -> float:
    if not events:
        return 0.0
    amounts = [abs(e.amount) for e in events]
    avg = sum(amounts) / len(amounts) if amounts else 0.0
    peak = max(amounts) if amounts else 0.0
    if avg == 0:
        return 0.0
    ratio = peak / max(avg, 1e-9)
    structured = sum(
        1 for a in amounts if STRUCTURING_RANGE[0] <= a <= STRUCTURING_RANGE[1]
    ) / len(amounts)
    return clamp(min(ratio / 10.0, 1.0) * 0.7 + structured * 0.3)


def graph_risk_score(
    events: List[TransactionEvent],
    watchlist_hits: int,
    sanctions_hits: int,
    finra_entity_hits: int,
    edgar_entity_hits: int,
) -> float:
    counterparties = {e.counterparty for e in events if e.counterparty}
    fanout = min(len(counterparties) / 20.0, 1.0)
    tags = [tag.lower() for e in events for tag in e.tags]
    suspicious_hits = sum(1 for t in tags if t in SUSPICIOUS_TAGS)
    tag_score = min(suspicious_hits / 5.0, 1.0)
    watch_score = min(watchlist_hits / 3.0, 1.0)
    sanctions_score = min(sanctions_hits / 1.0, 1.0)
    finra_score = min(finra_entity_hits / 3.0, 1.0)
    edgar_score = min(edgar_entity_hits / 3.0, 1.0)
    score = (
        0.25 * fanout
        + 0.20 * tag_score
        + 0.18 * watch_score
        + 0.17 * sanctions_score
        + 0.10 * finra_score
        + 0.10 * edgar_score
    )
    return clamp(score)


def same_minute_cluster_score(events: List[TransactionEvent]) -> float:
    buckets: Dict[int, int] = {}
    for e in events:
        minute = e.timestamp // 60
        buckets[minute] = buckets.get(minute, 0) + 1
    if not buckets:
        return 0.0
    clustered = sum(1 for count in buckets.values() if count >= 3)
    return min(clustered / max(len(buckets), 1), 1.0)


def temporal_pattern_score(events: List[TransactionEvent]) -> float:
    if len(events) < 3:
        return 0.0
    hours = [time.gmtime(e.timestamp).tm_hour for e in events]
    overnight = sum(1 for h in hours if h < 5 or h > 22) / len(hours)
    clustered = same_minute_cluster_score(events)
    return clamp(0.55 * overnight + 0.45 * clustered)


def identity_variance_score(geo_variance: float, device_variance: float) -> float:
    return clamp(0.55 * geo_variance + 0.45 * device_variance)


def manual_signal_score(signals: Dict[str, float]) -> float:
    if not signals:
        return 0.0
    vals = [clamp(float(v)) for v in signals.values()]
    return sum(vals) / len(vals)


def confidence_score(req: TriangulationRequest, components: Dict[str, float]) -> float:
    data_depth = min(len(req.events) / 15.0, 1.0)
    component_strength = sum(components.values()) / max(len(components), 1)
    return clamp(0.45 + 0.35 * data_depth + 0.20 * component_strength, 0.30, 0.99)


def classify_risk(risk_score: float) -> str:
    if risk_score >= RISK_THRESHOLDS["high"]:
        return "high_risk"
    if risk_score >= RISK_THRESHOLDS["elevated"]:
        return "elevated_risk"
    if risk_score >= RISK_THRESHOLDS["monitor"]:
        return "monitor"
    return "low_risk"


def build_flags(components: Dict[str, float], req: TriangulationRequest) -> List[str]:
    flags: List[str] = []
    if components["velocity"] > 0.7:
        flags.append("high_velocity_flow")
    if components["amount_anomaly"] > 0.65:
        flags.append("amount_anomaly")
    if components["graph_risk"] > 0.7:
        flags.append("graph_obfuscation")
    if components["temporal_pattern"] > 0.55:
        flags.append("temporal_layering")
    if components["identity_variance"] > 0.6:
        flags.append("identity_variance")
    if req.sanctions_hits > 0:
        flags.append("sanctions_exposure")
    if req.watchlist_hits > 0:
        flags.append("watchlist_exposure")
    if req.finra_entity_hits > 0:
        flags.append("finra_exposure")
    if req.edgar_entity_hits > 0:
        flags.append("edgar_exposure")
    if not flags and req.events:
        flags.append("monitor")
    return flags


def build_explanation(components: Dict[str, float], flags: List[str]) -> List[str]:
    explanation: List[str] = []
    if components["amount_anomaly"] > 0.6:
        explanation.append("Large or structured transaction behavior detected")
    if components["graph_risk"] > 0.6:
        explanation.append("Counterparty graph indicates obfuscation or regulatory linkage")
    if components["velocity"] > 0.6:
        explanation.append("High transaction velocity suggests rapid movement")
    if components["temporal_pattern"] > 0.5:
        explanation.append("Temporal clustering suggests layering or coordinated timing")
    if components["identity_variance"] > 0.5:
        explanation.append("Geo/device variance indicates identity inconsistency")
    if "sanctions_exposure" in flags:
        explanation.append("Sanctions exposure detected")
    if "watchlist_exposure" in flags:
        explanation.append("Watchlist exposure detected")
    if "finra_exposure" in flags:
        explanation.append("FINRA-related exposure signal detected")
    if "edgar_exposure" in flags:
        explanation.append("EDGAR-related entity exposure signal detected")
    if not explanation:
        explanation.append("No major risk indicators detected")
    return explanation


def analyze(req: TriangulationRequest, recognized_owner: bool = False) -> TriangulationResult:
    components = {
        "velocity": velocity_score(req.events),
        "amount_anomaly": amount_anomaly_score(req.events),
        "graph_risk": graph_risk_score(
            req.events,
            req.watchlist_hits,
            req.sanctions_hits,
            req.finra_entity_hits,
            req.edgar_entity_hits,
        ),
        "temporal_pattern": temporal_pattern_score(req.events),
        "identity_variance": identity_variance_score(
            req.geo_variance_score, req.device_variance_score
        ),
        "manual_signal": manual_signal_score(req.manual_signals),
    }

    weights = {
        "velocity": 0.16,
        "amount_anomaly": 0.18,
        "graph_risk": 0.26,
        "temporal_pattern": 0.14,
        "identity_variance": 0.12,
        "manual_signal": 0.14,
    }

    risk_score = clamp(sum(components[k] * weights[k] for k in components))
    confidence = confidence_score(req, components)
    trust_score = round(1.0 - risk_score, 6)
    risk_class = classify_risk(risk_score)
    flags = build_flags(components, req)
    flags.append(risk_class)
    explanation = build_explanation(components, flags)
    case_id = req.case_id or f"case_{uuid.uuid4().hex[:12]}"

    proof_payload = {
        "case_id": case_id,
        "wallet_address": req.wallet_address,
        "risk_score": round(risk_score, 6),
        "confidence": round(confidence, 6),
        "flags": flags,
        "components": components,
        "event_count": len(req.events),
        "recognized_owner": recognized_owner,
    }

    result = TriangulationResult(
        case_id=case_id,
        wallet_address=req.wallet_address,
        risk_score=round(risk_score, 6),
        confidence=round(confidence, 6),
        trust_score=trust_score,
        risk_class=risk_class,
        flags=flags,
        components={k: round(v, 6) for k, v in components.items()},
        explanation=explanation,
        proof_hash=build_proof_hash(proof_payload),
        recognized_owner=recognized_owner,
    )
    CASE_STORE[case_id] = result
    return result


@app.get("/")
def home() -> Dict[str, Any]:
    return {
        "status": "Wendell Running",
        "app": APP_NAME,
        "version": APP_VERSION,
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/login")
def login(req: LoginRequest) -> Dict[str, Any]:
    if req.username != OWNER_USERNAME or req.password != OWNER_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = secrets.token_hex(24)
    SESSIONS[token] = {
        "username": OWNER_USERNAME,
        "role": "owner",
        "created_at": time.time(),
    }
    return {
        "message": "Login successful",
        "session_token": token,
        "profile": {
            "username": OWNER_USERNAME,
            "display_name": "Keith",
            "role": "owner",
            "recognized": True,
        },
    }


@app.get("/me")
def me(x_session_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    session = require_session(x_session_token)
    return {
        "username": session["username"],
        "role": session["role"],
        "recognized": True,
    }


@app.post("/triangulate")
def triangulate(
    req: TriangulationRequest,
    x_session_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    session = require_session(x_session_token)
    recognized_owner = session["username"] == OWNER_USERNAME
    return asdict(analyze(req, recognized_owner=recognized_owner))


@app.get("/cases")
def cases(x_session_token: Optional[str] = Header(default=None)) -> List[Dict[str, Any]]:
    require_session(x_session_token)
    values = sorted(CASE_STORE.values(), key=lambda c: c.created_at, reverse=True)
    return [asdict(v) for v in values[:25]]


@app.get("/cases/{case_id}")
def case_detail(
    case_id: str,
    x_session_token: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    require_session(x_session_token)
    if case_id not in CASE_STORE:
        raise HTTPException(status_code=404, detail="Case not found")
    return asdict(CASE_STORE[case_id])
# ============================================================
# ELM LEARNING ENGINE (Backend Training + Certification)
# ============================================================

class Lesson(BaseModel):
    lesson_id: str
    title: str
    content: str
    duration_minutes: int = 10


class Module(BaseModel):
    module_id: str
    title: str
    description: str
    difficulty: str = "beginner"
    lessons: List[Lesson] = Field(default_factory=list)


class LearningPath(BaseModel):
    path_id: str
    title: str
    description: str
    modules: List[Module] = Field(default_factory=list)


class ProgressRecord(BaseModel):
    user_id: str
    path_id: str
    completed_lessons: List[str] = Field(default_factory=list)
    completed_modules: List[str] = Field(default_factory=list)
    score: float = 0.0
    certificate_awarded: bool = False


LEARNING_PATHS: Dict[str, LearningPath] = {}
PROGRESS_STORE: Dict[str, ProgressRecord] = {}


def seed_learning_paths() -> None:
    if LEARNING_PATHS:
        return

    basics = LearningPath(
        path_id="computer-literacy-001",
        title="Computer Literacy Foundations",
        description="Learn basic terminal, files, and app usage.",
        modules=[
            Module(
                module_id="module-1",
                title="File Navigation",
                description="Basic folder and file commands",
                lessons=[
                    Lesson(
                        lesson_id="lesson-1",
                        title="Using ls and pwd",
                        content="Use pwd to show location. Use ls to list files.",
                    ),
                    Lesson(
                        lesson_id="lesson-2",
                        title="Using cd",
                        content="Use cd foldername and cd .. to navigate.",
                    ),
                ],
            ),
            Module(
                module_id="module-2",
                title="Running Python",
                description="Execute scripts and run servers",
                lessons=[
                    Lesson(
                        lesson_id="lesson-3",
                        title="Run Python file",
                        content="Use python file.py",
                    ),
                    Lesson(
                        lesson_id="lesson-4",
                        title="Run FastAPI",
                        content="Use uvicorn main:app --host 0.0.0.0 --port 8000",
                    ),
                ],
            ),
        ],
    )

    azure = LearningPath(
        path_id="azure-deploy-001",
        title="Azure Deployment",
        description="Deploy backend to Azure App Service",
        modules=[
            Module(
                module_id="module-3",
                title="Prep Backend",
                description="Prepare files for deployment",
                lessons=[
                    Lesson(
                        lesson_id="lesson-5",
                        title="Create main.py",
                        content="Wrapper file for Azure startup",
                    ),
                    Lesson(
                        lesson_id="lesson-6",
                        title="Requirements",
                        content="Add fastapi, uvicorn, gunicorn",
                    ),
                ],
            )
        ],
    )



# ============================================================
# ELM LEARNING ENGINE (Backend Training + Certification)
# ============================================================

class Lesson(BaseModel):
    lesson_id: str
    title: str
    content: str
    duration_minutes: int = 10


class Module(BaseModel):
    module_id: str
    title: str
    description: str
    difficulty: str = "beginner"
    lessons: List[Lesson] = Field(default_factory=list)


class LearningPath(BaseModel):
    path_id: str
    title: str
    description: str
    modules: List[Module] = Field(default_factory=list)


class ProgressRecord(BaseModel):
    user_id: str
    path_id: str
    completed_lessons: List[str] = Field(default_factory=list)
    completed_modules: List[str] = Field(default_factory=list)
    score: float = 0.0
    certificate_awarded: bool = False


LEARNING_PATHS: Dict[str, LearningPath] = {}
PROGRESS_STORE: Dict[str, ProgressRecord] = {}


def seed_learning_paths() -> None:
    if LEARNING_PATHS:
        return

    basics = LearningPath(
        path_id="computer-literacy-001",
        title="Computer Literacy Foundations",
        description="Learn basic terminal, files, and app usage.",
        modules=[
            Module(
                module_id="module-1",
                title="File Navigation",
                description="Basic folder and file commands",
                lessons=[
                    Lesson(
                        lesson_id="lesson-1",
                        title="Using ls and pwd",
                        content="Use pwd to show location. Use ls to list files.",
                    ),
                    Lesson(
                        lesson_id="lesson-2",
                        title="Using cd",
                        content="Use cd foldername and cd .. to navigate.",
                    ),
                ],
            ),
            Module(
                module_id="module-2",
                title="Running Python",
                description="Execute scripts and run servers",
                lessons=[
                    Lesson(
                        lesson_id="lesson-3",
                        title="Run Python file",
                        content="Use python file.py",
                    ),
                    Lesson(
                        lesson_id="lesson-4",
                        title="Run FastAPI",
                        content="Use uvicorn main:app --host 0.0.0.0 --port 8000",
                    ),
                ],
            ),
        ],
    )

    azure = LearningPath(
        path_id="azure-deploy-001",
        title="Azure Deployment",
        description="Deploy backend to Azure App Service",
        modules=[
            Module(
                module_id="module-3",
                title="Prep Backend",
                description="Prepare files for deployment",
                lessons=[
                    Lesson(
                        lesson_id="lesson-5",
                        title="Create main.py",
                        content="Wrapper file for Azure startup",
                    ),
                    Lesson(
                        lesson_id="lesson-6",
                        title="Requirements",
                        content="Add fastapi, uvicorn, gunicorn",
                    ),
                ],
            )
        ],
    )

    LEARNING_PATHS[basics.path_id] = basics
    LEARNING_PATHS[azure.path_id] = azure


seed_learning_paths()


@app.get("/learning/paths")
def list_learning_paths():
    return {"paths": [p.model_dump() for p in LEARNING_PATHS.values()]}


@app.get("/learning/paths/{path_id}")
def get_learning_path(path_id: str):
    path = LEARNING_PATHS.get(path_id)
    if not path:
        raise HTTPException(status_code=404, detail="Path not found")
    return path.model_dump()


@app.post("/learning/progress/{user_id}/{path_id}")
def create_progress(user_id: str, path_id: str):
    if path_id not in LEARNING_PATHS:
        raise HTTPException(status_code=404, detail="Path not found")

    key = f"{user_id}:{path_id}"
    if key not in PROGRESS_STORE:
        PROGRESS_STORE[key] = ProgressRecord(user_id=user_id, path_id=path_id)

    return PROGRESS_STORE[key].model_dump()


@app.get("/learning/progress/{user_id}/{path_id}")
def get_progress(user_id: str, path_id: str):
    key = f"{user_id}:{path_id}"
    progress = PROGRESS_STORE.get(key)
    if not progress:
        raise HTTPException(status_code=404, detail="No progress found")
    return progress.model_dump()


@app.post("/learning/progress/{user_id}/{path_id}/complete/{lesson_id}")
def complete_lesson(user_id: str, path_id: str, lesson_id: str):
    key = f"{user_id}:{path_id}"

    progress = PROGRESS_STORE.get(key)
    if not progress:
        progress = ProgressRecord(user_id=user_id, path_id=path_id)
        PROGRESS_STORE[key] = progress

    if lesson_id not in progress.completed_lessons:
        progress.completed_lessons.append(lesson_id)

    total_lessons = sum(len(m.lessons) for m in LEARNING_PATHS[path_id].modules)
    completed = len(progress.completed_lessons)

    progress.score = round((completed / max(total_lessons, 1)) * 100, 2)

    if completed >= total_lessons:
        progress.certificate_awarded = True

    return progress.model_dump()
