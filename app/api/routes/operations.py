"""Safe, allow-listed operations for the monitoring frontend."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.core.exceptions import NotFoundError

router = APIRouter(prefix="/operations", tags=["operations"])
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


class OperationRequest(BaseModel):
    action: Literal["status", "train_direction", "walk_forward", "regime_report"]
    horizon: Literal["24h", "7d"] = "24h"


class OperationSummary(BaseModel):
    job_id: str
    action: str
    horizon: str
    status: Literal["running", "completed", "failed"]
    started_at: str
    finished_at: str | None = None
    output: str = ""
    return_code: int | None = None


def _command(request: OperationRequest) -> list[str]:
    if request.action == "status":
        return ["scripts/status_report.py"]
    if request.action == "train_direction":
        return ["scripts/train_direction_model.py", "--horizon", request.horizon]
    if request.action == "walk_forward":
        return ["scripts/walk_forward_report.py", "--horizon", request.horizon]
    return ["scripts/regime_report.py", "--horizon", request.horizon]


def _run(job_id: str, request: OperationRequest) -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    process = subprocess.Popen(
        [sys.executable, *_command(request)],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        output.append(line)
        with _lock:
            _jobs[job_id]["output"] = "".join(output)[-12000:]
    return_code = process.wait()
    with _lock:
        _jobs[job_id].update(
            status="completed" if return_code == 0 else "failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            return_code=return_code,
            output="".join(output)[-12000:],
        )


def _check_token(token: str | None) -> None:
    expected = os.getenv("OPERATIONS_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Operations are not configured. Set OPERATIONS_TOKEN on the backend.")
    if token != expected:
        raise HTTPException(status_code=401, detail="Invalid operations token")


@router.post("/run", response_model=OperationSummary, status_code=202)
def run_operation(request: OperationRequest, x_operations_token: str | None = Header(default=None)) -> OperationSummary:
    _check_token(x_operations_token)
    with _lock:
        if any(job["status"] == "running" for job in _jobs.values()):
            raise HTTPException(status_code=409, detail="Another operation is already running")
        job_id = uuid.uuid4().hex[:12]
        _jobs[job_id] = {
            "job_id": job_id,
            "action": request.action,
            "horizon": request.horizon,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "output": "",
            "return_code": None,
        }
    threading.Thread(target=_run, args=(job_id, request), daemon=True).start()
    return OperationSummary(**_jobs[job_id])


@router.get("/status", response_model=list[OperationSummary])
def operation_status() -> list[OperationSummary]:
    with _lock:
        return [OperationSummary(**job) for job in reversed(list(_jobs.values()))]


@router.get("/status/{job_id}", response_model=OperationSummary)
def operation_detail(job_id: str) -> OperationSummary:
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        raise NotFoundError(f"Operation {job_id} not found")
    return OperationSummary(**job)
