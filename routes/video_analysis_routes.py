"""Video analysis test API (mounted at ``/api/video``).

``POST /analyze`` takes several short clips, each paired with one instructed
action, and reports per clip whether (and how fast) the action was performed,
plus an overall score. See ``controllers/video_analysis.py`` for the method.
"""

import asyncio
import json
import math
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from controllers.video_analysis import (
    ACTIONS,
    ModelUnavailableError,
    analyze_video,
    ensure_model,
    list_actions,
    normalize_action,
    required_models,
    summarize_results,
)

router = APIRouter()

MAX_VIDEOS = 10
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_INSTRUCTION_OFFSET_S = 30.0
UPLOAD_CHUNK_BYTES = 1024 * 1024

# Shared by all requests, so at most 3 clips are analysed at once server-wide.
_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="video-analysis")


def _parse_list(raw: str) -> list:
    """A JSON list, or a comma-separated string."""
    raw = raw.strip()
    if raw.startswith("["):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"invalid JSON list: {exc.msg}") from exc
        if not isinstance(value, list):
            raise HTTPException(400, "expected a JSON list")
        return value
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_actions(raw: str, expected: int) -> list[str]:
    names = _parse_list(raw)
    if len(names) != expected:
        raise HTTPException(
            400, f"got {expected} video(s) but {len(names)} action(s); they must match one to one"
        )
    keys, unsupported = [], []
    for name in names:
        key = normalize_action(str(name))
        if key is None:
            unsupported.append(str(name))
        keys.append(key)
    if unsupported:
        raise HTTPException(
            400, f"unsupported action(s): {unsupported}. Supported: {sorted(ACTIONS)}"
        )
    return keys


def _parse_offsets(raw: str | None, expected: int) -> list[float]:
    if raw is None or not raw.strip():
        return [0.0] * expected
    try:
        single = float(raw)
    except ValueError:
        values = _parse_list(raw)
    else:
        values = [single] * expected
    if len(values) != expected:
        raise HTTPException(400, f"expected {expected} instruction offset(s), got {len(values)}")

    offsets = []
    for value in values:
        try:
            offset = float(value)
        except (TypeError, ValueError):
            raise HTTPException(400, f"invalid instruction offset: {value!r}") from None
        if not math.isfinite(offset) or not 0.0 <= offset <= MAX_INSTRUCTION_OFFSET_S:
            raise HTTPException(
                400, f"instruction offsets must be between 0 and {MAX_INSTRUCTION_OFFSET_S:g} s"
            )
        offsets.append(offset)
    return offsets


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix if re.fullmatch(r"\.[a-z0-9]{1,5}", suffix) else ".video"


async def _save_upload(upload: UploadFile, index: int) -> str:
    """Stream an upload to a temp file, enforcing the size limit."""
    fd, path = tempfile.mkstemp(prefix="neurosense_video_", suffix=_safe_suffix(upload.filename))
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_VIDEO_BYTES:
                    raise HTTPException(
                        400,
                        f"video {index + 1} ('{upload.filename}') is larger than "
                        f"{MAX_VIDEO_BYTES // (1024 * 1024)} MB",
                    )
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, f"video {index + 1} ('{upload.filename}') is empty")
    except BaseException:
        Path(path).unlink(missing_ok=True)
        raise
    return path


def _ensure_models(action_keys: list[str]) -> None:
    for name in required_models(action_keys):
        ensure_model(name)


@router.get("/actions")
def get_actions():
    """Supported actions with bilingual labels and instructions."""
    return {"actions": list_actions()}


@router.post("/analyze")
async def analyze(
    videos: list[UploadFile] | None = File(None, description="One clip per action"),
    actions: str = Form(..., description='JSON list or comma-separated, e.g. ["look left","close eyes"]'),
    instruction_offsets: str | None = Form(
        None, description="Seconds from clip start to the instruction: one number or a JSON list"
    ),
    mirrored: bool = Form(False, description="True if the clips were recorded mirrored (selfie view)"),
):
    """Analyse clips; results are returned in the same order as the uploads."""
    videos = [v for v in (videos or []) if v.filename]
    if not videos:
        raise HTTPException(400, "at least one video is required")
    if len(videos) > MAX_VIDEOS:
        raise HTTPException(400, f"at most {MAX_VIDEOS} videos per request")
    action_keys = _parse_actions(actions, len(videos))
    offsets = _parse_offsets(instruction_offsets, len(videos))

    try:
        await run_in_threadpool(_ensure_models, action_keys)
    except ModelUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc

    paths: list[str] = []
    try:
        for index, upload in enumerate(videos):
            paths.append(await _save_upload(upload, index))

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(_executor, analyze_video, path, key, offset, mirrored)
            for path, key, offset in zip(paths, action_keys, offsets)
        ]
        # Wait for every clip before cleaning up, even if one of them failed.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for outcome in results:
            if isinstance(outcome, ModelUnavailableError):
                raise HTTPException(503, str(outcome)) from outcome
            if isinstance(outcome, BaseException):
                raise outcome
    finally:
        for path in paths:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:  # e.g. still locked on Windows; the OS temp cleanup handles it
                pass
        for upload in videos:
            await upload.close()

    for upload, result in zip(videos, results):
        result["filename"] = upload.filename
    return {"results": results, "summary": summarize_results(results)}
