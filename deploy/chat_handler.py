"""AWS Lambda `/chat` adapter for the Motion Caddie coaching chatbot.

Thin, stateless adapter around Scripts/coaching_chat.py. One HTTP request = one
user turn. The browser holds the conversation and posts it back each turn; this
handler reconstructs a SwingContext from the (already-cached) scorecard for the
chosen clip, runs the tool-use loop, verifies grounding, and returns a small JSON.

TRUST BOUNDARY (important): the client's posted history is treated as UNTRUSTED
conversational context only. We accept prior turns as **plain text** (user
questions + the assistant's prior answers) and DROP any client-supplied tool_use /
tool_result blocks. Every number in the new answer is re-fetched via server-side
tools this turn and re-verified — so a forged history cannot inject a fake
measurement. See CHATBOT_PLAN.md §5 and the AWS handoff `/chat` section.

Local/offline testing: `chat_once(...)` takes a `backend_factory`, so tests inject
a ScriptedBackend (no ANTHROPIC_API_KEY, no network). Prod uses AnthropicBackend.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(ROOT / "Scripts"))
import coaching_chat as C  # noqa: E402

# ---- cost / abuse caps (a /chat turn is several model round-trips) --------- #
MAX_HISTORY_MSGS = int(os.environ.get("CHAT_MAX_HISTORY_MSGS", "20"))   # ~10 prior turns
MAX_QUESTION_CHARS = int(os.environ.get("CHAT_MAX_QUESTION_CHARS", "500"))
MAX_BODY_BYTES = int(os.environ.get("CHAT_MAX_BODY_BYTES", "32768"))
ALLOWED_CLIPS = set(json.loads(os.environ.get("ALLOWED_CLIPS", "[]")))

# where the demo path writes per-clip scorecards (built CPU-only from cached pose)
SCORECARD_DIR = ROOT / "Data" / "demo"

# processed UPLOADS: the pipeline publishes 03_outputs/<job>/scorecard.json,
# served publicly through the site's CloudFront distribution — fetch over HTTPS
# (no cross-bucket IAM needed) and cache in /tmp for the warm Lambda.
JOB_SCORECARD_BASE = os.environ.get("JOB_SCORECARD_BASE", "").rstrip("/")
_JOB_ID = __import__("re").compile(r"^[0-9a-f]{32}$")


def job_scorecard_path(job_id: str):
    """Fetch + cache an uploaded swing's scorecard. Returns a Path or None.
    CloudFront rewrites unknown paths to 200/index.html, so a JSON parse is the
    existence check — HTML means 'no scorecard for this job'."""
    import urllib.request
    cache = Path("/tmp") / f"job_scorecard_{job_id}.json"
    if cache.exists():
        return cache
    try:
        with urllib.request.urlopen(f"{JOB_SCORECARD_BASE}/{job_id}/scorecard.json",
                                    timeout=10) as r:
            raw = r.read()
        json.loads(raw)                      # reject the index.html rewrite
        cache.write_bytes(raw)
        return cache
    except Exception as e:
        print(f"[chat] job scorecard fetch failed for {job_id}: {e}")
        return None


def _sanitize_history(raw) -> list[dict]:
    """Keep only alternating user/assistant TEXT turns. Drop tool blocks and any
    non-string content the client may have injected."""
    clean: list[dict] = []
    for m in (raw or [])[-MAX_HISTORY_MSGS:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            clean.append({"role": role, "content": content.strip()[:MAX_QUESTION_CHARS * 4]})
    # ensure it doesn't end on a user turn (the new question is the user turn)
    while clean and clean[-1]["role"] == "user":
        clean.pop()
    return clean


def scorecard_path(clip_id: int) -> Path:
    return SCORECARD_DIR / str(clip_id) / f"{clip_id}_scorecard.json"


def chat_once(scorecard: str | Path, question: str, history: list[dict] | None = None,
              compare: str | Path | None = None,
              backend_factory: Callable[[], C.Backend] = C.AnthropicBackend) -> dict:
    """Run ONE grounded chat turn. Pure core — no HTTP, injectable backend."""
    if not question or not question.strip():
        return {"error": "empty question"}
    ctx = C.SwingContext.from_files(scorecard, compare)
    convo = C.Conversation(ctx, backend_factory())
    convo.messages = _sanitize_history(history)  # untrusted text-only context
    res = convo.ask(question.strip()[:MAX_QUESTION_CHARS])
    grounding = C.verify_chat_grounding(ctx, res)
    return {
        "answer": res.answer,
        "grounded": grounding["grounded"],
        "violations": grounding["violations"],
        "tools_used": [e["name"] for e in res.tool_log],
        # full trace (name + input + result) so the UI can show what each call returned
        "tool_log": [{"name": e["name"], "input": e["input"], "result": e["result"]}
                     for e in res.tool_log],
        "iterations": res.iterations,
        "stop": res.stopped_reason,
        # NB: full history is NOT stored server-side — the client appends this
        # answer to its transcript and posts it back next turn.
    }


def _resp(code: int, body: dict) -> dict:
    return {"statusCode": code,
            "headers": {"content-type": "application/json",
                        "access-control-allow-origin": "*"},
            "body": json.dumps(body)}


def handler(event, _ctx=None):
    """Lambda Function URL adapter: POST {clip_id, question, history:[{role,content}]}."""
    raw = event.get("body") or "{}"
    if len(raw.encode("utf-8")) > MAX_BODY_BYTES:
        return _resp(413, {"error": "request too large"})
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        return _resp(400, {"error": "invalid JSON"})

    raw_id = body.get("clip_id")
    clip_id: int | str
    sc = None
    if isinstance(raw_id, str) and _JOB_ID.match(raw_id):
        # a processed UPLOAD (hex job id) — scorecard comes from 03_outputs
        if not JOB_SCORECARD_BASE:
            return _resp(400, {"error": "uploaded-swing chat not enabled"})
        clip_id = raw_id
        sc = job_scorecard_path(raw_id)
        if sc is None:
            return _resp(404, {"error": "no scorecard for this upload (yet)"})
    else:
        try:
            clip_id = int(raw_id)
        except (TypeError, ValueError):
            return _resp(400, {"error": "clip_id required"})
        if ALLOWED_CLIPS and clip_id not in ALLOWED_CLIPS:
            return _resp(400, {"error": f"clip {clip_id} not in demo set"})

    question = (body.get("question") or "").strip()
    if not question:
        return _resp(400, {"error": "question required"})
    if len(question) > MAX_QUESTION_CHARS:
        return _resp(400, {"error": f"question exceeds {MAX_QUESTION_CHARS} chars"})

    if sc is None:
        sc = scorecard_path(clip_id)
    if not sc.exists():
        return _resp(404, {"error": f"no scorecard for clip {clip_id}"})

    compare_id = body.get("compare_clip_id")
    compare = scorecard_path(int(compare_id)) if compare_id else None
    if compare is not None and (not compare.exists() or
                                (ALLOWED_CLIPS and int(compare_id) not in ALLOWED_CLIPS)):
        return _resp(400, {"error": "invalid compare_clip_id"})

    try:
        out = chat_once(sc, question, history=body.get("history"), compare=compare)
    except Exception as e:  # never leak a stack trace; surface a request id in logs
        print(f"[chat] error: {type(e).__name__}: {e}")
        return _resp(502, {"error": "chat backend failed"})
    return _resp(200, {"clip_id": clip_id, **out})
