"""Local dev server for the Motion Caddie coaching-chatbot demo.

Serves the static demo page + the uploaded-swing videos AND exposes a LIVE
endpoint that runs the real tool-calling agent (`Scripts/coaching_chat.py`)
through the authed **Codex** CLI — so the browser talks to the actual agent,
not the in-page mock.

    ./.venv/Scripts/python.exe serve_demo.py           # http://localhost:8000
    ./.venv/Scripts/python.exe serve_demo.py --port 8080 --backend codex

Routes:
  GET  /                     -> coaching_chatbot_demo.html
  GET  /api/clips            -> the user's "uploaded" swing library (video + summary)
  POST /api/chat             -> {clip_id, question, history[], compare_clip_id?} -> grounded answer
  GET  /Data/... , *.mp4 ... -> static files (overlay videos, etc.)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "Scripts"))
sys.path.insert(0, str(ROOT / "deploy"))
import coaching_chat as C          # noqa: E402
import chat_handler as H           # noqa: E402

PAGE = "coaching_chatbot_demo.html"
LIBRARY_IDS = [0, 2, 4, 6, 8, 10, 1292]  # the "already uploaded" swings
BACKEND = "codex"                        # overridden by --backend

# fake but plausible upload dates so the library reads like a real history
_UPLOAD_DATES = {0: "Jun 2 2026", 2: "Jun 9 2026", 4: "Jun 14 2026", 6: "Jun 18 2026",
                 8: "Jun 23 2026", 10: "Jun 27 2026", 1292: "Jun 30 2026"}


def _video_for(cid: int) -> str | None:
    for p in (f"Data/handoff/{cid}/{cid}_overlay.mp4", f"Data/demo/{cid}/{cid}_overlay.mp4"):
        if (ROOT / p).exists():
            return p
    return None


def _library() -> list[dict]:
    items = []
    for cid in LIBRARY_IDS:
        sc_path = H.scorecard_path(cid)
        if not sc_path.exists():
            continue
        sc = json.loads(sc_path.read_text(encoding="utf-8"))
        meta_path = sc_path.parent / "_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else sc.get("meta", {})
        conf = C._load_confidence()
        inds = sc["indicators"]
        reliable = [k for k in inds if (inds[k].get("confidence_tier")
                    or conf.get(k, {}).get("tier", "med")) != "low"]
        in_range = [k for k in reliable
                    if inds[k]["pro_band"][0] <= inds[k]["value"] <= inds[k]["pro_band"][1]]
        flags = [f["indicator"] for f in sc.get("feedback", []) if f.get("severity") == "review"]
        items.append({
            "clip_id": cid,
            "player": str(meta.get("player", cid)),
            "club": str(meta.get("club", "driver")),
            "view": str(meta.get("view", "down-the-line")),
            "date": _UPLOAD_DATES.get(cid, ""),
            "video": _video_for(cid),
            "n_in_range": len(in_range),
            "n_reliable": len(reliable),
            "n_flagged": len(flags),
        })
    return items


class Handler(SimpleHTTPRequestHandler):
    def _json(self, code: int, obj: dict):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = self.path.split("?")[0]
        if route == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if route in ("/", "/index.html"):
            self.path = "/" + PAGE
            return super().do_GET()
        if route == "/api/clips":
            return self._json(200, {"clips": _library(), "backend": BACKEND})
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/chat":
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "bad request"})
        try:
            clip_id = int(body.get("clip_id"))
        except (TypeError, ValueError):
            return self._json(400, {"error": "clip_id required"})
        if clip_id not in LIBRARY_IDS:
            return self._json(400, {"error": "unknown clip"})
        question = (body.get("question") or "").strip()
        if not question:
            return self._json(400, {"error": "question required"})
        sc = H.scorecard_path(clip_id)
        if not sc.exists():
            return self._json(404, {"error": "no scorecard"})
        compare = None
        cmp_id = body.get("compare_clip_id")
        if cmp_id and int(cmp_id) in LIBRARY_IDS and int(cmp_id) != clip_id:
            compare = H.scorecard_path(int(cmp_id))
        try:
            out = H.chat_once(sc, question, history=body.get("history"), compare=compare,
                              backend_factory=lambda: C.make_backend(BACKEND))
        except Exception as e:  # keep the server up; report the failure
            print(f"[chat] {type(e).__name__}: {e}", flush=True)
            return self._json(502, {"error": f"agent failed: {type(e).__name__}"})
        return self._json(200, {"clip_id": clip_id, **out})

    def log_message(self, fmt, *args):  # quieter console: only log /api/ requests
        line = args[0] if args else ""
        if isinstance(line, str) and "/api/" in line:
            super().log_message(fmt, *args)


def main():
    global BACKEND
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--backend", choices=["auto", "codex", "anthropic"], default="codex")
    args = ap.parse_args()
    BACKEND = args.backend
    lib = _library()
    print(f"Motion Caddie live demo -> http://localhost:{args.port}")
    print(f"  backend: {BACKEND}   library: {len(lib)} swings "
          f"({', '.join(str(c['clip_id']) for c in lib)})")
    print("  open the URL, pick a swing, and ask away (live answers - Codex takes a few seconds).")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
