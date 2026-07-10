"""Smoke test for the deployed MotionCaddie site + API.

Checks, per demo clip: static assets fetchable, metrics are real (no placeholder
flag), and the live /chat answers with a grounding verdict. Exit 0 = all green.

Usage:  python Scripts/smoke_web.py            # live CloudFront + API Gateway
        python Scripts/smoke_web.py --base http://localhost:8790  # local build
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

SITE = "https://d3oak5k3h8fdvi.cloudfront.net"
API = "https://bk7s56lvq3.execute-api.us-east-1.amazonaws.com"


def get(url: str, timeout: float = 20.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        print(f"  !! {url}: {e}")
        return 0, b""


def post_json(url: str, body: dict, timeout: float = 60.0) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}
    except Exception as e:
        print(f"  !! {url}: {e}")
        return 0, {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=SITE, help="site origin to test")
    ap.add_argument("--api", default=API, help="chat API origin to test")
    ap.add_argument("--skip-chat", action="store_true", help="static assets only")
    args = ap.parse_args()

    failures: list[str] = []
    ok = lambda cond, label: (print(("  ok  " if cond else "  FAIL") + " " + label),
                              None if cond else failures.append(label))

    print(f"[site] {args.base}")
    code, body = get(args.base + "/")
    ok(code == 200 and b"MotionCaddie" in body, "landing page serves")
    code, body = get(args.base + "/assets/clips.json")
    ok(code == 200, "clips.json serves")
    clips = [c["id"] for c in json.loads(body or b'{"clips":[]}')["clips"]]
    ok(len(clips) >= 5, f"manifest lists {len(clips)} clips")

    for c in clips:
        base = f"{args.base}/assets/{c}"
        for f in ("raw.mp4", "overlay.mp4", "replay_3d.json", "metrics.json",
                  "explanation.json"):
            code, body = get(f"{base}/{f}")
            ok(code == 200, f"clip {c}: {f}")
            if f == "explanation.json" and code == 200:
                ok(not json.loads(body).get("placeholder"),
                   f"clip {c}: explanation is real (no placeholder flag)")

    if not args.skip_chat:
        print(f"[chat] {args.api}")
        code, d = post_json(args.api + "/chat", {})
        ok(code == 400, "guardrail rejects empty body (400)")
        for c in clips:
            code, d = post_json(args.api + "/chat",
                                {"clip_id": int(c), "question": "How was my tempo?"})
            ok(code == 200 and "grounded" in d,
               f"clip {c}: live answer with grounding verdict"
               + (f" (grounded={d.get('grounded')})" if code == 200 else ""))

    print(f"\n{'ALL GREEN' if not failures else str(len(failures)) + ' FAILURES'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
