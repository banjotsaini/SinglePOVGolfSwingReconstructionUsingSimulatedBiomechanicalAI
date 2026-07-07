"""SQS-triggered processing worker — the container-Lambda backend for real uploads.

Wiring (full_app.yaml): S3 upload to 01_inputs/uploads/<job_id>/<file>  ->  EventBridge
rule  ->  SQS ingest queue  ->  this handler. One message = one uploaded swing.

Per message it:
  1. resolves the uploaded video (bucket/key from the S3 event) and its job_id,
  2. downloads it to /tmp,
  3. runs the SAME pipeline the local demo runs (2D -> 3D lift -> smoothing ->
     overlay + 3D replay; then event detection + coaching scorecard; then the
     grounded Claude eval) as subprocesses — each step is torch-heavy and isolated,
  4. uploads the artifacts (overlay.mp4, replay_3d.json, metrics.json,
     explanation.json, scorecard.json) to 03_outputs/<job_id>/,
  5. writes queryable rows (swings/analyses/indicators/swing_events) via the RDS
     Data API, mirroring deploy/db/load_data.py's backfill shape.

Runs as a CONTAINER image (needs torch/mediapipe/mixste + the model bundle) — this
is the one piece that can't ship as a zip. Gated in full_app.yaml behind
HasProcessingImage until the image is built (deploy/chat/buildspec.yml pattern +
the processing Dockerfile, still to be authored).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import boto3

ARTIFACTS_BUCKET = os.environ.get("ARTIFACTS_BUCKET", "")
DB_CLUSTER_ARN = os.environ.get("DB_CLUSTER_ARN", "")
DB_SECRET_ARN = os.environ.get("DB_SECRET_ARN", "")
DB_NAME = os.environ.get("DB_NAME", "motioncaddie")
APP_ROOT = Path(os.environ.get("APP_ROOT", "/var/task"))
SCRIPTS = APP_ROOT / "Scripts"
PY = sys.executable
BACKBONE = os.environ.get("POSE_BACKBONE", "mediapipe_lite")
LIFTER = os.environ.get("LIFTER", "golfpose3d")

_s3 = boto3.client("s3")
_rd = boto3.client("rds-data") if DB_CLUSTER_ARN else None

# artifacts the front end consumes, keyed by the suffix each pipeline step emits.
ARTIFACTS = ["overlay.mp4", "replay_3d.json", "metrics.json", "explanation.json"]


# --------------------------------------------------------------------------
def _job_from_key(key: str) -> str:
    """01_inputs/uploads/<job_id>/<file>  ->  <job_id>."""
    parts = key.split("/")
    if len(parts) >= 3 and parts[0] == "01_inputs" and parts[1] == "uploads":
        return parts[2]
    return Path(key).stem  # fallback: filename stem


def _iter_s3_records(event: dict):
    """Yield (bucket, key) from an SQS batch whose bodies are S3/EventBridge events."""
    for rec in event.get("Records", []):
        try:
            body = json.loads(rec["body"])
        except (KeyError, json.JSONDecodeError):
            continue
        # EventBridge S3 "Object Created" shape
        if body.get("detail", {}).get("bucket"):
            yield body["detail"]["bucket"]["name"], body["detail"]["object"]["key"]
        # raw S3 notification shape (fallback)
        for r in body.get("Records", []):
            if "s3" in r:
                yield r["s3"]["bucket"]["name"], r["s3"]["object"]["key"]


def _run(cmd: list[str], desc: str) -> None:
    print(f"[proc] {desc}: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=str(SCRIPTS), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{desc} failed (rc={res.returncode}): {res.stderr[-800:]}")


def _pipeline(video: Path, work: Path) -> dict:
    """Run the 3-step pipeline; return the paths of the produced artifacts."""
    stem = video.stem
    _run([PY, str(SCRIPTS / "pipeline.py"), str(video),
          "--backbone", BACKBONE, "--lifter", LIFTER, "--out-dir", str(work)],
         "2D->3D pipeline + overlay + replay")
    parquet_3d = work / f"{stem}_3d.parquet"
    scorecard = work / f"{stem}_scorecard.json"
    _run([PY, str(SCRIPTS / "scorecard_step.py"), "--parquet", str(parquet_3d),
          "--out", str(scorecard)], "event detection + scorecard")
    # grounded Claude eval (needs ANTHROPIC_API_KEY in env); non-fatal if it fails
    try:
        _run([PY, str(SCRIPTS / "coaching_explain.py"), "--scorecard", str(scorecard),
              "--backend", "anthropic"], "coaching eval")
    except RuntimeError as e:
        print(f"[proc] coaching eval skipped: {e}", flush=True)
    return {"scorecard": scorecard, "work": work, "stem": stem}


def _upload_artifacts(work: Path, stem: str, job_id: str) -> list[str]:
    uploaded = []
    for name in ARTIFACTS:
        # pipeline writes some as <stem>_<name>; accept either exact or stem-prefixed
        for cand in (work / name, work / f"{stem}_{name}"):
            if cand.exists():
                dest = f"03_outputs/{job_id}/{name}"
                _s3.upload_file(str(cand), ARTIFACTS_BUCKET, dest)
                uploaded.append(dest)
                break
    return uploaded


def _sql(sql: str, params: list) -> None:
    _rd.execute_statement(resourceArn=DB_CLUSTER_ARN, secretArn=DB_SECRET_ARN,
                          database=DB_NAME, sql=sql, parameters=params)


def _p(name: str, **kv):
    (k, v), = kv.items()
    return {"name": name, "value": {k: v}}


def _write_db(job_id: str, scorecard: Path, raw_key: str) -> None:
    """Insert swings + analyses + indicators rows for the uploaded swing."""
    if _rd is None:
        print("[proc] no DB configured; skipping row write", flush=True)
        return
    sc = json.loads(scorecard.read_text(encoding="utf-8"))
    _sql("""INSERT INTO swings (swing_id, source, raw_video_s3_key, status)
            VALUES (:sid::uuid, 'upload', :key, 'done')
            ON CONFLICT (swing_id) DO NOTHING""",
         [_p("sid", stringValue=_as_uuid(job_id)), _p("key", stringValue=raw_key)])
    _sql("""INSERT INTO analyses (analysis_id, swing_id, pipeline_version, lifter, scorecard_json)
            VALUES (:aid::uuid, :sid::uuid, :ver, :lifter, :blob::jsonb)
            ON CONFLICT (analysis_id) DO NOTHING""",
         [_p("aid", stringValue=_as_uuid(job_id + "a")), _p("sid", stringValue=_as_uuid(job_id)),
          _p("ver", stringValue="mixste+oneeuro@2026-07"), _p("lifter", stringValue=LIFTER),
          _p("blob", stringValue=json.dumps(sc))])
    for ind in sc.get("indicators", []):
        _sql("""INSERT INTO indicators (analysis_id, indicator_key, value, percentile, confidence_tier)
                VALUES (:aid::uuid, :key, :val, :pct, :tier)
                ON CONFLICT (analysis_id, indicator_key) DO NOTHING""",
             [_p("aid", stringValue=_as_uuid(job_id + "a")), _p("key", stringValue=ind["indicator"]),
              _p("val", doubleValue=float(ind.get("value") or 0)),
              _p("pct", doubleValue=float(ind.get("percentile") or 0)),
              _p("tier", stringValue=str(ind.get("confidence_tier") or "unknown"))])


def _as_uuid(seed: str) -> str:
    """Deterministic UUID from the job id so retries are idempotent (uuid5)."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


# --------------------------------------------------------------------------
def handler(event, _ctx=None):
    processed = []
    for bucket, key in _iter_s3_records(event):
        job_id = _job_from_key(key)
        print(f"[proc] job {job_id}: s3://{bucket}/{key}", flush=True)
        work = Path("/tmp") / job_id
        work.mkdir(parents=True, exist_ok=True)
        video = work / Path(key).name
        _s3.download_file(bucket, key, str(video))
        out = _pipeline(video, work)
        uploaded = _upload_artifacts(work, out["stem"], job_id)
        _write_db(job_id, out["scorecard"], key)
        print(f"[proc] job {job_id} done; artifacts: {uploaded}", flush=True)
        processed.append(job_id)
    return {"processed": processed}
