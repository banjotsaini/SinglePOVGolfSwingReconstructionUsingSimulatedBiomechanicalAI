"""motioncaddie-elevenlabs-tts — narration audio for coach feedback.

Lawrence's ElevenLabs Lambda (Scottish coach voice), extended for the web app:
  * accepts BOTH direct invokes ({narration, swing_id}) and API Gateway HTTP
    API proxy events (same JSON in the request body) — POST /tts on
    motion-caddie-api routes here, CORS handled at the gateway
  * audio keys are content-hashed (audio/<swing_id>/<sha1>.mp3) and checked in
    S3 before calling ElevenLabs, so replaying the same narration is free
  * response includes audio_url served through the site's CloudFront
    distribution (audio/* behavior -> artifacts bucket via OAC)

Env: ELEVENLABS_SECRET_NAME, ELEVENLABS_VOICE_ID, ELEVENLABS_MODEL_ID,
     AUDIO_BUCKET, AUDIO_CDN_BASE (e.g. https://d3oak5k3h8fdvi.cloudfront.net)
"""
import base64
import hashlib
import json
import os
import re
import urllib.error
import urllib.request

import boto3

secrets_client = boto3.client("secretsmanager")
s3_client = boto3.client("s3")

SECRET_NAME = os.environ["ELEVENLABS_SECRET_NAME"]
VOICE_ID = os.environ["ELEVENLABS_VOICE_ID"]
AUDIO_BUCKET = os.environ["AUDIO_BUCKET"]
CDN_BASE = os.environ.get("AUDIO_CDN_BASE", "").rstrip("/")
MODEL_ID = os.environ.get(
    "ELEVENLABS_MODEL_ID",
    "eleven_multilingual_v2",
)

MAX_NARRATION_CHARS = 2400  # ~2.5 min of speech; guards runaway ElevenLabs spend

_api_key_cache = None


def get_api_key() -> str:
    global _api_key_cache
    if _api_key_cache is None:
        response = secrets_client.get_secret_value(SecretId=SECRET_NAME)
        secret = json.loads(response["SecretString"])
        api_key = secret.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is missing from the secret.")
        _api_key_cache = api_key
    return _api_key_cache


def generate_speech(text: str) -> bytes:
    api_key = get_api_key()

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.15,
            "use_speaker_boost": True
        }
    }

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
    )

    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ElevenLabs returned HTTP {error.code}: {error_body}"
        ) from error


def parse_event(event) -> dict:
    """Direct invoke passes the payload as the event itself; API Gateway proxy
    wraps the same JSON in event['body'] (possibly base64-encoded)."""
    if isinstance(event, dict) and "narration" in event:
        return event
    body = (event or {}).get("body")
    if not body:
        return {}
    if (event or {}).get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {}


def respond(status: int, payload: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }


def lambda_handler(event, context):
    params = parse_event(event)
    narration = str(params.get("narration", "")).strip()
    swing_id = str(params.get("swing_id", "test-swing"))

    if not narration:
        return respond(400, {"error": "narration is required"})
    if len(narration) > MAX_NARRATION_CHARS:
        return respond(400, {"error": f"narration too long (max {MAX_NARRATION_CHARS} chars)"})

    # key safety: swing_id comes from the browser on the /tts route
    swing_id = re.sub(r"[^A-Za-z0-9_-]", "", swing_id)[:64] or "unknown"

    # content-addressed key: same narration + voice + model -> same mp3, no
    # second ElevenLabs call and no stale audio when the voice changes
    digest = hashlib.sha1(
        f"{VOICE_ID}|{MODEL_ID}|{narration}".encode("utf-8")
    ).hexdigest()[:16]
    object_key = f"audio/{swing_id}/{digest}.mp3"

    # existence check goes through CloudFront, not s3:HeadObject — the
    # execution role is deliberately write-only on the bucket. The distribution
    # rewrites missing keys to the SPA's index.html with a 200, so a hit only
    # counts when the response is actually audio.
    cached = False
    if CDN_BASE:
        try:
            head = urllib.request.Request(f"{CDN_BASE}/{object_key}", method="HEAD")
            with urllib.request.urlopen(head, timeout=5) as r:
                ctype = r.headers.get("Content-Type", "")
                cached = r.status == 200 and ctype.startswith("audio/")
        except (urllib.error.URLError, OSError):
            pass

    if not cached:
        audio = generate_speech(narration)
        s3_client.put_object(
            Bucket=AUDIO_BUCKET,
            Key=object_key,
            Body=audio,
            ContentType="audio/mpeg"
        )

    return respond(200, {
        "bucket": AUDIO_BUCKET,
        "audio_key": object_key,
        "audio_url": f"{CDN_BASE}/{object_key}" if CDN_BASE else None,
        "cached": cached,
    })
