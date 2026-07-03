"""Eval harness for the coaching interpretation layer (research->prototype->evaluate loop).

This is FEATURE 1 of the loop: a fair, reusable scorer that any feature variant
is graded by. Design goals:

  - FAIR. Iterate on the DEV split; gate on the held-out TEST split of 200
    (player-disjoint — see build_eval_split.py). A feature ships only if it
    scores >= 0.8 composite on TEST.
  - PLUGGABLE. A "feature" is a small JSON config (Data/coaching/features/<id>.json)
    that overrides how the explanation is generated (prompt rules, KB on/off,
    backend, multimodal). Parallel agents add a feature by dropping in a config
    file — no edits to this core, no merge conflicts.
  - STRUCTURED GROUNDING. Scores off the v2 structured claims (deterministic
    verify_grounding), not fuzzy phrase-matching. Grounding/coverage/leakage are
    mechanically checked.
  - CHEAP TO RE-RUN. Generations cached per (feature, split, clip).

Composite score (each component in [0,1]):
    0.45 * grounded_rate        (no factual contradictions / no low-conf leakage)
  + 0.25 * coverage_mean        (mentions each review-flagged metric)
  + 0.15 * nonprescriptive_rate (no fix-it advice — MVP scope)
  + 0.15 * readable_rate        (Flesch reading-ease >= 60)

Usage:
  python coaching_eval_harness.py --feature baseline_v2 --split dev --workers 6
  python coaching_eval_harness.py --feature baseline_v2 --split test          # the 200-clip gate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
COACH = PROJECT_ROOT / "Data" / "coaching"
FEATURES_DIR = COACH / "features"
GEN_DIR = COACH / "eval_gen"
REPORTS_DIR = COACH / "eval_reports"
SPLIT_DIRS = {"dev": COACH / "dev_scorecards", "test": COACH / "test_scorecards"}
FRAME_DIRS = {"dev": COACH / "dev_frames", "test": COACH / "test_frames"}

sys.path.insert(0, str(Path(__file__).parent))
import coaching_llm_summary_v2 as v2

WEIGHTS = {"grounded": 0.45, "coverage": 0.25, "nonprescriptive": 0.15, "readable": 0.15}

PRESCRIPTIVE = [
    r"\byou should\b", r"\btry to\b", r"\btry \b", r"\bwork on\b", r"\bfocus on\b",
    r"\bkeep your\b", r"\bmake sure\b", r"\bto fix\b", r"\bto improve\b", r"\bpractice\b",
    r"\bdrill\b", r"\byou need to\b", r"\baim to\b", r"\bensure\b", r"\bremember to\b",
]

# Built-in default feature = current v2 system (codex + KB + v2 rules).
DEFAULT_FEATURE = {
    "id": "baseline_v2",
    "description": "v2 system as-is: Codex backend, full KB, structured-output grounding.",
    "backend": "codex",
    "model": None,
    "use_kb": True,
    "multimodal": False,
    "rules_override": None,
    "rules_append": None,
}


# --------------------------------------------------------------------------- #
# Readability (Flesch) — local copy so the harness is self-contained
# --------------------------------------------------------------------------- #

def _syllables(word: str) -> int:
    word = word.lower(); v = "aeiouy"; n = 0; prev = False
    for ch in word:
        is_v = ch in v
        if is_v and not prev:
            n += 1
        prev = is_v
    if word.endswith("e") and n > 1:
        n -= 1
    return max(1, n)


def flesch(text: str) -> float:
    sents = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[a-zA-Z]+", text)
    if not words:
        return 0.0
    syl = sum(_syllables(w) for w in words)
    return 206.835 - 1.015 * (len(words) / sents) - 84.6 * (syl / len(words))


# --------------------------------------------------------------------------- #
# Feature config
# --------------------------------------------------------------------------- #

def load_feature(feature_id: str) -> dict:
    if feature_id == "baseline_v2":
        return dict(DEFAULT_FEATURE)
    path = FEATURES_DIR / f"{feature_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"feature config not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg.setdefault("id", feature_id)
    # fill defaults
    for k, val in DEFAULT_FEATURE.items():
        cfg.setdefault(k, val)
    return cfg


def rules_for(cfg: dict) -> str:
    rules = cfg.get("rules_override") or v2.RULES
    if cfg.get("rules_append"):
        rules = rules + "\n" + cfg["rules_append"]
    return rules


# --------------------------------------------------------------------------- #
# Generation (per feature) with caching
# --------------------------------------------------------------------------- #

def generate(cfg: dict, sc: dict, kb: dict, image: str | None) -> dict:
    rules = rules_for(cfg)
    if cfg.get("backend") == "anthropic":
        import anthropic
        system = [{"type": "text", "text": rules}]
        if cfg.get("use_kb", True):
            system.append({"type": "text", "text": v2.build_kb_block(kb),
                           "cache_control": {"type": "ephemeral"}})
        content = []
        if image:
            import base64, mimetypes
            data = Path(image).read_bytes()
            media = mimetypes.guess_type(image)[0] or "image/png"
            content.append({"type": "image", "source": {"type": "base64", "media_type": media,
                            "data": base64.standard_b64encode(data).decode("ascii")}})
        content.append({"type": "text", "text": v2.build_scorecard_text(sc)})
        kwargs = dict(model=cfg.get("model") or v2.DEFAULT_ANTHROPIC_MODEL, max_tokens=4000,
                      system=system, messages=[{"role": "user", "content": content}],
                      output_config={"format": {"type": "json_schema", "schema": v2.OUTPUT_SCHEMA}})
        kwargs["thinking"] = {"type": "adaptive"}
        msg = anthropic.Anthropic().messages.create(**kwargs)
        return v2._coerce_json(next(b.text for b in msg.content if b.type == "text"))
    # codex (default)
    prompt = rules + "\n\n"
    if cfg.get("use_kb", True):
        prompt += v2.build_kb_block(kb) + "\n\n"
    prompt += v2.build_scorecard_text(sc)
    if image:
        prompt += ("\n\n(An image of the SAME swing is attached for visual context only. "
                   "Never let it override or contradict the measured metrics.)")
    return v2.call_codex(prompt, image=image, model=cfg.get("model"))


def config_hash(cfg: dict, kb: dict) -> str:
    """Short hash of everything that affects generation, so editing a feature's
    prompt/KB/backend auto-invalidates its cached generations during iteration."""
    sig = json.dumps({
        "rules": rules_for(cfg),
        "use_kb": cfg.get("use_kb", True),
        "kb_version": kb.get("_version"),
        "backend": cfg.get("backend"),
        "model": cfg.get("model"),
        "multimodal": cfg.get("multimodal", False),
    }, sort_keys=True)
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:10]


def gen_cached(cfg: dict, split: str, cid: int, sc: dict, kb: dict, image: str | None) -> dict | None:
    cdir = GEN_DIR / cfg["id"] / split / config_hash(cfg, kb)
    cdir.mkdir(parents=True, exist_ok=True)
    cache = cdir / f"{cid}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        out = generate(cfg, sc, kb, image)
    except Exception as e:
        print(f"  clip {cid}: generation failed: {repr(e)[:120]}")
        return None
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_clip(sc: dict, out: dict) -> dict:
    g = v2.verify_grounding(sc, out.get("claims", []))
    text = out.get("explanation", "")
    presc = [p for p in PRESCRIPTIVE if re.search(p, text.lower())]
    fre = flesch(text)
    return {
        "grounded": g["grounded"],
        "violations": g["violations"],
        "coverage": g["coverage"],          # None if no review flags
        "n_flagged": g["n_flagged"],
        "prescriptive_count": len(presc),
        "flesch": round(fre, 1),
        "word_count": len(re.findall(r"[a-zA-Z]+", text)),
    }


def composite(per_clip: list[dict]) -> dict:
    n = len(per_clip)
    grounded_rate = sum(c["grounded"] for c in per_clip) / n
    covs = [c["coverage"] for c in per_clip if c["coverage"] is not None]
    coverage_mean = (sum(covs) / len(covs)) if covs else 1.0   # nothing to miss -> 1.0
    nonpresc_rate = sum(c["prescriptive_count"] == 0 for c in per_clip) / n
    readable_rate = sum(c["flesch"] >= 60 for c in per_clip) / n
    comp = (WEIGHTS["grounded"] * grounded_rate
            + WEIGHTS["coverage"] * coverage_mean
            + WEIGHTS["nonprescriptive"] * nonpresc_rate
            + WEIGHTS["readable"] * readable_rate)
    return {
        "composite": round(comp, 4),
        "grounded_rate": round(grounded_rate, 4),
        "coverage_mean": round(coverage_mean, 4),
        "nonprescriptive_rate": round(nonpresc_rate, 4),
        "readable_rate": round(readable_rate, 4),
        "n_clips_with_flags": len(covs),
    }


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def clip_ids_for(split: str, n: int | None) -> list[int]:
    d = SPLIT_DIRS[split]
    ids = sorted(int(p.stem) for p in d.glob("*.json") if p.stem.isdigit())
    return ids[:n] if n else ids


def run(feature_id: str, split: str, n: int | None, workers: int) -> dict:
    cfg = load_feature(feature_id)
    kb = v2.load_kb()
    ids = clip_ids_for(split, n)
    if not ids:
        raise SystemExit(f"no scorecards in {SPLIT_DIRS[split]} — run build_eval_split.py first")
    print(f"[eval] feature={cfg['id']} split={split} clips={len(ids)} workers={workers} "
          f"backend={cfg['backend']} kb={cfg['use_kb']}")

    scs = {cid: json.loads((SPLIT_DIRS[split] / f"{cid}.json").read_text(encoding="utf-8")) for cid in ids}

    def _img_for(cid):
        if not cfg.get("multimodal"):
            return None
        p = FRAME_DIRS[split] / f"{cid}.png"
        return str(p) if p.exists() else None

    def _one(cid):
        out = gen_cached(cfg, split, cid, scs[cid], kb, _img_for(cid))
        if out is None:
            return None
        row = score_clip(scs[cid], out)
        row["clip_id"] = cid
        return row

    per_clip, done = [], 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, cid): cid for cid in ids}
        for fut in as_completed(futs):
            done += 1
            r = fut.result()
            if r is not None:
                per_clip.append(r)
            if done % 20 == 0:
                print(f"  {done}/{len(ids)}", flush=True)

    per_clip.sort(key=lambda r: r["clip_id"])
    agg = composite(per_clip)
    report = {
        "feature_id": cfg["id"], "description": cfg.get("description", ""),
        "split": split, "weights": WEIGHTS,
        "n_evaluated": len(per_clip), "n_requested": len(ids),
        "scores": agg,
        "passes_0.8": agg["composite"] >= 0.8,
        "config": cfg,
        "per_clip": per_clip,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"{cfg['id']}__{split}.json"
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  {cfg['id']}  [{split}]  n={len(per_clip)}")
    print("=" * 60)
    for k in ("grounded_rate", "coverage_mean", "nonprescriptive_rate", "readable_rate"):
        print(f"  {k:22s}: {agg[k]}")
    print(f"  {'COMPOSITE':22s}: {agg['composite']}   {'PASS (>=0.8)' if report['passes_0.8'] else 'below 0.8'}")
    print(f"\n[eval] wrote {out_path}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature", default="baseline_v2")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--n", type=int, default=None, help="cap clips (quick runs)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    run(args.feature, args.split, args.n, args.workers)
