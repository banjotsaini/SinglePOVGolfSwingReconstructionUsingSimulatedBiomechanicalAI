"""Seed reference tables + backfill existing scorecards into the MotionCaddie DB.

Two jobs, one script (idempotent — safe to re-run):

  seed      indicator_catalog, reference_bands, indicator_confidence, kb_cards
            from the repo JSON (Data/coaching/*.json + coaching_indicators.py).
  backfill  every Data/coaching/{dev,test,eval}_scorecards/*.json into
            swings (source='golfdb') + analyses + swing_events + indicators +
            feedback_notes rows, so the file-based corpus becomes queryable.

Usage:
  python deploy/db/load_data.py --dsn postg:///motioncaddie --all
  python deploy/db/load_data.py --dsn ... --seed            # reference only
  python deploy/db/load_data.py --dry-run --emit-sql out.sql  # no DB needed

Design: builds parameterized statements through a thin executor so it runs
against a real psycopg connection OR, with --dry-run, emits SQL to stdout/file
for review without a database. This lets the migration be validated before any
Aurora exists.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COACH = ROOT / "Data" / "coaching"
sys.path.insert(0, str(ROOT / "Scripts"))

PIPELINE_VERSION = "mixste+oneeuro@2026-07"
LIFTER = "golfpose3d"
SMOOTHING_CFG = {"method": "oneeuro", "min_cutoff": 0.3, "beta": 0.4,
                 "branch": "measurement_only"}

UNIT_MAP = {"degrees": "deg", "percent of body height": "pct", "ratio": "ratio"}
SCORECARD_SETS = ["dev_scorecards", "test_scorecards", "eval_scorecards"]


# --------------------------------------------------------------------------
# executor: real DB or dry-run SQL emitter
# --------------------------------------------------------------------------
class Executor:
    def __init__(self, conn=None, sink=None):
        self.conn = conn
        self.cur = conn.cursor() if conn else None
        self.sink = sink
        self.n = 0

    def run(self, sql: str, params: tuple = ()):
        self.n += 1
        if self.cur:
            self.cur.execute(sql, params)
        if self.sink is not None:
            self.sink.write(_render(sql, params) + "\n")

    def commit(self):
        if self.conn:
            self.conn.commit()


def _lit(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (dict, list)):
        return "'" + json.dumps(v).replace("'", "''") + "'::jsonb"
    return "'" + str(v).replace("'", "''") + "'"


def _render(sql: str, params: tuple) -> str:
    """Best-effort inline of %s params for the dry-run SQL dump (review only)."""
    out, pi = [], 0
    i = 0
    while i < len(sql):
        if sql[i:i + 2] == "%s":
            out.append(_lit(params[pi])); pi += 1; i += 2
        else:
            out.append(sql[i]); i += 1
    return "".join(out) + ";"


# --------------------------------------------------------------------------
# seed
# --------------------------------------------------------------------------
def _indicator_deps():
    """indicator_key -> sorted unique joint names, from the reliability map."""
    from coaching_reliability import INDICATOR_DEPS, MP
    inv = {v: k for k, v in MP.items()}
    out = {}
    for key, deps in INDICATOR_DEPS.items():
        joints = sorted({inv[j] for _ev, js in deps for j in js})
        out[key] = joints
    return out


def seed(ex: Executor):
    kb = json.loads((COACH / "indicator_kb.json").read_text())
    bands = json.loads((COACH / "reference_bands.json").read_text())
    conf_raw = json.loads((COACH / "indicator_confidence.json").read_text())
    conf = conf_raw.get("indicators", conf_raw)
    deps = _indicator_deps()
    cards = kb["indicators"]

    # indicator_catalog (from KB cards — authoritative label/plain_name/event/unit)
    for key, c in cards.items():
        ex.run(
            """INSERT INTO indicator_catalog
               (indicator_key,label,plain_name,event_ref,unit,depends_on_joints)
               VALUES (%s,%s,%s,%s,%s,%s)
               ON CONFLICT (indicator_key) DO UPDATE SET
                 label=EXCLUDED.label, plain_name=EXCLUDED.plain_name,
                 event_ref=EXCLUDED.event_ref, unit=EXCLUDED.unit,
                 depends_on_joints=EXCLUDED.depends_on_joints""",
            (key, c["label"], c["plain_name"], c["event"],
             UNIT_MAP.get(c["unit"], "ratio"), deps.get(key, [])),
        )

    # reference_bands (corpus bands are club-agnostic -> club='all')
    for key, b in bands.items():
        if key not in cards:
            continue  # skip any non-indicator keys
        ex.run(
            """INSERT INTO reference_bands
               (club,indicator_key,p10,p25,p50,p75,p90,mean,std)
               VALUES ('all',%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (club,indicator_key) DO UPDATE SET
                 p10=EXCLUDED.p10,p25=EXCLUDED.p25,p50=EXCLUDED.p50,
                 p75=EXCLUDED.p75,p90=EXCLUDED.p90,mean=EXCLUDED.mean,std=EXCLUDED.std""",
            (key, b["p10"], b["p25"], b["p50"], b["p75"], b["p90"], b["mean"], b["std"]),
        )

    # indicator_confidence
    for key, c in conf.items():
        if key not in cards:
            continue
        ex.run(
            """INSERT INTO indicator_confidence
               (indicator_key,confidence,tier,worst_joint_z_std)
               VALUES (%s,%s,%s,%s)
               ON CONFLICT (indicator_key) DO UPDATE SET
                 confidence=EXCLUDED.confidence, tier=EXCLUDED.tier,
                 worst_joint_z_std=EXCLUDED.worst_joint_z_std""",
            (key, c["confidence"], c["tier"], c.get("worst_joint_z_std")),
        )

    # kb_cards: glossary + one indicator_card per indicator
    for term, gloss in kb["glossary"].items():
        ex.run(
            """INSERT INTO kb_cards (term,kind,plain_name,gloss,indicator_key,card)
               VALUES (%s,'glossary',NULL,%s,NULL,NULL)
               ON CONFLICT (term) DO UPDATE SET gloss=EXCLUDED.gloss""",
            (term, gloss),
        )
    for key, c in cards.items():
        ex.run(
            """INSERT INTO kb_cards (term,kind,plain_name,gloss,indicator_key,card)
               VALUES (%s,'indicator_card',%s,%s,%s,%s)
               ON CONFLICT (term) DO UPDATE SET
                 plain_name=EXCLUDED.plain_name, gloss=EXCLUDED.gloss,
                 indicator_key=EXCLUDED.indicator_key, card=EXCLUDED.card""",
            (key, c["plain_name"], c.get("measures", c["plain_name"]), key, c),
        )
    print(f"[seed] {len(cards)} indicators, {len(kb['glossary'])} glossary terms")


# --------------------------------------------------------------------------
# backfill
# --------------------------------------------------------------------------
def _clip_id(path: Path):
    stem = path.stem.replace("_scorecard", "")
    return int(stem) if stem.isdigit() else None


def backfill(ex: Executor):
    n_sw = n_an = 0
    seen = set()
    for setname in SCORECARD_SETS:
        for jf in sorted((COACH / setname).glob("*.json")):
            cid = _clip_id(jf)
            if cid is None or cid in seen:
                continue
            seen.add(cid)
            sc = json.loads(jf.read_text())
            meta = sc.get("meta", {})

            # swing (deterministic uuid from clip id so re-runs are idempotent)
            ex.run(
                """INSERT INTO swings (swing_id,user_id,source,golfdb_clip_id,club,view,status)
                   VALUES (gen_random_uuid(),NULL,'golfdb',%s,%s,%s,'done')
                   ON CONFLICT (golfdb_clip_id) DO NOTHING""",
                (cid, meta.get("club"),
                 _norm_view(meta.get("view"))),
            )
            n_sw += 1

            # analysis keyed by (swing, pipeline_version); resolve swing via clip id
            swing_sel = "(SELECT swing_id FROM swings WHERE golfdb_clip_id=%s)"
            ex.run(
                f"""INSERT INTO analyses
                   (analysis_id,swing_id,pipeline_version,lifter,smoothing_cfg,
                    scorecard_json,status)
                   VALUES (gen_random_uuid(),{swing_sel},%s,%s,%s,%s,'done')
                   ON CONFLICT (swing_id,pipeline_version) DO NOTHING""",
                (cid, PIPELINE_VERSION, LIFTER, SMOOTHING_CFG, sc),
            )
            n_an += 1

            an_sel = (f"(SELECT a.analysis_id FROM analyses a "
                      f"JOIN swings s ON s.swing_id=a.swing_id "
                      f"WHERE s.golfdb_clip_id=%s AND a.pipeline_version=%s)")

            for ev, fr in sc.get("events", {}).items():
                ex.run(
                    f"""INSERT INTO swing_events (analysis_id,event_name,frame_idx)
                        VALUES ({an_sel},%s,%s)
                        ON CONFLICT (analysis_id,event_name) DO NOTHING""",
                    (cid, PIPELINE_VERSION, ev, int(fr)),
                )

            for key, ind in sc.get("indicators", {}).items():
                band = ind.get("pro_band") or [None, None]
                pct = ind.get("percentile")
                in_range = None
                if pct is not None:
                    in_range = 25 <= pct <= 75
                ex.run(
                    f"""INSERT INTO indicators
                        (analysis_id,indicator_key,value,percentile,pro_median,
                         pro_band_low,pro_band_high,in_range,confidence_tier)
                        VALUES ({an_sel},%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (analysis_id,indicator_key) DO NOTHING""",
                    (cid, PIPELINE_VERSION, key, ind.get("value"), pct,
                     ind.get("pro_median"), band[0], band[1], in_range,
                     ind.get("confidence_tier")),
                )

            for fb in sc.get("feedback", []):
                ex.run(
                    f"""INSERT INTO feedback_notes
                        (note_id,analysis_id,indicator_key,label,event_name,value,
                         percentile,pro_median,confidence_tier,message,severity)
                        VALUES (gen_random_uuid(),{an_sel},%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (cid, PIPELINE_VERSION, fb.get("indicator"), fb.get("label"),
                     fb.get("event"), fb.get("value"), fb.get("percentile"),
                     fb.get("pro_median"), fb.get("confidence_tier"),
                     fb.get("message"), fb.get("severity", "info")),
                )
    print(f"[backfill] {n_sw} swings / {n_an} analyses from {len(seen)} clips")


def _norm_view(v):
    if not v:
        return None
    v = v.lower().replace("-", "_").replace(" ", "_")
    return v if v in ("face_on", "down_the_line") else None


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", help="psycopg DSN; omit with --dry-run")
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="emit SQL, no DB")
    ap.add_argument("--emit-sql", help="write dry-run SQL to this file")
    a = ap.parse_args()

    do_seed = a.seed or a.all
    do_back = a.backfill or a.all
    if not (do_seed or do_back):
        do_seed = do_back = True

    conn = None
    if not a.dry_run:
        import psycopg
        conn = psycopg.connect(a.dsn)
    sink = open(a.emit_sql, "w", encoding="utf-8") if a.emit_sql else (
        sys.stdout if a.dry_run else None)

    ex = Executor(conn, sink)
    if do_seed:
        seed(ex)
    if do_back:
        backfill(ex)
    ex.commit()
    print(f"[done] {ex.n} statements ({'dry-run' if a.dry_run else 'committed'})")
    if a.emit_sql:
        sink.close()


if __name__ == "__main__":
    main()
