# MotionCaddie — database migration

Makes the implementation-guide data model real: schema + seed + backfill for
Aurora Serverless v2 / PostgreSQL 15+.

## Files
| File | Role |
|---|---|
| `migrations/0001_init.sql` | schema — 13 tables + `results_history` view, FKs, CHECK constraints, indexes |
| `load_data.py` | seed reference tables from repo JSON **and** backfill existing scorecards into rows |

## Apply

```bash
# 1. schema (once, on a fresh DB)
psql "$DSN" -f deploy/db/migrations/0001_init.sql

# 2. seed reference data + backfill the file-based scorecard corpus
python deploy/db/load_data.py --dsn "$DSN" --all      # needs: pip install psycopg
```

`$DSN` = `postgresql://user:pass@host:5432/motioncaddie`. The loader is
**idempotent** (every INSERT has `ON CONFLICT`), so re-running is safe.

## Validate without a database

No Aurora yet? The loader emits reviewable SQL instead of connecting:

```bash
python deploy/db/load_data.py --dry-run --all --emit-sql seed.sql
```

CI check (no DB, no key) — parse both the schema and the emitted seed:

```python
import sqlglot
sqlglot.parse(open("deploy/db/migrations/0001_init.sql").read(), read="postgres")
```

Verified 2026-07: DDL parses; dry-run seeds **15 indicators + 8 glossary terms**
and backfills **400 clips → 6000 indicator rows / 3200 event rows / 400 analyses**;
all emitted SQL parses as Postgres.

## What the seed/backfill maps
- `indicator_catalog / kb_cards` ← `Data/coaching/indicator_kb.json` (units normalized:
  `degrees→deg`, `percent of body height→pct`).
- `reference_bands` ← `reference_bands.json` (corpus bands are **club-agnostic** → seeded
  as `club='all'`; the PK leaves room for per-club bands later without a migration).
- `indicator_confidence` ← `indicator_confidence.json` (Austin's depth-reliability EDA).
- `depends_on_joints` ← derived from `coaching_reliability.INDICATOR_DEPS`.
- Backfill: `Data/coaching/{dev,test,eval}_scorecards/*.json` → `swings` (source=`golfdb`)
  + `analyses` (pipeline_version `mixste+oneeuro@2026-07`) + `swing_events` + `indicators`
  + `feedback_notes`.

## Not yet wired (open gaps)
- **Live upload path** writes `swings`/`analyses` from the GPU worker — that tier
  (SageMaker Async / Batch) doesn't exist yet. Backfill only covers the GolfDB corpus.
- `users` / `consents` populate from **Cognito** (not yet provisioned).
- Migration tooling is raw `.sql` + a Python loader; adopt Alembic/Flyway before there
  are multiple migrations to order.
