# Integrating Austin Lee's Depth-Reliability Findings into the Coaching Layer

**Author:** Banjo &nbsp;•&nbsp; **Credit:** depth-reliability empirics from Austin Lee's Phase-2 driver EDA

---

## What Austin found

In `AL EDA Investigation/`, Austin measured how *consistently* each body joint's
3D position — especially the **depth (z) axis** — can be recovered from
single-POV golf video. His key result (FINDINGS.md §6):

> **Most reliable depth: hip joints at Address / Toe-up. Least reliable: wrists
> and elbows throughout, and all joints at Finish — extremely high variance.**

This is the crux of the single-camera problem: a phone sees a 2D projection, so
depth is *estimated*, and some joints/phases are far more estimable than others.

## Why it matters for coaching

Our coaching scorecard computes 15 biomechanical indicators from the 3D pose.
Several of them — shoulder/hip turn, X-factor, arm extension — depend on the
**depth axis** to measure rotation. If we raise a coaching note off a joint
whose depth is unreliable (e.g. wrist position at the top of the backswing), we
risk telling a golfer something that's really just measurement noise.

Before this integration, the scorecard treated all 15 indicators as equally
trustworthy. That was the honest gap.

## What we built

`Scripts/coaching_reliability.py`:

1. Loads Austin's `pose_consistency_by_player_view.csv` (per-player, per-view,
   per-event, per-joint 3D landmarks, 86 driver clips).
2. Computes, for each (event, joint), the **within-player standard deviation of
   the z coordinate** — a direct empirical reliability measure (low std =
   reliably recovered depth).
3. Maps each of our 15 indicators onto the joints + events it depends on, and
   assigns a **confidence weight** = reliability of its *worst* constituent
   joint (an indicator is only as trustworthy as its shakiest input).
4. Writes `Data/coaching/indicator_confidence.json` (high / med / low tier).

The derived confidence reproduces Austin's finding exactly:

| Indicator | Confidence | Tier | (matches Austin?) |
|---|---|---|---|
| hip turn @ top / impact | 1.00 | high | ✓ hips reliable |
| weight shift (hip lateral) | 1.00 | high | ✓ |
| tempo | 1.00 | high | (timing only) |
| spine tilt @ address | 0.83 | high | ✓ |
| shoulder turn / X-factor | 0.64 | med | |
| posture / spine @ impact | 0.66 | med | |
| head sway / lift | 0.50 | med | |
| lead knee flex | 0.63 | med | |
| **lead/right arm bend @ top** | **0.11–0.34** | **low** | ✓ wrists noisy |

## How the scorecard uses it

`Scripts/coaching_scorecard.py` + `render_scorecard.py`:

- **Feedback gating:** a LOW-confidence indicator can never raise a coaching
  flag. If its value is out of band, it's surfaced as an *informational*
  "low-confidence measurement" note, not a coaching instruction.
- **Visual:** low-confidence indicators render grayed-out with a `⚠ low-confidence`
  tag and a gray marker, so the user (and the team) see exactly which numbers
  to trust.
- **Summary count:** the headline "N indicators outside tour range" counts only
  confidently-measured flags.

Concrete effect on the demo (clip 1292): "Lead-arm extension @ top" is now
correctly demoted to low-confidence (wrist depth), while the hip-based
"weight shift" note — high confidence — still stands.

## Honest caveats

- Austin's reliability sample is **3 players / 86 driver clips**. This is a
  reliability *prior*, not a population statistic. The relative ranking
  (hips > shoulders > wrists; address > finish) is robust and matches
  biomechanical intuition, but the exact confidence values would tighten with
  more players.
- His depth comes from MediaPipe *world landmarks*; our pipeline lifts depth
  with MotionBERT. The reliability ranking transfers as a prior (it reflects
  which joints/phases are inherently hard to see in depth from one camera), but
  is not a perfect 1:1 map to MotionBERT's error profile. Re-deriving the same
  table directly on MotionBERT output is a clean future refinement.

## Where else Austin's work converges with ours (not yet integrated)

- **Tempo:** his per-player `tempo_stats.csv` vs our corpus-wide median 3.4:1 —
  both confirm the ~3:1 tour norm. Could cross-validate per-player.
- **Multi-view infeasibility:** he and I independently concluded GolfDB clips
  sharing a youtube_id are not synchronized stereo pairs — closes off
  triangulation as a depth approach.
- **`pose_angles_speed_matched.csv`:** his per-(player,event,joint) angles could
  cross-validate our indicator extraction on overlapping joints.

## Files

| File | Role |
|---|---|
| `Scripts/coaching_reliability.py` | Derives confidence from Austin's data |
| `Data/coaching/indicator_confidence.json` | Per-indicator confidence tiers |
| `Scripts/coaching_scorecard.py` | Reliability-gated feedback |
| `Scripts/render_scorecard.py` | Low-confidence visual tagging |
| `AL EDA Investigation/.../pose_consistency_by_player_view.csv` | Austin's source data |
