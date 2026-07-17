"""Offline tests for ball_flight.py — pure physics, no network, no key.

    .venv/Scripts/python.exe Scripts/test_ball_flight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import ball_flight as BF

_PASS = _FAIL = 0


def check(name: str, cond: bool, extra: str = ""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {extra}")


# =========================================================================== #
print("\n[1] Simulated carries match TrackMan published tour averages (±6%)")
# (ball speed mph, launch deg, backspin rpm) -> published carry yd
PUBLISHED = {
    "driver": ((167.0, 10.9, 2686), 275),
    "7 iron": ((120.0, 16.3, 7097), 172),
    "PW":     ((102.0, 24.2, 9304), 136),
}
for club, (launch, pub_carry) in PUBLISHED.items():
    r = BF.simulate_flight(*launch)
    err = abs(r["carry_yd"] - pub_carry) / pub_carry
    check(f"{club}: carry {r['carry_yd']:.0f} vs published {pub_carry} ({err:.1%})", err <= 0.06)

r = BF.simulate_flight(167.0, 10.9, 2686)
check("driver apex plausible (25-40 yd)", 25 <= r["apex_yd"] <= 40, f"{r['apex_yd']:.1f}")
check("driver flight time plausible (5-8 s)", 5 <= r["flight_time_s"] <= 8, f"{r['flight_time_s']:.1f}")

# =========================================================================== #
print("\n[2] Curve direction follows the sidespin sign convention")
fade = BF.simulate_flight(167.0, 10.9, 2686, sidespin_rpm=800)
draw = BF.simulate_flight(167.0, 10.9, 2686, sidespin_rpm=-800)
straight = BF.simulate_flight(167.0, 10.9, 2686, sidespin_rpm=0)
check("positive sidespin curves right (fade)", fade["side_yd"] > 5, f"{fade['side_yd']:.1f}")
check("negative sidespin curves left (draw)", draw["side_yd"] < -5, f"{draw['side_yd']:.1f}")
check("fade/draw symmetric", abs(fade["side_yd"] + draw["side_yd"]) < 1.0)
check("zero sidespin flies straight", abs(straight["side_yd"]) < 0.5, f"{straight['side_yd']:.2f}")
check("sidespin costs a little carry", fade["carry_yd"] < straight["carry_yd"])

# =========================================================================== #
print("\n[3] Physics sanity")
# no spin -> no lift beyond drag ballistics; must fly far shorter than with backspin
nospin = BF.simulate_flight(167.0, 10.9, 550)   # near-minimum spin
check("low spin flies much shorter than tour spin",
      nospin["carry_yd"] < straight["carry_yd"] - 15,
      f"{nospin['carry_yd']:.0f} vs {straight['carry_yd']:.0f}")
# more ball speed -> more carry, monotone
c1 = BF.simulate_flight(120.0, 16.3, 7097)["carry_yd"]
c2 = BF.simulate_flight(140.0, 16.3, 7097)["carry_yd"]
check("carry increases with ball speed", c2 > c1 + 15)
# dt-halving convergence: dt=0.005 within 1 yd of dt=0.01
a = BF.simulate_flight(167.0, 10.9, 2686, dt=0.01)
b = BF.simulate_flight(167.0, 10.9, 2686, dt=0.005)
check("dt halving changes carry < 1 yd (converged)",
      abs(a["carry_yd"] - b["carry_yd"]) < 1.0,
      f"{a['carry_yd']:.2f} vs {b['carry_yd']:.2f}")
# integration always terminates (steep wedge lob)
lob = BF.simulate_flight(60.0, 49.0, 10000)
check("steep lob terminates with small carry", 0 < lob["carry_yd"] < 80)

# =========================================================================== #
print("\n[4] Club normalization + defaults table")
check("scorecard 'driver' resolves", BF.normalize_club("driver") == "driver")
check("scorecard 'iron' resolves to 7 iron", BF.normalize_club("iron") == "7_iron")
check("'Pitching Wedge' resolves", BF.normalize_club("Pitching Wedge") == "wedge")
check("'3-wood' resolves", BF.normalize_club("3-wood") == "3_wood")
check("unknown club -> None", BF.normalize_club("chopstick") is None)
check("empty/None -> None", BF.normalize_club(None) is None and BF.normalize_club("") is None)
for club, tiers in BF.CLUB_DEFAULTS.items():
    check(f"all tiers present for {club}", set(tiers) == set(BF.TIERS))
# tier ordering: tour >= lpga >= amateur carry for every club
for club in BF.CLUB_DEFAULTS:
    ct = BF.estimate_for_club(club, "tour")["carry_yd"]
    cl = BF.estimate_for_club(club, "lpga")["carry_yd"]
    ca = BF.estimate_for_club(club, "amateur")["carry_yd"]
    check(f"{club}: tour {ct} >= lpga {cl} >= amateur {ca}", ct >= cl >= ca)

# =========================================================================== #
print("\n[4b] Trajectory sampling")
tr = BF.simulate_flight(167.0, 10.9, 2686, trajectory_points=24)["trajectory"]
check("~24 trajectory points returned", 20 <= len(tr) <= 30, str(len(tr)))
check("trajectory starts at the tee", tr[0][0] == 0.0 and tr[0][1] == 0.0)
full = BF.simulate_flight(167.0, 10.9, 2686)
check("trajectory ends at the landing point (carry, ground)",
      abs(tr[-1][0] - full["carry_yd"]) < 1.0 and tr[-1][1] == 0.0, str(tr[-1]))
check("trajectory apex matches reported apex",
      abs(max(p[1] for p in tr) - full["apex_yd"]) < 2.0)
check("downrange is monotonically increasing",
      all(tr[i][0] <= tr[i+1][0] for i in range(len(tr)-1)))
check("no trajectory key unless requested", "trajectory" not in full)

print("\n[5] estimate_for_club tool-facing contract")
e = BF.estimate_for_club("driver", "tour")
check("estimated True with assumed_launch", e["estimated"] and "assumed_launch" in e)
check("display values are ints (verifier-friendly)",
      all(isinstance(e[k], int) for k in ("carry_yd", "side_yd", "apex_yd")))
check("flight time 1 decimal", e["flight_time_s"] == round(e["flight_time_s"], 1))
check("note marks it as an estimate", "estimate" in e["note"])
check("UI trajectory attached under _ui prefix",
      isinstance(e.get("_ui_trajectory"), list) and len(e["_ui_trajectory"]) >= 20)

miss = BF.estimate_for_club(None)
check("missing club -> estimated False + guidance", miss["estimated"] is False and "club" in miss["note"])
check("unknown tier falls back to tour",
      BF.estimate_for_club("driver", "banana")["skill_tier"] == "tour")

ov = BF.estimate_for_club("driver", "amateur",
                          overrides={"ball_speed_mph": 150, "sidespin_rpm": 700})
check("overrides applied and recorded",
      ov["assumed_launch"]["ball_speed_mph"] == 150 and
      set(ov["assumed_launch"]["overridden_by_golfer"]) == {"ball_speed_mph", "sidespin_rpm"})
check("override increases carry vs amateur default",
      ov["carry_yd"] > BF.estimate_for_club("driver", "amateur")["carry_yd"])
check("fade override reported as curve right", ov["curve"].startswith("right"))
crazy = BF.estimate_for_club("driver", "tour", overrides={"ball_speed_mph": 9999})
check("absurd override clamped to bounds",
      crazy["assumed_launch"]["ball_speed_mph"] <= BF._BOUNDS["ball_speed_mph"][1])
bad = BF.estimate_for_club("driver", "tour", overrides={"ball_speed_mph": "fast", "backspin_rpm": True})
check("non-numeric/bool overrides ignored",
      bad["assumed_launch"]["ball_speed_mph"] == 167.0 and
      bad["assumed_launch"]["overridden_by_golfer"] == [])

print("\n[6] Phys-NN engine (trained on public TrackMan data, envelope-gated)")
nn = BF._load_nn()
check("NN weights load", nn is not None and len(nn["layers"]) == 3)
check("driver launch inside training envelope", BF.nn_in_envelope(167.0, 10.9, 2686))
check("tour wedge spin outside envelope (falls back)",
      not BF.nn_in_envelope(102.0, 24.2, 9304))
check("absurd speed outside envelope", not BF.nn_in_envelope(210.0, 10.9, 2686))
e_nn = BF.estimate_for_club("driver", "tour")
check("driver estimate uses phys_nn engine", e_nn["engine"] == "phys_nn", str(e_nn.get("engine")))
e_pq = BF.estimate_for_club("wedge", "tour")
check("wedge estimate falls back to phys_q", e_pq["engine"] == "phys_q")
pq_drv = BF.simulate_flight(167.0, 10.9, 2686)
check("NN driver carry within 12% of Phys-Q (both track tour reality)",
      abs(e_nn["carry_yd"] - pq_drv["carry_yd"]) / pq_drv["carry_yd"] < 0.12,
      f"nn={e_nn['carry_yd']} pq={pq_drv['carry_yd']:.0f}")
check("NN path still ships a UI trajectory", len(e_nn["_ui_trajectory"]) >= 15)
nn_fade = BF._simulate_nn(167.0, 10.9, 2686, sidespin_rpm=800)
check("NN fade curves right", nn_fade["side_yd"] > 3, f"{nn_fade['side_yd']:.1f}")
# graceful degradation: without the weights the estimate still works on phys_q
_saved = dict(BF._NN_CACHE)
BF._NN_CACHE.update({"loaded": True, "nn": None})
e_no = BF.estimate_for_club("driver", "tour")
check("missing weights -> phys_q fallback, still estimates",
      e_no["engine"] == "phys_q" and e_no["estimated"])
BF._NN_CACHE.update(_saved)

print(f"\n{'='*50}\n  {_PASS} passed, {_FAIL} failed")
sys.exit(1 if _FAIL else 0)
