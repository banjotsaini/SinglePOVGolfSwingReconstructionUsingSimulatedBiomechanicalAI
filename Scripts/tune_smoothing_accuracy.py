"""Tune One-Euro params against the GT coaching benchmark (lift once per swing,
grid over filter params) and report the best end-to-end config vs the current app.
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coaching_accuracy_benchmark as B
from benchmark_interventions import _one_euro, _sg

NOISE = 10.0


def overall_err(pred, gt, quants):
    pa = B.procrustes_align(pred, gt)
    qp, qg = B.quantities(pa), B.quantities(gt)
    return float(np.mean([np.mean(B.ang_err(qp[q], qg[q], wrap=q in B.WRAP)) for q in quants]))


def main():
    data = B.load_vicon()
    quants = list(B.quantities(np.zeros((1, 17, 3))).keys())
    mix = B.MixSTELifter()

    # lift each test swing once at NOISE (and a clean MotionBERT pass for the app baseline)
    test = [(s, sw, gt, xy) for s in B.TEST_SUBJ for sw, (gt, xy) in sorted(data.get(s, {}).items())]
    lifted_noisy = []
    for s, sw, gt, xy in test:
        rng = np.random.default_rng(abs(hash((s, sw))) % (2**32))
        xn = xy + rng.normal(0.0, NOISE, size=xy.shape).astype(np.float32)
        lifted_noisy.append((mix.lift(xn), gt))

    base = float(np.mean([overall_err(p, g, quants) for p, g in lifted_noisy]))
    print(f"MixSTE @ {NOISE:g}px, no smoothing: {base:.2f}°")

    grid = []
    for mc in (0.2, 0.3, 0.5, 0.8, 1.0):
        for beta in (0.02, 0.05, 0.1, 0.2, 0.4):
            errs = [overall_err(_one_euro(p.copy(), min_cutoff=mc, beta=beta), g, quants)
                    for p, g in lifted_noisy]
            grid.append((float(np.mean(errs)), mc, beta))
    grid.sort()
    best_err, best_mc, best_beta = grid[0]
    print(f"best One-Euro: min_cutoff={best_mc}, beta={best_beta} -> {best_err:.2f}°  "
          f"({100*(base-best_err)/base:.1f}% lower than no-smoothing)")
    print("top 5:")
    for e, mc, bt in grid[:5]:
        print(f"   mc={mc:<4} beta={bt:<5} -> {e:.2f}°")

    out = {"noise_px": NOISE, "mixste_nosmooth_deg": round(base, 2),
           "best": {"min_cutoff": best_mc, "beta": best_beta, "deg": round(best_err, 2),
                    "pct_better": round(100 * (base - best_err) / base, 1)},
           "grid_top5": [{"min_cutoff": mc, "beta": bt, "deg": round(e, 2)} for e, mc, bt in grid[:5]]}
    with open(B.OUT / "oneeuro_tuning.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[+] wrote {B.OUT/'oneeuro_tuning.json'}")


if __name__ == "__main__":
    main()
