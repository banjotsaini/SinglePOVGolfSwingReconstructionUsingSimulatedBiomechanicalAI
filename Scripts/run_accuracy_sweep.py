"""Iteration engine: sweep noise levels x interventions against the Vicon
ground-truth coaching benchmark, and report which fixes measurably reduce
coaching-measurement error.

Lifts each swing once per noise level (the slow step), then applies every
intervention to the cached 3D (cheap). Produces a results matrix + figure.
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coaching_accuracy_benchmark as B
from benchmark_interventions import INTERVENTIONS

OUT = B.OUT
NOISE_LEVELS = [0.0, 6.0, 12.0]          # clean, moderate, heavy 2D jitter (px)
INTERV = ["none", "sg", "oneeuro", "bonelock", "sg_bonelock", "oneeuro_bonelock"]


def measure(pred, gt, quants):
    pa = B.procrustes_align(pred, gt)
    qp, qg = B.quantities(pa), B.quantities(gt)
    return {q: np.mean(B.ang_err(qp[q], qg[q], wrap=q in B.WRAP)) for q in quants}


def main(backend="mixste"):
    data = B.load_vicon()
    lifter = {"mixste": B.MixSTELifter, "motionbert": B.MotionBERTLifter}[backend]()
    quants = list(B.quantities(np.zeros((1, 17, 3))).keys())
    test = [(s, sw, gt, xy) for s in B.TEST_SUBJ for sw, (gt, xy) in sorted(data.get(s, {}).items())]
    print(f"[lifter] {lifter.name}  |  {len(test)} test swings  |  noise {NOISE_LEVELS}")

    # results[noise][interv] = {quant: err, _overall: x}
    results = {}
    for noise in NOISE_LEVELS:
        # lift each swing once at this noise level
        lifted = []
        for s, sw, gt, xy in test:
            xin = xy
            if noise > 0:
                rng = np.random.default_rng(abs(hash((s, sw))) % (2**32))
                xin = xy + rng.normal(0.0, noise, size=xy.shape).astype(np.float32)
            lifted.append((lifter.lift(xin), gt))
        results[noise] = {}
        for iv in INTERV:
            fn = (lambda p, ctx: p) if iv == "none" else INTERVENTIONS[iv]
            per = {q: [] for q in quants}
            for pred, gt in lifted:
                e = measure(fn(pred.copy(), {}), gt, quants)
                for q in quants:
                    per[q].append(e[q])
            row = {q: round(float(np.mean(per[q])), 2) for q in quants}
            row["_overall"] = round(float(np.mean([row[q] for q in quants])), 2)
            results[noise][iv] = row
            print(f"  noise={noise:>4}  {iv:<18} overall={row['_overall']:5.2f}°")

    with open(OUT / f"sweep_{backend}.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- summary table (overall °) ----
    print("\n" + "=" * 66)
    print(f"OVERALL coaching-measurement error (deg) — backend={lifter.name}")
    print("=" * 66)
    hdr = "intervention".ljust(20) + "".join(f"{('noise '+str(int(n))+'px'):>12}" for n in NOISE_LEVELS)
    print(hdr)
    for iv in INTERV:
        line = iv.ljust(20) + "".join(f"{results[n][iv]['_overall']:>12.2f}" for n in NOISE_LEVELS)
        print(line)

    # ---- best improvement at the heaviest noise ----
    heavy = NOISE_LEVELS[-1]
    base = results[heavy]["none"]["_overall"]
    best_iv = min((iv for iv in INTERV if iv != "none"),
                  key=lambda iv: results[heavy][iv]["_overall"])
    best = results[heavy][best_iv]["_overall"]
    print(f"\nAt {heavy:g}px noise: baseline {base:.2f}° -> best ({best_iv}) {best:.2f}°  "
          f"= {100*(base-best)/base:.1f}% lower error")

    make_figure(results, backend, lifter.name)
    return results


def make_figure(results, backend, lifter_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

    # Panel 1: overall error vs noise, per intervention
    for iv in INTERV:
        ys = [results[n][iv]["_overall"] for n in NOISE_LEVELS]
        style = "--" if iv == "none" else "-"
        lw = 2.4 if iv == "none" else 1.8
        ax[0].plot(NOISE_LEVELS, ys, style, marker="o", lw=lw, label=iv)
    ax[0].set_xlabel("Injected 2D jitter (px std)")
    ax[0].set_ylabel("Mean coaching-measurement error (deg)")
    ax[0].set_title("Smoothing recovers accuracy under detection noise")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    # Panel 2: per-quantity, baseline vs best, at heaviest noise
    heavy = NOISE_LEVELS[-1]
    quants = [q for q in results[heavy]["none"] if q != "_overall"]
    best_iv = min((iv for iv in INTERV if iv != "none"),
                  key=lambda iv: results[heavy][iv]["_overall"])
    base = [results[heavy]["none"][q] for q in quants]
    best = [results[heavy][best_iv][q] for q in quants]
    x = np.arange(len(quants)); w = 0.38
    ax[1].bar(x - w / 2, base, w, label="baseline (no smoothing)", color="#d9822b")
    ax[1].bar(x + w / 2, best, w, label=f"+ {best_iv}", color="#2a9d3a")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([q.replace("_deg", "").replace("_", "\n") for q in quants], fontsize=7)
    ax[1].set_ylabel("error (deg)")
    ax[1].set_title(f"Per-measurement @ {heavy:g}px noise")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, axis="y")

    fig.suptitle(f"Coaching-measurement accuracy vs Vicon truth — {lifter_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"sweep_{backend}.png", dpi=150)
    plt.close(fig)
    print(f"[+] wrote {OUT / f'sweep_{backend}.png'} and sweep_{backend}.json")


if __name__ == "__main__":
    bk = sys.argv[1] if len(sys.argv) > 1 else "mixste"
    main(bk)
