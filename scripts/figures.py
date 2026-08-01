"""Every figure in the writeup, from the JSON the eval scripts emit.

    python scripts/compare.py --ckpt runs/s3_joint/best.pt      # -> results/compare.json
    python scripts/benchmark.py --compile                       # -> results/benchmark.json
    python scripts/figures.py                                   # -> figures/*.png + *.csv

Figures
-------
1. depth histogram        how depth is distributed, not just its mean
2. survival curve         what fraction of tokens is still computing at each layer
3. perplexity vs FLOPs    the Pareto plot -- the headline
4. router confidence      is the router decisive or is it hedging at 0.5
5. throughput             what the FLOP saving is actually worth in wall clock

Colour
------
Four arms, one fixed slot each, assigned by entity and never by rank -- so a figure
that drops an arm does not repaint the others. The four hexes were checked with the
palette validator on the all-pairs list (scatter is an all-pairs form), which is also
why the fourth slot is violet rather than the more obvious yellow: yellow beside
orange fails the normal-vision separation floor.

Light mode only, deliberately. These are figures for a paper, printed on white; there
is no viewer theme to respond to. The `aqua` slot sits below 3:1 against the surface,
so every figure using it carries a legend and direct labels -- which is the relief the
palette rule requires, and is what a paper figure should have anyway.

Every figure also writes its own `.csv`. That is the table view: the numbers behind a
figure should be readable without decoding a colour.
"""

import os
import sys

# `python scripts/x.py` puts scripts/ on sys.path, not the repo root, so `agpt` is not
# importable without this. Two lines here beats requiring `pip install -e .` before the
# first run.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402

# -- the design system, as parameters ---------------------------------------
ARM_COLOR = {
    "adaptive": "#2a78d6",   # slot 1, blue
    "fixed":    "#eb6834",   # slot 2, orange
    "random":   "#1baf7a",   # slot 3, aqua
    "dense":    "#4a3aa7",   # slot 7, violet
}
ARM_ORDER = ["dense", "random", "fixed", "adaptive"]
ARM_LABEL = {"dense": "Dense (12 layers)", "random": "Random skip",
             "fixed": "Fixed exit", "adaptive": "AdaptiveGPT"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 8,
        "figure.dpi": 160,
    })


def finish(fig, ax, path, csv_rows=None, csv_header=None):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    if csv_rows is not None:
        with open(os.path.splitext(path)[0] + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            if csv_header:
                w.writerow(csv_header)
            w.writerows(csv_rows)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------

def fig_depth_histogram(data, out_dir):
    """How many layers each token actually got.

    The mean is the number people quote and the distribution is the number that says
    whether anything adaptive happened: 6.2 layers/token means one thing if every
    token got 6 and something entirely different if half got 3 and half got 10. If
    this histogram has one spike, the model has learned a fixed depth and the `fixed`
    baseline is going to match it.
    """
    arm = data["arms"]["adaptive"]
    hist = arm["depth_hist"]
    layers = list(range(len(hist)))

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.bar(layers, [h * 100 for h in hist], color=ARM_COLOR["adaptive"], width=0.72)
    ax.axvline(arm["avg_depth"], color=INK_2, linestyle="--", linewidth=1.2)
    ax.annotate(f"mean {arm['avg_depth']:.2f}", (arm["avg_depth"], ax.get_ylim()[1]),
                xytext=(4, -10), textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_xlabel("layers computed")
    ax.set_ylabel("% of tokens")
    ax.set_title("Depth spent per token")
    ax.set_xticks([l for l in layers if l % 2 == 0])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    finish(fig, ax, os.path.join(out_dir, "fig1_depth_histogram.png"),
           [[l, round(h * 100, 3)] for l, h in enumerate(hist)],
           ["layers_computed", "percent_of_tokens"])


def fig_survival(data, out_dir):
    """Fraction of tokens still computing when each layer starts.

    This is the same information as the histogram, integrated -- and it is the curve
    the FLOP model consumes directly, since the cost of layer l is set by how many
    tokens enter it.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    rows = []
    # Drawn in reverse so the adaptive arm sits on top: the matched baselines are
    # *designed* to land on the same depth, so these curves coincide often and the
    # one the figure is about should be the one that stays visible.
    for z, name in enumerate(reversed(ARM_ORDER)):
        arm = data["arms"].get(name)
        if not arm:
            continue
        y = [f * 100 for f in arm["active_fracs"]]
        x = list(range(len(y)))
        ax.plot(x, y, color=ARM_COLOR[name], label=ARM_LABEL[name],
                marker="o", markersize=4, zorder=3 + z)
        rows += [[name, l, round(f, 5)] for l, f in enumerate(arm["active_fracs"])]

    # No end-of-line labels here, deliberately. Depth-matched arms end at the same
    # point by construction, so direct labels overprint each other in the normal case
    # rather than in an edge case; the legend carries identity instead.
    ax.set_xlabel("layer")
    ax.set_ylabel("% of tokens still active")
    ax.set_title("Tokens surviving to each layer")
    ax.set_ylim(-4, 106)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", ncol=2)
    finish(fig, ax, os.path.join(out_dir, "fig2_survival.png"), rows,
           ["arm", "layer", "active_fraction"])


def fig_pareto(data, out_dir, sweep=None):
    """Perplexity against FLOPs -- the headline.

    Down and to the left is better. The claim this project makes is that the adaptive
    point sits below the line joining dense to fixed: that per-token depth buys
    quality a uniformly shallower model of the same cost cannot.

    Plotted on LAYER FLOPs. The output head is ~44% of total forward FLOPs at this
    width and routing cannot touch it, so the total-FLOP axis compresses every
    difference toward 1.0 and hides the effect being measured. The total-FLOP number
    is in the CSV and in the results table; it belongs in the text, not in this
    figure's geometry.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    rows, placed = [], []
    for name in ARM_ORDER:
        arm = data["arms"].get(name)
        if not arm:
            continue
        x = arm["flops_frac_layers"] * 100
        y = arm["ppl"]
        ax.scatter([x], [y], color=ARM_COLOR[name], s=90, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        # Stack the label above any already-placed label it would collide with. The
        # depth-matched baselines routinely land on the same point -- that is what
        # "matched" means -- so overlapping labels are the normal case here, and a
        # figure that renders them on top of each other is unreadable exactly when the
        # controls worked.
        dy = 11
        while any(abs(x - px) < 6 and abs(dy - pdy) < 12 for px, pdy in placed):
            dy += 13
        placed.append((x, dy))
        ax.annotate(ARM_LABEL[name], (x, y), xytext=(0, dy),
                    textcoords="offset points", color=ARM_COLOR[name],
                    fontsize=9, ha="center")
        rows.append([name, round(x, 3), round(arm["flops_frac_total"] * 100, 3),
                     round(y, 4), round(arm["avg_depth"], 3)])

    if sweep:
        xs = [p["flops_frac_layers"] * 100 for p in sweep]
        ys = [p["ppl"] for p in sweep]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                color=ARM_COLOR["adaptive"], linewidth=1.6, alpha=0.55, zorder=2,
                label="AdaptiveGPT, depth-penalty sweep")
        rows += [["sweep", round(x, 3), "", round(y, 4), ""] for x, y in zip(xs, ys)]
        ax.legend(loc="upper right")

    ax.set_xlabel("layer FLOPs per token (% of dense)")
    ax.set_ylabel("validation perplexity")
    ax.set_title("Quality against compute")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    finish(fig, ax, os.path.join(out_dir, "fig3_pareto.png"), rows,
           ["arm", "layer_flops_pct", "total_flops_pct", "ppl", "avg_depth"])


def fig_router_confidence(conf, out_dir):
    """Distribution of the routers' continue probability.

    Mass piled at 0 and 1 means the router has made up its mind. Mass piled at 0.5
    means it has not, and an average depth that looks reasonable is then an artefact
    of the threshold rather than a decision -- the failure mode that a mean depth
    cannot show you.
    """
    edges, counts = conf["edges"], conf["counts"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    total = sum(counts) or 1

    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.bar(centers, [c / total * 100 for c in counts],
           width=(edges[1] - edges[0]) * 0.9, color=ARM_COLOR["adaptive"])
    ax.axvline(0.5, color=INK_2, linestyle="--", linewidth=1.2)
    ax.annotate("exit threshold", (0.5, ax.get_ylim()[1]), xytext=(5, -10),
                textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_xlabel("P(continue), pooled over routers")
    ax.set_ylabel("% of decisions")
    ax.set_title("Router confidence")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    finish(fig, ax, os.path.join(out_dir, "fig4_router_confidence.png"),
           [[round(c, 4), round(n / total * 100, 4)] for c, n in zip(centers, counts)],
           ["p_continue_bin_center", "percent_of_decisions"])


def fig_throughput(bench, out_dir):
    """Tokens per second by batch size.

    The number that decides whether any of this was worth doing. A FLOP saving that
    does not appear here is a FLOP saving the hardware declined to give you.
    """
    arms = [a for a in ARM_ORDER if a in bench["arms"]]
    batches = sorted({r["batch_size"] for a in arms
                      for r in bench["arms"][a]["throughput"]})
    width = 0.8 / max(len(arms), 1)

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    rows = []
    for i, name in enumerate(arms):
        by_b = {r["batch_size"]: r for r in bench["arms"][name]["throughput"]}
        xs = [j + i * width - 0.4 + width / 2 for j in range(len(batches))]
        ys = [by_b.get(b, {}).get("tokens_per_sec", 0) / 1000 for b in batches]
        ax.bar(xs, ys, width=width * 0.92, color=ARM_COLOR[name],
               label=ARM_LABEL[name])
        rows += [[name, b, round(by_b.get(b, {}).get("tokens_per_sec", 0), 1),
                  round(by_b.get(b, {}).get("peak_mb", 0), 1)] for b in batches]

    ax.set_xticks(range(len(batches)))
    ax.set_xticklabels([f"B={b}" for b in batches])
    ax.set_ylabel("thousand tokens / sec")
    ax.set_title("Throughput"
                 + ("" if bench.get("compiled") else "  (EAGER — not a valid claim)"))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper left")
    finish(fig, ax, os.path.join(out_dir, "fig5_throughput.png"), rows,
           ["arm", "batch_size", "tokens_per_sec", "peak_mb"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", default="results/compare.json")
    ap.add_argument("--benchmark", default="results/benchmark.json")
    ap.add_argument("--confidence", default="results/confidence.json",
                    help="written by scripts/evaluate.py --confidence")
    ap.add_argument("--sweep", default=None,
                    help="JSON list of {ppl, flops_frac_layers} for the Pareto curve")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()

    style()
    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.compare):
        with open(args.compare) as f:
            data = json.load(f)
        sweep = None
        if args.sweep and os.path.exists(args.sweep):
            with open(args.sweep) as f:
                sweep = json.load(f)
        fig_depth_histogram(data, args.out_dir)
        fig_survival(data, args.out_dir)
        fig_pareto(data, args.out_dir, sweep)
    else:
        print(f"skipped figures 1-3: no {args.compare} (run scripts/compare.py)")

    if os.path.exists(args.confidence):
        with open(args.confidence) as f:
            fig_router_confidence(json.load(f), args.out_dir)
    else:
        print(f"skipped figure 4: no {args.confidence} "
              "(run scripts/evaluate.py --confidence)")

    if os.path.exists(args.benchmark):
        with open(args.benchmark) as f:
            fig_throughput(json.load(f), args.out_dir)
    else:
        print(f"skipped figure 5: no {args.benchmark} (run scripts/benchmark.py)")


if __name__ == "__main__":
    main()
