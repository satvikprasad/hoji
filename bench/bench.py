#!/usr/bin/env python3
"""Benchmark the hoji matmul across shapes, with statistics, plots and
run-to-run comparison.

    cmake --build build --target bench          # build libs + run
    ./venv/bin/python bench/bench.py            # run against build/
    ./venv/bin/python bench/bench.py --save-baseline
    ./venv/bin/python bench/bench.py            # now shows delta vs baseline

Every case is checked against numpy before timing; a mismatch is reported
instead of a throughput number. Results are written to build/bench_results.json
so successive runs can be diffed as the kernel changes.

The register tile (_CPU_MATMUL_TILE_M x _CPU_MATMUL_TILE_N) is a compile-time
macro, so CMake builds one shared library per tile and this script sweeps them.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import load_all  # noqa: E402

# Measured ceilings for this machine (see notes below); override on the CLI.
#   PEAK_GFLOPS: 16 independent vector FMAs, no memory traffic
#   PEAK_GBS   : single-thread streaming read of 256 MB
PEAK_GFLOPS = 63.0
PEAK_GBS = 33.6

# name, M, K, N, reps -- Qwen2-0.5B geometry plus shapes that probe edge cases
CASES = [
    ("decode gate_proj", 1, 896, 4864, 60),
    ("decode o_proj", 1, 896, 896, 200),
    ("decode qkv", 1, 896, 1152, 200),
    ("prefill gate_proj", 50, 896, 4864, 10),
    ("prefill o_proj", 50, 896, 896, 40),
    ("attn QK^T 1 head", 50, 64, 50, 500),
    ("attn PV 1 head", 50, 50, 64, 500),
    ("lm_head", 1, 896, 4096, 60),
    ("awkward K (=101)", 8, 101, 512, 200),
    ("square 256", 256, 256, 256, 40),
]


def measure(fn, reps):
    """Returns a list of per-call seconds, after one warm-up call."""
    fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def run_case(backend, m, k, n, reps, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, (m, k)).astype(np.float32)
    b_trans = rng.uniform(-1.0, 1.0, (n, k)).astype(np.float32)
    out = np.zeros((m, n), dtype=np.float32)

    backend.matmul(a, b_trans, out)
    want = a @ b_trans.T
    max_diff = float(np.max(np.abs(out.astype(np.float64) - want.astype(np.float64))))
    tol = 1e-5 * k                      # error grows with the reduction length
    if max_diff > tol:
        return {"ok": False, "max_diff": max_diff, "tol": tol}

    s = measure(lambda: backend.matmul(a, b_trans, out), reps)
    flops = 2.0 * m * k * n
    nbytes = (m * k + n * k + m * n) * 4
    lo, med = min(s), statistics.median(s)
    return {
        "ok": True,
        "max_diff": max_diff,
        "reps": reps,
        "ms_best": lo * 1e3,
        "ms_median": med * 1e3,
        "ms_stdev": (statistics.stdev(s) * 1e3) if len(s) > 1 else 0.0,
        "cv_pct": (statistics.stdev(s) / med * 100) if len(s) > 1 and med else 0.0,
        "gflops": flops / lo / 1e9,
        "gflops_median": flops / med / 1e9,
        "gbs": nbytes / lo / 1e9,
        "ai": flops / nbytes,
        "pct_roof": 100.0 * (flops / lo / 1e9) / min(PEAK_GFLOPS, PEAK_GBS * flops / nbytes),
    }


def sweep(build_dir, cases=CASES):
    backends = load_all(build_dir)
    tiles = [b.tile for b in backends]
    results = {}
    for name, m, k, n, reps in cases:
        results[name] = {"M": m, "K": k, "N": n,
                         "by_tile": {b.tile: run_case(b, m, k, n, reps)
                                     for b in backends}}
    return {"tiles": tiles, "cases": results,
            "peak_gflops": PEAK_GFLOPS, "peak_gbs": PEAK_GBS}


def print_table(data, baseline=None):
    tiles = data["tiles"]
    print(f"\n  matmul: out(M x N) = a(M x K) @ b_trans(N x K).T"
          f"      roofline: {data['peak_gflops']:.0f} GFLOP/s / {data['peak_gbs']:.0f} GB/s")
    head = (f"  {'case':<19}{'M':>5}{'K':>6}{'N':>6}{'AI':>6}"
            + "".join(f"{t:>9}" for t in tiles)
            + f"{'best':>8}{'vs ' + tiles[0]:>9}{'%roof':>7}{'cv%':>6}")
    if baseline:
        head += f"{'vs base':>9}"
    print(head)
    print("  " + "-" * (len(head) - 2))

    for name, c in data["cases"].items():
        row = c["by_tile"]
        cells, best_g, best_t = "", 0.0, None
        for t in tiles:
            r = row[t]
            if not r["ok"]:
                cells += f"{'WRONG':>9}"
                continue
            cells += f"{r['gflops']:>9.2f}"
            if r["gflops"] > best_g:
                best_g, best_t = r["gflops"], t
        if best_t is None:
            print(f"  {name:<19}{c['M']:>5}{c['K']:>6}{c['N']:>6}{'':>6}{cells}")
            continue
        b = row[best_t]
        naive = row[tiles[0]]
        gain = (f"{best_g / naive['gflops']:.2f}x" if naive.get("ok") else "-")
        line = (f"  {name:<19}{c['M']:>5}{c['K']:>6}{c['N']:>6}{b['ai']:>6.1f}{cells}"
                f"{best_t:>8}{gain:>9}{b['pct_roof']:>6.0f}%{b['cv_pct']:>6.1f}")
        if baseline:
            prev = baseline.get("cases", {}).get(name, {}).get("by_tile", {})
            pg = max((v["gflops"] for v in prev.values() if v.get("ok")), default=None)
            line += f"{best_g / pg:>8.2f}x" if pg else f"{'-':>9}"
        print(line)

    oks = [r for c in data["cases"].values() for r in c["by_tile"].values() if r["ok"]]
    bad = sum(1 for c in data["cases"].values() for r in c["by_tile"].values() if not r["ok"])
    if oks:
        print(f"\n  {len(oks)} configs correct, {bad} wrong;"
              f" best overall {max(r['gflops'] for r in oks):.2f} GFLOP/s")


def plot(data, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib missing; skipping plot -- pip install matplotlib)")
        return

    tiles = data["tiles"]
    names = list(data["cases"].keys())
    fig, (ax_b, ax_s, ax_r) = plt.subplots(1, 3, figsize=(21, 6))

    # ---- grouped bars: GFLOP/s per case per folding factor
    width = 0.8 / len(tiles)
    for i, t in enumerate(tiles):
        ys = [data["cases"][n]["by_tile"][t].get("gflops", 0.0) for n in names]
        ax_b.bar([x + i * width for x in range(len(names))], ys, width, label=t)
    ax_b.axhline(data["peak_gflops"], ls="--", c="k", lw=1,
                 label=f"FMA peak {data['peak_gflops']:.0f}")
    ax_b.set_xticks([x + 0.4 - width / 2 for x in range(len(names))])
    ax_b.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax_b.set_ylabel("GFLOP/s")
    ax_b.set_title("throughput by shape and register tile (T_M x T_N)")
    ax_b.legend(fontsize=8, ncol=2)
    ax_b.grid(axis="y", alpha=0.3)

    # ---- speedup over the naive (smallest) tile, per shape
    base_tile = tiles[0]
    x = list(range(len(tiles)))
    for n in names:
        row = data["cases"][n]["by_tile"]
        b0 = row[base_tile]
        if not b0.get("ok") or not b0["gflops"]:
            continue
        ys = [row[t]["gflops"] / b0["gflops"] if row[t].get("ok") else float("nan")
              for t in tiles]
        ax_s.plot(x, ys, marker="o", ms=4, lw=1.4, label=n)
        ax_s.annotate(f"{ys[-1]:.1f}x" if ys[-1] == ys[-1] else "",
                      (x[-1], ys[-1]), fontsize=7,
                      textcoords="offset points", xytext=(5, -2))
    ax_s.axhline(1.0, ls="--", c="k", lw=1)
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(tiles)
    ax_s.set_xlabel("register tile (T_M x T_N)")
    ax_s.set_ylabel(f"speedup over {base_tile}")
    ax_s.set_title(f"improvement over naive ({base_tile}) tile")
    ax_s.grid(alpha=0.3)
    ax_s.legend(fontsize=7, ncol=2)

    # ---- roofline: where each shape sits against the two ceilings
    ai = [10 ** (x / 20.0) for x in range(-40, 61)]
    ax_r.plot(ai, [min(data["peak_gflops"], data["peak_gbs"] * x) for x in ai],
              "k-", lw=1.5, label="roofline")
    for n in names:
        row = data["cases"][n]["by_tile"]
        good = [r for r in row.values() if r["ok"]]
        if not good:
            continue
        b = max(good, key=lambda r: r["gflops"])
        ax_r.plot(b["ai"], b["gflops"], "o", ms=7)
        ax_r.annotate(n, (b["ai"], b["gflops"]), fontsize=7,
                      textcoords="offset points", xytext=(6, -3))
    ax_r.set_xscale("log"); ax_r.set_yscale("log")
    ax_r.set_xlabel("arithmetic intensity (flop/byte)")
    ax_r.set_ylabel("GFLOP/s")
    ax_r.set_title("roofline (best tile per shape)")
    ax_r.grid(which="both", alpha=0.3)
    ax_r.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  plot -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--save-baseline", action="store_true",
                    help="record this run as the comparison baseline")
    ap.add_argument("--no-compare", action="store_true")
    ap.add_argument("--peak-gflops", type=float, default=PEAK_GFLOPS)
    ap.add_argument("--peak-gbs", type=float, default=PEAK_GBS)
    args = ap.parse_args()

    globals()["PEAK_GFLOPS"] = args.peak_gflops
    globals()["PEAK_GBS"] = args.peak_gbs

    base_path = os.path.join(args.build_dir, "bench_baseline.json")
    baseline = None
    if not args.no_compare and os.path.exists(base_path):
        with open(base_path) as fh:
            baseline = json.load(fh)

    data = sweep(args.build_dir)
    print_table(data, baseline)

    out_json = args.json or os.path.join(args.build_dir, "bench_results.json")
    with open(out_json, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"  json -> {out_json}")
    if args.save_baseline:
        with open(base_path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"  baseline -> {base_path}")

    plot(data, args.plot or os.path.join(args.build_dir, "bench_matmul.png"))


if __name__ == "__main__":
    main()
