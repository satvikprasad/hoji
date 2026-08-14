#!/usr/bin/env python3
"""Benchmark the hoji backend across folding factors and plot the result.

    cmake --build build --target bench
    ./venv/bin/python bench/bench.py --build-dir build

Data generation, the correctness reference (numpy), and timing all live here;
the shared libraries expose only the kernels. Measured ctypes call overhead is
~150 ns, under 0.4% of the smallest kernel timed, so Python-side timing is fine.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend import load_all  # noqa: E402

# name, M, K, N, reps -- Qwen2-0.5B geometry, plus one awkward K to exercise the
# fold remainder path (896 and 64 are both divisible by 8, so they never do).
CASES = [
    ("decode gate_proj", 1, 896, 4864, 100),
    ("decode o_proj", 1, 896, 896, 500),
    ("prefill gate_proj", 50, 896, 4864, 5),
    ("prefill o_proj", 50, 896, 896, 30),
    ("attn QK^T 1 head", 50, 64, 50, 2000),
    ("awkward K (=101)", 8, 101, 512, 500),
]


def make_inputs(m, k, n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-1.0, 1.0, size=(m, k)).astype(np.float32)
    b_trans = rng.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32)
    return a, b_trans


def best_seconds(fn, reps):
    """Minimum is the least noisy estimator: interference only ever slows a run."""
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run_case(backend, m, k, n, reps):
    a, b_trans = make_inputs(m, k, n)
    out = np.zeros((m, n), dtype=np.float32)

    backend.matmul(a, b_trans, out)
    want = a @ b_trans.T                      # numpy is the reference
    max_diff = float(np.max(np.abs(out.astype(np.float64) - want.astype(np.float64))))
    tol = 1e-5 * k                            # error grows with reduction length

    s = best_seconds(lambda: backend.matmul(a, b_trans, out), reps)
    flops = 2.0 * m * k * n
    nbytes = (m * k + n * k + m * n) * 4
    return {
        "ok": max_diff <= tol,
        "max_diff": max_diff,
        "ms": s * 1e3,
        "gflops": flops / s / 1e9,
        "gbs": nbytes / s / 1e9,
        "ai": flops / nbytes,
    }


def sweep(build_dir, cases=CASES):
    backends = load_all(build_dir)
    folds = [b.folding_factor for b in backends]
    results = {name: {b.folding_factor: run_case(b, m, k, n, reps)
                      for b in backends}
               for name, m, k, n, reps in cases}
    return results, folds


def print_table(results, folds):
    head = f"  {'case':<18}{'M':<5}{'K':<6}{'N':<6}" + "".join(
        f"{'f=' + str(f):>9}" for f in folds)
    print("matmul GFLOP/s: out(M x N) = a(M x K) @ b_trans(N x K).T")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, m, k, n, _ in CASES:
        row = results[name]
        cells = "".join(f"{row[f]['gflops']:>9.2f}" if row[f]["ok"]
                        else f"{'WRONG':>9}" for f in folds)
        print(f"  {name:<18}{m:<5}{k:<6}{n:<6}{cells}")


def plot(results, folds, path):
    try:
        import matplotlib
        matplotlib.use("Agg")            # no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib not installed; skipping plot -- "
              "pip install matplotlib)")
        return

    fig, (ax_g, ax_s) = plt.subplots(1, 2, figsize=(12, 5))
    for name, *_ in CASES:
        row = results[name]
        ys = [row[f]["gflops"] if row[f]["ok"] else float("nan") for f in folds]
        ax_g.plot(folds, ys, marker="o", label=name)
        base = ys[0]
        ax_s.plot(folds, [y / base for y in ys], marker="o", label=name)

    ax_g.set_xlabel("_CPU_MATMUL_FOLDING_FACTOR")
    ax_g.set_ylabel("GFLOP/s")
    ax_g.set_title("matmul throughput vs folding factor")
    ax_s.set_xlabel("_CPU_MATMUL_FOLDING_FACTOR")
    ax_s.set_ylabel(f"speedup over f={folds[0]}")
    ax_s.set_title("relative speedup")
    for ax in (ax_g, ax_s):
        ax.set_xticks(folds)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\n  plot written to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--plot", default=None,
                    help="output image path (default: <build-dir>/bench_matmul.png)")
    args = ap.parse_args()

    results, folds = sweep(args.build_dir)
    print_table(results, folds)
    plot(results, folds, args.plot or os.path.join(args.build_dir,
                                                   "bench_matmul.png"))


if __name__ == "__main__":
    main()
