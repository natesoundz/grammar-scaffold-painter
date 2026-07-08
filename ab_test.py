"""
ab_test.py — A/B test: compiled rules vs. fallback defaults.

    python ab_test.py
    python ab_test.py --test-file test_inputs.txt --max 100
"""

import argparse
import json
import time

from analyzer import analyze
from generator import generate
from discover import load_substrate


def run_batch(inputs, sub, use_transforms):
    transforms = None if use_transforms else []
    results = []
    for inp in inputs:
        state = analyze(inp, [], sub, max_layers=5, frontier_cap=400,
                        checkpoint="ab_checkpoint.json")
        result = generate(state, sub, n=6, transforms=transforms)
        results.append({
            "input":    inp,
            "matched":  result["rule_matched"],
            "conf":     result["rule_confidence"],
            "join":     result["join_used"],
            "output":   result["output"],
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default="inputs.txt")
    ap.add_argument("--max",       type=int, default=50,
                    help="Number of inputs to test (default 50 for speed)")
    ap.add_argument("--out",       default="ab_results.json")
    args = ap.parse_args()

    with open(args.test_file, encoding="utf-8") as f:
        all_inputs = [l.strip() for l in f
                      if l.strip() and not l.startswith("#")]
    test_set = all_inputs[:args.max]

    print(f"A/B TEST: Compiled Rules vs Fallback")
    print(f"Test set: {len(test_set)} inputs from {args.test_file}")
    print()

    sub = load_substrate("substrate.min.bin")

    print("[ A ] Running WITH compiled rules...")
    t0 = time.time()
    results_a = run_batch(test_set, sub, use_transforms=True)
    t_a = time.time() - t0

    print("[ B ] Running WITHOUT rules (fallback only)...")
    t0 = time.time()
    results_b = run_batch(test_set, sub, use_transforms=False)
    t_b = time.time() - t0

    n         = len(test_set)
    matched_a = sum(1 for r in results_a if r["matched"])
    matched_b = sum(1 for r in results_b if r["matched"])
    avg_conf  = sum(r["conf"] for r in results_a) / n

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n[A] With compiled rules:")
    print(f"  Time:          {t_a:.1f}s  ({1000*t_a/n:.0f}ms/input)")
    print(f"  Rules matched: {matched_a}/{n}  ({100*matched_a/n:.1f}%)")
    print(f"  Avg conf:      {avg_conf:.3f}")
    if results_a:
        print(f"  Sample output:\n{results_a[0]['output']}")

    print(f"\n[B] Fallback only:")
    print(f"  Time:          {t_b:.1f}s  ({1000*t_b/n:.0f}ms/input)")
    print(f"  Rules matched: {matched_b}/{n}  (none — no rules loaded)")
    if results_b:
        print(f"  Sample output:\n{results_b[0]['output']}")

    print(f"\nOverhead of rule matching: "
          f"{'+' if t_a > t_b else ''}{1000*(t_a-t_b)/n:.1f}ms/input")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "test_size": n,
            "with_rules": {
                "matched": matched_a, "time_sec": t_a, "avg_conf": avg_conf,
                "samples": results_a[:5],
            },
            "without_rules": {
                "matched": matched_b, "time_sec": t_b,
                "samples": results_b[:5],
            },
        }, f, indent=2)
    print(f"\nDetailed results saved to {args.out}")

    if matched_a == 0:
        print()
        print("No rules matched — transforms.jsonl is empty or missing.")
        print("Complete Sprint 2 first: collect_artifacts.ps1 + transform_compiler.py")


if __name__ == "__main__":
    main()
