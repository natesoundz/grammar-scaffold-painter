"""
measure_system.py — Final system measurement: correctness, determinism, performance.

    python measure_system.py

Runs after all three sprints are complete.
"""

import json
import time

from analyzer import analyze
from generator import generate
from discover import load_substrate


def main():
    sub = load_substrate("substrate.min.bin")

    print("=" * 60)
    print("SYSTEM MEASUREMENT")
    print("=" * 60)

    # ---------------------------------------------------------------- Determinism
    print("\n[DETERMINISM]")
    test_input = "How do I fix a broken engine?"

    state1 = analyze(test_input, [], sub, max_layers=5, checkpoint="measure_chk.json")
    result1 = generate(state1, sub)
    output1 = result1["output"]

    state2 = analyze(test_input, [], sub, max_layers=5, checkpoint="measure_chk.json")
    result2 = generate(state2, sub)
    output2 = result2["output"]

    if output1 == output2:
        print("  [OK] DETERMINISTIC — same output on repeated runs")
    else:
        print("  [--] NON-DETERMINISTIC — outputs differ:")
        print(f"       Run 1: {output1[:60]}...")
        print(f"       Run 2: {output2[:60]}...")

    # ---------------------------------------------------------------- Rule matching
    print("\n[RULE MATCHING]")
    try:
        with open("ab_results.json", encoding="utf-8") as f:
            ab_data = json.load(f)
        matched = ab_data["with_rules"]["matched"]
        total   = ab_data["test_size"]
        pct     = 100 * matched / total if total else 0
        print(f"  {matched}/{total} inputs matched a compiled rule ({pct:.1f}%)")
        if pct >= 70:
            print("  [OK] Good coverage (>= 70%)")
        elif pct >= 40:
            print("  [--] Moderate coverage (40-70%) — collect more artifacts")
        else:
            print("  [--] Low coverage (<40%) — Sprint 2 incomplete")
    except FileNotFoundError:
        print("  ab_results.json not found — run ab_test.py first")
        pct = 0

    # ---------------------------------------------------------------- Performance
    print("\n[PERFORMANCE]")
    inputs = [
        "How do I fix a broken engine?",
        "The mechanic repaired the broken engine.",
        "Tighten the bolt clockwise until snug.",
        "The shaft cracked because of cyclic fatigue.",
        "First remove the bolt, then clean the threads.",
    ]
    t0 = time.time()
    for inp in inputs:
        state  = analyze(inp, [], sub, max_layers=5, checkpoint="measure_chk.json")
        result = generate(state, sub)
    elapsed = time.time() - t0
    ms_per  = 1000 * elapsed / len(inputs)

    print(f"  {len(inputs)} inputs in {elapsed:.1f}s  ({ms_per:.0f}ms/input)")
    if ms_per < 200:
        print("  [OK] Performance acceptable (<200ms/input)")
    else:
        print(f"  [--] {ms_per:.0f}ms/input — consider reducing --layers or --frontier-cap")

    # ---------------------------------------------------------------- Verdict
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    checks = {
        "Deterministic":            output1 == output2,
        "Performance <200ms/input": ms_per < 200,
        "Rule coverage >=70%":      pct >= 70,
    }
    for label, ok in checks.items():
        print(f"  {'[OK]' if ok else '[--]'} {label}")

    if all(checks.values()):
        print("\n  SYSTEM READY.")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"\n  Outstanding: {failed}")


if __name__ == "__main__":
    main()
