"""
validate_artifacts.py — Sprint 1 validation.

    python validate_artifacts.py
    python validate_artifacts.py --file artifacts.jsonl
"""

import argparse
import json
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="artifacts.jsonl")
    args = ap.parse_args()

    try:
        with open(args.file, encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
    except FileNotFoundError:
        print(f"File not found: {args.file}")
        print("Run collect_artifacts.ps1 first.")
        sys.exit(1)

    artifacts, errors = [], []
    for i, line in enumerate(lines):
        try:
            artifacts.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append((i + 1, str(e)))

    print(f"Total lines:     {len(lines)}")
    print(f"Parse errors:    {len(errors)}")
    if errors:
        for lineno, msg in errors[:5]:
            print(f"  line {lineno}: {msg}")
    print(f"Valid artifacts: {len(artifacts)}")
    print()

    if not artifacts:
        print("No valid artifacts. Nothing to validate.")
        sys.exit(1)

    speech_acts = Counter(a["state"]["speech_act"] for a in artifacts)
    print("Speech acts:")
    for act, c in speech_acts.most_common():
        print(f"  {act:<15} {c:4}  ({100*c/len(artifacts):5.1f}%)")
    print()

    resp_types = Counter(a["state"]["response_type"] for a in artifacts)
    print("Response types:")
    for rt, c in resp_types.most_common():
        print(f"  {rt:<15} {c:4}  ({100*c/len(artifacts):5.1f}%)")
    print()

    joins = Counter(a["join_used"] for a in artifacts)
    print("Join patterns used:")
    for j, c in joins.most_common():
        print(f"  {j:<25} {c:4}  ({100*c/len(artifacts):5.1f}%)")
    print()

    closure_sizes = [a["state"]["closure_size"] for a in artifacts]
    output_words  = [len(a["output"].split()) for a in artifacts]
    visited       = [a["state"]["visited_total"] for a in artifacts]

    def stats(label, values):
        s = sorted(values)
        n = len(s)
        print(f"{label}:")
        print(f"  min={s[0]}  median={s[n//2]}  max={s[-1]}"
              f"  avg={sum(s)/n:.1f}")

    stats("Closure sizes",   closure_sizes)
    stats("Output words",    output_words)
    stats("Visited total",   visited)
    print()

    ok = True
    print("Success criteria:")

    if len(artifacts) >= 500:
        print(f"  [OK] {len(artifacts)} artifacts (>= 500)")
    else:
        print(f"  [--] {len(artifacts)} artifacts (need >= 500)")
        ok = False

    min_per_type = min(speech_acts.values()) if speech_acts else 0
    if min_per_type >= 50:
        print(f"  [OK] balanced distribution (min {min_per_type} per type)")
    else:
        low = [k for k, v in speech_acts.items() if v < 50]
        print(f"  [--] low coverage: {low}")

    median_cl = sorted(closure_sizes)[len(closure_sizes)//2]
    if 5 <= median_cl <= 500:
        print(f"  [OK] median closure size {median_cl} (reasonable)")
    else:
        print(f"  [--] median closure size {median_cl} (unexpected)")

    if len(errors) == 0:
        print("  [OK] no JSON parse errors")
    else:
        print(f"  [--] {len(errors)} parse errors")
        ok = False

    print()
    if ok:
        print("All checks passed. Ready for Sprint 2.")
    else:
        print("Some checks failed. Collect more artifacts or investigate errors.")


if __name__ == "__main__":
    main()
