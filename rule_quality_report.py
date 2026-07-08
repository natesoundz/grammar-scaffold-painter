"""
rule_quality_report.py — Sprint 2 inspection.

    python rule_quality_report.py
    python rule_quality_report.py --file transforms.jsonl
"""

import argparse
import json
import sys
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="transforms.jsonl")
    args = ap.parse_args()

    try:
        with open(args.file, encoding="utf-8") as f:
            rules = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        print(f"File not found: {args.file}")
        print("Run: python transform_compiler.py artifacts.jsonl --validate")
        sys.exit(1)

    print("RULE QUALITY REPORT")
    print("=" * 70)
    print(f"Total rules: {len(rules)}")
    print()

    by_resp = defaultdict(list)
    for r in rules:
        by_resp[r["response_type"]].append(r)

    for resp_type in sorted(by_resp.keys()):
        group = by_resp[resp_type]
        avg_freq    = sum(r["frequency"]        for r in group) / len(group)
        avg_entropy = sum(r["entropy"]          for r in group) / len(group)
        avg_conf    = sum(r["confidence"]       for r in group) / len(group)
        best  = min(group, key=lambda r: r["entropy"])
        worst = max(group, key=lambda r: r["entropy"])
        print(f"{resp_type.upper()}")
        print(f"  Rules:       {len(group)}")
        print(f"  Avg freq:    {avg_freq:.1f}")
        print(f"  Avg entropy: {avg_entropy:.3f}")
        print(f"  Avg conf:    {avg_conf:.3f}")
        print(f"  Best  (ent): {best['join']}"
              f"  bucket={best['closure_bucket']}"
              f"  ent={best['entropy']:.3f}")
        print(f"  Worst (ent): {worst['join']}"
              f"  bucket={worst['closure_bucket']}"
              f"  ent={worst['entropy']:.3f}")
        print()

    print("=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    missing_bridges = [r for r in rules if not r["bridge_words"]]
    if missing_bridges:
        print(f"  [--] {len(missing_bridges)} rules have no bridge words")
        for r in missing_bridges[:3]:
            print(f"       {r['speech_act']} + {r['response_type']} + {r['closure_bucket']}")
    else:
        print("  [OK] all rules have bridge words")

    freq_list    = [r["frequency"]  for r in rules]
    entropy_list = [r["entropy"]    for r in rules]
    conf_list    = [r["confidence"] for r in rules]

    if rules:
        print(f"  [OK] frequency range: {min(freq_list)}-{max(freq_list)}")
        if max(entropy_list) < 0.6:
            print(f"  [OK] entropy range: {min(entropy_list):.3f}-{max(entropy_list):.3f}")
        else:
            high = [r for r in rules if r["entropy"] >= 0.6]
            print(f"  [--] {len(high)} rules with entropy >= 0.6 (noisy classes)")
            for r in high[:3]:
                print(f"       {r['speech_act']} + {r['response_type']}"
                      f" ent={r['entropy']:.3f} freq={r['frequency']}")
        print(f"  [OK] confidence range: {min(conf_list):.3f}-{max(conf_list):.3f}")

    print()
    ok = (len(rules) >= 10
          and (not rules or max(entropy_list) < 0.6)
          and not missing_bridges)
    if ok:
        print("Rules ready for integration into generator.py.")
    else:
        print("Some issues found. Adjust thresholds and recompile.")


if __name__ == "__main__":
    main()
