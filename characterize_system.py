"""
characterize_system.py — Exhaustive behavioral characterization.

Reveals what the system actually does, not what we hope it does.

    python characterize_system.py
    python characterize_system.py --families 1 2 3        # run specific families
    python characterize_system.py --families 6             # edge cases only
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from analyzer import analyze
from discover import load_substrate
from generator import generate
from transform_compiler import load_rules, closure_bucket

# ---------------------------------------------------------------- setup

def _load_sub():
    print("Loading substrate...", end=" ", flush=True)
    s = load_substrate(os.path.join(_HERE, "substrate.min.bin"))
    print(f"OK  ({len(s['records']):,} headwords)")
    return s

def load_artifacts(path=None):
    if path is None:
        path = os.path.join(_HERE, "artifacts.jsonl")
    arts = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    arts.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return arts

def _match_from_artifact(artifact, rules):
    """Rule matching on raw artifact dict — no mock objects needed."""
    sa     = artifact["state"]["speech_act"]
    rt     = artifact["state"]["response_type"]
    bucket = closure_bucket(artifact["state"]["closure_size"])

    exact   = [r for r in rules
                if r["speech_act"] == sa
                and r["response_type"] == rt
                and r["closure_bucket"] == bucket]
    if exact:
        return max(exact, key=lambda r: r["confidence"])

    partial = [r for r in rules
               if r["speech_act"] == sa
               and r["response_type"] == rt]
    if partial:
        return max(partial, key=lambda r: r["confidence"])

    return None

def _bridge_stability_for(artifact):
    """Fraction of input_field words that appear verbatim in output."""
    input_words = set(artifact["state"]["input_field"])
    output_low  = artifact["output"].lower()
    bridges     = {w for w in input_words if w in output_low}
    return len(bridges) / len(input_words) if input_words else 0.0

def _five_bucket(closure_size):
    if closure_size <  30: return "tiny"
    if closure_size <  60: return "small"
    if closure_size < 120: return "medium"
    if closure_size < 250: return "large"
    return "huge"

DIVIDER = "=" * 80


# ================================================================ FAMILY 1

def family_1_determinism(sub):
    print(f"\n{DIVIDER}")
    print("FAMILY 1 — DETERMINISM VERIFICATION")
    print(DIVIDER)

    # ---- 1.1 Byte-for-byte repetition ----------------------------------
    print("\n[1.1] Byte-for-byte repetition (20 runs × 5 inputs)")
    print("-" * 60)

    TEST_INPUTS = [
        "How do I fix a broken engine?",
        "The bearing failed because lubrication was inadequate.",
        "What happens if the bolt fails?",
        "Tighten the bolt clockwise until snug.",
        "Why does the gasket fail?",
    ]

    all_deterministic = True
    for input_text in TEST_INPUTS:
        hashes  = []
        outputs = []
        for _ in range(20):
            state  = analyze(input_text, [], sub, max_layers=5,
                             frontier_cap=400, checkpoint="_det_chk.json")
            result = generate(state, sub, n=6)
            text   = result["output"]
            outputs.append(text)
            hashes.append(hashlib.sha256(text.encode()).hexdigest())

        unique = set(hashes)
        is_det = len(unique) == 1
        if not is_det:
            all_deterministic = False

        print(f"\nInput: {input_text}")
        print(f"  Unique outputs : {len(unique)}/20")
        print(f"  Deterministic  : {'YES' if is_det else 'NO'}")
        if not is_det:
            for h, c in Counter(hashes).most_common(3):
                print(f"    {h[:20]}... ({c} runs)")
        else:
            print(f"  Output: {outputs[0]}")

    print(f"\nOVERALL: {'DETERMINISTIC' if all_deterministic else 'NON-DETERMINISTIC'}")

    # ---- 1.2 Closure stability -----------------------------------------
    print(f"\n[1.2] Closure stability (20 runs, same input)")
    print("-" * 60)

    TEST_INPUT = "The bearing failed because lubrication was inadequate."
    closures, visited_counts = [], []

    for _ in range(20):
        state = analyze(TEST_INPUT, [], sub, max_layers=5,
                        frontier_cap=400, checkpoint="_cls_chk.json")
        closures.append(frozenset(state.closure))
        visited_counts.append(state.trace.get("visited_total", 0))

    unique_closures = set(closures)
    sizes = [len(c) for c in closures]

    print(f"Input: {TEST_INPUT}")
    print(f"  Unique closure sets : {len(unique_closures)}")
    print(f"  Closure size        : min={min(sizes)}  max={max(sizes)}  mean={sum(sizes)/len(sizes):.1f}")
    print(f"  Visited             : min={min(visited_counts)}  max={max(visited_counts)}"
          f"  mean={sum(visited_counts)/len(visited_counts):.1f}")
    verdict = ("IDENTICAL" if len(unique_closures) == 1
               else "MINOR variation" if len(unique_closures) <= 3
               else "SIGNIFICANT variation")
    print(f"  Verdict             : {verdict}")


# ================================================================ FAMILY 2

def family_2_rule_matching(artifacts, rules):
    print(f"\n{DIVIDER}")
    print("FAMILY 2 — RULE MATCHING DISTRIBUTION")
    print(DIVIDER)

    by_sa      = defaultdict(lambda: {"matched": 0, "total": 0})
    by_rt      = defaultdict(lambda: {"matched": 0, "total": 0})
    by_bucket  = defaultdict(lambda: {"matched": 0, "total": 0, "conf": []})
    total_matched = 0

    for a in artifacts:
        sa     = a["state"]["speech_act"]
        rt     = a["state"]["response_type"]
        bucket = closure_bucket(a["state"]["closure_size"])

        rule = _match_from_artifact(a, rules)
        hit  = rule is not None

        by_sa[sa]["total"]     += 1
        by_rt[rt]["total"]     += 1
        by_bucket[bucket]["total"] += 1

        if hit:
            total_matched += 1
            by_sa[sa]["matched"]     += 1
            by_rt[rt]["matched"]     += 1
            by_bucket[bucket]["matched"] += 1
            by_bucket[bucket]["conf"].append(rule["confidence"])

    print("\nBy speech act:")
    for sa in sorted(by_sa):
        m = by_sa[sa]
        pct = 100 * m["matched"] / m["total"]
        print(f"  {sa:<15} {m['matched']:3}/{m['total']:3}  ({pct:5.1f}%)")

    print("\nBy response type:")
    for rt in sorted(by_rt):
        m = by_rt[rt]
        pct = 100 * m["matched"] / m["total"]
        print(f"  {rt:<15} {m['matched']:3}/{m['total']:3}  ({pct:5.1f}%)")

    print("\nBy closure bucket:")
    for bucket in ("small", "medium", "large"):
        if bucket in by_bucket:
            m   = by_bucket[bucket]
            pct = 100 * m["matched"] / m["total"]
            avg_conf = (statistics.mean(m["conf"]) if m["conf"] else 0.0)
            print(f"  {bucket:<8} {m['matched']:3}/{m['total']:3}  ({pct:5.1f}%)"
                  f"  avg_conf={avg_conf:.3f}")

    pct = 100 * total_matched / len(artifacts)
    print(f"\nOverall coverage: {total_matched}/{len(artifacts)}  ({pct:.1f}%)")


# ================================================================ FAMILY 3

def family_3_bridge_stability(artifacts):
    print(f"\n{DIVIDER}")
    print("FAMILY 3 — BRIDGE WORD STABILITY BY CLOSURE SIZE")
    print(DIVIDER)

    buckets = {name: [] for name in ("tiny", "small", "medium", "large", "huge")}
    ranges  = {"tiny": (0,30), "small": (30,60), "medium": (60,120),
               "large": (120,250), "huge": (250,99999)}

    for a in artifacts:
        name = _five_bucket(a["state"]["closure_size"])
        buckets[name].append(_bridge_stability_for(a))

    print()
    any_data = False
    for name in ("tiny", "small", "medium", "large", "huge"):
        stabs = buckets[name]
        if not stabs:
            continue
        any_data = True
        lo, hi = ranges[name]
        print(f"{name.upper()}  (closure {lo}-{hi})  n={len(stabs)}")
        print(f"  Min    : {min(stabs):6.1%}")
        if len(stabs) >= 4:
            q = statistics.quantiles(stabs, n=4)
            print(f"  Q1     : {q[0]:6.1%}")
        print(f"  Median : {statistics.median(stabs):6.1%}")
        if len(stabs) >= 4:
            print(f"  Q3     : {q[2]:6.1%}")
        print(f"  Max    : {max(stabs):6.1%}")
        print(f"  Mean   : {statistics.mean(stabs):6.1%}")
        if len(stabs) >= 2:
            print(f"  Stdev  : {statistics.stdev(stabs):6.1%}")
        print()

    if not any_data:
        print("  No artifacts found.")

    # pattern summary
    medians = {name: statistics.median(buckets[name])
               for name in ("tiny","small","medium","large","huge")
               if buckets[name]}
    if len(medians) >= 2:
        vals = list(medians.values())
        trend = "DECLINING" if vals[-1] < vals[0] else "STABLE" if vals[-1] == vals[0] else "RISING"
        print(f"Pattern: Bridge stability {trend} with closure size")
        first_below_50 = next(
            (name for name in ("tiny","small","medium","large","huge")
             if medians.get(name, 1.0) < 0.5), None)
        print(f"Drops below 50% at: {first_below_50 or 'none'}")


# ================================================================ FAMILY 4

def family_4_definition_accuracy(artifacts):
    print(f"\n{DIVIDER}")
    print("FAMILY 4 — DEFINITION ACCURACY (MANUAL REVIEW SAMPLE)")
    print(DIVIDER)

    random.seed(42)
    shuffled = list(artifacts)
    random.shuffle(shuffled)

    # 3 from each five-bucket
    sample = []
    for name in ("tiny", "small", "medium", "large", "huge"):
        bucket = [a for a in shuffled if _five_bucket(a["state"]["closure_size"]) == name]
        sample.extend(bucket[:3])

    print(f"\nShowing {len(sample)} samples (3 per closure bucket)\n")
    for i, a in enumerate(sample, 1):
        bucket = _five_bucket(a["state"]["closure_size"])
        print(f"[{i:2d}]  closure={a['state']['closure_size']:4d}  bucket={bucket:<6}"
              f"  sa={a['state']['speech_act']}")
        print(f"  Input  : {a['input']}")
        print(f"  Output :")
        print(f"  {a['output']}")
        print()


# ================================================================ FAMILY 5

def family_5_discourse_effects(artifacts):
    print(f"\n{DIVIDER}")
    print("FAMILY 5 — DISCOURSE TYPE EFFECTS ON OUTPUT")
    print(DIVIDER)

    random.seed(42)
    groups = defaultdict(list)
    for a in artifacts:
        key = (a["state"]["speech_act"], a["state"]["response_type"])
        groups[key].append(a)

    for (sa, rt), group in sorted(groups.items(), key=lambda x: -len(x[1]))[:8]:
        stabs = [_bridge_stability_for(a) for a in group]
        closures = [a["state"]["closure_size"] for a in group]
        joins = Counter(a["join_used"] for a in group)

        print(f"\n{sa.upper()} → {rt.upper()}  (n={len(group)})")
        print(f"  Join distribution : {dict(joins.most_common(3))}")
        print(f"  Bridge stability  : median={statistics.median(stabs):.1%}"
              f"  mean={statistics.mean(stabs):.1%}")
        print(f"  Closure size      : median={statistics.median(closures):.0f}"
              f"  min={min(closures)}  max={max(closures)}")

        for i, a in enumerate(random.sample(group, min(2, len(group))), 1):
            print(f"\n  Sample {i}: {a['input']}")
            print(f"  Output : {a['output']}")


# ================================================================ FAMILY 6

def family_6_edge_cases(sub):
    print(f"\n{DIVIDER}")
    print("FAMILY 6 — EDGE CASES AND BREAKING POINTS")
    print(DIVIDER)

    EDGE_CASES = [
        ("bolt",
         "Single word"),
        ("xyzabc qwerty failure",
         "Unknown words"),
        ("the a an to be",
         "Only function words"),
        ("being being being",
         "Repeated polysemous word"),
        ("The bearing failed because the lubrication was inadequate and the temperature "
         "was excessive and the pressure was too high and the alignment was wrong and "
         "the material was weak and the design was flawed and the maintenance was poor "
         "and the assembly was incorrect.",
         "Extremely long"),
        ("THE BEARING FAILED!!! Why??? HOW???",
         "Extreme punctuation"),
        ("The bolt (M8x1.25) failed @500rpm with +/-0.5% tolerance",
         "Technical notation"),
        ("The bearing both failed and succeeded simultaneously.",
         "Logical contradiction"),
    ]

    results = []
    for input_text, description in EDGE_CASES:
        print(f"\n[{description}]")
        print(f"Input: {input_text}")
        try:
            state  = analyze(input_text, [], sub, max_layers=5,
                             frontier_cap=400, checkpoint="_edge_chk.json")
            result = generate(state, sub, n=6)
            print(f"  Status        : SUCCESS")
            print(f"  Speech act    : {state.frame.speech_act}")
            print(f"  Closure size  : {len(state.closure)}")
            print(f"  Output        :")
            print(f"  {result['output']}")
            results.append({"desc": description, "ok": True,
                            "closure": len(state.closure)})
        except Exception as e:
            print(f"  Status  : FAILED — {type(e).__name__}: {e}")
            results.append({"desc": description, "ok": False, "error": str(e)[:120]})

    print(f"\n{DIVIDER}")
    ok  = sum(1 for r in results if r["ok"])
    bad = len(results) - ok
    print(f"Results: {ok} succeeded  {bad} failed")
    if bad:
        print("Failures:")
        for r in results:
            if not r["ok"]:
                print(f"  - {r['desc']}: {r.get('error','?')}")


# ================================================================ MAIN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", type=int,
                    default=[1, 2, 3, 4, 5, 6],
                    help="Which test families to run (1-6)")
    ap.add_argument("--artifacts", default=os.path.join(_HERE, "artifacts.jsonl"))
    ap.add_argument("--transforms", default=os.path.join(_HERE, "transforms.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    needs_sub       = bool(set(args.families) & {1, 6})
    needs_artifacts = bool(set(args.families) & {2, 3, 4, 5})
    needs_rules     = bool(set(args.families) & {2})

    sub       = _load_sub() if needs_sub else None
    artifacts = load_artifacts(args.artifacts) if needs_artifacts else []
    rules     = load_rules(args.transforms)     if needs_rules     else []

    if needs_artifacts:
        print(f"Artifacts loaded: {len(artifacts)}")
    if needs_rules:
        print(f"Rules loaded: {len(rules)}")

    if 1 in args.families:
        family_1_determinism(sub)
    if 2 in args.families:
        family_2_rule_matching(artifacts, rules)
    if 3 in args.families:
        family_3_bridge_stability(artifacts)
    if 4 in args.families:
        family_4_definition_accuracy(artifacts)
    if 5 in args.families:
        family_5_discourse_effects(artifacts)
    if 6 in args.families:
        family_6_edge_cases(sub)

    print(f"\n{DIVIDER}")
    print("CHARACTERIZATION COMPLETE")
    print(DIVIDER)


if __name__ == "__main__":
    main()
