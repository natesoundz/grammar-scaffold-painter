"""
_scaffold_test.py — Empirical test: does scaffold compatibility drive selection?

Compares scaffold compatibility of SELECTED terms vs RANDOM samples
from the same closure, under the same join operator.

If selected terms score higher, the system is selecting by grammar.
If not, something else is steering selection.

Usage:
    python _scaffold_test.py "fire soot ash burn heat" --layers 8 --select 8
    python _scaffold_test.py "prove" --layers 8 --select 8 --trials 100
"""

import os
import sys
import random
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import load_substrate
from analyzer import analyze
from generator import generate
from scaffold import ScaffoldAnalyzer

SUB_PATH = os.path.join(_HERE, "substrate.min.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="fire soot ash burn heat")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--frontier-cap", type=int, default=600)
    ap.add_argument("--select", type=int, default=8)
    ap.add_argument("--trials", type=int, default=200, help="Random sample trials")
    ap.add_argument("--join", default="", help="Force a specific join")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for trial sampling")
    args = ap.parse_args()

    random.seed(args.seed)

    print("Loading substrate ...", end=" ", flush=True)
    sub = load_substrate(SUB_PATH)
    records = sub["records"]
    function_words = set(sub["relations"]["function_words"])
    print(f"{len(records):,} headwords\n")

    sa = ScaffoldAnalyzer()

    # Run analysis + generation
    state = analyze(
        input_text=args.text,
        context_texts=[],
        sub=sub,
        max_layers=args.layers,
        frontier_cap=args.frontier_cap,
    )

    result = generate(
        state=state,
        sub=sub,
        n=args.select,
        join_name=args.join or None,
    )

    join_name = result["join_used"]
    selected = [h for h, *_ in result["selected"]]
    closure = list(state.closure - function_words)

    print(f"Input:      {args.text}")
    print(f"Join:       {join_name}")
    print(f"Closure:    {len(closure)} headwords")
    print(f"Selected:   {len(selected)} headwords")
    print()

    # ---- Compute scaffold compatibility for SELECTED terms
    selected_compat = []
    print("SELECTED TERMS:")
    for h in selected:
        if h not in records or not records[h]["senses"]:
            continue
        defn = records[h]["senses"][0]["definition"]
        scaffold = sa.extract_scaffold(defn)
        compat = sa.scaffold_compatibility(scaffold, join_name)
        sig = sa.scaffold_signature(scaffold)
        selected_compat.append(compat)
        print(f"  {h:<20} compat={compat:.3f}  {sig}")
    
    selected_mean = sum(selected_compat) / len(selected_compat) if selected_compat else 0
    print(f"\n  Selected mean compatibility: {selected_mean:.3f}")
    print()

    # ---- Compute scaffold compatibility for RANDOM samples from same closure
    # Run many trials: pick N random headwords from closure, measure mean compat
    n = len(selected)
    trial_means = []
    random.seed(42)
    
    for _ in range(args.trials):
        if len(closure) <= n:
            sample = closure
        else:
            sample = random.sample(closure, n)
        
        compat_values = []
        for h in sample:
            if h not in records or not records[h]["senses"]:
                continue
            defn = records[h]["senses"][0]["definition"]
            scaffold = sa.extract_scaffold(defn)
            compat = sa.scaffold_compatibility(scaffold, join_name)
            compat_values.append(compat)
        
        if compat_values:
            trial_means.append(sum(compat_values) / len(compat_values))
    
    random_mean = sum(trial_means) / len(trial_means) if trial_means else 0
    random_min = min(trial_means) if trial_means else 0
    random_max = max(trial_means) if trial_means else 0
    
    # What percentile is the selected mean?
    below = sum(1 for t in trial_means if t < selected_mean)
    percentile = 100 * below / len(trial_means) if trial_means else 0
    
    print(f"RANDOM SAMPLES ({args.trials} trials, {n} per trial, from same closure):")
    print(f"  Random mean compatibility:  {random_mean:.3f}")
    print(f"  Random min:                 {random_min:.3f}")
    print(f"  Random max:                 {random_max:.3f}")
    print(f"  Selected is at percentile:  {percentile:.1f}%")
    print()

    # ---- Also test with ALL join operators to see which fits best
    print("SCAFFOLD COMPATIBILITY BY JOIN OPERATOR:")
    print(f"  {'Join':<25} {'Selected':>10} {'Random':>10} {'Delta':>10} {'Verdict':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    from scaffold import JOIN_SCAFFOLD_AFFINITY
    
    for jtype in JOIN_SCAFFOLD_AFFINITY:
        sel_vals = []
        for h in selected:
            if h not in records or not records[h]["senses"]:
                continue
            defn = records[h]["senses"][0]["definition"]
            scaffold = sa.extract_scaffold(defn)
            compat = sa.scaffold_compatibility(scaffold, jtype)
            sel_vals.append(compat)
        
        rnd_vals = []
        random.seed(42)
        for _ in range(min(100, args.trials)):
            if len(closure) <= n:
                sample = closure
            else:
                sample = random.sample(closure, n)
            vals = []
            for h in sample:
                if h not in records or not records[h]["senses"]:
                    continue
                defn = records[h]["senses"][0]["definition"]
                scaffold = sa.extract_scaffold(defn)
                compat = sa.scaffold_compatibility(scaffold, jtype)
                vals.append(compat)
            if vals:
                rnd_vals.append(sum(vals) / len(vals))
        
        sel_m = sum(sel_vals) / len(sel_vals) if sel_vals else 0
        rnd_m = sum(rnd_vals) / len(rnd_vals) if rnd_vals else 0
        delta = sel_m - rnd_m
        verdict = "GRAMMAR" if delta > 0.05 else ("WEAK" if delta > 0 else "NONE")
        active = " <-- ACTIVE" if jtype == join_name else ""
        print(f"  {jtype:<25} {sel_m:>10.3f} {rnd_m:>10.3f} {delta:>+10.3f} {verdict:>10}{active}")
    
    print()

    # ---- Language device distribution for selected vs closure
    from _scaffold_lab import classify_device, skeleton_pattern
    
    print("LANGUAGE DEVICE DISTRIBUTION:")
    print(f"  {'Device':<25} {'Selected':>10} {'Closure':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    
    from collections import Counter
    sel_devices = Counter()
    clo_devices = Counter()
    
    for h in selected:
        if h not in records or not records[h]["senses"]:
            continue
        defn = records[h]["senses"][0]["definition"]
        scaffold = sa.extract_scaffold(defn)
        pattern = skeleton_pattern(scaffold)
        device = classify_device(pattern)
        sel_devices[device] += 1
    
    for h in closure:
        if h not in records or not records[h]["senses"]:
            continue
        defn = records[h]["senses"][0]["definition"]
        scaffold = sa.extract_scaffold(defn)
        pattern = skeleton_pattern(scaffold)
        device = classify_device(pattern)
        clo_devices[device] += 1
    
    all_devices = set(sel_devices) | set(clo_devices)
    for device in sorted(all_devices, key=lambda d: clo_devices[d], reverse=True):
        sel_pct = 100 * sel_devices[device] / max(1, len(selected))
        clo_pct = 100 * clo_devices[device] / max(1, len(closure))
        marker = " ***" if sel_pct > clo_pct * 1.5 else ""
        print(f"  {device:<25} {sel_pct:>9.1f}% {clo_pct:>9.1f}%{marker}")
    
    print()
    
    # ---- Final verdict
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    if selected_mean > random_mean and percentile > 75:
        print(f"  SELECTED > RANDOM (p={percentile:.1f}%)")
        print(f"  The system IS selecting by scaffold compatibility.")
        print(f"  Grammar selects worlds. Content selects objects.")
    elif selected_mean > random_mean:
        print(f"  SELECTED > RANDOM (weak, p={percentile:.1f}%)")
        print(f"  Scaffold compatibility contributes but is not dominant.")
    else:
        print(f"  SELECTED <= RANDOM (p={percentile:.1f}%)")
        print(f"  Scaffold compatibility is NOT driving selection.")
        print(f"  Something else (anchor rank, tier, domain) is steering.")
    print()


if __name__ == "__main__":
    main()
