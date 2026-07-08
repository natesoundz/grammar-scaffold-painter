"""
_scaffold_fill.py — Build scaffold neighbors to fill structural gaps.

Given a set of selected headwords, analyze which scaffold roles are
present and which are missing relative to the join operator's affinity.
Then search the closure for headwords whose definitions supply those
missing roles — structural complements, not semantic neighbors.

This produces outputs where the selected set collectively covers all
relational roles the join operator expects.

Usage:
    python _scaffold_fill.py "fire soot ash burn heat" --layers 8 --select 6
"""

import os
import sys
import argparse
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import load_substrate
from analyzer import analyze
from generator import generate
from scaffold import ScaffoldAnalyzer, JOIN_SCAFFOLD_AFFINITY, _SCAFFOLD_ROLES, _FUNCTION_WORD_RE

SUB_PATH = os.path.join(_HERE, "substrate.min.bin")


def find_scaffold_fillers(selected, closure, records, join_name, sa, n_fill=3):
    """Find headwords in the closure whose scaffolds supply roles
    that the selected set is missing relative to the join's affinity.
    
    Returns list of (headword, scaffold, missing_roles_filled, compatibility).
    """
    # What roles does the join want?
    affinity = JOIN_SCAFFOLD_AFFINITY.get(join_name, set())
    if not affinity:
        return []
    
    # What roles do the selected terms already cover?
    covered_roles = set()
    for h in selected:
        if h not in records or not records[h]["senses"]:
            continue
        defn = records[h]["senses"][0]["definition"]
        scaffold = sa.extract_scaffold(defn)
        covered_roles.update(scaffold)
    
    # What's missing?
    missing_roles = affinity - covered_roles
    if not missing_roles:
        return []
    
    # Search the closure for headwords that supply missing roles
    candidates = []
    selected_set = set(selected)
    
    for h in closure:
        if h in selected_set:
            continue
        if h not in records or not records[h]["senses"]:
            continue
        defn = records[h]["senses"][0]["definition"]
        scaffold = sa.extract_scaffold(defn)
        if not scaffold:
            continue
        
        scaffold_roles = set(scaffold)
        # How many missing roles does this headword fill?
        fills = scaffold_roles & missing_roles
        if not fills:
            continue
        
        # Compatibility with the join
        compat = sa.scaffold_compatibility(scaffold, join_name)
        
        # Score: number of missing roles filled, weighted by compatibility
        score = len(fills) * compat
        
        candidates.append({
            "headword": h,
            "scaffold": scaffold,
            "sig": sa.scaffold_signature(scaffold),
            "fills": sorted(fills),
            "compatibility": compat,
            "score": score,
            "definition": defn[:100],
        })
    
    # Sort by score (most missing roles filled * highest compat)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    
    # Deduplicate: don't pick two headwords that fill the same roles
    chosen = []
    remaining_missing = set(missing_roles)
    
    for c in candidates:
        if not remaining_missing:
            break
        new_fills = set(c["fills"]) & remaining_missing
        if not new_fills:
            continue
        c["new_fills"] = sorted(new_fills)
        chosen.append(c)
        remaining_missing -= new_fills
    
    return chosen, missing_roles, covered_roles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="fire soot ash burn heat")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--frontier-cap", type=int, default=600)
    ap.add_argument("--select", type=int, default=6)
    ap.add_argument("--join", default="")
    ap.add_argument("--n-fill", type=int, default=4, help="Max scaffold filler terms")
    args = ap.parse_args()

    print("Loading substrate ...", end=" ", flush=True)
    sub = load_substrate(SUB_PATH)
    records = sub["records"]
    function_words = set(sub["relations"]["function_words"])
    print(f"{len(records):,} headwords\n")

    sa = ScaffoldAnalyzer()

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

    print(f"Input:    {args.text}")
    print(f"Join:     {join_name}")
    print(f"Closure:  {len(closure)} headwords")
    print()

    # ---- Show selected terms and their scaffold roles
    affinity = JOIN_SCAFFOLD_AFFINITY.get(join_name, set())
    print(f"JOIN AFFINITY ROLES: {sorted(affinity)}")
    print()

    print("SELECTED TERMS — scaffold roles:")
    covered_roles = set()
    for h in selected:
        if h not in records or not records[h]["senses"]:
            continue
        defn = records[h]["senses"][0]["definition"]
        scaffold = sa.extract_scaffold(defn)
        roles = set(scaffold)
        covered_roles.update(roles)
        matching = roles & affinity
        sig = sa.scaffold_signature(scaffold)
        print(f"  {h:<20} {sig}")
        print(f"    {'':20} matching: {sorted(matching) if matching else 'NONE'}")
        print(f"    {'':20} defn: {defn[:80]}")
    
    print()

    # ---- Find missing roles
    missing_roles = affinity - covered_roles
    print(f"ROLES COVERED BY SELECTION:  {sorted(covered_roles & affinity)}")
    print(f"ROLES MISSING FROM SELECTION: {sorted(missing_roles) if missing_roles else 'NONE — full coverage'}")
    print()

    if not missing_roles:
        print("No structural gaps — selected set covers all join affinity roles.")
        return

    # ---- Find scaffold fillers
    fillers, missing, covered = find_scaffold_fillers(
        selected, closure, records, join_name, sa, n_fill=args.n_fill
    )

    if not fillers:
        print("No closure members supply the missing roles.")
        return

    print(f"SCAFFOLD FILLERS — {len(fillers)} terms that fill structural gaps:")
    print()

    for i, f in enumerate(fillers):
        print(f"  [{i+1}] {f['headword']:<20} fills: {f['new_fills']}")
        print(f"      scaffold: {f['sig']}")
        print(f"      compat:   {f['compatibility']:.3f}")
        print(f"      defn:     {f['definition']}")
        print()

    # ---- Simulate the augmented output
    print("=" * 60)
    print("AUGMENTED OUTPUT — original + scaffold fillers")
    print("=" * 60)
    print()

    # Get the original output
    print(f"ORIGINAL (join={join_name}):")
    print(f"  {result['output']}")
    print()

    # Build augmented set: original selected + fillers
    augmented = selected + [f["headword"] for f in fillers]
    
    # Re-check coverage
    new_covered = set(covered_roles)
    for f in fillers:
        new_covered.update(f["scaffold"])
    
    new_missing = affinity - new_covered
    print(f"AUGMENTED SET: {augmented}")
    print(f"  Roles now covered: {sorted(new_covered & affinity)}")
    print(f"  Roles still missing: {sorted(new_missing) if new_missing else 'NONE — full coverage'}")
    print()

    # Show what the filler definitions look like in context
    print("FILLER DEFINITIONS IN CONTEXT:")
    from generator import _statement_for, _statement_composite, MODE_ROTATION, NEIGHBOR_MODE_DEFAULT
    
    relations = sub["relations"]
    oppositions = relations.get("oppositions", {})
    crossrefs = relations.get("crossrefs", {})
    morphological = relations.get("morphological", {})
    
    for f in fillers:
        h = f["headword"]
        if h not in records or not records[h]["senses"]:
            continue
        sense = records[h]["senses"][0]
        defn = sense["definition"]
        base = defn.split(";")[0].strip().rstrip(".")
        if base:
            core = f"{h.capitalize()}: {base[0].lower() + base[1:]}"
        else:
            core = f"{h.capitalize()}: {defn[:60]}"
        print(f"  {core}")
    
    print()

    # ---- Show the reskin possibility: take the combined scaffold and 
    #      show what structure the full set represents
    print("=" * 60)
    print("COMBINED SCAFFOLD SIGNATURE")
    print("=" * 60)
    print()
    
    all_scaffolds = []
    for h in augmented:
        if h not in records or not records[h]["senses"]:
            continue
        defn = records[h]["senses"][0]["definition"]
        scaffold = sa.extract_scaffold(defn)
        all_scaffolds.append((h, scaffold, sa.scaffold_signature(scaffold)))
    
    print(f"  {'Headword':<20} {'Scaffold signature'}")
    print(f"  {'-'*20} {'-'*40}")
    for h, scaffold, sig in all_scaffolds:
        print(f"  {h:<20} {sig}")
    
    # Aggregate role distribution
    role_counts = Counter()
    for _, scaffold, _ in all_scaffolds:
        role_counts.update(scaffold)
    
    print()
    print(f"  AGGREGATE ROLE DISTRIBUTION:")
    for role, count in role_counts.most_common():
        marker = " ***" if role in affinity else ""
        bar = "#" * min(count, 30)
        print(f"    {role:<15} {count:>3} {bar}{marker}")
    
    print()
    
    # ---- Reskin experiment: take the most common scaffold pattern
    #      and fill it with content from a different domain
    print("=" * 60)
    print("RESKIN: borrow the dominant scaffold, swap content")
    print("=" * 60)
    print()
    
    from _scaffold_lab import extract_skeleton, reskin, skeleton_pattern, classify_device
    
    # Find the scaffold with highest join compatibility
    best_h = None
    best_compat = -1
    best_scaffold = None
    for h, scaffold, sig in all_scaffolds:
        c = sa.scaffold_compatibility(scaffold, join_name)
        if c > best_compat:
            best_compat = c
            best_h = h
            best_scaffold = scaffold
    
    if best_h and best_h in records:
        defn = records[best_h]["senses"][0]["definition"]
        skeleton, slots = extract_skeleton(defn)
        pattern = skeleton_pattern(best_scaffold)
        device = classify_device(pattern)
        
        print(f"  Donor: {best_h} (compat={best_compat:.3f}, device={device})")
        print(f"  Original: {defn[:80]}")
        print(f"  Skeleton: {skeleton}")
        print(f"  Pattern:  {pattern}")
        print()
        
        # Fill with content from the filler headwords' definitions
        import re
        all_content = []
        for f in fillers:
            h = f["headword"]
            if h in records and records[h]["senses"]:
                d = records[h]["senses"][0]["definition"]
                words = [w for w in re.findall(r'[a-zA-Z]+', d)
                        if w.lower() not in set(_SCAFFOLD_ROLES.keys()) and len(w) > 2]
                all_content.extend(words[:4])
        
        n_slots = sum(1 for s in slots if s)
        if all_content and n_slots > 0:
            print(f"  Content source: filler definitions")
            print(f"  Slots to fill: {n_slots}")
            print()
            
            for trial in range(min(4, len(all_content) // max(1, n_slots))):
                start = trial * n_slots
                if start + n_slots <= len(all_content):
                    content = all_content[start:start+n_slots]
                    reskinned = reskin(skeleton, slots, content)
                    print(f"  [{trial}] {reskinned}")
            print()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"  Original selected: {len(selected)} terms, {len(covered_roles & affinity)}/{len(affinity)} roles covered")
    print(f"  Scaffold fillers:  {len(fillers)} terms, filling {len(affinity - new_missing)}/{len(missing_roles)} gaps")
    print(f"  Final coverage:    {len(new_covered & affinity)}/{len(affinity)} roles")
    if new_missing:
        print(f"  Still missing:     {sorted(new_missing)}")
    else:
        print(f"  COMPLETE COVERAGE — all join affinity roles represented")


if __name__ == "__main__":
    main()
