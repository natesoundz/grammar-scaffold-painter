"""
_scaffold_only.py — Decisive experiment: can scaffold-only signatures
(without any content words) recover neighborhood structure?

Strip every content word from definitions. Keep only function-word scaffolds.
Then measure: do definitions in the same semantic neighborhood share scaffold
signatures more than definitions in different neighborhoods?

Test 1: Given a headword, retrieve its closure neighbors using ONLY scaffold
        similarity (no content words). Compare to content-based retrieval.

Test 2: Can scaffold-only signatures distinguish domains?
        Group: fuel, engine, tool, weapon, instrument (physical/functional)
        Group: justice, beauty, sorrow, virtue, truth (abstract/evaluative)
        Do their scaffold signatures cluster?

Test 3: At what closure richness does topology hand control to grammar?
        Run scaffold compatibility at increasing closure sizes and find
        the transition point where grammar-driven selection dominates.

Usage:
    python _scaffold_only.py --group1 "fuel engine tool weapon instrument" --group2 "justice beauty sorrow virtue truth"
    python _scaffold_only.py --word fuel --layers 8
    python _scaffold_only.py --transition "fire heat burn" --max-layers 12
"""

import os
import sys
import re
import argparse
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import load_substrate
from scaffold import ScaffoldAnalyzer, _SCAFFOLD_ROLES, _FUNCTION_WORD_RE, JOIN_SCAFFOLD_AFFINITY

SUB_PATH = os.path.join(_HERE, "substrate.min.bin")


def strip_to_scaffold(definition):
    """Strip all content words. Keep only function words as role sequence.
    
    "any matter used to produce heat by burning"
    → ['QUANTIFY', 'DIRECT', 'CAUSE']
    (any=QUANTIFY, to=DIRECT, by=CAUSE — matter/used/produce/heat/burning removed)
    """
    roles = []
    for m in _FUNCTION_WORD_RE.finditer(definition):
        word = m.group(1).lower()
        role = _SCAFFOLD_ROLES.get(word)
        if role:
            roles.append(role)
    return roles


def scaffold_signature(roles):
    """Compact signature from role list."""
    abbrev = {
        "COMPOSE": "CMP", "CONTAIN": "CON", "CAUSE": "CAU", "DIRECT": "DIR",
        "IDENTIFY": "IDN", "SPECIFY": "SPC", "CONTRAST": "CTN", "NEGATE": "NEG",
        "SEQUENCE": "SEQ", "CONDITION": "CND", "ARTICLE": "ART", "QUANTIFY": "QNT",
        "CONJOIN": "CNJ", "PURPOSE": "PUR", "ROLE": "ROL",
    }
    return "·".join(abbrev.get(r, r[:3]) for r in roles)


def scaffold_similarity(sig_a, sig_b):
    """Jaccard similarity between two scaffold signatures (as role multisets)."""
    a = Counter(sig_a)
    b = Counter(sig_b)
    intersection = sum((a & b).values())
    union = sum((a | b).values())
    return intersection / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group1", default="fuel engine tool weapon instrument",
                    help="First semantic group (physical/functional)")
    ap.add_argument("--group2", default="justice beauty sorrow virtue truth",
                    help="Second semantic group (abstract/evaluative)")
    ap.add_argument("--word", default="", help="Single word scaffold retrieval test")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--frontier-cap", type=int, default=600)
    ap.add_argument("--transition", default="", help="Transition point experiment")
    ap.add_argument("--max-layers", type=int, default=12)
    args = ap.parse_args()

    print("Loading substrate ...", end=" ", flush=True)
    sub = load_substrate(SUB_PATH)
    records = sub["records"]
    relations = sub["relations"]
    function_words = set(relations["function_words"])
    print(f"{len(records):,} headwords\n")

    sa = ScaffoldAnalyzer()

    # ================================================================
    # TEST 1: Scaffold signatures for two semantic groups
    # ================================================================
    print("=" * 70)
    print("TEST 1: SCAFFOLD-ONLY SIGNATURES FOR TWO SEMANTIC GROUPS")
    print("=" * 70)
    print()

    group1 = args.group1.lower().split()
    group2 = args.group2.lower().split()

    print(f"Group 1 (physical/functional): {group1}")
    print(f"Group 2 (abstract/evaluative): {group2}")
    print()

    # Extract scaffolds for each group
    group1_scaffolds = {}
    group2_scaffolds = {}

    print("GROUP 1 — scaffold signatures:")
    for h in group1:
        if h not in records or not records[h]["senses"]:
            print(f"  {h}: NOT FOUND")
            continue
        defn = records[h]["senses"][0]["definition"]
        scaffold = strip_to_scaffold(defn)
        sig = scaffold_signature(scaffold)
        group1_scaffolds[h] = scaffold
        print(f"  {h:<15} {sig}")
        print(f"  {'':15} defn: {defn[:70]}")
        print(f"  {'':15} stripped: {' '.join(_SCAFFOLD_ROLES.get(w, '___') for w in _FUNCTION_WORD_RE.findall(defn))}")
        print()

    print("GROUP 2 — scaffold signatures:")
    for h in group2:
        if h not in records or not records[h]["senses"]:
            print(f"  {h}: NOT FOUND")
            continue
        defn = records[h]["senses"][0]["definition"]
        scaffold = strip_to_scaffold(defn)
        sig = scaffold_signature(scaffold)
        group2_scaffolds[h] = scaffold
        print(f"  {h:<15} {sig}")
        print(f"  {'':15} defn: {defn[:70]}")
        print(f"  {'':15} stripped: {' '.join(_SCAFFOLD_ROLES.get(w, '___') for w in _FUNCTION_WORD_RE.findall(defn))}")
        print()

    # ================================================================
    # TEST 2: Within-group vs between-group scaffold similarity
    # ================================================================
    print("=" * 70)
    print("TEST 2: WITHIN-GROUP vs BETWEEN-GROUP SCAFFOLD SIMILARITY")
    print("=" * 70)
    print()

    # Within-group similarities
    within1 = []
    g1_words = list(group1_scaffolds.keys())
    for i in range(len(g1_words)):
        for j in range(i+1, len(g1_words)):
            sim = scaffold_similarity(group1_scaffolds[g1_words[i]],
                                      group1_scaffolds[g1_words[j]])
            within1.append(sim)
            print(f"  G1: {g1_words[i]:<12} ~ {g1_words[j]:<12} = {sim:.3f}")

    within2 = []
    g2_words = list(group2_scaffolds.keys())
    for i in range(len(g2_words)):
        for j in range(i+1, len(g2_words)):
            sim = scaffold_similarity(group2_scaffolds[g2_words[i]],
                                      group2_scaffolds[g2_words[j]])
            within2.append(sim)
            print(f"  G2: {g2_words[i]:<12} ~ {g2_words[j]:<12} = {sim:.3f}")

    # Between-group similarities
    between = []
    for h1 in g1_words:
        for h2 in g2_words:
            sim = scaffold_similarity(group1_scaffolds[h1], group2_scaffolds[h2])
            between.append(sim)
            print(f"  X:  {h1:<12} ~ {h2:<12} = {sim:.3f}")

    within_mean = sum(within1 + within2) / max(1, len(within1 + within2))
    between_mean = sum(between) / max(1, len(between))
    separation = within_mean - between_mean

    print()
    print(f"  Within-group mean similarity:  {within_mean:.3f}")
    print(f"  Between-group mean similarity: {between_mean:.3f}")
    print(f"  Separation (within - between): {separation:.3f}")
    print()

    if separation > 0.05:
        print(f"  VERDICT: SCAFFOLD-ONLY RETRIEVAL SEPARATES GROUPS")
        print(f"  Grammar alone carries enough signal to distinguish domains.")
    elif separation > 0:
        print(f"  VERDICT: WEAK SEPARATION")
        print(f"  Scaffold similarity is slightly higher within groups.")
    else:
        print(f"  VERDICT: NO SEPARATION")
        print(f"  Scaffold signatures do not distinguish these groups.")
    print()

    # ================================================================
    # TEST 3: Scaffold-only retrieval — can we recover neighbors?
    # ================================================================
    print("=" * 70)
    print("TEST 3: SCAFFOLD-ONLY NEIGHBOR RETRIEVAL")
    print("=" * 70)
    print()

    # For each group1 word, find the top-5 most scaffold-similar words
    # in the entire dictionary (using only scaffold signatures)
    # Then check: are other group1 words in the top results?

    print("For each Group 1 word, find top-5 scaffold-similar headwords")
    print("in the entire dictionary. Check if other Group 1 members appear.")
    print()

    # Precompute all scaffolds (sample if dictionary is large)
    all_scaffolds = {}
    for h, rec in records.items():
        if not rec.get("senses"):
            continue
        defn = rec["senses"][0].get("definition", "")
        if not defn or len(defn) < 10:
            continue
        scaffold = strip_to_scaffold(defn)
        if scaffold:
            all_scaffolds[h] = scaffold

    print(f"  Computed scaffolds for {len(all_scaffolds):,} headwords")
    print()

    g1_set = set(group1)
    g2_set = set(group2)

    retrieval_hits = 0
    retrieval_total = 0

    for h in g1_words:
        if h not in all_scaffolds:
            continue
        query_sig = all_scaffolds[h]
        
        # Find top-5 most similar (excluding self)
        sims = []
        for other, other_sig in all_scaffolds.items():
            if other == h:
                continue
            sim = scaffold_similarity(query_sig, other_sig)
            sims.append((other, sim))
        
        sims.sort(key=lambda x: x[1], reverse=True)
        top5 = sims[:5]
        
        hits = [other for other, _ in top5 if other in g1_set - {h}]
        misses_g2 = [other for other, _ in top5 if other in g2_set]
        
        print(f"  Query: {h}  (scaffold: {scaffold_signature(query_sig)})")
        for other, sim in top5:
            marker = ""
            if other in g1_set - {h}:
                marker = " *** G1 HIT"
            elif other in g2_set:
                marker = " (G2)"
            print(f"    {other:<20} sim={sim:.3f}{marker}")
        
        retrieval_hits += len(hits)
        retrieval_total += 4  # other 4 group1 members
        print(f"    Hits: {len(hits)}/4 group members found in top-5")
        print()

    print(f"  Overall scaffold-only retrieval: {retrieval_hits}/{retrieval_total} group members recovered")
    print()

    # ================================================================
    # TEST 4: Dominant scaffold patterns per group
    # ================================================================
    print("=" * 70)
    print("TEST 4: DOMINANT SCAFFOLD PATTERNS PER GROUP")
    print("=" * 70)
    print()

    g1_patterns = Counter()
    g2_patterns = Counter()

    for h in g1_words:
        if h in all_scaffolds:
            g1_patterns[scaffold_signature(all_scaffolds[h])] += 1
    for h in g2_words:
        if h in all_scaffolds:
            g2_patterns[scaffold_signature(all_scaffolds[h])] += 1

    print("Group 1 patterns:")
    for sig, count in g1_patterns.most_common():
        print(f"  [{count}] {sig}")

    print()
    print("Group 2 patterns:")
    for sig, count in g2_patterns.most_common():
        print(f"  [{count}] {sig}")

    # Check overlap
    g1_sig_set = set(g1_patterns.keys())
    g2_sig_set = set(g2_patterns.keys())
    overlap = g1_sig_set & g2_sig_set
    print()
    print(f"  Shared signatures: {len(overlap)}/{len(g1_sig_set | g2_sig_set)}")
    if overlap:
        print(f"  Overlapping: {overlap}")
    print()

    # ================================================================
    # TEST 5: Transition point — when does grammar take over?
    # ================================================================
    if args.transition:
        print("=" * 70)
        print("TEST 5: TRANSITION POINT — WHEN DOES GRAMMAR DOMINATE?")
        print("=" * 70)
        print()

        from analyzer import analyze
        from generator import generate
        import random

        transition_words = args.transition.lower().split()
        print(f"Input: {transition_words}")
        print()

        print(f"  {'Layers':>6}  {'Closure':>8}  {'Sel compat':>12}  {'Rand compat':>12}  {'Delta':>8}  {'Verdict':>10}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*10}")

        for n_layers in range(2, args.max_layers + 1, 2):
            state = analyze(
                input_text=" ".join(transition_words),
                context_texts=[],
                sub=sub,
                max_layers=n_layers,
                frontier_cap=600,
            )
            closure = list(state.closure - function_words)
            if len(closure) < 8:
                continue

            result = generate(state=state, sub=sub, n=8)
            join_name = result["join_used"]
            selected = [h for h, *_ in result["selected"]]

            # Selected compatibility
            sel_compat = []
            for h in selected:
                if h not in records or not records[h]["senses"]:
                    continue
                defn = records[h]["senses"][0]["definition"]
                scaffold = sa.extract_scaffold(defn)
                c = sa.scaffold_compatibility(scaffold, join_name)
                sel_compat.append(c)

            # Random baseline
            random.seed(42)
            rnd_compat = []
            for _ in range(100):
                sample = random.sample(closure, min(8, len(closure)))
                vals = []
                for h in sample:
                    if h not in records or not records[h]["senses"]:
                        continue
                    defn = records[h]["senses"][0]["definition"]
                    scaffold = sa.extract_scaffold(defn)
                    c = sa.scaffold_compatibility(scaffold, join_name)
                    vals.append(c)
                if vals:
                    rnd_compat.append(sum(vals) / len(vals))

            sel_mean = sum(sel_compat) / len(sel_compat) if sel_compat else 0
            rnd_mean = sum(rnd_compat) / len(rnd_compat) if rnd_compat else 0
            delta = sel_mean - rnd_mean
            verdict = "GRAMMAR" if delta > 0.05 else ("WEAK" if delta > 0 else "NONE")

            print(f"  {n_layers:>6}  {len(closure):>8}  {sel_mean:>12.3f}  {rnd_mean:>12.3f}  {delta:>+8.3f}  {verdict:>10}")

        print()

    # ================================================================
    # SUMMARY
    # ================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"  Within-group scaffold similarity:  {within_mean:.3f}")
    print(f"  Between-group scaffold similarity: {between_mean:.3f}")
    print(f"  Separation:                        {separation:.3f}")
    print(f"  Scaffold-only retrieval:           {retrieval_hits}/{retrieval_total} group members recovered")
    print()
    if separation > 0.05 and retrieval_hits > retrieval_total * 0.3:
        print("  SCAFFOLD-ONLY RETRIEVAL RECOVERS NEIGHBORHOOD STRUCTURE.")
        print("  Grammar alone carries sufficient semantic signal.")
        print("  The function-word skeleton IS the ontology.")
    elif separation > 0:
        print("  Partial signal in scaffold signatures.")
        print("  Grammar contributes but does not fully determine neighborhood.")
    else:
        print("  Scaffold signatures alone do not separate semantic groups.")
        print("  Content words carry the primary semantic load.")
    print()


if __name__ == "__main__":
    main()
