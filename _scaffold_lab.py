"""
_scaffold_lab.py — Experimental scaffold laboratory.

Extracts function-word skeletons from real definitions, classifies
language devices by scaffold pattern, and re-skins skeletons with
different content words to generate novel text.

Usage:
    python _scaffold_lab.py
    python _scaffold_lab.py --word fire
    python _scaffold_lab.py --reskin fire --theme power
"""

import os
import re
import sys
import json
import lzma
import msgpack
from collections import Counter, defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scaffold import ScaffoldAnalyzer, _SCAFFOLD_ROLES, _FUNCTION_WORD_RE
from discover import load_substrate

SUB_PATH = os.path.join(_HERE, "substrate.min.bin")

# ---------------------------------------------------------------- skeleton extraction

_FUNC_WORDS = set(_SCAFFOLD_ROLES.keys())

def extract_skeleton(definition):
    """Split a definition into (skeleton, content_slots).
    
    skeleton: list of function words in order, with None placeholders for content
    content_slots: list of content word groups between function words
    
    "a force in nature which is recognized in various effects"
    → skeleton: ['a', 'in', 'which', 'is', 'in']
    → content_slots: [['force'], ['nature'], [], ['recognized'], ['various', 'effects']]
    """
    tokens = definition.split()
    skeleton = []
    slots = []
    current_slot = []
    
    for tok in tokens:
        clean = re.sub(r'[^a-zA-Z]', '', tok).lower()
        if clean in _FUNC_WORDS:
            if current_slot:
                slots.append(current_slot)
                current_slot = []
            elif skeleton:
                slots.append([])
            skeleton.append(clean)
        else:
            current_slot.append(clean)
    
    if current_slot:
        slots.append(current_slot)
    elif skeleton and not slots:
        slots.append([])
    
    # Pad: if no leading function word, first slot is leading content
    if skeleton and definition and not definition.split()[0].lower() in _FUNC_WORDS:
        slots = [current_slot] + slots if current_slot else slots
    
    return skeleton, slots

def skeleton_pattern(skeleton):
    """Convert function word list to role pattern string."""
    return ' '.join(_SCAFFOLD_ROLES.get(w, w) for w in skeleton)

def render_skeleton(skeleton, slots):
    """Render skeleton with slot markers.
    
    'a force in nature which is recognized in various effects'
    → '[a] ___ [in] ___ [which] [is] ___ [in] ___'
    """
    parts = []
    si = 0
    for i, fw in enumerate(skeleton):
        if i < len(slots) and slots[i]:
            parts.append(f"___ [{fw}]")
        else:
            parts.append(f"[{fw}]")
    # trailing content
    if len(slots) > len(skeleton) and slots[-1]:
        parts.append("___")
    return ' '.join(parts)

def reskin(skeleton, slots, new_content):
    """Fill a skeleton with new content words.
    
    new_content: flat list of words to fill slots in order.
    Returns the re-skinned definition string.
    """
    result = []
    ci = 0
    for i, fw in enumerate(skeleton):
        if i < len(slots) and slots[i]:
            n_needed = len(slots[i])
            replacement = new_content[ci:ci+n_needed]
            ci += n_needed
            if replacement:
                result.append(' '.join(replacement))
        result.append(fw)
    # trailing content
    if len(slots) > len(skeleton) and slots[-1]:
        replacement = new_content[ci:ci+len(slots[-1])]
        if replacement:
            result.append(' '.join(replacement))
    return ' '.join(result)

# ---------------------------------------------------------------- language device classification

def classify_device(skeleton_roles):
    """Classify a scaffold pattern as a language device."""
    roles = skeleton_roles.split()
    role_set = set(roles)
    
    # Check patterns
    has_article = "ARTICLE" in role_set
    has_compose = "COMPOSE" in role_set
    has_contain = "CONTAIN" in role_set
    has_cause = "CAUSE" in role_set
    has_direct = "DIRECT" in role_set
    has_identify = "IDENTIFY" in role_set
    has_specify = "SPECIFY" in role_set
    has_negate = "NEGATE" in role_set
    has_contrast = "CONTRAST" in role_set
    has_condition = "CONDITION" in role_set
    has_purpose = "PURPOSE" in role_set
    has_role = "ROLE" in role_set
    
    if has_negate and has_contrast:
        return "negation_contrast"
    if has_negate:
        return "negation"
    if has_contrast:
        return "contrast"
    if has_condition:
        return "conditional"
    if has_cause and has_direct and not has_identify:
        return "causal_chain"
    if has_cause and has_compose:
        return "causal_composition"
    if has_compose and has_contain and has_specify:
        return "nested_specification"
    if has_compose and has_contain:
        return "compositional_containment"
    if has_compose and has_identify:
        return "definitional_composition"
    if has_compose and has_article:
        return "taxonomic"
    if has_purpose and has_direct:
        return "purposive"
    if has_role and has_compose:
        return "role_assignment"
    if has_direct and has_article and not has_compose:
        return "directive"
    if has_identify and has_specify:
        return "identificational"
    if has_article and has_compose:
        return "classificatory"
    if has_direct:
        return "action"
    if has_article:
        return "bare_nominal"
    return "other"

# ---------------------------------------------------------------- main

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--word", default="", help="Inspect a specific word's scaffold")
    ap.add_argument("--reskin", default="", help="Word whose skeleton to borrow")
    ap.add_argument("--theme", default="", help="Theme words to fill the skeleton")
    ap.add_argument("--n", type=int, default=20, help="Number of examples per device")
    args = ap.parse_args()
    
    print("Loading substrate ...", end=" ", flush=True)
    sub = load_substrate(SUB_PATH)
    records = sub["records"]
    print(f"{len(records):,} headwords\n")
    
    sa = ScaffoldAnalyzer()
    
    if args.reskin:
        # ---- RESKIN MODE: borrow a skeleton, fill with new content
        h = args.reskin.lower()
        if h not in records:
            print(f"'{h}' not found")
            return
        defn = records[h]["senses"][0]["definition"]
        skeleton, slots = extract_skeleton(defn)
        pattern = skeleton_pattern(skeleton)
        device = classify_device(pattern)
        
        print(f"Skeleton donor: {h}")
        print(f"  Definition: {defn}")
        print(f"  Skeleton: {skeleton}")
        print(f"  Pattern: {pattern}")
        print(f"  Device: {device}")
        print(f"  Slots: {slots}")
        print()
        
        if args.theme:
            # Parse theme words from substrate
            theme_words = [w.strip().lower() for w in args.theme.split()]
            # For each slot, pick a theme word
            n_slots = sum(1 for s in slots if s)
            print(f"Filling {n_slots} content slots with theme: {theme_words}")
            print()
            
            # Generate multiple reskins by rotating through theme words
            for offset in range(min(5, len(theme_words))):
                content = []
                for i in range(n_slots):
                    w = theme_words[(offset + i) % len(theme_words)]
                    content.append(w)
                reskinned = reskin(skeleton, slots, content)
                print(f"  [{offset}] {reskinned}")
            print()
            
            # Also try: find definitions of theme words and use their content
            print("Using theme word definitions as content source:")
            all_content = []
            for tw in theme_words:
                if tw in records:
                    d = records[tw]["senses"][0]["definition"]
                    content_words = [w for w in re.findall(r'[a-zA-Z]+', d) 
                                   if w.lower() not in _FUNC_WORDS and len(w) > 2]
                    all_content.extend(content_words[:5])
            
            if all_content:
                for trial in range(3):
                    start = trial * n_slots
                    if start + n_slots <= len(all_content):
                        content = all_content[start:start+n_slots]
                        reskinned = reskin(skeleton, slots, content)
                        print(f"  [def-trial {trial}] {reskinned}")
            print()
        return
    
    if args.word:
        # ---- SINGLE WORD MODE
        h = args.word.lower()
        if h not in records:
            print(f"'{h}' not found")
            return
        for i, sense in enumerate(records[h]["senses"][:3]):
            defn = sense["definition"]
            skeleton, slots = extract_skeleton(defn)
            scaffold = sa.extract_scaffold(defn)
            pattern = skeleton_pattern(skeleton)
            device = classify_device(pattern)
            sig = sa.scaffold_signature(scaffold)
            print(f"Sense {i+1}: {defn}")
            print(f"  Skeleton: {skeleton}")
            print(f"  Roles: {sig}")
            print(f"  Device: {device}")
            print(f"  Slots: {slots}")
            print()
        return
    
    # ---- SURVEY MODE: scan all definitions, classify devices, show examples
    print("Scanning definitions for scaffold patterns ...\n")
    
    device_examples = defaultdict(list)
    device_counts = Counter()
    pattern_counts = Counter()
    
    for h, rec in records.items():
        if not rec["senses"]:
            continue
        defn = rec["senses"][0]["definition"]
        if not defn or len(defn) < 20:
            continue
        skeleton, slots = extract_skeleton(defn)
        if not skeleton:
            continue
        scaffold = sa.extract_scaffold(defn)
        pattern = skeleton_pattern(scaffold)
        device = classify_device(pattern)
        
        device_counts[device] += 1
        pattern_counts[pattern] += 1
        
        if len(device_examples[device]) < args.n:
            device_examples[device].append({
                "headword": h,
                "definition": defn,
                "skeleton": skeleton,
                "slots": slots,
                "pattern": pattern,
                "sig": sa.scaffold_signature(scaffold),
            })
    
    # Print device distribution
    print("=" * 70)
    print("LANGUAGE DEVICE DISTRIBUTION")
    print("=" * 70)
    for device, count in device_counts.most_common():
        pct = 100 * count / sum(device_counts.values())
        print(f"  {device:<25} {count:>7,}  ({pct:.1f}%)")
    print()
    
    # Print top scaffold patterns
    print("=" * 70)
    print("TOP SCAFFOLD PATTERNS")
    print("=" * 70)
    for pattern, count in pattern_counts.most_common(20):
        print(f"  [{count:>6,}] {pattern}")
    print()
    
    # Print examples per device
    print("=" * 70)
    print("EXAMPLES PER LANGUAGE DEVICE")
    print("=" * 70)
    for device in sorted(device_examples.keys()):
        examples = device_examples[device]
        print(f"\n--- {device} ({device_counts[device]:,} total) ---")
        for ex in examples[:5]:
            print(f"  {ex['headword']}: {ex['definition'][:80]}")
            print(f"    skeleton: {ex['sig']}")
            # Show the skeleton with content removed
            skel_render = render_skeleton(ex['skeleton'], ex['slots'])
            print(f"    structure: {skel_render[:80]}")
            print()
    
    # ---- RESKIN DEMO: take a few interesting skeletons and swap content
    print("=" * 70)
    print("RESKIN DEMO: same skeleton, different content")
    print("=" * 70)
    
    # Pick a few rich skeletons from different devices
    demo_devices = ["causal_chain", "nested_specification", "taxonomic", "contrast"]
    for device in demo_devices:
        if device not in device_examples:
            continue
        ex = device_examples[device][0]
        skeleton = ex["skeleton"]
        slots = ex["slots"]
        n_slots = sum(1 for s in slots if s)
        
        print(f"\nDevice: {device}")
        print(f"  Donor: {ex['headword']} — {ex['definition'][:60]}")
        print(f"  Skeleton: {ex['sig']}")
        
        # Pick 3 random other headwords as content sources
        import random
        random.seed(42)
        donors = random.sample(list(records.keys()), min(5, n_slots * 3))
        
        # Fill slots with content from donor definitions
        all_content = []
        for d in donors:
            if records[d]["senses"]:
                d_defn = records[d]["senses"][0]["definition"]
                words = [w for w in re.findall(r'[a-zA-Z]+', d_defn) 
                        if w.lower() not in _FUNC_WORDS and len(w) > 2]
                all_content.extend(words[:3])
        
        if len(all_content) >= n_slots:
            for trial in range(3):
                start = trial * n_slots
                if start + n_slots <= len(all_content):
                    content = all_content[start:start+n_slots]
                    reskinned = reskin(skeleton, slots, content)
                    print(f"  [{trial}] {reskinned}")
        print()


if __name__ == "__main__":
    main()
