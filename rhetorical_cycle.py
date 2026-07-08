"""
rhetorical_cycle.py — Recursive discourse cycling.

Takes any output, transforms it to its adjacent rhetorical form,
feeds it back as the next input. Repeats for n cycles.

The adjacency ring:

    assertion → question → instruction → narrative → explanation → assertion → ...

Each cycle changes the discourse frame, which changes the closure geometry,
which changes the selected headwords, which changes the output content.

    python rhetorical_cycle.py "The bearing failed because lubrication was inadequate."
    python rhetorical_cycle.py "How do I fix a broken engine?" --cycles 5
    python rhetorical_cycle.py "Tighten the bolt clockwise." --cycles 3 --start instruction
"""

import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import load_substrate
from analyzer import analyze
from generator import generate, NEIGHBOR_MODE_DEFAULT

# Mode rotation cycle — each cycle exposes a different graph relation
_MODE_ROTATION = [
    "definition_chain",
    "plus_opposition",
    "exemplify",
    "plus_morphological",
    "primary_only",
    "definition_chain",
]


# ---------------------------------------------------------------- adjacency ring

RING = ["assertion", "question", "instruction", "narrative", "explanation"]

ADJACENT = {
    "assertion":   "question",
    "question":    "instruction",
    "instruction": "narrative",
    "narrative":   "explanation",
    "explanation": "assertion",
    "counterfactual": "question",
    "hypothesis":  "assertion",
}


# ---------------------------------------------------------------- rhetorical transforms
# Each function converts a text string into the surface form that will
# be classified as the target discourse type by the analyzer.

def _lower1(s):
    return s[0].lower() + s[1:] if s else s

def _strip_markers(text):
    """Remove join connectives so the core statement survives."""
    t = text
    for marker in [
        "Because of this, ", "Specifically, ", "Then, ",
        "Yet ", "If ", ", then ", "Take into account that ",
        "What follows from the fact that ", "Were this otherwise, ",
        "It may be that ",
    ]:
        t = t.replace(marker, "")
    return re.sub(r"\s{2,}", " ", t).strip()

def to_question(text):
    """assertion → question: 'What follows from the fact that X?'"""
    core = _lower1(_strip_markers(text).rstrip("."))
    return f"What follows from the fact that {core}?"

def to_instruction(text):
    """question → instruction: reframe question as an imperative directive."""
    core = _strip_markers(text).rstrip("?").rstrip(".")
    core = re.sub(r"^(what follows from the fact that|how do i|why does|what is)\s*",
                  "", core, flags=re.I).strip()
    return f"Determine and apply: {_lower1(core)}."

def to_narrative(text):
    """instruction → narrative: cast as a temporal sequence of events."""
    core = _lower1(_strip_markers(text).rstrip("."))
    return f"First observe that {core}, then consider what follows from it."

def to_explanation(text):
    """narrative → explanation: frame as a causal account."""
    core = _lower1(_strip_markers(text).rstrip("."))
    return f"This occurred because {core}."

def to_assertion(text):
    """explanation → assertion: extract headwords only, drop definition bodies.

    Input:  "Bearing: the manner in which... Because of this, fail: miscarriage..."
    Output: "Bearing fail lubrication inadequate adequate."

    Discarding the definition bodies prevents the closure from exploding on
    the next cycle (30-word definition blob → 6-word headword list).
    """
    segments = re.split(
        r'\s+(?:Because of this|Specifically|Then|Yet),?\s+',
        text, flags=re.I
    )
    headwords = []
    for seg in segments:
        seg = re.sub(
            r'^(?:If\s+)?(?:Take into account that\s+)?(?:This occurred because\s+)?',
            '', seg, flags=re.I
        ).strip()
        if ':' in seg:
            hw = seg.split(':', 1)[0].strip()
            if hw and len(hw.split()) <= 3:
                headwords.append(hw)

    if headwords:
        return " ".join(headwords) + "."
    core = _strip_markers(text).strip().rstrip(".")
    return (core[0].upper() + core[1:] + ".") if core else text

TRANSFORMS = {
    "question":    to_question,
    "instruction": to_instruction,
    "narrative":   to_narrative,
    "explanation": to_explanation,
    "assertion":   to_assertion,
}


# ---------------------------------------------------------------- cycle engine

def run_cycle(input_text, sub, n_cycles=5, verbose=True, layers=5,
              frontier_cap=400, select=6,
              cycle_mode="refine", exclusion_window=5,
              stop_on_saturation=False, saturation_patience=10):
    """
    Run n_cycles of rhetorical transformation.

    Each cycle:
      1. Excludes headwords used in prior cycles → forces fresh selection
      2. Rotates neighbor_mode → different graph relations become visible
      3. Feeds selected headwords forward (not output text) → no noise loop
      4. Gravity anchor keeps traversal centered on the original input

    cycle_mode:
      "explore" — all-time exclusion: maximizes novelty, outward semantic
                  discovery from a fixed anchor. Traverses the dictionary.
      "refine"  — rolling exclusion (exclusion_window cycles): permits return
                  to high-value anchor-adjacent concepts after cooldown,
                  preventing global dictionary drift.

    Returns list of dicts.
    """
    history = []
    current = input_text
    used_headwords = set()    # exclusion set (all-time or rolling depending on mode)
    _saturation_counter = 0   # consecutive cycles with 0 new headwords

    # Gravity anchor: extract the original input's headwords once and
    # always include them in context so the BFS pool keeps the traversal
    # centered on the original topic.
    from discover import resolve, words
    records = sub["records"]
    function_words = set(sub["relations"]["function_words"])
    anchor_heads = []
    for surface in words(input_text):
        head = resolve(surface, records)
        if head and head not in function_words:
            anchor_heads.append(head)
    anchor_str = " ".join(anchor_heads) if anchor_heads else ""

    for i in range(n_cycles):
        # Context = gravity anchor (always) + previous cycle's headwords
        prior_heads = [s for s in [anchor_str, " ".join(
            history[-1]["selected"]
        )] if s] if history else ([anchor_str] if anchor_str else [])

        state  = analyze(current, prior_heads, sub,
                         max_layers=layers, frontier_cap=frontier_cap,
                         checkpoint=f"_cycle_{i}.json")

        # Rotate neighbor mode per cycle
        forced_mode = _MODE_ROTATION[i % len(_MODE_ROTATION)]

        result = generate(state, sub, n=select, exclude=used_headwords,
                          force_neighbor_mode=forced_mode)

        output  = result["output"]
        sa      = state.frame.speech_act
        rt      = state.frame.response_type
        closure = len(state.closure)
        selected = [h for h, _, _, _ in result["selected"]]

        # Update exclusion set
        new_this_cycle = set(selected) - used_headwords
        used_headwords.update(selected)

        # Rolling exclusion: forget headwords older than exclusion_window cycles
        if cycle_mode == "refine" and i >= exclusion_window:
            old_cycle = history[i - exclusion_window]
            used_headwords -= set(old_cycle["selected"])

        # Saturation detection: consecutive cycles with no new headwords
        if stop_on_saturation:
            if new_this_cycle:
                _saturation_counter = 0
            else:
                _saturation_counter += 1
                if _saturation_counter >= saturation_patience:
                    if verbose:
                        print(f"\n  [saturation] {saturation_patience} consecutive cycles "
                              f"with 0 new headwords — stopping at cycle {i+1}")
                    break

        # find adjacent type and apply transform
        next_type    = ADJACENT.get(sa, "assertion")
        transform_fn = TRANSFORMS[next_type]
        transformed  = transform_fn(output)

        record = {
            "cycle":          i + 1,
            "input_text":     current,
            "speech_act":     sa,
            "response_type":  rt,
            "closure_size":   closure,
            "tier_sizes":     result["tier_sizes"],
            "join_used":      result["join_used"],
            "neighbor_mode":  forced_mode,
            "selected":       selected,
            "output":         output,
            "transformed_to": next_type,
            "transform_text": transformed,
            "cycle_mode":     cycle_mode,
        }
        history.append(record)

        if verbose:
            print(f"\n{'-'*70}")
            print(f"CYCLE {i+1}  ({cycle_mode}, mode={forced_mode}, "
                  f"excluded={len(used_headwords)} heads)")
            print(f"{'-'*70}")
            print(f"  Discourse in  : {sa} → {rt}")
            print(f"  Closure size  : {closure}")
            print(f"  Selected      : {selected}")
            print(f"  Join          : {result['join_used']}")
            print(f"\n  OUTPUT:")
            print(f"  {output}")
            print(f"\n  → Transformed to [{next_type}]:")
            print(f"  {transformed}")

        # Feed headwords forward via the context built at top of next iteration
        current = transformed

    return history


def convergence_report(history):
    """Report on whether successive outputs share vocabulary."""
    print(f"\n{'='*70}")
    print("CONVERGENCE REPORT")
    print(f"{'='*70}")
    print(f"\n{'Cycle':<6} {'SA':<15} {'Closure':>7} {'Tier A':>6}  Output words")

    words_history = []
    for rec in history:
        words = set(rec["output"].lower().split())
        words_history.append(words)
        print(f"  {rec['cycle']:<4} {rec['speech_act']:<15}"
              f" {rec['closure_size']:>6}  {rec['tier_sizes']['tier_a']:>5}"
              f"  {len(words)} words")

    if len(words_history) >= 2:
        print(f"\nOverlap between consecutive cycle outputs:")
        for i in range(1, len(words_history)):
            a, b = words_history[i-1], words_history[i]
            overlap = len(a & b)
            union   = len(a | b)
            jaccard = overlap / union if union else 0
            print(f"  Cycle {i} → {i+1}:  {overlap} shared words"
                  f"  Jaccard={jaccard:.3f}")

    # Discourse cycle detection
    acts = [r["speech_act"] for r in history]
    print(f"\nDiscourse trajectory: {' → '.join(acts)}")

    unique_acts = len(set(acts))
    if unique_acts == len(RING):
        print("Full discourse ring traversed.")
    elif unique_acts == 1:
        print("WARNING: discourse type did not change — classifier may be stuck.")
    else:
        print(f"{unique_acts}/{len(RING)} discourse types visited.")


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*",
                     default=["The bearing failed because lubrication was inadequate."])
    ap.add_argument("--substrate",  default=os.path.join(_HERE, "substrate.min.bin"))
    ap.add_argument("--cycles",     type=int, default=5)
    ap.add_argument("--layers",     type=int, default=5)
    ap.add_argument("--select",     type=int, default=6)
    ap.add_argument("--quiet",      action="store_true")
    ap.add_argument("--mode",       choices=["explore", "refine"], default="refine",
                     help="explore: all-time exclusion (semantic discovery). "
                          "refine: rolling exclusion (iterative clarification).")
    ap.add_argument("--exclusion-window", type=int, default=5,
                     help="Cycles before headwords expire from exclusion (refine mode only)")
    args = ap.parse_args()

    input_text = " ".join(args.text)

    print(f"{'='*70}")
    print("RHETORICAL CYCLE")
    print(f"{'='*70}")
    print(f"Input    : {input_text}")
    print(f"Cycles   : {args.cycles}")
    print(f"Layers   : {args.layers}")
    print(f"Mode     : {args.mode}")
    if args.mode == "refine":
        print(f"Exclusion window: {args.exclusion_window} cycles")

    sub = load_substrate(args.substrate)

    history = run_cycle(
        input_text=input_text,
        sub=sub,
        n_cycles=args.cycles,
        verbose=not args.quiet,
        layers=args.layers,
        select=args.select,
        cycle_mode=args.mode,
        exclusion_window=args.exclusion_window,
    )

    convergence_report(history)


if __name__ == "__main__":
    main()
