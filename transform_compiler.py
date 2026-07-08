"""
transform_compiler.py — Second-order compilation of rhetorical transformations.

    python transform_compiler.py artifacts.jsonl
    python transform_compiler.py artifacts.jsonl --min-freq 3 --max-entropy 0.5

This is the second-order analysis described in the architecture:
  First order:   language  → communicative state  (discover.py / analyzer.py)
  Second order:  many states → recurring trajectories → compiled transformations

Algorithm:
  1. Load artifacts (input → state → output triples)
  2. Group by contingency class: (speech_act, response_type, closure_bucket)
  3. Per class: find the dominant join pattern, measure consistency (entropy),
     measure bridge stability (do the same concepts recur across instances?)
  4. Compile classes that meet quality thresholds into transformation rules
  5. Write rules to JSONL

The compiled rules are reusable: at inference time, the generator looks up
the best matching rule and uses its join pattern instead of the default.

Rule schema:
    {
      "speech_act":        "question",
      "response_type":     "instruction",
      "closure_bucket":    "medium",
      "join":              "cause_effect",
      "frequency":         47,
      "entropy":           0.23,
      "bridge_stability":  0.92,
      "bridge_words":      ["engine", "fix", "broken"],
      "confidence":        0.71
    }

confidence = (1 - entropy) * bridge_stability  — a composite of consistency
and concept stability. Not used as a ranking score; only used to choose
between two rules that both match a query state.
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ---------------------------------------------------------------- bucket

def closure_bucket(size: int) -> str:
    if size < 10:
        return "small"
    if size < 50:
        return "medium"
    return "large"


# ---------------------------------------------------------------- entropy

def join_entropy(join_counts: Counter) -> float:
    """Normalized Shannon entropy over join pattern distribution.
    0 = one pattern used exclusively. 1 = uniform over all 6 joins."""
    total = sum(join_counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in join_counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    max_h = math.log2(6)  # 6 possible join patterns
    return h / max_h if max_h > 0 else 0.0


# ---------------------------------------------------------------- bridge words

def extract_bridges(artifact: dict) -> set:
    """Words that appear in BOTH the input activation field AND the
    selected headwords — concepts that traversed from input to output."""
    input_set   = set(artifact["state"].get("input_field", []))
    selected    = set(artifact.get("selected_headwords", []))
    return input_set & selected


def bridge_stability(artifacts_in_class: list) -> tuple:
    """Fraction of artifacts that have at least one bridge word, plus
    the most common bridge words across the class.
    Returns (stability_float, sorted_common_bridges)."""
    bridge_counts = Counter()
    has_bridge = 0
    for a in artifacts_in_class:
        b = extract_bridges(a)
        if b:
            has_bridge += 1
            bridge_counts.update(b)
    stability = has_bridge / len(artifacts_in_class) if artifacts_in_class else 0.0
    common = [w for w, _ in bridge_counts.most_common(5)]
    return stability, common


# ---------------------------------------------------------------- neighbor mode assignment

_NEIGHBOR_MODE_MAP = {
    # (speech_act, response_type) -> neighbor_mode
    # definition_chain: chains through definition text — works without explicit
    #   crossref edges (important: Webster’s crossref table is sparse, ~0.1/headword)
    # plus_opposition:  contrast boundary — fires from 5,557 opposition links
    # exemplify:        usage example over abstract definition — instruction-optimised
    ("explanation", "explanation"):  "definition_chain",
    ("assertion",   "explanation"):  "definition_chain",
    ("question",    "explanation"):  "definition_chain",
    ("question",    "instruction"):  "definition_chain",
    ("instruction", "assertion"):    "exemplify",
    ("narrative",   "narrative"):    "definition_chain",
    ("assertion",   "assertion"):    "plus_opposition",
}

_NEIGHBOR_MODE_BUCKET_OVERRIDE = {
    # Small closures: fewer neighbours to chain through — stay primary
    ("explanation", "explanation", "small"): "primary_only",
    ("assertion",   "explanation", "small"): "primary_only",
    ("question",    "explanation", "small"): "primary_only",
    ("question",    "instruction", "small"): "primary_only",
    # Medium question explanation: plus_opposition establishes concept boundary
    ("question",    "explanation", "medium"): "plus_opposition",
}

def _assign_neighbor_mode(speech_act, response_type, bucket):
    key3 = (speech_act, response_type, bucket)
    if key3 in _NEIGHBOR_MODE_BUCKET_OVERRIDE:
        return _NEIGHBOR_MODE_BUCKET_OVERRIDE[key3]
    key2 = (speech_act, response_type)
    return _NEIGHBOR_MODE_MAP.get(key2, "primary_only")


# ---------------------------------------------------------------- compiler

def compile_rules(artifacts: list, min_freq: int, max_entropy: float,
                  min_stability: float) -> tuple:
    """Group artifacts into contingency classes and compile rules.
    Returns (rules, stats_dict)."""

    # Group
    classes = defaultdict(list)
    for a in artifacts:
        s = a["state"]
        key = (s["speech_act"], s["response_type"],
               closure_bucket(s["closure_size"]))
        classes[key].append(a)

    rules, skipped_freq, skipped_entropy, skipped_stability = [], 0, 0, 0

    for (speech_act, response_type, bucket), group in sorted(classes.items()):
        freq = len(group)
        if freq < min_freq:
            skipped_freq += 1
            continue

        join_counts = Counter(a["join_used"] for a in group)
        entropy     = join_entropy(join_counts)
        if entropy > max_entropy:
            skipped_entropy += 1
            continue

        stability, bridges = bridge_stability(group)
        if stability < min_stability:
            skipped_stability += 1
            continue

        dominant_join = join_counts.most_common(1)[0][0]
        confidence    = round((1.0 - entropy) * stability, 3)
        neighbor_mode = _assign_neighbor_mode(speech_act, response_type, bucket)

        rules.append({
            "speech_act":       speech_act,
            "response_type":    response_type,
            "closure_bucket":   bucket,
            "join":             dominant_join,
            "neighbor_mode":    neighbor_mode,
            "frequency":        freq,
            "entropy":          round(entropy, 4),
            "bridge_stability": round(stability, 4),
            "bridge_words":     bridges,
            "confidence":       confidence,
        })

    rules.sort(key=lambda r: r["frequency"], reverse=True)

    stats = {
        "total_artifacts":   len(artifacts),
        "total_classes":     len(classes),
        "rules_compiled":    len(rules),
        "skipped_freq":      skipped_freq,
        "skipped_entropy":   skipped_entropy,
        "skipped_stability": skipped_stability,
    }
    return rules, stats


# ---------------------------------------------------------------- load + match (used by generator)

def load_rules(path: str) -> list:
    """Load rules from JSONL. Returns list of rule dicts."""
    p = Path(path)
    if not p.exists():
        return []
    rules = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rules.append(json.loads(line))
    return rules


def match_rule(state, rules: list) -> dict | None:
    """Find the best matching rule for a ListenerState.

    Match criteria (in priority order):
      1. speech_act + response_type + closure_bucket  (exact)
      2. speech_act + response_type  (any bucket)
    Among ties, highest confidence wins.

    Returns the matched rule dict, or None if nothing matches.
    """
    if not rules:
        return None

    bucket = closure_bucket(len(state.closure))
    sa     = state.frame.speech_act
    rt     = state.frame.response_type

    # Tier 1: exact match
    exact = [r for r in rules
             if r["speech_act"] == sa
             and r["response_type"] == rt
             and r["closure_bucket"] == bucket]
    if exact:
        return max(exact, key=lambda r: r["confidence"])

    # Tier 2: speech_act + response_type, any bucket
    partial = [r for r in rules
               if r["speech_act"] == sa
               and r["response_type"] == rt]
    if partial:
        return max(partial, key=lambda r: r["confidence"])

    return None


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Compile transformation rules from artifacts")
    ap.add_argument("artifacts",       default=os.path.join(_HERE, "artifacts.jsonl"), nargs="?")
    ap.add_argument("--output",        default=os.path.join(_HERE, "transforms.jsonl"))
    ap.add_argument("--min-freq",      type=int,   default=3)
    ap.add_argument("--max-entropy",   type=float, default=0.5)
    ap.add_argument("--min-stability", type=float, default=0.0)
    ap.add_argument("--validate",      action="store_true")
    args = ap.parse_args()

    try:
        with open(args.artifacts, encoding="utf-8-sig") as f:
            raw = [l.strip() for l in f if l.strip()]
        artifacts = []
        for l in raw:
            try:
                artifacts.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    except FileNotFoundError:
        print(f"Artifacts file not found: {args.artifacts}")
        print("Run collect_artifacts.ps1 first.")
        sys.exit(1)

    print("=" * 70)
    print("TRANSFORMATION COMPILER")
    print("=" * 70)
    print(f"Loaded {len(artifacts)} artifacts")
    print(f"Quality thresholds:")
    print(f"  min_frequency    = {args.min_freq}")
    print(f"  max_entropy      = {args.max_entropy}")
    print(f"  min_stability    = {args.min_stability}")
    print()

    rules, stats = compile_rules(artifacts, args.min_freq,
                                 args.max_entropy, args.min_stability)

    print(f"Grouped into {stats['total_classes']} contingency classes")
    print()
    print(f"{stats['rules_compiled']} rules compiled")
    print(f"Skipped:")
    print(f"  {stats['skipped_freq']}  (frequency < {args.min_freq})")
    print(f"  {stats['skipped_entropy']}  (entropy > {args.max_entropy})")
    print(f"  {stats['skipped_stability']}  (stability < {args.min_stability})")

    if args.validate and rules:
        print()
        print("=" * 70)
        print("RULE VALIDATION")
        print("=" * 70)
        print(f"\nTop rules by frequency:\n")
        for r in rules[:10]:
            print(f"  {r['speech_act']:<14} + {r['response_type']:<14} + {r['closure_bucket']}")
            print(f"    Join:       {r['join']}")
            print(f"    Frequency:  {r['frequency']}")
            print(f"    Entropy:    {r['entropy']:.3f}")
            print(f"    Stability:  {r['bridge_stability']*100:.1f}%")
            print(f"    Confidence: {r['confidence']:.3f}")
            if r['bridge_words']:
                print(f"    Bridges:    {r['bridge_words']}")
            print()

    print()
    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    if rules:
        avg_freq    = sum(r["frequency"]        for r in rules) / len(rules)
        avg_entropy = sum(r["entropy"]          for r in rules) / len(rules)
        avg_stab    = sum(r["bridge_stability"] for r in rules) / len(rules)
        avg_conf    = sum(r["confidence"]       for r in rules) / len(rules)
        freq_dist   = Counter()
        for r in rules:
            freq_dist["low"  if r["frequency"] < 5
                       else "med" if r["frequency"] < 15
                       else "high"] += 1
        print(f"Total rules:    {len(rules)}")
        print(f"Avg frequency:  {avg_freq:.1f}")
        print(f"Avg entropy:    {avg_entropy:.3f}")
        print(f"Avg stability:  {avg_stab*100:.1f}%")
        print(f"Avg confidence: {avg_conf:.3f}")
        print(f"Frequency distribution:")
        for tier in ("low", "med", "high"):
            c = freq_dist[tier]
            print(f"  {tier}: {c:3} ({100*c/len(rules):5.1f}%)")
    else:
        print("No rules compiled.")

    with open(args.output, "w", encoding="utf-8") as f:
        for r in rules:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"Stored {len(rules)} rules to {args.output}")


if __name__ == "__main__":
    main()
