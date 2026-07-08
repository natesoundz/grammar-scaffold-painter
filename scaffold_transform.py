"""
scaffold_transform.py — Compiled function-word transformations between
rhetorical devices.

Each transform is a scaffold-level rewrite: change the function-word
skeleton from one language device to another while preserving content.

Example transforms:
  taxonomic → causal_chain:
    "a [X] of [Y]" → "to [X] by [Y]"
    ART·CMP        → DIR·CAU

  elaboration → contrast:
    "a [X] which is [Y]" → "[X] but not [Y]"
    ART·SPC·IDN          → CTN·NEG

  taxonomic → purposive:
    "a [X] of [Y]" → "[X] for [Y]"
    ART·CMP        → PUR

The transforms operate on function words only. Content words pass through
unchanged. This is the compiled method-space: HOW to rewrite definitions
from one discourse frame to another.

Usage:
    python scaffold_transform.py "fire" --to causal_chain
    python scaffold_transform.py "authority" --to contrast
    python scaffold_transform.py --list
"""

import os
import sys
import re
import argparse
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scaffold import ScaffoldAnalyzer, _SCAFFOLD_ROLES, _FUNCTION_WORD_RE
from _scaffold_lab import extract_skeleton, classify_device, skeleton_pattern

# ---------------------------------------------------------------- transform table

# Each transform: (source_device, target_device) → function-word rewrite
# The rewrite is a pattern: (source_fw_pattern, target_fw_pattern)
# where each pattern is a list of function words (or None for content slots)

SCAFFOLD_TRANSFORMS = {
    # ---- taxonomic → causal_chain
    # "a [X] of [Y]" → "to [X] by [Y]"
    ("taxonomic", "causal_chain"): {
        "source_fw": ["a", "of"],
        "target_fw": ["to", "by"],
        "source_roles": ["ARTICLE", "COMPOSE"],
        "target_roles": ["DIRECT", "CAUSE"],
        "description": "Genus classification → causal agency",
        "example": '"a force of nature" → "to force by nature"',
    },
    # "the [X] of [Y]" → "to [X] by [Y]"
    ("taxonomic", "causal_chain"): {
        "source_fw": ["a", "of"],
        "target_fw": ["to", "by"],
        "source_roles": ["ARTICLE", "COMPOSE"],
        "target_roles": ["DIRECT", "CAUSE"],
        "description": "Genus classification → causal agency",
        "example": '"a force of nature" → "to force by nature"',
    },

    # ---- taxonomic → contrast
    # "a [X] of [Y]" → "[X] but [Y]"
    ("taxonomic", "contrast"): {
        "source_fw": ["a", "of"],
        "target_fw": [None, "but"],
        "source_roles": ["ARTICLE", "COMPOSE"],
        "target_roles": ["CONTRAST"],
        "description": "Genus classification → contrastive opposition",
        "example": '"a force of nature" → "force but nature"',
    },

    # ---- taxonomic → purposive
    # "a [X] of [Y]" → "[X] for [Y]"
    ("taxonomic", "purposive"): {
        "source_fw": ["a", "of"],
        "target_fw": [None, "for"],
        "source_roles": ["ARTICLE", "COMPOSE"],
        "target_roles": ["PURPOSE"],
        "description": "Genus classification → purposive function",
        "example": '"a force of nature" → "force for nature"',
    },

    # ---- taxonomic → conditional
    # "a [X] of [Y]" → "if [X] then [Y]"
    ("taxonomic", "conditional"): {
        "source_fw": ["a", "of"],
        "target_fw": ["if", "then"],
        "source_roles": ["ARTICLE", "COMPOSE"],
        "target_roles": ["CONDITION", "DIRECT"],
        "description": "Genus classification → conditional hypothesis",
        "example": '"a force of nature" → "if force then nature"',
    },

    # ---- compositional_containment → causal_chain
    # "[X] of [Y] in [Z]" → "to [X] by [Y] in [Z]"
    ("compositional_containment", "causal_chain"): {
        "source_fw": [None, "of", None, "in", None],
        "target_fw": ["to", "by", None, "in", None],
        "source_roles": ["COMPOSE", "CONTAIN"],
        "target_roles": ["DIRECT", "CAUSE", "CONTAIN"],
        "description": "Compositional containment → causal containment",
        "example": '"evolution of heat in bodies" → "to evolve by heat in bodies"',
    },

    # ---- compositional_containment → contrast
    # "[X] of [Y] in [Z]" → "[X] but [Y] in [Z]"
    ("compositional_containment", "contrast"): {
        "source_fw": [None, "of", None, "in", None],
        "target_fw": [None, "but", None, "in", None],
        "source_roles": ["COMPOSE", "CONTAIN"],
        "target_roles": ["CONTRAST", "CONTAIN"],
        "description": "Compositional containment → contrastive containment",
        "example": '"evolution of heat in bodies" → "evolution but heat in bodies"',
    },

    # ---- causal_chain → taxonomic
    # "to [X] by [Y]" → "a [X] of [Y]"
    ("causal_chain", "taxonomic"): {
        "source_fw": ["to", "by"],
        "target_fw": ["a", "of"],
        "source_roles": ["DIRECT", "CAUSE"],
        "target_roles": ["ARTICLE", "COMPOSE"],
        "description": "Causal agency → genus classification",
        "example": '"to forbid by authority" → "a forbidding of authority"',
    },

    # ---- causal_chain → contrast
    # "to [X] by [Y]" → "[X] despite [Y]"
    ("causal_chain", "contrast"): {
        "source_fw": ["to", "by"],
        "target_fw": [None, "despite"],
        "source_roles": ["DIRECT", "CAUSE"],
        "target_roles": ["CONTRAST"],
        "description": "Causal agency → concessive contrast",
        "example": '"to forbid by authority" → "forbid despite authority"',
    },

    # ---- causal_chain → conditional
    # "to [X] by [Y]" → "if [X] then [Y]"
    ("causal_chain", "conditional"): {
        "source_fw": ["to", "by"],
        "target_fw": ["if", "then"],
        "source_roles": ["DIRECT", "CAUSE"],
        "target_roles": ["CONDITION", "DIRECT"],
        "description": "Causal agency → conditional hypothesis",
        "example": '"to forbid by authority" → "if forbid then authority"',
    },

    # ---- bare_nominal → taxonomic
    # "a [X]" → "a [X] of [Y]" (requires finding a Y from context)
    ("bare_nominal", "taxonomic"): {
        "source_fw": ["a"],
        "target_fw": ["a", "of"],
        "source_roles": ["ARTICLE"],
        "target_roles": ["ARTICLE", "COMPOSE"],
        "description": "Bare nominal → genus classification (adds composition)",
        "example": '"a force" → "a force of [context_word]"',
    },

    # ---- bare_nominal → causal_chain
    # "a [X]" → "to [X] by [Y]"
    ("bare_nominal", "causal_chain"): {
        "source_fw": ["a"],
        "target_fw": ["to", "by"],
        "source_roles": ["ARTICLE"],
        "target_roles": ["DIRECT", "CAUSE"],
        "description": "Bare nominal → causal agency",
        "example": '"a force" → "to force by [context_word]"',
    },

    # ---- negation → contrast
    # "not [X]" → "[X] but [Y]"
    ("negation", "contrast"): {
        "source_fw": ["not", None],
        "target_fw": [None, "but"],
        "source_roles": ["NEGATE"],
        "target_roles": ["CONTRAST"],
        "description": "Negation → contrastive opposition",
        "example": '"not free" → "free but [context_word]"',
    },

    # ---- negation → causal_chain
    # "not [X]" → "to [X] by [Y]"
    ("negation", "causal_chain"): {
        "source_fw": ["not", None],
        "target_fw": ["to", "by"],
        "source_roles": ["NEGATE"],
        "target_roles": ["DIRECT", "CAUSE"],
        "description": "Negation → causal agency (negated → caused)",
        "example": '"not free" → "to free by [context_word]"',
    },

    # ---- nested_specification → causal_chain
    # "a [X] which [Y] in [Z]" → "to [X] by [Y] in [Z]"
    ("nested_specification", "causal_chain"): {
        "source_fw": ["a", "which", None, "in", None],
        "target_fw": ["to", "by", None, "in", None],
        "source_roles": ["ARTICLE", "SPECIFY", "CONTAIN"],
        "target_roles": ["DIRECT", "CAUSE", "CONTAIN"],
        "description": "Nested specification → causal containment",
        "example": '"a force which acts in nature" → "to force by acts in nature"',
    },

    # ---- nested_specification → contrast
    # "a [X] which [Y]" → "[X] but [Y]"
    ("nested_specification", "contrast"): {
        "source_fw": ["a", "which", None],
        "target_fw": [None, "but", None],
        "source_roles": ["ARTICLE", "SPECIFY"],
        "target_roles": ["CONTRAST"],
        "description": "Nested specification → contrastive",
        "example": '"a force which acts" → "force but acts"',
    },
}


def apply_transform(definition, source_fw, target_fw, context_words=None):
    """Apply a function-word transform to a definition.

    Replaces function words in source_fw with those in target_fw,
    preserving content words in their positions.

    Args:
        definition: the original definition text
        source_fw: list of function words to find (None = content slot)
        target_fw: list of function words to replace with (None = keep content)
        context_words: optional list to fill newly-created slots

    Returns:
        (transformed_text, content_extracted)
    """
    tokens = definition.split()
    
    # Find the function words in the definition
    fw_positions = []
    for i, tok in enumerate(tokens):
        clean = re.sub(r'[^a-zA-Z]', '', tok).lower()
        if clean in _SCAFFOLD_ROLES:
            fw_positions.append((i, clean))

    # Match source_fw pattern against the function words
    func_words_in_def = [fw for _, fw in fw_positions]
    
    # Simple approach: find and replace each source function word
    # with its target, keeping content words in place
    result = []
    fw_idx = 0
    ctx_idx = 0
    
    for tok in tokens:
        clean = re.sub(r'[^a-zA-Z]', '', tok).lower()
        if clean in _SCAFFOLD_ROLES and fw_idx < len(source_fw):
            # Check if this matches the current source pattern position
            if source_fw[fw_idx] is not None and clean == source_fw[fw_idx]:
                # Replace with target
                if target_fw[fw_idx] is not None:
                    result.append(target_fw[fw_idx])
                # else: drop the function word (None = remove)
                fw_idx += 1
            elif source_fw[fw_idx] is None:
                # Content slot — keep the content word
                result.append(tok)
                fw_idx += 1
            else:
                # No match — keep original
                result.append(tok)
                fw_idx += 1
        else:
            # Content word or unmatched — keep
            result.append(tok)
    
    # If target has more slots than source (e.g., bare_nominal → taxonomic
    # adds an "of" + needs a content word), append from context
    while fw_idx < len(target_fw):
        if target_fw[fw_idx] is not None:
            result.append(target_fw[fw_idx])
        elif context_words and ctx_idx < len(context_words):
            result.append(context_words[ctx_idx])
            ctx_idx += 1
        fw_idx += 1
    
    return " ".join(result)


def transform_definition(definition, target_device, context_words=None):
    """Transform a definition to a target language device.
    
    1. Extract scaffold from definition
    2. Classify current device
    3. Look up transform (current_device → target_device)
    4. Apply function-word rewrite
    5. Return transformed text
    
    Returns (transformed, source_device, transform_info) or (original, None, None)
    """
    sa = ScaffoldAnalyzer()
    scaffold = sa.extract_scaffold(definition)
    pattern = skeleton_pattern(scaffold)
    source_device = classify_device(pattern)
    
    key = (source_device, target_device)
    if key not in SCAFFOLD_TRANSFORMS:
        return None, source_device, None
    
    transform = SCAFFOLD_TRANSFORMS[key]
    transformed = apply_transform(
        definition,
        transform["source_fw"],
        transform["target_fw"],
        context_words=context_words,
    )
    
    return transformed, source_device, transform


def main():
    ap = argparse.ArgumentParser(description="Scaffold transform compiler")
    ap.add_argument("word", nargs="?", default="fire",
                    help="Headword to transform")
    ap.add_argument("--to", dest="target", default="causal_chain",
                    help="Target language device")
    ap.add_argument("--context", default="", help="Context words for slot filling")
    ap.add_argument("--all", action="store_true", help="Transform through all devices")
    ap.add_argument("--list", action="store_true", help="List all transforms")
    args = ap.parse_args()

    if args.list:
        print("=" * 70)
        print("COMPILED SCAFFOLD TRANSFORMS")
        print("=" * 70)
        print()
        for (src, tgt), t in sorted(SCAFFOLD_TRANSFORMS.items()):
            print(f"  {src:<25} → {tgt:<25}")
            print(f"    {t['description']}")
            print(f"    {t['example']}")
            print(f"    roles: {'·'.join(t['source_roles'])} → {'·'.join(t['target_roles'])}")
            print()
        return

    from discover import load_substrate
    sub = load_substrate(os.path.join(_HERE, "substrate.min.bin"))
    records = sub["records"]

    h = args.word.lower()
    if h not in records:
        print(f"'{h}' not found")
        return

    context_words = args.context.lower().split() if args.context else None

    print(f"Headword: {h}")
    print()

    for i, sense in enumerate(records[h]["senses"][:3]):
        defn = sense["definition"]
        sa = ScaffoldAnalyzer()
        scaffold = sa.extract_scaffold(defn)
        pattern = skeleton_pattern(scaffold)
        source_device = classify_device(pattern)
        sig = sa.scaffold_signature(scaffold)

        print(f"Sense {i+1}: {defn[:80]}")
        print(f"  Device: {source_device}")
        print(f"  Scaffold: {sig}")
        print()

        if args.all:
            # Transform to every target device
            targets = sorted(set(tgt for _, tgt in SCAFFOLD_TRANSFORMS.keys()))
            for target in targets:
                transformed, src, info = transform_definition(defn, target, context_words)
                if transformed:
                    print(f"  → {target}: {transformed}")
                    print(f"    {info['description']}")
                else:
                    print(f"  → {target}: (no transform from {source_device})")
                print()
        else:
            target = args.target
            transformed, src, info = transform_definition(defn, target, context_words)
            if transformed:
                print(f"  → {target}: {transformed}")
                print(f"    Transform: {info['description']}")
                print(f"    Example: {info['example']}")
            else:
                print(f"  → {target}: (no transform from {source_device})")
                print(f"    Available from {source_device}:")
                available = [tgt for (s, tgt) in SCAFFOLD_TRANSFORMS if s == source_device]
                if available:
                    for tgt in available:
                        print(f"      → {tgt}")
                else:
                    print(f"      (none)")
            print()


if __name__ == "__main__":
    main()
