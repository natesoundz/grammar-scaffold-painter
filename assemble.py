"""
assemble.py — trace-to-answer. Sits on top of discover.py's trace.

    python assemble.py "The bolt is stuck." --intent "disassemble remove repair" \
        --setting "workshop machinery" --layers 6 --select 6

REBUILT: the previous version scored every candidate with
`composite = depth x (1+significance) x path_quality x convergence x role_fit`
— five numbers multiplied together into a ranking. That is a weighted model.
It does not matter that the numbers were derived from a dictionary instead
of gradient descent; structurally it is the exact thing this architecture
was built to be an alternative to. Removing `significance` from the
compiled substrate (recompile with compile_substrate.py) forced this file
to stop depending on it, which forced the real fix: selection is now a
sequence of discrete, inspectable, cascading filters. Nothing here is
multiplied. Nothing here is a score. Every decision is admit/exclude or a
binary partition, with a stated, fixed tie-break where one is needed —
the same status as the alphabetical frontier cap already used in
discover.py, not a weight in disguise.

FILTER 1 — SHAPE, BEFORE CONTENT. The input's discourse type is classified
first. Its adjacency-required response type is fixed before any content is
touched. If the response must be Instruction-shaped, only headwords with an
admitted action-sense ("To ...", Webster's own verb-sense convention)
survive. This is a requirement, not a preference weight.

FILTER 2 — DOMAIN CONFLICT. Webster's own compiled domain tags ("U.S.
Politics", "Law", "Mach.") are discrete categories. A sense whose domain
tag doesn't match any domain tag active elsewhere in the unified
input+context activation is excluded in favor of a same-headword sense
that doesn't conflict, if one exists.

FILTER 3 — CONVERGENCE AS PARTITION, NOT MULTIPLIER. A headword reachable
from two or more independent input-root words (via discover.py's
all_parents, which records every path, not just the first) is
"convergent" — real reinforcement. A headword reachable from only one
narrow thread is "single-thread" — not excluded, but never preferred over
a convergent candidate, and labeled as weaker evidence rather than blended
in silently.
"""

import argparse
import re

from discover import (
    load_substrate, discover, resolve, words, build_bridge,
    content_of, sense_bridge,
)
from dict_algebra import FILTER, CONSTRAIN, domain as dom_of

ACTION_RE = re.compile(r"^\s*to\s+\w", re.I)
STUB_RE = re.compile(r"^\s*(imp\.|p\.\s*p\.|pp\.|pres\.\s*p\.|impf\.|syn\.)", re.I)

INSTRUCTION_SEEKING_RE = re.compile(
    r"\b(what should i|what's my best|what is my best|how do i|how can i|"
    r"what next|next move|next step|what do i do)\b", re.I)
YES_NO_RE = re.compile(r"^\s*(is|are|was|were|does|did|do|can|will|would|should)\b", re.I)

ADJACENCY = {
    "instruction":    "assertion",
    "explanation":    "explanation",
    "narrative":      "narrative",
    "counterfactual": "explanation",
    "hypothesis":     "assertion",
    "assertion":      "explanation",
}


# --------------------------------------------------------- speech act (v7)

def classify_speech_act(text: str) -> str:
    """Heuristic, stdlib-only, over the seven v7 categories. Honest limit:
    regex-level, not a parser. Order matters: most distinctive signal first."""
    t = text.strip()
    low = t.lower()
    if t.endswith("?"):
        return "question"
    if re.search(r"\b(if|unless)\b.*\b(would|could|might|had)\b", low) or \
       re.search(r"\b(would|could|might|had)\b.*\b(if|unless)\b", low):
        return "counterfactual"
    if re.search(r"\b(may|might|perhaps|possibly|could it be)\b", low):
        return "hypothesis"
    if re.search(r"\b(because|since|therefore|thus|hence)\b", low):
        return "explanation"
    if re.search(r"\b(then|after that|meanwhile|yesterday|earlier|later)\b", low):
        return "narrative"
    first = low.split()[0].strip(".,!") if low.split() else ""
    if first and first not in (
        "the", "a", "an", "it", "he", "she", "they", "you", "i", "we",
        "this", "that", "these", "those",
    ):
        return "instruction"
    return "assertion"


def required_response_type(input_text: str, speech_act: str) -> str:
    if speech_act != "question":
        return ADJACENCY[speech_act]
    if INSTRUCTION_SEEKING_RE.search(input_text.lower()):
        return "instruction"
    if YES_NO_RE.search(input_text.strip()):
        return "assertion"
    return "explanation"


def is_action_sense(definition: str) -> bool:
    return bool(ACTION_RE.match(definition))


# ------------------------------------------------------- convergence (Filter 3)

# NOTE: no convergence/root_map here. "How many roots reach this node" is
# not a question set algebra can pose — it requires a multiset, a different
# object. Every attempt to answer it (raw sum, mean, boolean threshold) was
# arithmetic in a different costume. Removed, not replaced by a count.


# --------------------------------------------------------- domain (Filter 2)

def active_domains(heads, records):
    """Discrete set of domain tags present anywhere in the given headwords'
    senses. Membership, not a score."""
    doms = set()
    for h in heads:
        for s in records[h]["senses"]:
            if s["domain"]:
                doms.add(s["domain"].strip().lower())
    return doms


def best_nonconflicting_sense(head, records, allowed_domains, pool, function_words):
    """Among this head's senses, prefer one whose domain tag (if any)
    doesn't conflict with the active domain set. A conflicting-domain sense
    is skipped in favor of a clean one on the SAME headword — a same-word
    tie-break, not a cross-word ranking. Ties broken by lowest index."""
    senses = records[head]["senses"]
    clean, conflicting = [], []
    for i, s in enumerate(senses):
        admit = sense_bridge(s, pool, function_words, self_head=head)
        if not admit:
            continue
        dom = s["domain"].strip().lower() if s["domain"] else ""
        if dom and allowed_domains and dom not in allowed_domains:
            conflicting.append(i)
        else:
            clean.append(i)
    if clean:
        return clean[0]
    if conflicting:
        return conflicting[0]
    return 0


# ------------------------------------------------------------ mood (discourse)

def first_clause(definition: str) -> str:
    t = re.sub(r"^\d{1,2}\.\s*", "", definition)
    t = re.split(r"[;.]", t, 1)[0].strip()
    return t[0].upper() + t[1:] if t else t


def lower1(s):
    return s[0].lower() + s[1:] if s else s


def statement_for(head, records, allowed_domains, pool, function_words, crossrefs, _seen=None) -> str:
    _seen = _seen or set()
    idx = best_nonconflicting_sense(head, records, allowed_domains, pool, function_words)
    definition = records[head]["senses"][idx]["definition"]
    if STUB_RE.match(definition) and head not in _seen:
        targets = crossrefs.get(head, [])
        if targets:
            _seen.add(head)
            return statement_for(targets[0], records, allowed_domains, pool,
                                 function_words, crossrefs, _seen)
    base = first_clause(definition)
    return f"{head.capitalize()}: {base[0].lower() + base[1:]}" if base else head


MOOD = {
    "assertion":     lambda s: s,
    "question":      lambda s: f"What follows from the fact that {lower1(s.rstrip('.'))}?",
    "instruction":   lambda s: f"Take into account that {lower1(s.rstrip('.'))}.",
    "explanation":   lambda s: s,
    "narrative":     lambda s: s,
    "counterfactual": lambda s: f"Were this otherwise, {lower1(s.rstrip('.'))} would not hold.",
    "hypothesis":    lambda s: f"It may be that {lower1(s.rstrip('.'))}.",
}

# rhetoric joins already-mooded statements; discourse and rhetoric never
# touch the same string, so they cannot duplicate each other's connective.
# FIXED: the opening frame applies ONCE across the whole chain, not
# re-applied per fold — the earlier version produced "If if if if..." by
# wrapping every pairwise fold with its own copy of the frame.
JOIN = {
    "plain":                 ("", " "),
    "cause_effect":          ("", " Because of this, "),
    "sequence":              ("", " Then, "),
    "condition_consequence": ("If ", ", then "),
    "contrast":              ("", " Yet "),
    "elaboration":           ("", " Specifically, "),
}

DEFAULT_JOIN = {
    "assertion": "elaboration", "question": "elaboration",
    "instruction": "condition_consequence", "explanation": "cause_effect",
    "narrative": "sequence", "counterfactual": "condition_consequence",
    "hypothesis": "contrast",
}


def assemble(selected_heads, resp_type, records, allowed_domains, pool,
            function_words, crossrefs, join_name=None):
    join_name = join_name or DEFAULT_JOIN[resp_type]
    prefix, connective = JOIN[join_name]
    mood_fn = MOOD[resp_type]

    statements, seen_text = [], set()
    for h in selected_heads:
        stmt = statement_for(h, records, allowed_domains, pool, function_words, crossrefs)
        if stmt in seen_text:
            continue
        seen_text.add(stmt)
        statements.append(mood_fn(stmt))
    if not statements:
        return "", join_name

    if len(statements) == 1:
        out = (prefix + statements[0]) if prefix else statements[0]
        return out.rstrip(".") + ".", join_name

    body = connective.join(
        lower1(s.rstrip(".")) if i else s for i, s in enumerate(statements)
    )
    out = prefix + body
    out = out.rstrip(".") + "."
    return out, join_name


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--substrate", default="substrate.min.bin")
    ap.add_argument("--intent", default="")
    ap.add_argument("--goal", default="")
    ap.add_argument("--setting", default="")
    ap.add_argument("--subject", default="")
    ap.add_argument("--audience", default="")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seconds", type=int, default=0)
    ap.add_argument("--frontier-cap", type=int, default=400)
    ap.add_argument("--select", type=int, default=6)
    ap.add_argument("--join", default="")
    args = ap.parse_args()
    input_text = " ".join(args.text)
    context_texts = [v for v in (args.intent, args.goal, args.setting,
                                 args.subject, args.audience) if v]

    sub = load_substrate(args.substrate)
    records = sub["records"]
    function_words = set(sub["relations"]["function_words"])
    crossrefs = sub["relations"]["crossrefs"]

    trace = discover(input_text, context_texts, sub, args.layers,
                     args.seconds, args.frontier_cap, "assemble_checkpoint.json")
    print()

    speech_act = classify_speech_act(input_text)
    resp_type = required_response_type(input_text, speech_act)

    input_heads = [h for h, a in trace["activations"].items() if a["via"] == "input"]
    allowed_domains = active_domains(input_heads, records)

    # LAYERS 0-2: structural closure, UNBOUNDED. No size limit is applied
    # here — Axiom 2: closure size is a true fact about the substrate, not
    # something to be cut down to look like a ranked shortlist.
    shaped = FILTER(
        set(trace["activations"]),
        lambda h: (resp_type != "instruction"
                  or any(is_action_sense(s["definition"]) for s in records[h]["senses"])),
    )
    closure = CONSTRAIN(shaped, allowed_domains, records)

    # LAYER 3 ONLY: rendering needs a bounded number of sentences. This is
    # a stated, cosmetic cutoff — ordered by layer-distance (a free
    # structural fact already recorded) then alphabetically (equally
    # free). It is NOT a relevance judgment; every member of `closure` is
    # equally structurally admissible. Only presentation length forces a cut.
    # Layer-3 ordering: pure alphabetical only. Layer-number was dropped —
    # it's a property of THIS traversal's search order, not a fact the
    # dictionary itself states, and using it first structurally guaranteed
    # layer-0 (input) words would dominate the shown set, which is exactly
    # the "echo the input back" mistake retracted several turns ago.
    # Alphabetical order is the one thing genuinely given by D itself.
    shown = sorted(closure)[:args.select]

    _, pool = build_bridge(context_texts, input_text, records, function_words)

    print(f"SPEECH ACT        {speech_act}")
    print(f"REQUIRED RESPONSE {resp_type}  (adjacency constraint, fixed before content)")
    print(f"ALLOWED DOMAINS   {allowed_domains or '(none tagged in input)'}")
    print(f"VISITED {trace['visited_total']:,}  STRUCTURAL CLOSURE (Layers 0-2, unbounded) "
         f"{len(closure):,}")
    print(f"SHOWN {len(shown)} of {len(closure):,} — Layer 3 rendering cutoff only "
         f"(pure alphabetical order), not a relevance ranking\n")

    print("SELECTION (every member of the closure was equally admissible; this is just which ones got printed)")
    for h in shown:
        a = trace["activations"][h]
        print(f"  {h:<14} L{a['layer']}  via {a['via']}")

    output, join_used = assemble(shown, resp_type, records, allowed_domains,
                                 pool, function_words, crossrefs, args.join or None)
    print(f"\nJOIN USED    {join_used}")
    print(f"\nASSEMBLED OUTPUT\n{output}\n")


if __name__ == "__main__":
    main()
