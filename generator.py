"""
generator.py — Reverse traversal: Listener State  →  Text

Generation reverses the analysis. It begins with the listener state —
not with text. It traverses the same substrate in the opposite direction.

  Forward  (analysis):    Text  →  definitions outward  →  State
  Reverse  (generation):  State  →  definitions inward   →  Headword selection  →  Text

The "inward" direction: instead of following a word's definition outward
to find more concepts, we find which closure members' definitions contain
the input/context concepts already in the listener state. Those headwords
are the substrate-central concepts for this communicative situation — the
words that, when read, direct attention BACK to what the speaker said.

All selection decisions are binary membership tests (∩ ≠ ∅), not counts.
This keeps generation within the same algebra as analysis.

  Tier A  definitions reference input-field concepts    (∩ input_field ≠ ∅)
  Tier B  definitions reference context-field concepts  (∩ context_field ≠ ∅)
  Tier C  remaining closure members

Within each tier: anchor-rank ordering — direct lemma match to input/context
first, then definition coverage of input terms, then domain match, then
lower traversal depth, then shorter definition, then alphabetical.
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import resolve, words, sense_bridge, build_bridge
from dict_algebra import (
    definition_words, FILTER, EXPAND, FOLLOW, INTERSECT, CONSTRAIN,
    PROJECT, relation,
)
from listener_state import ListenerState
from transform_compiler import load_rules, match_rule
from scaffold import ScaffoldAnalyzer

_ACTION_RE = re.compile(r"^\s*to\s+\w", re.I)

# Grammatical function words that Webster's defines as headwords but the
# substrate's function_words set misses. These pollute Tier A selection
# because their definitions reference common input words.
_EXTRA_FUNC_WORDS = frozenset({
    "because", "then", "how", "until", "why", "should", "would", "could",
    "might", "must", "does", "did", "though", "although", "unless", "since",
    "while", "whereas", "therefore", "however", "thus", "yet", "still",
    "whether", "what", "when", "where", "which", "who", "shall", "may",
    "if", "than", "except",
})

# ---------------------------------------------------------------- mood and join tables

_STUB_RE = re.compile(r"^\s*(imp\.|p\.\s*p\.|pp\.|pres\.\s*p\.|impf\.|syn\.)", re.I)

MOOD = {
    "assertion":      lambda s: s,
    "question":       lambda s: f"What follows from the fact that {_lower1(s.rstrip('.'))}?",
    "instruction":    lambda s: f"Take into account that {_lower1(s.rstrip('.'))}.",
    "explanation":    lambda s: s,
    "narrative":      lambda s: s,
    "counterfactual": lambda s: f"Were this otherwise, {_lower1(s.rstrip('.'))} would not hold.",
    "hypothesis":     lambda s: f"It may be that {_lower1(s.rstrip('.'))}.",
}

JOIN = {
    "plain":                 ("", " "),
    "cause_effect":          ("", " Because of this, "),
    "sequence":              ("", " Then, "),
    "condition_consequence": ("If ", ", then "),
    "contrast":              ("", " Yet "),
    "elaboration":           ("", " Specifically, "),
}

DEFAULT_JOIN = {
    "assertion":      "elaboration",
    "question":       "elaboration",
    "instruction":    "condition_consequence",
    "explanation":    "cause_effect",
    "narrative":      "sequence",
    "counterfactual": "condition_consequence",
    "hypothesis":     "contrast",
}


# ---------------------------------------------------------------- neighbor transform modes

NEIGHBOR_MODE_DEFAULT = {
    "assertion":      "plus_crossref",
    "question":       "plus_crossref",
    "instruction":    "exemplify",
    "explanation":    "definition_chain",
    "narrative":      "definition_chain",
    "counterfactual": "plus_opposition",
    "hypothesis":     "plus_opposition",
}

# Mode rotation for multi-headword output: each headword gets a different
# relation type, so one output reveals multiple neighborhood dimensions.
# The anchor (i=0) gets the discourse-default mode; subsequent headwords
# cycle through the remaining modes.
MODE_ROTATION = [
    "definition_chain",
    "plus_opposition",
    "exemplify",
    "plus_morphological",
    "plus_crossref",
    "primary_only",
]


# ---------------------------------------------------------------- helpers

def _lower1(s):
    return s[0].lower() + s[1:] if s else s


def _first_clause(definition: str, max_words: int = 14) -> str:
    t = re.sub(r"^\d{1,2}\.\s*", "", definition)
    t = re.split(r"[;.]", t, 1)[0].strip()
    if not t:
        return t
    ws = t.split()
    if len(ws) > max_words:
        t = " ".join(ws[:max_words])
    return t[0].upper() + t[1:]


def _best_sense_idx(head, records, allowed_domains, pool, function_words,
                    response_type=""):
    """Pick the sense that best matches discourse shape then domain constraints.

    Shape takes priority over domain:
      instruction  → prefer verb senses  (definition starts with "To ...")
      all others   → prefer noun/adjective senses

    Within the shape-preferred set, domain conflict is the tiebreak.
    """
    senses = records[head]["senses"]
    want_verb = (response_type == "instruction")

    preferred, other, conflicting = [], [], []
    for i, s in enumerate(senses):
        dom = s["domain"].strip().lower() if s["domain"] else ""
        is_verb = bool(_ACTION_RE.match(s["definition"]))
        domain_ok = not (dom and allowed_domains and dom not in allowed_domains)
        shape_ok = (want_verb and is_verb) or (not want_verb and not is_verb)

        if not domain_ok:
            conflicting.append(i)
        elif shape_ok:
            preferred.append(i)
        else:
            other.append(i)

    if preferred:
        return preferred[0]
    if other:
        return other[0]
    return conflicting[0] if conflicting else 0


def _statement_for(head, records, allowed_domains, pool, function_words, crossrefs,
                   response_type="", _seen=None,
                   neighbor_mode="primary_only", oppositions=None,
                   morphological=None):
    _seen = _seen or set()
    oppositions = oppositions or {}
    morphological = morphological or {}

    idx = _best_sense_idx(head, records, allowed_domains, pool, function_words,
                          response_type)
    sense = records[head]["senses"][idx]
    definition = sense["definition"]

    if _STUB_RE.match(definition) and head not in _seen:
        targets = crossrefs.get(head, [])
        if targets:
            _seen.add(head)
            return _statement_for(targets[0], records, allowed_domains, pool,
                                  function_words, crossrefs, response_type, _seen,
                                  neighbor_mode, oppositions)

    # ---- exemplify: prefer Webster's usage examples (instruction-optimised)
    if neighbor_mode == "exemplify" or (response_type == "instruction" and sense.get("usage")):
        if sense.get("usage"):
            surface = sense["usage"][0].strip().rstrip(".,;")
            if surface:
                return f"{head.capitalize()}: {surface[0].lower() + surface[1:]}"

    base = _first_clause(definition)
    core = f"{head.capitalize()}: {base[0].lower() + base[1:]}" if base else head

    # ---- primary_only: stop here
    if neighbor_mode == "primary_only":
        return core

    # ---- plus_crossref: append most relevant crossref definition
    if neighbor_mode == "plus_crossref":
        for ref in crossrefs.get(head, [])[:4]:
            if ref in records and ref not in _seen and ref in pool:
                ref_sense = records[ref]["senses"][0]
                ref_clause = _first_clause(ref_sense["definition"])
                if ref_clause and ref_clause.lower() != base.lower():
                    return f"{core} (→ {ref}: {_lower1(ref_clause)})"
        return core

    # ---- plus_opposition: append antonym as contrast boundary
    if neighbor_mode == "plus_opposition":
        opps = oppositions.get(head, [])
        if opps:
            opp = opps[0]
            if opp in records:
                opp_sense = records[opp]["senses"][0]
                opp_clause = _first_clause(opp_sense["definition"])
                if opp_clause:
                    return f"{core} (≠ {opp}: {_lower1(opp_clause)})"
        return core

    # ---- definition_chain: expand via the most relevant definition word
    if neighbor_mode == "definition_chain":
        def_words = words(definition)
        for w in def_words:
            h2 = w if w in records else resolve(w, records)
            if not h2 or h2 == head or h2 in _seen or h2 not in pool:
                continue
            if h2 in records:
                ch_sense = records[h2]["senses"][0]
                ch_clause = _first_clause(ch_sense["definition"])
                if ch_clause and ch_clause.lower() != base.lower():
                    return f"{core} — {h2}: {_lower1(ch_clause)}"
        return core

    # ---- plus_morphological: append a morphological neighbor's definition
    if neighbor_mode == "plus_morphological":
        for mneigh in morphological.get(head, [])[:4]:
            if mneigh in records and mneigh not in _seen and mneigh in pool:
                m_sense = records[mneigh]["senses"][0]
                m_clause = _first_clause(m_sense["definition"])
                if m_clause and m_clause.lower() != base.lower():
                    return f"{core} (~ {mneigh}: {_lower1(m_clause)})"
        return core

    return core


def _statement_composite(head, records, allowed_domains, pool, function_words,
                          crossrefs, response_type, modes, oppositions,
                          morphological, _seen=None):
    """Apply multiple neighbor modes to one headword, stacking relations.

    Each mode that fires appends its relation annotation. This makes a
    single headword reveal multiple neighborhood dimensions at once:

        Carbon: ... (≠ nitrogen: ...) (~ carbonic: ...) (→ diamond: ...)

    Modes are applied in order. If a mode finds no neighbor, it is skipped
    silently — the output shows only relations that actually fired.
    """
    _seen = _seen or set()
    _seen.add(head)

    idx = _best_sense_idx(head, records, allowed_domains, pool, function_words,
                          response_type)
    sense = records[head]["senses"][idx]
    definition = sense["definition"]

    if _STUB_RE.match(definition):
        targets = crossrefs.get(head, [])
        if targets:
            return _statement_composite(
                targets[0], records, allowed_domains, pool, function_words,
                crossrefs, response_type, modes, oppositions,
                morphological, _seen)

    # exemplify as base
    base = _first_clause(definition)
    if "exemplify" in modes and sense.get("usage"):
        surface = sense["usage"][0].strip().rstrip(".,;")
        if surface:
            core = f"{head.capitalize()}: {surface[0].lower() + surface[1:]}"
        else:
            core = f"{head.capitalize()}: {base[0].lower() + base[1:]}" if base else head
    else:
        core = f"{head.capitalize()}: {base[0].lower() + base[1:]}" if base else head

    annotations = []

    for mode in modes:
        if mode in ("primary_only", "exemplify"):
            continue

        if mode == "plus_opposition":
            opps = oppositions.get(head, [])
            for opp in opps[:2]:
                if opp in records and opp not in _seen:
                    opp_sense = records[opp]["senses"][0]
                    opp_clause = _first_clause(opp_sense["definition"])
                    if opp_clause:
                        annotations.append(f"≠ {opp}: {_lower1(opp_clause)}")
                        _seen.add(opp)
                        break

        elif mode == "plus_morphological":
            for mneigh in morphological.get(head, [])[:4]:
                if mneigh in records and mneigh not in _seen and mneigh in pool:
                    m_sense = records[mneigh]["senses"][0]
                    m_clause = _first_clause(m_sense["definition"])
                    if m_clause and m_clause.lower() != base.lower():
                        annotations.append(f"~ {mneigh}: {_lower1(m_clause)}")
                        _seen.add(mneigh)
                        break

        elif mode == "plus_crossref":
            for ref in crossrefs.get(head, [])[:4]:
                if ref in records and ref not in _seen and ref in pool:
                    ref_sense = records[ref]["senses"][0]
                    ref_clause = _first_clause(ref_sense["definition"])
                    if ref_clause and ref_clause.lower() != base.lower():
                        annotations.append(f"→ {ref}: {_lower1(ref_clause)}")
                        _seen.add(ref)
                        break

        elif mode == "definition_chain":
            def_words = words(definition)
            for w in def_words:
                h2 = w if w in records else resolve(w, records)
                if not h2 or h2 == head or h2 in _seen or h2 not in pool:
                    continue
                if h2 in records:
                    ch_sense = records[h2]["senses"][0]
                    ch_clause = _first_clause(ch_sense["definition"])
                    if ch_clause and ch_clause.lower() != base.lower():
                        annotations.append(f"— {h2}: {_lower1(ch_clause)}")
                        _seen.add(h2)
                        break

    if annotations:
        return core + " " + " ".join(f"({a})" for a in annotations)
    return core


def _algebraic_distill(selected_heads, state, sub, neighbor_mode):
    """Use the dictionary algebra to post-process generation output.

    Three operations, each a pure algebraic expression:

    DISTILL — keep only annotations whose neighbor definitions reference
        back to the input field. This is INTERSECT(neighbor_def_words, input_field).
        Removes annotations that are structurally disconnected from the speaker's
        actual words.

    EXPAND — show what's reachable from the selected set via each relation type.
        This is EXPAND(selected, r) for each r in the active relation set.
        Gives the user a "next layer" preview without running another cycle.

    EMPHASIZE — mark which relation types actually fired for each headword.
        This is PROJECT(selected, r) — shows the structure-preserving map,
        not just a collapsed union.
    """
    records = sub["records"]
    relations = sub["relations"]
    function_words = set(relations["function_words"])
    input_field = state.fields.input_field
    allowed_domains = state.allowed_domains

    if not selected_heads or not input_field:
        return None

    selected_set = set(selected_heads)

    # ---- DISTILL: which selected headwords have definitions referencing input?
    # This is the algebraic test: INTERSECT(definition_words(h), input_field) ≠ ∅
    distilled = {}
    for h in selected_heads:
        dw = definition_words(h, records, function_words)
        overlap = INTERSECT(dw, input_field)
        if overlap:
            distilled[h] = sorted(overlap)[:5]

    # ---- EXPAND: one-hop reachable set from selected, per relation type
    relation_types = ["crossref", "opposition", "morphological"]
    expansion = {}
    for r in relation_types:
        reachable = EXPAND(selected_set, r, records, relations)
        # CONSTRAIN to allowed domains
        constrained = CONSTRAIN(reachable, allowed_domains, records)
        # INTERSECT with closure to keep only structurally relevant
        in_closure = INTERSECT(constrained, state.closure)
        new_only = in_closure - selected_set
        if new_only:
            expansion[r] = sorted(new_only)[:8]

    # ---- EMPHASIZE: which relations fired for each headword?
    emphasis = {}
    for h in selected_heads:
        fired = []
        for r in relation_types:
            edges = FOLLOW(h, r, records, relations)
            if edges:
                fired.append((r, len(edges), sorted(edges)[:3]))
        if fired:
            emphasis[h] = fired

    return {
        "distilled": distilled,
        "expansion": expansion,
        "emphasis": emphasis,
    }


# ---------------------------------------------------------------- join-affinity vocabulary

JOIN_AFFINITY = {
    "cause_effect":          {"because", "therefore", "hence", "result", "cause",
                              "effect", "due", "thus", "consequently", "reason",
                              "produce", "lead", "induce", "occasion"},
    "elaboration":           {"specifically", "namely", "type", "kind", "form",
                              "variety", "species", "genus", "class", "example",
                              "instance", "particular", "sort", "manner"},
    "sequence":              {"first", "then", "next", "finally", "step", "process",
                              "procedure", "stage", "phase", "order", "succession",
                              "follow", "precede", "subsequent"},
    "condition_consequence": {"if", "when", "condition", "provided", "unless",
                              "should", "assume", "given", "depend", "require",
                              "suppose", "case", "circumstance"},
    "contrast":              {"but", "however", "yet", "although", "whereas",
                              "contrary", "opposite", "unlike", "differ",
                              "distinguish", "oppose", "contrast", "despite"},
    "plain":                 set(),
}


# ---------------------------------------------------------------- anchor ranking

def _anchor_rank(head, records, input_field, context_field,
                allowed_domains, activations, join_name="plain"):
    """Sort key for a closure headword. Returns tuple — lower = higher priority.

    Priority:
      1. direct lemma match to input field          +1000
      2. direct lemma match to context field        +800
      3. definition covers input terms              +120 per term
      4. definition covers context terms            +80  per term
      5. domain match to allowed domains            +150
      6. join-affinity: definition contains join    +60  per affinity word
         vocabulary words for the active join
      7. lower traversal layer                      -10  per layer
      8. shorter definition (less dilution)         -0.01 per char
      9. alphabetical as tiebreak
    """
    rec    = records.get(head, {})
    senses = rec.get("senses", [])
    defn   = senses[0]["definition"].lower() if senses else ""
    dom    = (senses[0]["domain"] or "").strip().lower() if senses else ""
    layer  = activations.get(head, {}).get("layer", 9)

    score = 0
    w = head.lower()

    if w in input_field:   score += 1000
    if w in context_field: score += 800

    score += 120 * sum(1 for t in input_field   if t in defn)
    score += 80  * sum(1 for t in context_field if t in defn)

    if allowed_domains and dom and dom in allowed_domains:
        score += 150

    affinity_vocab = JOIN_AFFINITY.get(join_name, set())
    score += 60 * sum(1 for av in affinity_vocab if av in defn)

    score -= 10  * layer
    score -= 0.01 * len(defn)

    return (-score, layer, w)


# ---------------------------------------------------------------- reverse traversal selection

def _reverse_select(state: ListenerState, records, function_words, n: int,
                    join_name: str = "plain", exclude: set = None):
    """Partition the closure by which members' definitions point back toward
    the input or context concepts. Binary membership only — no counts.

    Tier A: definition_words(h) ∩ input_field  ≠ ∅
    Tier B: definition_words(h) ∩ context_field ≠ ∅  (and not in A)
    Tier C: everything else in closure

    Returns (ordered_list[:n], tier_sizes, tier_assignment_dict, constraints).
    constraints[h] = {
        "tier": "A"|"B"|"C",
        "input_overlap": [...],      # which input words this def references
        "context_overlap": [...],    # which context words this def references
        "domain": str,               # this headword's domain
        "domain_allowed": bool,      # whether domain is in allowed_domains
        "layer": int,                # traversal depth where discovered
        "via": str,                  # edge type that discovered it
        "is_function_word": bool,    # whether it's a grammatical word
        "anchor_score": float,       # raw ranking score (lower = better)
    }
    """
    input_field = state.fields.input_field
    context_field = state.fields.context_field

    # Filter function words from closure before tiering — words like
    # 'because', 'then', 'how', 'until', 'why' are Webster's headwords
    # but are grammatical, not content. Without this they win Tier A
    # because their definitions reference common input words.
    closure = state.closure - function_words

    tier_a = FILTER(
        closure,
        lambda h: bool(definition_words(h, records, function_words) & input_field),
    )
    tier_b = FILTER(
        closure - tier_a,
        lambda h: bool(definition_words(h, records, function_words) & context_field),
    )
    tier_c = closure - tier_a - tier_b

    assignment = {}
    for h in tier_a:
        assignment[h] = "A"
    for h in tier_b:
        assignment[h] = "B"
    for h in tier_c:
        assignment[h] = "C"

    activations   = state.trace.get("activations", {})
    input_field   = state.fields.input_field
    context_field = state.fields.context_field
    allowed       = state.allowed_domains

    def rank(h):
        return _anchor_rank(h, records, input_field, context_field,
                            allowed, activations, join_name=join_name)

    exclude = exclude or set()
    ordered = (sorted(tier_a, key=rank)
             + sorted(tier_b, key=rank)
             + sorted(tier_c, key=rank))
    ordered = [h for h in ordered if h not in exclude]
    sizes = {"tier_a": len(tier_a), "tier_b": len(tier_b), "tier_c": len(tier_c)}

    # ---- Per-term constraint satisfaction record
    constraints = {}
    for h in ordered[:n]:
        act = activations.get(h, {})
        dw = definition_words(h, records, function_words)
        inp_overlap = sorted(dw & input_field)
        ctx_overlap = sorted(dw & context_field)
        dom = records[h]["senses"][0].get("domain", "").strip().lower() if records[h]["senses"] else ""
        dom_ok = not (dom and allowed and dom not in allowed)
        constraints[h] = {
            "tier":            assignment[h],
            "input_overlap":   inp_overlap,
            "context_overlap": ctx_overlap,
            "domain":          dom,
            "domain_allowed":  dom_ok,
            "layer":           act.get("layer", "?"),
            "via":             act.get("via", "?"),
            "anchor_score":    rank(h)[0],
        }

    return ordered[:n], sizes, assignment, constraints


# ---------------------------------------------------------------- main entry point

_RULES_CACHE = None
_RULES_PATH  = os.path.join(_HERE, "transforms.jsonl")


def generate(state: ListenerState, sub, n: int = 6, join_name: str = None,
             transforms=None, exclude: set = None,
             force_neighbor_mode: str = None) -> dict:
    """
    Generate text from a ListenerState via reverse substrate traversal.

    Returns a dict with keys:
      output          — the assembled text
      join_used       — which join pattern was applied
      selected        — list of (head, tier, layer, via)
      tier_sizes      — {"tier_a": int, "tier_b": int, "tier_c": int}
      rule_matched    — True if a compiled transformation rule was applied
      rule_confidence — confidence score of the matched rule (0 if none)
    """
    global _RULES_CACHE

    records = sub["records"]
    function_words = set(sub["relations"]["function_words"]) | _EXTRA_FUNC_WORDS
    crossrefs  = sub["relations"]["crossrefs"]
    oppositions = sub["relations"].get("oppositions", {})
    morphological = sub["relations"].get("morphological", {})

    input_text = state.trace["input"]
    context_texts = state.trace["context"]
    _, pool = build_bridge(context_texts, input_text, records, function_words)

    resp_type = state.frame.response_type

    # ---- Load and match compiled transformation rule
    if transforms is None:
        if _RULES_CACHE is None:
            _RULES_CACHE = load_rules(_RULES_PATH)
        transforms = _RULES_CACHE

    matched_rule = match_rule(state, transforms) if transforms else None
    rule_matched     = matched_rule is not None
    rule_confidence  = matched_rule["confidence"] if matched_rule else 0.0

    # Caller-supplied join > matched rule > default
    if join_name:
        resolved_join = join_name
    elif matched_rule:
        resolved_join = matched_rule["join"]
    else:
        resolved_join = DEFAULT_JOIN.get(resp_type, "elaboration")

    join_name = resolved_join
    prefix, connective = JOIN[join_name]
    mood_fn = MOOD[resp_type]

    # neighbor_mode: forced > matched rule > discourse default
    neighbor_mode = force_neighbor_mode or \
                    (matched_rule.get("neighbor_mode")
                     if matched_rule else None) or \
                    NEIGHBOR_MODE_DEFAULT.get(resp_type, "primary_only")

    selected_heads, tier_sizes, assignment, constraints = _reverse_select(
        state, records, function_words, n, join_name=join_name,
        exclude=exclude or set(),
    )

    statements, seen_text = [], set()
    # Build the mode list for composite rendering: active mode + all others
    all_modes = [neighbor_mode] + [m for m in MODE_ROTATION if m != neighbor_mode]

    for i, h in enumerate(selected_heads):
        if i == 0:
            # Anchor: composite — stack all relation types on one headword
            stmt = _statement_composite(
                h, records, state.allowed_domains, pool, function_words,
                crossrefs, resp_type, all_modes, oppositions, morphological,
            )
        else:
            # Remaining headwords: rotate through modes so each reveals
            # a different neighborhood dimension
            effective_mode = MODE_ROTATION[(i - 1) % len(MODE_ROTATION)]
            stmt = _statement_for(
                h, records, state.allowed_domains, pool, function_words, crossrefs,
                response_type=resp_type,
                neighbor_mode=effective_mode,
                oppositions=oppositions,
                morphological=morphological,
            )
        if stmt in seen_text:
            continue
        seen_text.add(stmt)
        statements.append(mood_fn(stmt))

    if not statements:
        output = ""
    elif len(statements) == 1:
        raw = (prefix + statements[0]) if prefix else statements[0]
        output = raw.rstrip(".") + "."
    else:
        body = connective.join(
            _lower1(s.rstrip(".")) if i else s
            for i, s in enumerate(statements)
        )
        output = (prefix + body).rstrip(".") + "."

    selected = []
    for h in selected_heads:
        a = state.trace["activations"].get(h, {})
        selected.append((h, assignment.get(h, "?"), a.get("layer", "?"), a.get("via", "?")))

    # ---- Algebraic post-processing: distill, expand, emphasize
    algebra = _algebraic_distill(selected_heads, state, sub, neighbor_mode)

    # ---- Scaffold analysis: validate grammatical frame compatibility
    _scaffold_analyzer = ScaffoldAnalyzer()
    scaffold_report = _scaffold_analyzer.validate_selection(
        selected, records, join_name,
    )

    return {
        "output":          output,
        "join_used":       join_name,
        "neighbor_mode":   neighbor_mode,
        "selected":        selected,
        "tier_sizes":      tier_sizes,
        "rule_matched":    rule_matched,
        "rule_confidence": rule_confidence,
        "algebra":         algebra,
        "constraints":     constraints,
        "scaffold":        scaffold_report,
    }
