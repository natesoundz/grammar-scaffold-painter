"""
dict_algebra.py — the dictionary algebra, as executable definitions, not prose.

    D = the compiled substrate. w ∈ D is a headword.

    PRIMITIVE OBJECTS (all reads of already-compiled data, zero computation):
        senses(w)          the ordered tuple of sense-records for w
        domain(w)          the SET of domain tags w's senses carry (may be ∅)
        definition_words(w) the SET of content words across w's own definitions
        crossref(w)        the compiled SET of headwords w points to
        opposition(w)      the compiled SET of headwords w opposes
        phrase(w)          the compiled SET of compound-phrase entries under w
        ordered_before(w)  the immediate alphabetical predecessor of w in D
        ordered_after(w)   the immediate alphabetical successor of w in D
        relation(w, r)     the named edge-set of type r from w

    All of the above return SETS or fixed values. None return a count, a
    sum, a mean, or anything compared against a numeric cutoff. This is the
    whole point: a set does not remember how many paths reached it, or how
    many times an element was produced. "How many roots converge on this
    node" is not a question this algebra can even pose — that requires a
    multiset (a bag with counts), which is a different mathematical object.
    Convergence-via-cardinality failed in every form tried (raw sum, mean,
    boolean threshold) because every one of those forms first builds
    something with a count and then measures it. The fix isn't a better
    measurement. It's staying inside an algebra where measurement of a
    collection's SIZE is not an available operation at all.

    PRIMITIVE OPERATORS (all total functions: sets and predicates in, sets
    out — never a scalar):
        ACTIVATE(w)         -> the full compiled record for w (a lookup)
        FILTER(S, pred)     -> {x ∈ S : pred(x)}            (keep/discard)
        INTERSECT(A, B)     -> A ∩ B
        FOLLOW(w, r)        -> relation(w, r)                (one named edge)
        EXPAND(S, r)        -> ⋃_{w∈S} FOLLOW(w, r)           (set union)
        CONSTRAIN(S, doms)  -> FILTER(S, w: domain(w)⊆doms or domain(w)=∅)
        PROJECT(S, r)       -> {w: relation(w, r) for w ∈ S}  (structure-preserving map)
        TRACE(w)            -> the parent-edge audit chain (discover.py's chain_to)
        ASSEMBLE(S)         -> Layer 3 only. Render S to text. NOT part of
                              the structural algebra — see the axiom below.

    AXIOMS (explicit, stipulated rules ABOUT the algebra — not facts
    Layer 0 states, and must not be smuggled in as though they were):

        AXIOM 1 (domain compatibility): an untagged sense (domain(w) = ∅)
        is compatible with every constraint set. The dictionary does not
        say this; we are choosing to interpret absence-of-tag as
        non-conflict. Stated here, not hidden inside CONSTRAIN's code.

        AXIOM 2 (no bound in Layers 0–2): structural closure under EXPAND
        has no built-in size limit. For a rich input, the honestly-closed
        relevant set may be large — Webster's defining vocabulary is
        densely self-referential, and that may be a true fact about the
        substrate, not a bug in the algebra. Layer 3 (ASSEMBLE) is where a
        practical, EXPLICITLY COSMETIC bound is applied — using only free
        structural facts (ordered_before/after, layer-distance from
        input) — never smuggled in as a relevance judgment. This is the
        one place a count is allowed to exist, and it must be labeled as
        a rendering constraint, not a filter.

    LAYER SEPARATION:
        Layer 0  Dictionary exists (compiled, immutable).
        Layer 1  Navigate: ACTIVATE / FOLLOW / EXPAND / TRACE.
        Layer 2  Assemble compatible structures: FILTER / INTERSECT / CONSTRAIN.
        Layer 3  Produce language: ASSEMBLE. The only layer permitted to
                 apply a size bound, and only as a stated rendering limit.
"""

from dataclasses import dataclass


# ------------------------------------------------------- primitive objects

def senses(w, records):
    return records[w]["senses"]


def domain(w, records):
    return {s["domain"].strip().lower() for s in senses(w, records) if s["domain"]}


def definition_words(w, records, function_words):
    from discover import words as _words
    out = set()
    for s in senses(w, records):
        out |= {t for t in _words(s["definition"]) if t not in function_words}
    return out


def crossref(w, relations):
    return set(relations["crossrefs"].get(w, []))


def opposition(w, relations):
    return set(relations["oppositions"].get(w, []))


def phrase(w, records):
    return {p["phrase"] for s in senses(w, records) for p in s["phrases"]}


def ordered_before(w, sorted_headwords, index):
    i = index.get(w)
    return sorted_headwords[i - 1] if i and i > 0 else None


def ordered_after(w, sorted_headwords, index):
    i = index.get(w)
    return (sorted_headwords[i + 1]
           if i is not None and i + 1 < len(sorted_headwords) else None)


def relation(w, r, records, relations):
    return {
        "crossref": lambda: crossref(w, relations),
        "opposition": lambda: opposition(w, relations),
        "phrase": lambda: phrase(w, records),
        "morphological": lambda: set(relations.get("morphological", {}).get(w, [])),
    }[r]()


# --------------------------------------------------------- primitive operators

def ACTIVATE(w, records):
    """Lookup. Returns the record, or None if w isn't a headword."""
    return records.get(w)


def FILTER(S, pred):
    """{x ∈ S : pred(x)}. A predicate is a fixed boolean test — never a
    threshold on a count derived from S itself."""
    return {x for x in S if pred(x)}


def INTERSECT(A, B):
    return A & B


def FOLLOW(w, r, records, relations):
    return relation(w, r, records, relations)


def EXPAND(S, r, records, relations):
    """⋃_{w∈S} FOLLOW(w, r). Pure set union — multiplicity is discarded by
    construction. This is the direct replacement for convergence-counting:
    it answers 'what is reachable', never 'how many ways is it reachable'."""
    out = set()
    for w in S:
        out |= FOLLOW(w, r, records, relations)
    return out


def CONSTRAIN(S, allowed_domains, records):
    """AXIOM 1 applied: keep w if its domain tags are a subset of the
    allowed set, OR w carries no domain tag at all (untagged = compatible
    by stipulation, not by discovery)."""
    def ok(w):
        d = domain(w, records)
        return (not d) or d.issubset(allowed_domains)
    return FILTER(S, ok)


def PROJECT(S, r, records, relations):
    """Structure-preserving map: keeps the association between each w and
    its own relation-set, rather than collapsing into one union."""
    return {w: relation(w, r, records, relations) for w in S}


def TRACE(w, trace):
    from discover import chain_to
    return chain_to(w, trace)


@dataclass
class Substrate:
    records: dict
    relations: dict
    sorted_headwords: list
    index: dict

    @classmethod
    def load(cls, path):
        from discover import load_substrate
        sub = load_substrate(path)
        sh = sorted(sub["records"].keys())
        return cls(
            records=sub["records"],
            relations=sub["relations"],
            sorted_headwords=sh,
            index={w: i for i, w in enumerate(sh)},
        )
