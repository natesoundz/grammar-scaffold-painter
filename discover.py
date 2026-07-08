"""
discover.py — model-free auto-discovery engine over the compiled substrate.

    python discover.py "The bolt is stuck." --intent "disassemble remove repair" \
        --setting "workshop machinery" --layers 8

No model anywhere. The run is pure traversal of compiled structure:

  LAYER 0   the input statement.
  LAYER N   content words of layer N-1's statements resolve to headwords;
            context-aligned senses fire; their definitions become the new
            statements. Crossrefs are followed. Oppositions are followed —
            the antonym pole ("loose", "free") is reachable structurally.
  BRIDGE    context filter, exactly as established: a sense joins the
            traversal only if its definition shares content substance with
            the context pool (binary alignment, not a score). No context =
            everything aligned.
  STOP      saturation (a layer yields no unvisited headwords), or the
            --layers / --seconds limit.
  TRACE     every activation records (word, via-relation, parent, layer,
            sense). "Why did galling fire at layer 4?" is answered by
            walking the trace, not by trusting the machine.
  TRANSFORM the discourse x rhetoric x language-device mechanism is compiled
            capacity: applied on demand to any traced statement, never
            pre-generated. Content words invariant; scaffolds shift.

Checkpoints every layer (JSON, never pickle). SIGINT saves and exits loud.
Frontier is capped per layer for tractability — deterministically
(alphabetical), stated here rather than hidden.
"""

import argparse
import json
import lzma
import re
import signal
import sys
import time
from collections import deque
from pathlib import Path

import msgpack

WORD_RE = re.compile(r"[a-z]+")
XZ_MAGIC = b"\xfd7zXZ\x00"


def words(text):
    return WORD_RE.findall(text.lower())


def load_substrate(path):
    raw = Path(path).read_bytes()
    if raw[:6] == XZ_MAGIC:
        raw = lzma.decompress(raw)
    sub = msgpack.unpackb(raw, strict_map_key=False)
    if sub["meta"]["format"] < 2:
        raise RuntimeError("substrate format 2 required (relations missing) — recompile")
    return sub


def resolve(surface, records):
    w = surface.lower()
    if w in records:
        return w
    cands = []
    if w.endswith("ies"): cands.append(w[:-3] + "y")
    if w.endswith("es"):  cands += [w[:-2], w[:-1]]
    if w.endswith("s"):   cands.append(w[:-1])
    if w.endswith("ed"):
        cands += [w[:-2], w[:-1], w[:-2] + "e"]
        if len(w) > 4 and w[-3] == w[-4]: cands.append(w[:-3])
    if w.endswith("ing"):
        cands += [w[:-3], w[:-3] + "e"]
        if len(w) > 5 and w[-4] == w[-5]: cands.append(w[:-4])
    for c in cands:
        if c in records:
            return c
    return None


# ------------------------------------------------- context bridge (filter)

def content_of(text, function_words):
    return {w for w in words(text) if w not in function_words}


def build_bridge(context_texts, input_text, records, function_words):
    """Mutual selection. CORE = the content headwords of the context AND the
    input themselves. POOL = core + content words of only those context
    senses that align with the core — so an archaic sense of 'repair'
    ('to go, betake oneself') contributes nothing when the core is
    mechanical. Context selects input; input selects context."""
    core = set()
    for text in context_texts + [input_text]:
        for surface in words(text):
            head = resolve(surface, records)
            if head and head not in function_words:
                core.add(head)

    pool = set(core)
    for text in context_texts:
        for surface in words(text):
            head = resolve(surface, records)
            if head is None or head in function_words:
                continue
            for sense in records[head]["senses"]:
                c = content_of(sense["definition"], function_words)
                if (c - {head}) & core:            # mutual alignment, no self-admission
                    pool |= c
    return core, pool


def sense_bridge(sense, pool, function_words, self_head=""):
    """Which pool words admit this sense? Empty set = not admitted.
    A sense may NEVER bridge on its own headword — otherwise every sense of
    'repair' self-admits because listings routinely repeat their headword,
    and quotation debris floods the pool (measured: 'dunsinane' entered the
    pool through exactly this hole)."""
    if not pool:
        return {"*"}
    return (content_of(sense["definition"], function_words) & pool) - {self_head}


def word_gate(candidate, pool, records, function_words, _memo={}):
    """A word extracted from an admitted sense joins the frontier only if
    its OWN listing shares substance with the pool — the same bridge,
    applied one level deeper. Self-justification excluded here too.
    Returns the justifying words (empty = deny)."""
    if candidate in _memo:
        return _memo[candidate]
    if candidate in pool:
        _memo[candidate] = {candidate}
        return _memo[candidate]
    shared = set()
    for sense in records[candidate]["senses"]:
        shared |= (content_of(sense["definition"], function_words) & pool) - {candidate}
        if shared:
            break
    _memo[candidate] = shared
    return shared


# --------------------------------------------------- transformation capacity

DISCOURSE = {
    "statement":   lambda s: s,
    "question":    lambda s: f"Is it the case that {s.rstrip('.')}?",
    "instruction": lambda s: f"Ensure that {s.rstrip('.')}.",
    "explanation": lambda s: f"This holds because {s.rstrip('.')}.",
    "argument":    lambda s: f"It must be accepted that {s.rstrip('.')}.",
}
RHETORIC = {
    "plain":                 lambda s: s,
    "problem_solution":      lambda s: f"{s.rstrip('.')} — therefore a remedy is required.",
    "cause_effect":          lambda s: f"Because {s[0].lower() + s[1:].rstrip('.')}, consequences follow.",
    "sequence":              lambda s: f"First, {s[0].lower() + s[1:].rstrip('.')}; what follows comes next.",
    "condition_consequence": lambda s: f"If {s[0].lower() + s[1:].rstrip('.')}, then the outcome is determined.",
    "contrast":              lambda s: f"{s.rstrip('.')}, yet the opposite condition is conceivable.",
    "elaboration":           lambda s: f"{s.rstrip('.')} — specifically, in every particular the definition names.",
}
LANGUAGE = {
    "literal":         lambda s: s,
    "personification": lambda s: s.replace(" is ", " insists on being ", 1) if " is " in s else s,
    "understatement":  lambda s: f"{s.rstrip('.')}, to put it mildly.",
    "hyperbole":       lambda s: f"{s.rstrip('.')}, beyond all measure.",
}


def transform(statement, discourse="statement", rhetoric="plain", language="literal"):
    """Compiled capacity: content words invariant, scaffolds shift. Composable
    because the three axes touch only function structure."""
    s = LANGUAGE[language](statement)
    s = RHETORIC[rhetoric](s)
    s = DISCOURSE[discourse](s)
    return s


# ------------------------------------------------- frontier relevance scoring

def _frontier_score(head, records, core, pool):
    """Score a frontier candidate by semantic proximity to the input core.

    Higher score = keep in frontier when cap is applied.
    Deterministic: no randomness; tiebreak is alphabetical in the caller.

    core = direct input headwords (strongest signal)
    pool = core + context-aligned words (weaker signal)
    """
    rec    = records.get(head, {})
    senses = rec.get("senses", [])
    if not senses:
        return 0
    defn_words = set(WORD_RE.findall(senses[0].get("definition", "").lower()))
    core_hits = len(defn_words & core)
    pool_hits = len(defn_words & pool) - core_hits   # don't double-count
    # head itself is in core: strong bonus
    self_bonus = 200 if head in core else 0
    return self_bonus + core_hits * 100 + pool_hits * 10


# ------------------------------------------------------------- the traversal

def discover(input_text, context_texts, sub, max_layers, max_seconds,
             frontier_cap, checkpoint_path):
    records = sub["records"]
    rel = sub["relations"]
    function_words = set(rel["function_words"])
    oppositions = rel["oppositions"]
    crossrefs = rel["crossrefs"]
    morphological = rel.get("morphological", {})

    core, pool = build_bridge(context_texts, input_text, records, function_words)

    trace = {
        "input": input_text,
        "context": context_texts,
        "bridge_core": sorted(core),
        "bridge_pool_size": len(pool),
        "layers": [],
        "activations": {},   # head -> {layer, via, parent, bridge_words} (first discovery)
        "all_parents": {},   # head -> [(parent, via, layer), ...] EVERY reaching path,
                             # not just the first — this is what convergence is measured
                             # from downstream. Discarding this was the actual gap.
    }

    visited = set()
    frontier = []
    for surface in words(input_text):
        head = resolve(surface, records)
        if head and head not in function_words and head not in visited:
            visited.add(head)
            frontier.append(head)
            trace["activations"][head] = {
                "layer": 0, "via": "input", "parent": None,
                "senses_fired": list(range(len(records[head]["senses"]))),
            }
            trace["all_parents"][head] = [(None, "input", 0)]
    trace["layers"].append({"layer": 0, "new_headwords": sorted(frontier)})

    def record_edge(h_from, h_to, via, layer):
        trace["all_parents"].setdefault(h_to, []).append((h_from, via, layer))

    interrupted = {"flag": False}

    def on_sigint(signum, frame):
        interrupted["flag"] = True
    prev = signal.signal(signal.SIGINT, on_sigint)

    start = time.time()
    layer = 0
    try:
        while frontier and layer < max_layers:
            if max_seconds and time.time() - start > max_seconds:
                trace["stopped"] = "time limit"
                break
            if interrupted["flag"]:
                trace["stopped"] = "interrupted"
                break
            layer += 1
            next_frontier = []
            new_this_layer = []

            for head in sorted(frontier):
                record = records[head]
                fired = []
                for si, sense in enumerate(record["senses"]):
                    admit = sense_bridge(sense, pool, function_words, self_head=head)
                    if not admit:
                        continue
                    fired.append(si)
                    for w in words(sense["definition"]):
                        h2 = w if w in records else resolve(w, records)
                        if not h2 or h2 in function_words:
                            continue
                        gate = word_gate(h2, pool, records, function_words)
                        if not gate:
                            continue
                        # record this edge regardless of visited status —
                        # convergence needs every path, not just the first
                        record_edge(head, h2, "definition", layer)
                        if h2 in visited:
                            continue
                        visited.add(h2)
                        next_frontier.append(h2)
                        new_this_layer.append(h2)
                        trace["activations"][h2] = {
                            "layer": layer, "via": "definition",
                            "parent": head, "sense": si,
                            "bridge_words": sorted(gate),
                        }
                if fired:
                    trace["activations"][head]["senses_fired"] = fired

                for h2 in crossrefs.get(head, []):
                    if h2 in function_words:
                        continue
                    gate = word_gate(h2, pool, records, function_words)
                    if not gate:
                        continue
                    record_edge(head, h2, "crossref", layer)
                    if h2 in visited:
                        continue
                    visited.add(h2)
                    next_frontier.append(h2)
                    new_this_layer.append(h2)
                    trace["activations"][h2] = {
                        "layer": layer, "via": "crossref", "parent": head,
                        "bridge_words": sorted(gate),
                    }
                for h2 in oppositions.get(head, []):
                    if h2 in function_words:
                        continue
                    # oppositions are pole jumps: always admissible — the
                    # antonym of an aligned word is structurally relevant
                    record_edge(head, h2, "opposition", layer)
                    if h2 in visited:
                        continue
                    visited.add(h2)
                    next_frontier.append(h2)
                    new_this_layer.append(h2)
                    trace["activations"][h2] = {
                        "layer": layer, "via": "opposition", "parent": head,
                        "bridge_words": ["(pole jump)"],
                    }
                # morphological neighbors: words sharing prefix/suffix.
                # These are structural form-links, not semantic — but they
                # connect word families (compose/composition/composer) that
                # definitions alone may miss. Gate by pool membership to
                # prevent morphological noise from flooding the frontier.
                for h2 in morphological.get(head, []):
                    if h2 in function_words or h2 in visited:
                        continue
                    gate = word_gate(h2, pool, records, function_words)
                    if not gate:
                        continue
                    record_edge(head, h2, "morphological", layer)
                    visited.add(h2)
                    next_frontier.append(h2)
                    new_this_layer.append(h2)
                    trace["activations"][h2] = {
                        "layer": layer, "via": "morphological", "parent": head,
                        "bridge_words": sorted(gate),
                    }

            # relevance-scored cap: keep nodes closest to input core.
            # tiebreak alphabetical — deterministic.
            deduped = sorted(set(next_frontier))   # alphabetical tiebreak
            if len(deduped) > frontier_cap:
                deduped.sort(
                    key=lambda h: _frontier_score(h, records, core, pool),
                    reverse=True,
                )
            next_frontier = deduped[:frontier_cap]

            elapsed = time.time() - start
            per = elapsed / layer
            eta = per * (max_layers - layer)
            trace["layers"].append({
                "layer": layer,
                "new_headwords_count": len(new_this_layer),
                "frontier_carried": len(next_frontier),
                "sample": sorted(new_this_layer)[:12],
            })
            sys.stdout.write(
                f"\r[discover] layer {layer}/{max_layers}  "
                f"visited {len(visited):>6,}  new {len(new_this_layer):>6,}  "
                f"elapsed {elapsed:5.1f}s eta {eta:5.1f}s"
            )
            sys.stdout.flush()

            Path(checkpoint_path).write_text(json.dumps(trace))

            if not new_this_layer:
                trace["stopped"] = "saturation"
                break
            frontier = next_frontier
        else:
            trace["stopped"] = trace.get("stopped", "layer limit")
    finally:
        signal.signal(signal.SIGINT, prev)
        print()

    trace["visited_total"] = len(visited)
    trace["elapsed_seconds"] = round(time.time() - start, 2)
    Path(checkpoint_path).write_text(json.dumps(trace))
    return trace


def chain_to(target, trace):
    """Audit: walk parents from target back to the input. The path IS the
    explanation."""
    if target not in trace["activations"]:
        return None
    path, cur = [], target
    while cur is not None:
        a = trace["activations"][cur]
        path.append((cur, a["via"], a["layer"]))
        cur = a.get("parent")
    return list(reversed(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="+")
    ap.add_argument("--substrate", default="substrate.min.bin")
    ap.add_argument("--intent", default="")
    ap.add_argument("--goal", default="")
    ap.add_argument("--setting", default="")
    ap.add_argument("--subject", default="")
    ap.add_argument("--audience", default="")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=0, help="0 = no time limit")
    ap.add_argument("--frontier-cap", type=int, default=400)
    ap.add_argument("--trace-out", default="discovery_trace.json")
    ap.add_argument("--target", action="append", default=[],
                    help="headword(s) to report chains for")
    args = ap.parse_args()
    input_text = " ".join(args.text)
    context_texts = [v for v in (args.intent, args.goal, args.setting,
                                 args.subject, args.audience) if v]

    sub = load_substrate(args.substrate)
    m = sub["meta"]
    print(f"substrate: {m['headwords']:,} listings, {m['senses']:,} senses, "
          f"{m['oppositions']:,} opposition links, {m['crossrefs']:,} crossref links")

    trace = discover(input_text, context_texts, sub, args.layers,
                     args.seconds, args.frontier_cap, args.trace_out)

    print(f"\nstopped: {trace['stopped']}   headwords visited: "
          f"{trace['visited_total']:,}   elapsed: {trace['elapsed_seconds']}s")
    print(f"trace written: {args.trace_out}")

    for tgt in args.target:
        path = chain_to(tgt, trace)
        if path is None:
            print(f"\n'{tgt}' was NOT reached.")
            continue
        print(f"\nCHAIN to '{tgt}' (the path is the explanation):")
        for head, via, layer in path:
            print(f"  L{layer}  {head:<14} via {via}")
        # demonstrate compiled transformation capacity on the terminal sense
        rec = sub["records"][tgt]
        base = rec["senses"][0]["definition"].split(".")[0].strip()
        stmt = f"{tgt} is {base[0].lower() + base[1:]}"
        print(f"\n  transform demo (content invariant, scaffolds composed):")
        print(f"    {transform(stmt, 'explanation', 'cause_effect', 'literal')}")
        print(f"    {transform(stmt, 'instruction', 'condition_consequence', 'understatement')}")


if __name__ == "__main__":
    main()
