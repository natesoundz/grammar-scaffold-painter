# The Compiled Listener — Dictionary-Substrate Semantics

A natural language understanding and generation system built entirely on
Webster's 1913 Dictionary (102,217 headwords, 147,456 senses), compiled
into a traversable graph substrate. No neural networks, no embeddings,
no learned weights, no floating-point scoring in the core pipeline.
Every decision in Layers 0–2 is a binary set-membership test.

Two layers of work live here:

1. **The Compiled Listener** — analysis (outward traversal) and generation
   (inward/reverse traversal) over the substrate graph, with a second-order
   rule compiler that learns rhetorical join + neighborhood policies from
   observed discourse classes. See `arxiv_draft.md` for the full paper.
2. **Grammar-Scaffold Painting** — a follow-on empirical investigation into
   whether the *function-word skeleton* of a definition (independent of its
   content words) carries semantic structure on its own, and what it's
   actually useful for. Four falsifiable claims, tested and reported below.

---

## Quick start

```powershell
python assemble.py "your sentence." --intent "..." --setting "..." --layers 6 --select 8
python listen.py "prove" --select 8
python painter.py "command authority" --across
python scaffold_transform.py "prohibit" --all
```

All scripts load `substrate.min.bin` (compiled Webster's 1913) at runtime.
Nothing here calls an external API or downloads anything.

---

## Part 1 — Compiled Listener architecture

### Files, in dependency order
- `substrate.min.bin` — compiled Webster's 1913 (records + opposition/crossref/
  function-word relations; no significance/idf table — the last surviving
  weighted-scoring mechanism was removed entirely).
- `dict_algebra.py` — formal primitives (`ACTIVATE`/`FILTER`/`INTERSECT`/
  `FOLLOW`/`EXPAND`/`CONSTRAIN`/`PROJECT`/`TRACE`) everything else composes from.
- `discover.py` — traversal engine (activation, bridge, sense admission,
  crossref/opposition following, full-parent-edge tracking, chain audit).
- `analyzer.py` / `listener_state.py` — outward traversal, produces a
  `ListenerState` (closure, discourse frame, activation trace).
- `generator.py` — reverse traversal: partitions the closure into tiers by
  whether definitions reference the input/context, selects by anchor rank,
  renders with a rhetorical join operator (`elaboration`, `cause_effect`,
  `sequence`, `condition_consequence`, `contrast`, `plain`).
- `assemble.py` — entry point combining both directions end-to-end.
- `transform_compiler.py` — second-order compiler: groups artifacts by
  `(speech_act, response_type, closure_bucket)`, compiles rules predicting
  join + neighborhood policy from entropy/bridge-stability thresholds.
- `rhetorical_cycle.py` — recursive discourse cycling (output → adjacent
  discourse frame → next input), with convergence reporting.

Only needed to recompile from source (not used at runtime):
- `compile_substrate.py` — the compiler.

### Governing principle
No weights, no scores, no counts compared against a cutoff, anywhere in
Layers 0–2. Every selection decision is either a fixed-category membership
test (domain tags), a named-edge lookup (crossref/opposition/definition),
or a shape requirement (discourse-adjacency, fixed before content is
touched). Full architecture, formal algebra, and 691-artifact behavioral
characterization: see `arxiv_draft.md`.

---

## Part 2 — Grammar-Scaffold Painting: four falsifiable claims

Every dictionary definition decomposes as `D(headword) = (content, scaffold)`:
content words (nouns, verbs, adjectives) and a scaffold — the function-word
skeleton (`a ___ of ___`, `to ___ by ___`) that encodes relational structure.
`scaffold.py` extracts this skeleton and maps function words to relational
roles (ARTICLE, COMPOSE, CONTAIN, CAUSE, DIRECT, CONJOIN, SPECIFY, IDENTIFY,
PURPOSE, ROLE, CONDITION, CONTRAST, NEGATE).

The question: does the scaffold alone carry semantic information, or is it
just a template that needs topology (graph reachability) to mean anything?

### Claim 1 — Scaffold similarity alone recovers semantic neighbors → **REJECTED**

Tool: `_scaffold_only.py`. Two probe groups (`fuel engine tool weapon
instrument` vs `justice beauty sorrow virtue truth`), scaffold-only
retrieval (no content words at all).

```
Within-group scaffold similarity:   0.280
Between-group scaffold similarity:  0.277
Separation:                         0.003
Top-5 scaffold-nearest-neighbor retrieval: 0/20 group members recovered
```

`fuel` → nearest scaffold neighbor is `resurrectionist`. `weapon` → nearest
is `anharmonic`. Grammar alone does not recover ontology — a skeleton like
`ART·CMP` ("a ___ of ___") is shared by ~20% of the entire dictionary and is
essentially domain-agnostic.

### Claim 2 — Topology alone produces coherent selection → **REJECTED**

Tool: `rhetorical_cycle.py`. Unfiltered reachability through definition/
opposition edges drifts across domains within a few hops, even though every
individual hop is locally valid.

```
python rhetorical_cycle.py "The bearing failed because lubrication was inadequate." --cycles 3
```
```
C1: bearing fail lubrication inadequate adequate reign
C2: deficient disproportionate insufficient inadequacy solomon peter
C3: sufficient proportion equal wanting enough wisdom
```

A machinery-failure input reaches biblical names (`solomon`, `peter`) within
two cycles via opposition and definition edges.

### Claim 3 — Grammar-sensitive selection emerges above a closure-size threshold → **SUPPORTED**

Tool: `_scaffold_test.py` (`--seed 42` for reproducible random baselines).
Selected-term scaffold compatibility vs. random-closure-sample compatibility,
swept across traversal depth:

```
Layers   |Closure|   Δ (selected − random)
  2         103        +0.007
  4         280        +0.079
  6         347        +0.095
  8         367        +0.090
```

Below ~250 reachable nodes, grammar signal is indistinguishable from noise.
Above ~300, it stabilizes around +0.08–0.10. Grammar becomes an active
selector only once topology supplies enough structurally diverse candidates
to select among.

**Counter-example** (`prove`, `|closure|=62`, below threshold): the active
join (`elaboration`) is *anti*-correlated with scaffold fit (Δ=−0.070),
while `sequence` (Δ=+0.139) and `condition_consequence` (Δ=+0.124) would
have fit better. Selection instead falls back to raw anchor rank:
`prove(−1118.9) → approve(−108.9) → disprove(−108.8) → refute(−97.6) →
confute(−93.9) → evincive(−79.2) → convict(−78.8) → experiment(−56.0)`.
Below threshold, join choice is externally forced and grammar has nothing
to filter.

### Claim 4 — Scaffold transforms preserve content while changing relational operators → **SUPPORTED**

Tool: `scaffold_transform.py` — 15 compiled function-word rewrites between
scaffold devices (taxonomic ↔ causal_chain, → contrast, → conditional, →
purposive, etc). Content tokens are held fixed; only the function words
change.

```
python scaffold_transform.py "prohibit" --all
```
```
Source (causal_chain, DIR·CAU·DIR): "To forbid by authority; to interdict"
→ conditional: "if forbid then authority; to interdict"
→ contrast:    "forbid despite authority; to interdict"
→ taxonomic:   "a forbid of authority; to interdict"
```

Content `{forbid, authority}` is preserved across all four framings; only
the relational operator changes (causation → implication → concession →
classification). **This is token preservation, not semantic equivalence** —
"forbid by authority" and "forbid despite authority" assert different
relations. Measuring which transforms preserve *meaning* is open future work.

`painter.py` demonstrates the corollary: painting the same context
(`command authority`, `|closure|=316`) across six structural templates
produces six different coherent renderings, with profundity computed as a
**multiplicative** intersection `role_fit × ctx_fit × proximity` (not
additive — an additive metric masks incomplete role coverage that the
multiplicative one correctly penalizes).

### The decomposition

```
M = (R, T, I)

R  Reachability      — topology; solves RETRIEVAL      (Claim 1: grammar fails)
T  Transformability  — scaffold operators; solves TRANSFORMATION
                        (Claim 2: topology fails; Claim 3: T activates ≥ ~250–300 nodes)
I  Instantiation     — lexical content filling scaffold slots
                        (Claim 4: preserved under T-rewrites)
```

**The graph retrieves. The grammar transforms. The content instantiates.**
Neither grammar nor topology alone is "meaning" — meaning is what happens
at their measured, multiplicative intersection, and that intersection has
a switch-on threshold you can find with a for-loop over traversal depth.

### Reproducibility

Every number above was independently regenerated from the exact commands
shown. Deterministic quantities (closure size, selected terms, anchor
scores, scaffold compatibility of specific terms, transform outputs,
profundity) reproduce bit-for-bit across runs. Stochastic quantities
(random-baseline means, percentiles from Monte Carlo trials) require the
fixed `--seed 42` in `_scaffold_test.py` — added specifically so these
results are reproducible rather than merely plausible.

### Grammar-scaffold files

| File | Purpose |
|---|---|
| `scaffold.py` | Function-word → relational-role extraction, compatibility scoring |
| `scaffold_transform.py` | 15 compiled device-to-device function-word rewrites |
| `painter.py` | `StructurePainter` — fills scaffold templates with context-constrained content |
| `_scaffold_lab.py` | Language-device census and classification across the dictionary |
| `_scaffold_test.py` | Claim 3 — selected vs. random scaffold-compatibility test (seeded) |
| `_scaffold_fill.py` | Finds closure members that fill missing scaffold roles |
| `_scaffold_only.py` | Claim 1 — the scaffold-only retrieval falsification experiment |

---

## License

Dual-licensed: **free for noncommercial use** (personal projects, research,
education, hobby use — see `LICENSE`, PolyForm Noncommercial 1.0.0). Any
**commercial use** — selling this software, offering it as a paid service,
or incorporating it into a commercial product — requires a separate
commercial license. Contact [natesoundz](https://github.com/natesoundz)
to purchase one.

---

## Known open problems (stated honestly)

- Structural closure for a rich input can be large (1,600+ members on one
  tested case) and mostly undifferentiated by any zero-arithmetic ordering
  available — alphabetical-first is genuinely free of scoring but also
  genuinely uninformative about topical relevance.
- The ~250–300 node grammar-activation threshold (Claim 3) is observed on
  four sweep points on one substrate; it is a reported regime shift, not a
  precisely located critical point, and may be substrate-specific.
- Semantic equivalence under scaffold transformation (Claim 4) is
  explicitly not established — only token preservation and operator change
  are verified. Whether any transform preserves *meaning* is open.
