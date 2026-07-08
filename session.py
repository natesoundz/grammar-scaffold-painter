"""
session.py — Session manager for the Compiled Listener.

A Session holds a substrate reference (loaded once), maintains a turn
history, and threads the most recent outputs as context into each new
analysis. Compound questions are segmented and processed independently,
with each segment's output feeding into the next as live context.

    from session import Session
    from discover import load_substrate

    sub = load_substrate("substrate.min.bin")
    s   = Session(sub)

    print(s.ask("What is a bearing?"))
    print(s.ask("Why does it fail under cyclic load?"))
    print(s.ask("How do I inspect and replace it?"))

The third query automatically carries the first two outputs as context,
so the closure of "inspect" now includes bearing-specific vocabulary
from the previous turns.

Architecture
------------
  Turn = (input_segment, ListenerState, generate_result)
  Session.turns = list[Turn]

  ask(text):
    1. Segment text into sub-questions (compound question splitting)
    2. For each segment:
       a. Build context from last context_depth turn outputs
       b. analyze() → ListenerState
       c. generate() → result
       d. Append turn; append output to live context for next segment
    3. Return joined output string

Isolation
---------
  Each Session gets a unique checkpoint path derived from session_id so
  concurrent sessions never share or corrupt a checkpoint file.
"""

import os
import re
import uuid

_HERE = os.path.dirname(os.path.abspath(__file__))

from discover import load_substrate
from analyzer import analyze
from generator import generate


# ---------------------------------------------------------------- compound question splitter

_COMPOUND_SEP = re.compile(
    r"""
    (?<=[?])          # after a question mark
    \s+               # whitespace
    (?=[A-Z])         # followed by capital (new sentence)
    |
    (?<=\w)           # after a word character
    \s*;\s*           # semicolon separator
    """,
    re.VERBOSE,
)

_SUB_QUESTION_RE = re.compile(
    r"\b(what|why|how|when|where|which|who|does|is|are|can|should|will)\b",
    re.I,
)


def _segment(text: str) -> list:
    """Split compound questions into processable segments.

    Splits on:
      - '? ' followed by a capital letter (two separate questions)
      - '; ' between clauses that each contain a question word
      - ' and ' joining two question-word clauses

    Returns a list with at least one element (the original text if no
    split applies).
    """
    text = text.strip()

    # Split on '? ' + capital — unambiguous sentence boundary
    parts = _COMPOUND_SEP.split(text)
    if len(parts) > 1:
        return [p.strip() for p in parts if p.strip()]

    # Split 'X and Y' where both halves contain a question word
    and_split = re.split(r"\s+and\s+", text, maxsplit=1, flags=re.I)
    if (len(and_split) == 2
            and _SUB_QUESTION_RE.search(and_split[0])
            and _SUB_QUESTION_RE.search(and_split[1])):
        return [p.strip() for p in and_split if p.strip()]

    return [text]


# ---------------------------------------------------------------- turn record

class Turn:
    __slots__ = ("input", "state", "result")

    def __init__(self, input_text, state, result):
        self.input  = input_text
        self.state  = state
        self.result = result

    @property
    def output(self):
        return self.result["output"]

    @property
    def join_used(self):
        return self.result["join_used"]

    @property
    def neighbor_mode(self):
        return self.result.get("neighbor_mode", "primary_only")

    def __repr__(self):
        return (f"Turn(input={self.input!r:.40}, "
                f"closure={len(self.state.closure)}, "
                f"join={self.join_used}, mode={self.neighbor_mode})")


# ---------------------------------------------------------------- session

class Session:
    """Multi-turn session with automatic context threading."""

    def __init__(self, sub, session_id=None, context_depth=3,
                 max_layers=6, frontier_cap=400, n=6):
        self.sub           = sub
        self.id            = session_id or uuid.uuid4().hex[:8]
        self.context_depth = context_depth
        self.max_layers    = max_layers
        self.frontier_cap  = frontier_cap
        self.n             = n
        self.turns: list   = []

        # Each session uses its own checkpoint to avoid concurrency collisions
        self._ckpt = os.path.join(_HERE, f"_session_{self.id}.ckpt.json")

    # ---- public interface

    def ask(self, text: str, n: int = None, join_name: str = None,
            intent: str = "", goal: str = "", setting: str = "",
            audience: str = "") -> str:
        """Process one or more questions; return the assembled output."""
        n = n or self.n
        segments = _segment(text)

        # Seed context from selected headwords only — never from realized output text.
        # Using output text feeds definition bodies and connectives back as new input,
        # creating a compounding noise loop that drowns the actual user input by turn 4.
        live_context = [
            " ".join(h for h, _tier, _layer, _via in t.result["selected"])
            for t in self.turns[-self.context_depth:]
        ]

        outputs = []
        for seg in segments:
            state = analyze(
                input_text=seg,
                context_texts=live_context,
                sub=self.sub,
                max_layers=self.max_layers,
                frontier_cap=self.frontier_cap,
                intent=intent,
                goal=goal,
                setting=setting,
                audience=audience,
                checkpoint=self._ckpt,
            )
            result = generate(
                state=state,
                sub=self.sub,
                n=n,
                join_name=join_name,
            )
            turn = Turn(seg, state, result)
            self.turns.append(turn)
            # Thread headwords forward, not the rendered sentence.
            live_context.append(
                " ".join(h for h, _tier, _layer, _via in result["selected"])
            )
            outputs.append(result["output"])

        # Collect algebra from last segment's result
        last_result = self.turns[-1].result if self.turns else None
        return " ".join(outputs), last_result

    # ---- introspection

    def last(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def history(self) -> list:
        return [
            {
                "turn":          i + 1,
                "input":         t.input,
                "closure_size":  len(t.state.closure),
                "join":          t.join_used,
                "neighbor_mode": t.neighbor_mode,
                "output":        t.output,
            }
            for i, t in enumerate(self.turns)
        ]

    def display_history(self):
        sep = "=" * 70
        print(sep)
        print(f"SESSION  {self.id}  ({len(self.turns)} turns)")
        print(sep)
        for h in self.history():
            print(f"\n[Turn {h['turn']}]  {h['input']}")
            print(f"  closure={h['closure_size']}  "
                  f"join={h['join']}  mode={h['neighbor_mode']}")
            print(f"  {h['output']}")
        print()

    def __repr__(self):
        return f"Session(id={self.id!r}, turns={len(self.turns)})"


# ---------------------------------------------------------------- CLI convenience

def _load_sub_cached(path):
    return load_substrate(path)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compiled Listener — Interactive Session")
    ap.add_argument("--substrate",     default=os.path.join(_HERE, "substrate.min.bin"))
    ap.add_argument("--context-depth", type=int, default=3)
    ap.add_argument("--layers",        type=int, default=6)
    ap.add_argument("--frontier-cap",  type=int, default=400)
    ap.add_argument("--select",        type=int, default=6)
    ap.add_argument("--setting",       default="", help="Domain / setting")
    args = ap.parse_args()

    print("Loading substrate ...", end=" ", flush=True)
    sub = _load_sub_cached(args.substrate)
    m = sub["meta"]
    print(f"{m['headwords']:,} headwords  {m['senses']:,} senses\n")

    session = Session(
        sub,
        context_depth=args.context_depth,
        max_layers=args.layers,
        frontier_cap=args.frontier_cap,
        n=args.select,
    )

    print("Type your questions. Empty line to exit.\n")
    while True:
        try:
            text = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        output, last_result = session.ask(text, setting=args.setting)
        t = session.last()
        print(f"[closure={len(t.state.closure)} "
              f"join={t.join_used} "
              f"mode={t.neighbor_mode}]")
        print(output)

        # ---- Algebraic post-processing display
        if last_result and last_result.get("algebra"):
            alg = last_result["algebra"]
            if alg.get("distilled"):
                print("\n  [distill] input-referencing headwords:")
                for h, refs in alg["distilled"].items():
                    print(f"    {h}: ← {', '.join(refs)}")
            if alg.get("expansion"):
                print("\n  [expand] reachable from selected (one hop, in closure):")
                for rtype, heads in alg["expansion"].items():
                    print(f"    {rtype}: {', '.join(heads)}")
            if alg.get("emphasis"):
                print("\n  [emphasize] active relations per headword:")
                for h, fired in alg["emphasis"].items():
                    parts = [f"{r}({n})" for r, n, _ in fired]
                    print(f"    {h}: {', '.join(parts)}")

        # ---- Per-term constraint satisfaction
        if last_result and last_result.get("constraints"):
            cons = last_result["constraints"]
            print("\n  [constraints] per-term satisfaction:")
            for h, c in cons.items():
                inp = ', '.join(c["input_overlap"]) or "—"
                ctx = ', '.join(c["context_overlap"]) or "—"
                dom = c["domain"] or "—"
                dom_ok = "✓" if c["domain_allowed"] else "✗"
                print(f"    {h}: tier={c['tier']} L{c['layer']} via={c['via']} "
                      f"inp=[{inp}] ctx=[{ctx}] dom={dom}{dom_ok} score={c['anchor_score']:.1f}")

        # ---- Scaffold analysis: grammatical frame compatibility
        if last_result and last_result.get("scaffold"):
            scaf = last_result["scaffold"]
            from scaffold import ScaffoldAnalyzer as _SA
            _sa = _SA()
            print(f"\n  [scaffold] join={scaf['join_type']} "
                  f"mean_compat={scaf['mean_compatibility']:.2f} "
                  f"compatible={scaf['compatible_count']}/{scaf['total']} "
                  f"verdict={scaf['verdict']}")
            for h, info in scaf["per_term"].items():
                sig = _sa.scaffold_signature(info["scaffold"])
                print(f"    {h}: {sig} compat={info['compatibility']:.2f}")
        print()

    session.display_history()


if __name__ == "__main__":
    main()
