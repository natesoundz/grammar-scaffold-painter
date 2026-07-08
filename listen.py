"""
listen.py — Compiled Listener Architecture. Entry point.

    python listen.py "your sentence." --intent "..." --setting "..." --layers 6 --select 8

Analysis and generation as reversible traversals of the same substrate.

  Analysis:   Text  →  Listener State   (discover outward through substrate)
  Generation: Listener State  →  Text   (reverse traversal inward through closure)

The same substrate. The same structure. Opposite directions.
"""

import argparse

from discover import load_substrate
from analyzer import analyze
from generator import generate


def main():
    ap = argparse.ArgumentParser(
        description="Compiled Listener Architecture — Analysis and Generation"
    )
    ap.add_argument("text", nargs="*",
                    default=["The bearing failed because lubrication was inadequate."],
                    help="Input sentence (optional — defaults to bearing failure example)")
    ap.add_argument("--substrate", default="substrate.min.bin")
    ap.add_argument("--intent",    default="", help="Speaker's intent")
    ap.add_argument("--goal",      default="", help="Speaker's goal")
    ap.add_argument("--setting",   default="", help="Domain / setting")
    ap.add_argument("--audience",  default="", help="Intended audience")
    ap.add_argument("--layers",    type=int, default=6)
    ap.add_argument("--frontier-cap", type=int, default=400)
    ap.add_argument("--select",    type=int, default=6,
                    help="How many headwords to realize in output")
    ap.add_argument("--join",      default="",
                    help="Override join pattern (plain / cause_effect / sequence / "
                         "condition_consequence / contrast / elaboration)")
    args = ap.parse_args()

    input_text = " ".join(args.text)
    context_texts = [v for v in
                     (args.intent, args.goal, args.setting, args.audience) if v]

    print(f"\nINPUT    {input_text}")
    if context_texts:
        print(f"CONTEXT  {context_texts}")

    # ---------------------------------------------------------------- load substrate once
    sub = load_substrate(args.substrate)
    m = sub["meta"]
    print(f"SUBSTRATE  {m['headwords']:,} headwords  {m['senses']:,} senses  "
          f"{m['oppositions']:,} opposition links  {m['crossrefs']:,} crossref links\n")

    # ---------------------------------------------------------------- ANALYSIS
    print("[ ANALYSIS — outward traversal through substrate ]")
    state = analyze(
        input_text=input_text,
        context_texts=context_texts,
        sub=sub,
        max_layers=args.layers,
        frontier_cap=args.frontier_cap,
        intent=args.intent,
        goal=args.goal,
        setting=args.setting,
        audience=args.audience,
    )
    print()
    state.display()

    # ---------------------------------------------------------------- GENERATION
    print("\n[ GENERATION — reverse traversal through closure ]")
    print()
    print("  Reverse traversal partitions the closure by which members'")
    print("  definitions reference back to the input or context concepts.")
    print("  All decisions are binary membership tests, not scores.")
    print()

    result = generate(
        state=state,
        sub=sub,
        n=args.select,
        join_name=args.join or None,
    )

    ts = result["tier_sizes"]
    print(f"  CLOSURE PARTITION  ({len(state.closure):,} total)")
    print(f"    Tier A  definitions reference input concepts    {ts['tier_a']:>5}")
    print(f"    Tier B  definitions reference context concepts  {ts['tier_b']:>5}")
    print(f"    Tier C  no reference back to input or context  {ts['tier_c']:>5}")
    print()

    shown = result["selected"]
    print(f"  SELECTED  ({len(shown)} of {len(state.closure):,} — "
          f"Tier A first, then B, then C; alphabetical within tier)")
    for head, tier, layer, via in shown:
        print(f"    {head:<18}  Tier {tier}  L{layer}  via {via}")

    print(f"\n  JOIN     {result['join_used']}")
    print(f"\nOUTPUT\n{result['output']}\n")

    # ---- Scaffold analysis
    if result.get("scaffold"):
        scaf = result["scaffold"]
        from scaffold import ScaffoldAnalyzer as _SA
        _sa = _SA()
        print(f"[SCAFFOLD] join={scaf['join_type']} "
              f"mean_compat={scaf['mean_compatibility']:.2f} "
              f"compatible={scaf['compatible_count']}/{scaf['total']} "
              f"verdict={scaf['verdict']}")
        for h, info in scaf["per_term"].items():
            sig = _sa.scaffold_signature(info["scaffold"])
            print(f"  {h}: {sig} compat={info['compatibility']:.2f}")
        print()

    # ---- Algebra
    if result.get("algebra"):
        alg = result["algebra"]
        if alg.get("distilled"):
            print("[DISTILL] input-referencing headwords:")
            for h, refs in alg["distilled"].items():
                print(f"  {h}: <- {', '.join(refs)}")
        if alg.get("expansion"):
            print("[EXPAND] reachable from selected (one hop, in closure):")
            for rtype, heads in alg["expansion"].items():
                print(f"  {rtype}: {', '.join(heads)}")
        if alg.get("emphasis"):
            print("[EMPHASIZE] active relations per headword:")
            for h, fired in alg["emphasis"].items():
                parts = [f"{r}({n})" for r, n, _ in fired]
                print(f"  {h}: {', '.join(parts)}")
        print()

    # ---- Constraints
    if result.get("constraints"):
        cons = result["constraints"]
        print("[CONSTRAINTS] per-term satisfaction:")
        for h, c in cons.items():
            inp = ', '.join(c["input_overlap"]) or "-"
            ctx = ', '.join(c["context_overlap"]) or "-"
            dom = c["domain"] or "-"
            dom_ok = "Y" if c["domain_allowed"] else "N"
            print(f"  {h}: tier={c['tier']} L{c['layer']} via={c['via']} "
                  f"inp=[{inp}] ctx=[{ctx}] dom={dom}{dom_ok} score={c['anchor_score']:.1f}")
        print()


if __name__ == "__main__":
    main()
