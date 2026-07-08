"""
multi_join_collect.py — Sprint 4: competing join strategies.

For each input sentence, runs generate() with ALL 6 joins and records the
bridge_stability per join. This gives the compiler non-tautological training
signal so it can learn WHICH join maximises bridge retention per class.

    python multi_join_collect.py
    python multi_join_collect.py --artifacts artifacts.jsonl --out multi_join.jsonl --max 200
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import load_substrate
from analyzer import analyze
from generator import generate
from transform_compiler import closure_bucket

ALL_JOINS = [
    "cause_effect",
    "elaboration",
    "sequence",
    "condition_consequence",
    "contrast",
    "plain",
]


def bridge_stability(input_field: set, output: str) -> float:
    if not input_field:
        return 0.0
    low = output.lower()
    hits = sum(1 for w in input_field if w in low)
    return round(hits / len(input_field), 4)


def collect_multi_join(artifacts_path, sub, out_path,
                       max_inputs=None, layers=5, frontier_cap=400, select=6):

    with open(artifacts_path, encoding="utf-8-sig") as f:
        artifacts = [json.loads(l) for l in f if l.strip()]

    if max_inputs:
        artifacts = artifacts[:max_inputs]

    seen_inputs = set()
    records_written = 0

    with open(out_path, "w", encoding="utf-8") as out:
        for i, art in enumerate(artifacts):
            inp = art["input"]
            if inp in seen_inputs:
                continue
            seen_inputs.add(inp)

            ctx = art.get("context", [])

            state = analyze(inp, ctx, sub,
                            max_layers=layers,
                            frontier_cap=frontier_cap,
                            checkpoint=f"_mj_{i}.json")

            input_field = state.fields.input_field
            sa     = state.frame.speech_act
            rt     = state.frame.response_type
            bucket = closure_bucket(len(state.closure))

            best_join = None
            best_bs   = -1.0
            join_scores = {}

            for join in ALL_JOINS:
                result = generate(state, sub, n=select, join_name=join)
                bs = bridge_stability(input_field, result["output"])
                join_scores[join] = bs
                if bs > best_bs:
                    best_bs   = bs
                    best_join = join

            record = {
                "input":          inp,
                "speech_act":     sa,
                "response_type":  rt,
                "closure_bucket": bucket,
                "closure_size":   len(state.closure),
                "input_field":    sorted(input_field),
                "join_scores":    join_scores,
                "best_join":      best_join,
                "best_bs":        best_bs,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            records_written += 1

            print(f"[{records_written:4}] {sa:<14} cl={len(state.closure):4}  "
                  f"best={best_join:<22} bs={best_bs:.3f}  {inp[:60]}")

    return records_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate",  default=os.path.join(_HERE, "substrate.min.bin"))
    ap.add_argument("--artifacts",  default=os.path.join(_HERE, "artifacts.jsonl"))
    ap.add_argument("--out",        default=os.path.join(_HERE, "multi_join.jsonl"))
    ap.add_argument("--max",        type=int, default=None,
                    help="Max unique inputs to process (default: all)")
    ap.add_argument("--layers",     type=int, default=5)
    ap.add_argument("--select",     type=int, default=6)
    args = ap.parse_args()

    print("=" * 70)
    print("MULTI-JOIN COLLECTOR  (Sprint 4)")
    print("=" * 70)
    print(f"Artifacts : {args.artifacts}")
    print(f"Output    : {args.out}")
    print(f"Joins     : {ALL_JOINS}")
    print()

    sub = load_substrate(args.substrate)
    n = collect_multi_join(
        artifacts_path=args.artifacts,
        sub=sub,
        out_path=args.out,
        max_inputs=args.max,
        layers=args.layers,
        select=args.select,
    )

    print(f"\nDone. {n} unique inputs processed → {args.out}")


if __name__ == "__main__":
    main()
