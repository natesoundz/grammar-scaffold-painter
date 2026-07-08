"""
listen_instrumented.py — Batch artifact collection for transformation compilation.

    python listen_instrumented.py "sentence" --artifacts artifacts.jsonl

Runs the full analysis + generation pipeline and appends one artifact record
to the JSONL file. Designed to be called in a loop by collect_artifacts.ps1.

Artifact schema:
    {
      "input":    "sentence",
      "context":  [],
      "state": {
        "speech_act":    "question",
        "response_type": "instruction",
        "closure_size":  45,
        "allowed_domains": ["mach."],
        "input_field":   ["fix", "broken", "engine"],
        "context_field": []
      },
      "output":         "...",
      "join_used":      "cause_effect",
      "tier_sizes":     {"tier_a": 30, "tier_b": 5, "tier_c": 10},
      "selected_headwords": ["engine", "broken", ...]
    }
"""

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import load_substrate
from analyzer import analyze
from generator import generate

_sub_cache = {}


def _load_sub(path):
    if path not in _sub_cache:
        _sub_cache[path] = load_substrate(path)
    return _sub_cache[path]


def collect(input_text, context_texts, sub,
            max_layers=6, frontier_cap=400,
            intent="", goal="", setting="", audience="",
            select=6):

    state = analyze(
        input_text=input_text,
        context_texts=context_texts,
        sub=sub,
        max_layers=max_layers,
        frontier_cap=frontier_cap,
        intent=intent,
        goal=goal,
        setting=setting,
        audience=audience,
        checkpoint="instrumented_checkpoint.json",
    )

    result = generate(state=state, sub=sub, n=select)

    artifact = {
        "input": input_text,
        "context": context_texts,
        "state": {
            "speech_act":     state.frame.speech_act,
            "response_type":  state.frame.response_type,
            "closure_size":   len(state.closure),
            "allowed_domains": sorted(state.allowed_domains),
            "input_field":    sorted(state.fields.input_field),
            "context_field":  sorted(state.fields.context_field),
            "visited_total":  state.trace["visited_total"],
        },
        "output":           result["output"],
        "join_used":        result["join_used"],
        "tier_sizes":       result["tier_sizes"],
        "selected_headwords": [h for h, _t, _l, _v in result["selected"]],
    }

    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*",
                     default=["The bearing failed because lubrication was inadequate."])
    ap.add_argument("--substrate", default=os.path.join(_HERE, "substrate.min.bin"))
    ap.add_argument("--intent",       default="")
    ap.add_argument("--goal",         default="")
    ap.add_argument("--setting",      default="")
    ap.add_argument("--audience",     default="")
    ap.add_argument("--layers",       type=int, default=6)
    ap.add_argument("--frontier-cap", type=int, default=400)
    ap.add_argument("--select",       type=int, default=6)
    ap.add_argument("--artifacts",    default=os.path.join(_HERE, "artifacts.jsonl"),
                    help="JSONL file to append artifact to")
    ap.add_argument("--quiet",        action="store_true",
                    help="Suppress traversal progress output")
    args = ap.parse_args()

    input_text = " ".join(args.text)
    context_texts = [v for v in (args.intent, args.goal,
                                 args.setting, args.audience) if v]

    if args.quiet:
        import io
        sys.stdout = io.StringIO()

    sub = _load_sub(args.substrate)

    try:
        artifact = collect(
            input_text=input_text,
            context_texts=context_texts,
            sub=sub,
            max_layers=args.layers,
            frontier_cap=args.frontier_cap,
            intent=args.intent,
            goal=args.goal,
            setting=args.setting,
            audience=args.audience,
            select=args.select,
        )
    finally:
        if args.quiet:
            sys.stdout = sys.__stdout__

    with open(args.artifacts, "a", encoding="utf-8") as f:
        f.write(json.dumps(artifact, ensure_ascii=False) + "\n")

    print(f"[artifact] {artifact['state']['speech_act']:12} "
          f"closure={artifact['state']['closure_size']:3}  "
          f"{input_text}")


if __name__ == "__main__":
    main()
