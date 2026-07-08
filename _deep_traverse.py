"""
_deep_traverse.py — What happens when we let the traversal run deep?

Runs the same input at progressively deeper layer counts, reporting
closure size, time, and output at each depth.
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import load_substrate, discover
from analyzer import analyze
from generator import generate
from listener_state import ListenerState

sub = load_substrate(os.path.join(_HERE, "substrate.min.bin"))

INPUT = "Why does metal fatigue under cyclic loading?"
CKPT  = os.path.join(_HERE, "_deep_ck.json")

print(f"INPUT: {INPUT}\n")
print(f"  {'LAYERS':<8}  {'FRONTIER_CAP':<14}  {'CLOSURE':>8}  {'TIME':>7}  OUTPUT")
print("  " + "-" * 110)

configs = [
    (4,  200),
    (5,  400),
    (6,  400),
    (8,  600),
    (10, 800),
    (12, 1200),
]

for layers, fcap in configs:
    t0 = time.perf_counter()
    state = analyze(
        input_text=INPUT,
        context_texts=[],
        sub=sub,
        max_layers=layers,
        frontier_cap=fcap,
        checkpoint=CKPT,
    )
    result = generate(state=state, sub=sub, n=6)
    elapsed = time.perf_counter() - t0

    out = result["output"]
    out_short = out[:90] + "…" if len(out) > 90 else out
    print(f"  {layers:<8}  {fcap:<14}  {len(state.closure):>8}  {elapsed:>6.2f}s  {out_short}")

print()
print("Full output at deepest config:")
print(result["output"])
