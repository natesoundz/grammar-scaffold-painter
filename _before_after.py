import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from discover import load_substrate
from analyzer import analyze
from generator import generate

_HERE = os.path.dirname(os.path.abspath(__file__))
sub = load_substrate(os.path.join(_HERE, "substrate.min.bin"))

QUERIES = [
    "How do I repair a seized bearing?",
    "The bearing failed because lubrication was inadequate.",
    "Tighten the bolt clockwise until snug.",
    "Why does metal fatigue under cyclic loading?",
]

label = sys.argv[1] if len(sys.argv) > 1 else "RUN"

print(f"{label} — selector state")
print("=" * 70)
for q in QUERIES:
    state  = analyze(q, [], sub, max_layers=5, frontier_cap=400,
                     checkpoint="_ba_chk.json")
    result = generate(state, sub, n=6)
    heads  = [h for h, t, l, v in result["selected"]]
    ta     = result["tier_sizes"]["tier_a"]
    print(f"\nInput   : {q}")
    print(f"Closure : {len(state.closure)}  TierA={ta}")
    print(f"Selected: {heads}")
    print(f"Output  : {result['output']}")
