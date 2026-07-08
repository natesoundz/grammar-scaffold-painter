import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discover import load_substrate, discover

sub = load_substrate(os.path.join(os.path.dirname(os.path.abspath(__file__)), "substrate.min.bin"))

ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fanout_ck.json")

print("Single-word fan-out (5 layers, cap=400):")
for w in ["how", "bake", "bread", "do", "i", "example", "provide", "assertion"]:
    trace = discover(w, [], sub, 5, 0, 400, ckpt)
    n = len(trace["activations"])
    print(f"  {w:<14}  {n:>6} headwords")

print()
ctx = "example provide repeat reiterate rehearse recapitulate say tell invariant variant"
trace_ctx = discover("", [ctx], sub, 5, 0, 400, ckpt)
print(f"Context-only (10 words, no input):  {len(trace_ctx['activations']):>6} headwords")

trace_c = discover("How do I bake bread", [ctx], sub, 5, 0, 400, ckpt)
print(f"Combined input + context:           {len(trace_c['activations']):>6} headwords")

trace_bare = discover("How do I bake bread", [], sub, 5, 0, 400, ckpt)
print(f"Input alone (no context):           {len(trace_bare['activations']):>6} headwords")
