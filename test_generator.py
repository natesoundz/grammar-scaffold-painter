"""
test_generator.py — Sprint 3 test suite.

    python test_generator.py

Runs the full analysis + generation pipeline on a fixed set of test cases
and reports speech act classification, rule matching, and output.
"""

from analyzer import analyze
from generator import generate
from discover import load_substrate

sub = load_substrate("substrate.min.bin")

TEST_CASES = [
    ("How do I fix a broken engine?",                  "question",     "instruction"),
    ("The mechanic repaired the broken engine.",        "assertion",    "explanation"),
    ("Engines require regular maintenance.",            "assertion",    "explanation"),
    ("What happens if the bolt fails?",                 "question",     "explanation"),
    ("Tighten the bolt clockwise until snug.",          "instruction",  "assertion"),
    ("The shaft cracked because of cyclic fatigue.",    "explanation",  "explanation"),
    ("Why does the gasket fail?",                       "question",     "explanation"),
    ("First remove the bolt, then clean the threads.",  "narrative",    "narrative"),
    ("Replace the seal if the lip is worn.",            "instruction",  "assertion"),
    ("Metal fatigue occurs under repeated loading.",    "assertion",    "explanation"),
]

print("=" * 70)
print("GENERATOR TEST SUITE")
print("=" * 70)

results = []
for input_text, expected_act, expected_resp in TEST_CASES:
    print(f"\nInput: {input_text}")

    state = analyze(
        input_text=input_text,
        context_texts=[],
        sub=sub,
        max_layers=5,
        frontier_cap=400,
        checkpoint="test_checkpoint.json",
    )

    result = generate(state, sub, n=6)

    act_ok  = state.frame.speech_act    == expected_act
    resp_ok = state.frame.response_type == expected_resp

    print(f"  Speech act:      {state.frame.speech_act:<14}"
          f"  {'OK' if act_ok else 'MISMATCH (expected ' + expected_act + ')'}")
    print(f"  Response type:   {state.frame.response_type:<14}"
          f"  {'OK' if resp_ok else 'MISMATCH (expected ' + expected_resp + ')'}")
    print(f"  Closure:         {len(state.closure)}")
    print(f"  Rule matched:    {result['rule_matched']}"
          f"  confidence={result['rule_confidence']:.3f}")
    print(f"  Join:            {result['join_used']}")
    print(f"  Output:\n{result['output']}")

    results.append({
        "input":    input_text,
        "act_ok":   act_ok,
        "resp_ok":  resp_ok,
        "matched":  result["rule_matched"],
        "conf":     result["rule_confidence"],
    })

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
n         = len(results)
act_ok    = sum(1 for r in results if r["act_ok"])
resp_ok   = sum(1 for r in results if r["resp_ok"])
matched   = sum(1 for r in results if r["matched"])
avg_conf  = sum(r["conf"] for r in results) / n

print(f"Speech act correct:   {act_ok}/{n}")
print(f"Response type correct:{resp_ok}/{n}")
print(f"Rules matched:        {matched}/{n}")
print(f"Avg confidence:       {avg_conf:.3f}")

if matched == 0:
    print()
    print("No rules matched. This is expected before Sprint 2 is complete.")
    print("Run collect_artifacts.ps1 then transform_compiler.py to build rules.")
