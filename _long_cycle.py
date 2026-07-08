"""
_long_cycle.py — Full dictionary traversal via rhetorical cycling.

Runs cycles until the entire dictionary is covered or saturation is reached.
Uses explore mode (all-time exclusion) so every cycle forces fresh headwords.
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import load_substrate
from rhetorical_cycle import run_cycle

sub = load_substrate(os.path.join(_HERE, "substrate.min.bin"))
total_headwords = sub["meta"]["headwords"]

INPUT = "The bearing failed because lubrication was inadequate."

# explore mode = all-time exclusion, so every cycle must find new headwords
# stop_on_saturation: if 15 consecutive cycles find 0 new words, the dictionary
# is exhausted from this anchor
N = 100000
SATURATION_PATIENCE = 15

print(f"Dictionary: {total_headwords:,} headwords")
print(f"Mode: explore (all-time exclusion, no rolling)")
print(f"Saturation patience: {SATURATION_PATIENCE} consecutive empty cycles\n")

t0 = time.perf_counter()
history = run_cycle(
    INPUT, sub,
    n_cycles=N,
    verbose=False,
    layers=5,
    select=6,
    cycle_mode="explore",
    stop_on_saturation=True,
    saturation_patience=SATURATION_PATIENCE,
)
elapsed = time.perf_counter() - t0

actual_cycles = len(history)

print(f"{'CY':>5}  {'SA':<14}  {'CLOSURE':>7}  {'SELECTED':<50}  {'NEW':>4}  {'COV%':>5}")
print("-" * 92)

all_ever = set()
for rec in history:
    sel = rec["selected"]
    new = len(set(sel) - all_ever)
    all_ever.update(sel)
    cov = len(all_ever) / total_headwords * 100
    print(f"  {rec['cycle']:>3}  {rec['speech_act']:<14}  {rec['closure_size']:>7}  "
          f"{' '.join(sel):<50}  {new:>4}  {cov:>4.1f}%")

print(f"\n{'='*92}")
print(f"Cycles run:           {actual_cycles:,}")
print(f"Total unique headwords: {len(all_ever):,} / {total_headwords:,} "
      f"({len(all_ever)/total_headwords*100:.1f}%)")
print(f"Elapsed:              {elapsed:.1f}s  ({elapsed/actual_cycles:.3f}s/cycle)")
print(f"Stopped by:           {history[-1].get('cycle_mode', 'explore')} mode")

empties = sum(1 for rec in history if len(rec["selected"]) < 6)
print(f"Cycles with <6 selected: {empties}")

