"""
_compare_modes.py — Compare explore vs refine cycle modes.
"""
import os, sys, time
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import load_substrate
from rhetorical_cycle import run_cycle

sub = load_substrate(os.path.join(_HERE, "substrate.min.bin"))
INPUT = "The bearing failed because lubrication was inadequate."
N = 50

for mode in ["explore", "refine"]:
    t0 = time.perf_counter()
    history = run_cycle(INPUT, sub, n_cycles=N, verbose=False, layers=5, select=6,
                        cycle_mode=mode, exclusion_window=5)
    elapsed = time.perf_counter() - t0

    all_ever = set()
    print(f"\n{'='*80}")
    print(f"MODE: {mode.upper()}  ({N} cycles, {elapsed:.1f}s)")
    print(f"{'='*80}")
    print(f"{'CY':>4}  {'SA':<14}  {'CLOS':>5}  {'EXCL':>4}  SELECTED")
    print("-" * 80)

    for rec in history:
        sel = rec["selected"]
        all_ever.update(sel)
        # Count how many of the original anchor heads appear in this cycle's selected
        anchor_overlap = len(set(sel) & {"bearing", "fail", "lubrication", "inadequate", "adequate"})
        print(f"  {rec['cycle']:>2}  {rec['speech_act']:<14}  {rec['closure_size']:>5}  "
              f"{len(set().union(*[set(r['selected']) for r in history[:rec['cycle']]])):>4}  "
              f"{' '.join(sel)}")

    # Count revisits (headwords that appear in multiple cycles)
    from collections import Counter
    all_sel = []
    for rec in history:
        all_sel.extend(rec["selected"])
    counts = Counter(all_sel)
    revisits = {h: c for h, c in counts.items() if c > 1}

    print(f"\nTotal unique headwords: {len(all_ever)}")
    print(f"Revisited headwords: {len(revisits)}")
    if revisits:
        top = sorted(revisits.items(), key=lambda x: -x[1])[:10]
        print(f"  Top revisits: {top}")
    print(f"Avg closure size: {sum(r['closure_size'] for r in history)/N:.0f}")
