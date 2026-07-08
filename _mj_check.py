import json, os
from collections import defaultdict, Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(_HERE, "multi_join.jsonl")

records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

# Per discourse class: which join wins most?
class_winners = defaultdict(Counter)
all_identical = 0

for r in records:
    key = (r["speech_act"], r["response_type"], r["closure_bucket"])
    class_winners[key][r["best_join"]] += 1

    scores = list(r["join_scores"].values())
    if len(set(scores)) == 1:
        all_identical += 1

print(f"Records: {len(records)}")
print(f"All-join-identical bridge scores: {all_identical}/{len(records)}")
print()
print(f"{'Class':<45}  {'Winner':22}  n")
for key, ctr in sorted(class_winners.items()):
    winner, count = ctr.most_common(1)[0]
    label = f"{key[0]}+{key[1]}+{key[2]}"
    print(f"  {label:<43}  {winner:<22}  {count}/{sum(ctr.values())}")
