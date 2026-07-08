import json, os
rules = [json.loads(l) for l in open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "transforms.jsonl"),
    encoding="utf-8") if l.strip()]

hdr = f"  {'CLASS':<43}  {'JOIN':<24}  {'NEIGHBOR_MODE':<20}  CONF"
print(hdr)
print("-" * len(hdr))
for r in rules:
    cls = f"{r['speech_act']}+{r['response_type']}+{r['closure_bucket']}"
    print(f"  {cls:<43}  {r['join']:<24}  {r['neighbor_mode']:<20}  {r['confidence']:.3f}")
