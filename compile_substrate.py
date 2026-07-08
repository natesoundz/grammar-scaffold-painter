"""
compile_substrate.py — compiles Webster's 1913 into a binary substrate.

    python compile_substrate.py                     # expects websters1913.json
    python compile_substrate.py --source path.json

Output:
    substrate.bin       msgpack of the full compiled substrate (fast load)
    substrate.min.bin   duplicated, lzma-minimized, losslessly — verified by
                        DIRECT structural comparison (==), no hashes

What "compiled" means here: every headword listing is parsed into the same
category structure a listing actually has —

    record = {
      "preamble":   text before the first numbered sense (POS notes, "imp. of")
      "senses": [ {
          "block":      part-of-speech block index (numbering restarts at 1)
          "domain":     leading label like (Arch.) / (Naut.) if present
          "definition": the defining statement(s)
          "usage":      inline usage examples ("; as, to bolt food")
          "examples":   quotation sentences with attributions (Shak., Milton)
          "phrases":    compound entries ("Bolt and nut, a metallic pin ...")
          "crossrefs":  "See X" references
          "notes":      "Note: ..." material
      } ]
    }

— and function words are identified structurally at compile time (ubiquity
across listings — a discrete threshold, never exposed as a per-word weight).

Serialization is msgpack (never pickle). Interrupts save partial progress.
Segmentation of examples/phrases is heuristic where Webster's formatting is
irregular — stated plainly, not hidden.
"""

import argparse
import json
import lzma
import math
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import msgpack

WORD_RE = re.compile(r"[a-z]+")
SENSE_NUM_RE = re.compile(r"(?<![\w.])(\d{1,2})\.\s")
DOMAIN_RE = re.compile(r"^\((?P<d>[^)]{1,40})\)\s*")
SENTENCE_SPLIT_RE = re.compile(r"(?<=\.)\s+")
PHRASE_RE = re.compile(r"^([A-Z][A-Za-z'-]*(?:\s+[a-z'-]+){0,4}),\s+(.*)$")
CROSSREF_RE = re.compile(r"^(?:Note:\s*)?See\s+([A-Z][A-Za-z, &]+)")
ATTRIBUTION_RE = re.compile(
    r"(?:[A-Z][A-Za-z]*\.\s*){0,2}[A-Z][A-Za-z]{1,14}\.$"
)


def words(text: str):
    return WORD_RE.findall(text.lower())


def split_sense_spans(entry_text: str):
    """Numbered sense spans, with block tracking. Numbering restarts at 1
    for each new part-of-speech block; citation numbers can't split because
    they are neither the expected next number nor 1."""
    marks, expected, block = [], 1, 0
    for m in SENSE_NUM_RE.finditer(entry_text):
        n = int(m.group(1))
        if n == expected:
            marks.append((m.start(), block))
            expected = n + 1
        elif n == 1 and marks:
            block += 1
            marks.append((m.start(), block))
            expected = 2
    return marks


def classify_sense(sense_text: str, block: int):
    sense = {
        "block": block, "domain": "", "definition": "", "usage": [],
        "examples": [], "phrases": [], "crossrefs": [], "notes": [],
    }
    body = re.sub(r"^\d{1,2}\.\s*", "", sense_text).strip()

    dm = DOMAIN_RE.match(body)
    if dm:
        sense["domain"] = dm.group("d")
        body = body[dm.end():]

    # " -- " separates trailing compound-phrase entries from each other
    segments = body.split(" -- ")
    definition_parts = []

    for gi, seg in enumerate(segments):
        sentences = SENTENCE_SPLIT_RE.split(seg.strip())
        for si, s in enumerate(sentences):
            s = s.strip()
            if not s:
                continue
            if s.startswith("Note:"):
                sense["notes"].append(s[5:].strip())
                continue
            cr = CROSSREF_RE.match(s)
            if cr:
                sense["crossrefs"].extend(
                    x.strip() for x in re.split(r",| and ", cr.group(1)) if x.strip()
                )
                continue
            pm = PHRASE_RE.match(s)
            if pm and (gi > 0 or si > 0):
                sense["phrases"].append({"phrase": pm.group(1), "gloss": pm.group(2)})
                continue
            # inline usage: "...; as, to bolt food."
            if "; as," in s or ", as," in s:
                head, _, tail = re.split(r"(;\s*as,|,\s*as,)", s, 1)[0:3]
                definition_parts.append(head.strip().rstrip(";,"))
                sense["usage"].append(tail.strip())
                continue
            # quotation with attribution, only after a definition exists
            if definition_parts and si > 0 and ATTRIBUTION_RE.search(s):
                sense["examples"].append(s)
                continue
            definition_parts.append(s)

    sense["definition"] = " ".join(definition_parts).strip()
    return sense


def parse_listing(entry_text: str):
    marks = split_sense_spans(entry_text)
    if not marks:
        return {"preamble": "", "senses": [classify_sense(entry_text, 0)]}
    preamble = entry_text[: marks[0][0]].strip()
    senses, bounds = [], [p for p, _ in marks] + [len(entry_text)]
    for (pos, block), end in zip(marks, bounds[1:]):
        senses.append(classify_sense(entry_text[pos:end], block))
    return {"preamble": preamble, "senses": senses}


def sense_all_text(sense) -> str:
    parts = [sense["definition"], *sense["usage"], *sense["examples"],
             *sense["notes"]]
    parts += [f'{p["phrase"]} {p["gloss"]}' for p in sense["phrases"]]
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="websters1913.json")
    ap.add_argument("--out", default="substrate.bin")
    args = ap.parse_args()

    raw = Path(args.source).read_bytes()          # missing file = crash
    entries = json.loads(raw)
    n = len(entries)
    print(f"source: {args.source}  {len(raw):,} bytes, {n:,} headword listings")

    records, df = {}, Counter()
    state = {"done": 0}

    def on_sigint(signum, frame):
        Path(args.out + ".partial").write_bytes(
            msgpack.packb({"done": state["done"], "records": records})
        )
        print(f"\ninterrupted at {state['done']}/{n}; partial compile saved",
              file=sys.stderr)
        raise KeyboardInterrupt

    prev = signal.signal(signal.SIGINT, on_sigint)
    start = time.time()
    try:
        for i, (head, body) in enumerate(entries.items(), 1):
            rec = parse_listing(body)
            records[head] = rec
            listing_words = set(words(body))
            df.update(listing_words)
            state["done"] = i
            if i % 2000 == 0 or i == n:
                el = time.time() - start
                eta = el / i * (n - i)
                bar = "#" * int(30 * i / n)
                sys.stdout.write(
                    f"\r[compile {bar:<30}] {i}/{n} "
                    f"elapsed {el:5.1f}s eta {eta:5.1f}s"
                )
                sys.stdout.flush()
        print()
    finally:
        signal.signal(signal.SIGINT, prev)

    # NOTE: no significance/idf table. That was the last surviving piece of
    # continuous weighted scoring in this substrate — a number derived from
    # word frequency, multiplied into rankings downstream. It contradicts
    # the architecture's own governing principle: discrete category
    # membership (domain tags, oppositions, crossrefs, adjacency-required
    # shape), not scored ranking. df is still computed above, but only to
    # threshold function_words (a binary category assignment: ubiquitous
    # or not), never exposed as a per-word weight.
    print("compiling relations (oppositions, crossrefs, function-word neighborhood)...")

    OPP_RE = re.compile(
        r"(?:opposed to|opposite of|contrary to|reverse of)\s+"
        r"([a-z]+(?:[,;]?\s+(?:or\s+)?[a-z]+){0,2})", re.I)
    NEG_RE = re.compile(r"\bnot\s+([a-z]+(?:,\s*[a-z]+){0,3})")
    OF_REF_RE = re.compile(r"\bof\s+([A-Z][a-z]+)")

    oppositions, crossrefs = {}, {}
    ubiquitous = {w for w, c in df.items() if c > n * 0.01}
    for head, rec in records.items():
        opp = set()
        # scan only DEFINITION text (already separated from notes/examples
        # by classify_sense) — never the raw entry body. The raw body mixes
        # in editorial asides ("Walker condemns the pronunciation woond...
        # opposed to an important principle of our language") that have
        # nothing to do with the headword's meaning; scanning it produced
        # spurious oppositions like wound<->important, wound<->principle.
        def_text = " ".join(s["definition"] for s in rec["senses"])
        for m in OPP_RE.finditer(def_text):
            for w in re.findall(r"[a-z]+", m.group(1)):
                if w in entries and w != head and w not in ubiquitous:
                    opp.add(w)
        for m in NEG_RE.finditer(def_text[:400]):
            for w in re.findall(r"[a-z]+", m.group(1)):
                if w in entries and w != head and w not in ubiquitous:
                    opp.add(w)
        for pre in ("un", "dis", "in"):
            if head.startswith(pre) and head[len(pre):] in entries:
                opp.add(head[len(pre):])
        if opp:
            oppositions[head] = sorted(opp)

        refs = set()
        ref_texts = [rec.get("preamble", "")]
        for s in rec["senses"]:
            ref_texts.append(s["definition"][:120])
            ref_texts.extend(e[:60] for e in s["examples"])
        for t in ref_texts:
            for m in OF_REF_RE.finditer(t):
                w = m.group(1).lower()
                if w in entries and w != head and w not in ubiquitous:
                    refs.add(w)
        for s in rec["senses"]:
            for c in s["crossrefs"]:
                w = c.strip().lower()
                if w in entries and w != head:
                    refs.add(w)
        if refs:
            crossrefs[head] = sorted(refs)

    # function-word neighborhood: structurally derived — words so ubiquitous
    # across listings that their role is grammatical management, not content.
    # This is one of the latent neighborhoods (functional vs content), made
    # explicit at compile time. Threshold: present in > 2% of all listings.
    function_words = sorted(w for w, c in df.items() if c > n * 0.01)

    total_senses = sum(len(r["senses"]) for r in records.values())
    total_examples = sum(len(s["examples"]) for r in records.values() for s in r["senses"])
    total_usage = sum(len(s["usage"]) for r in records.values() for s in r["senses"])
    total_phrases = sum(len(s["phrases"]) for r in records.values() for s in r["senses"])

    substrate = {
        "meta": {
            "format": 3,
            "headwords": n,
            "senses": total_senses,
            "examples": total_examples,
            "usage": total_usage,
            "phrases": total_phrases,
            "oppositions": len(oppositions),
            "crossrefs": len(crossrefs),
            "function_words": len(function_words),
            "source_bytes": len(raw),
        },
        "records": records,
        "relations": {
            "oppositions": oppositions,
            "crossrefs": crossrefs,
            "function_words": function_words,
        },
    }

    packed = msgpack.packb(substrate)
    Path(args.out).write_bytes(packed)

    minimized = lzma.compress(packed, preset=9)
    min_path = args.out.replace(".bin", ".min.bin")
    Path(min_path).write_bytes(minimized)

    # lossless verification by DIRECT structural comparison — no hashes
    back_full = msgpack.unpackb(Path(args.out).read_bytes(), strict_map_key=False)
    back_min = msgpack.unpackb(lzma.decompress(Path(min_path).read_bytes()),
                               strict_map_key=False)
    lossless = (back_full == back_min == msgpack.unpackb(packed, strict_map_key=False))
    if not lossless:
        raise RuntimeError("minimized duplicate is NOT identical — refusing to ship")

    print(f"\ncompiled: {n:,} listings, {total_senses:,} senses, "
          f"{total_examples:,} examples, {total_usage:,} usage, "
          f"{total_phrases:,} phrases")
    print(f"function words: {len(function_words):,} (ubiquity threshold, not weighted)")
    print(f"{args.out:<18} {len(packed):>12,} bytes")
    print(f"{min_path:<18} {len(minimized):>12,} bytes "
          f"({len(minimized)/len(packed)*100:.1f}% of full)")
    print(f"lossless duplicate verified by direct comparison: {lossless}")


if __name__ == "__main__":
    main()
