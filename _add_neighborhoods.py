"""
_add_neighborhoods.py — Integrate morphological + etymological neighborhoods
from the dictionary probe into the listener substrate.

Morphological: words sharing a prefix (4+ chars) or suffix (4+ chars).
  Caps each word's neighbor list to 20 to prevent explosion.

Etymological: parsed from Etym: markers in websters_dictionary.txt.
  Links words sharing the same non-native etymological root (L., F., Gr., etc.).

Output: substrate.min.bin is updated in-place with two new relation types:
  relations["morphological"] = {head: [neighbor, ...]}
  relations["etymological"]  = {head: [neighbor, ...]}
"""
import json
import lzma
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import msgpack

_HERE = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = r"c:\Users\idz9m\Downloads\hscm_workspace\websters_dictionary.txt"
SUB_PATH = os.path.join(_HERE, "substrate.min.bin")

XZ_MAGIC = b"\xfd7zXZ\x00"

# ---------------------------------------------------------------- morphological

def compute_morphological(headwords, max_neighbors=20):
    """Group words by shared prefix (4+ chars) and suffix (4+ chars)."""
    by_prefix = defaultdict(list)
    by_suffix = defaultdict(list)

    for h in headwords:
        if len(h) < 5:
            continue
        by_prefix[h[:4]].append(h)
        by_suffix[h[-4:]].append(h)

    morpho = {}
    for h in headwords:
        if len(h) < 5:
            continue
        neighbors = set()
        pref_key = h[:4]
        suff_key = h[-4:]
        for w in by_prefix.get(pref_key, []):
            if w != h:
                neighbors.add(w)
        for w in by_suffix.get(suff_key, []):
            if w != h:
                neighbors.add(w)
        if neighbors:
            morpho[h] = sorted(neighbors)[:max_neighbors]

    return morpho


# ---------------------------------------------------------------- etymological

ETYM_RE = re.compile(r"Etym:\s*\[([^\]]+)\]")
# Language codes: L. = Latin, F. = French, Gr. = Greek, G. = German,
# AS. = Anglo-Saxon, D. = Dutch, Sp. = Spanish, It. = Italian, Pg. = Portuguese,
# Ar. = Arabic, Pers. = Persian, Hind. = Hindi, Skr. = Sanskrit, Heb. = Hebrew,
# W. = Welsh, Ir. = Irish, Gael. = Gaelic, Pol. = Polish, Russ. = Russian,
# Sw. = Swedish, Dan. = Danish, Icel. = Icelandic, OE. = Old English

LANG_CODES = {
    "L.", "F.", "Gr.", "G.", "AS.", "D.", "Sp.", "It.", "Pg.", "Ar.",
    "Pers.", "Hind.", "Skr.", "Heb.", "W.", "Ir.", "Gael.", "Pol.",
    "Russ.", "Sw.", "Dan.", "Icel.", "OE.", "OL.", "OF.", "LL.",
    "NL.", "Sp.", "Pr.", "Prov.", "Corn.", "Armor.", "Skt.",
}

ETYM_LANG_RE = re.compile(
    r"(L\.|F\.|Gr\.|G\.|AS\.|D\.|Sp\.|It\.|Pg\.|Ar\.|Pers\.|Hind\.|"
    r"Skr\.|Heb\.|W\.|Ir\.|Gael\.|Pol\.|Russ\.|Sw\.|Dan\.|Icel\.|"
    r"OE\.|OL\.|OF\.|LL\.|NL\.|Pr\.|Prov\.|Corn\.|Armor\.|Skt\.)"
)

# Root extraction: after the language code, capture the root word(s)
# e.g. "[L. articulus]" -> lang=L., root=articulus
# e.g. "[F. abaisser, fr. L. bassus]" -> lang=F., root=abaisser
ROOT_RE = re.compile(
    r"(?:L\.|F\.|Gr\.|G\.|AS\.|D\.|Sp\.|It\.|Pg\.|Ar\.|Pers\.|Hind\.|"
    r"Skr\.|Heb\.|W\.|Ir\.|Gael\.|Pol\.|Russ\.|Sw\.|Dan\.|Icel\.|"
    r"OE\.|OL\.|OF\.|LL\.|NL\.|Pr\.|Prov\.|Corn\.|Armor\.|Skt\.)"
    r"\s+([a-z]+)"
)


def parse_etymology(raw_text):
    """Extract (language, root) from Etym: markers AND bare language codes.

    Webster's 1913 has two patterns:
      1. "Etym: [L. articulus]" — explicit etymology block
      2. "[L.]" or "[F. abaisser]" — inline language tag in definition text

    Returns list of (lang, root) tuples, or empty list if none found.
    """
    results = []
    seen = set()

    # Pattern 1: Etym: [...] blocks
    for m in ETYM_RE.finditer(raw_text):
        inner = m.group(1)
        lang_match = ETYM_LANG_RE.search(inner)
        if not lang_match:
            continue
        lang = lang_match.group(1)
        root_match = ROOT_RE.search(inner)
        root = root_match.group(1).strip().lower() if root_match else ""
        if root and len(root) >= 3:
            key = (lang, root)
            if key not in seen:
                seen.add(key)
                results.append(key)
        elif lang:
            key = (lang, "")
            if key not in seen:
                seen.add(key)
                results.append(key)

    # Pattern 2: bare [L. word] / [F. word] / [Gr. word] inline tags
    # Look for language code followed by a word in brackets
    INLINE_RE = re.compile(
        r"\[(?:" + "|".join(re.escape(c) for c in LANG_CODES) + r")"
        r"\s+([a-z]{3,})\]"
    )
    for m in INLINE_RE.finditer(raw_text):
        lang_match = ETYM_LANG_RE.search(raw_text[m.start():m.end()])
        if not lang_match:
            continue
        lang = lang_match.group(1)
        root = m.group(1).strip().lower()
        key = (lang, root)
        if key not in seen:
            seen.add(key)
            results.append(key)

    # Pattern 3: "Same as <word>" or "See <word>" — cross-linguistic borrowing
    # These don't give us etymology roots, so skip.

    return results


def compute_etymological(headwords_set, dict_path, max_neighbors=20):
    """Parse the raw dictionary text for Etym: markers and link words
    sharing the same etymological root."""
    if not os.path.exists(dict_path):
        print(f"  [etym] dictionary file not found: {dict_path}")
        return {}

    print(f"  [etym] parsing {dict_path} ...")
    raw = Path(dict_path).read_text(encoding="utf-8", errors="replace")

    # Parse entries: "headword: definition text"
    # Each entry is on a line starting after the previous headword
    entry_re = re.compile(r"^([a-z]+[a-z\'-]*):\s", re.MULTILINE)

    head_to_etym = {}
    entries = list(entry_re.finditer(raw))
    print(f"  [etym] found {len(entries):,} entries in raw text")

    for i, m in enumerate(entries):
        head = m.group(1).lower()
        if head not in headwords_set:
            continue
        start = m.end()
        end = entries[i + 1].start() if i + 1 < len(entries) else len(raw)
        entry_text = raw[start:end]
        etym_pairs = parse_etymology(entry_text)
        if etym_pairs:
            head_to_etym[head] = etym_pairs

    print(f"  [etym] {len(head_to_etym):,} headwords with etymology")

    # Group by (lang, root) to find neighbors
    by_root = defaultdict(set)
    for head, pairs in head_to_etym.items():
        for lang, root in pairs:
            if root:
                by_root[(lang, root)].add(head)

    # Build etymological neighborhood
    etym = {}
    for head, pairs in head_to_etym.items():
        neighbors = set()
        for lang, root in pairs:
            if root:
                for w in by_root.get((lang, root), set()):
                    if w != head:
                        neighbors.add(w)
        if neighbors:
            etym[head] = sorted(neighbors)[:max_neighbors]

    print(f"  [etym] {len(etym):,} headwords with etymological neighbors")
    return etym


# ---------------------------------------------------------------- main

def main():
    print("Loading substrate ...")
    raw = Path(SUB_PATH).read_bytes()
    if raw[:6] == XZ_MAGIC:
        raw = lzma.decompress(raw)
    sub = msgpack.unpackb(raw, strict_map_key=False)

    headwords = list(sub["records"].keys())
    headwords_set = set(headwords)
    print(f"  {len(headwords):,} headwords")

    # ---- Morphological
    print("\nComputing morphological neighborhoods ...")
    t0 = time.time()
    morpho = compute_morphological(headwords)
    total_morpho_edges = sum(len(v) for v in morpho.values())
    print(f"  {len(morpho):,} words with morphological neighbors, "
          f"{total_morpho_edges:,} total edges  ({time.time()-t0:.1f}s)")

    # ---- Etymological
    print("\nComputing etymological neighborhoods ...")
    t0 = time.time()
    etym = compute_etymological(headwords_set, DICT_PATH)
    total_etym_edges = sum(len(v) for v in etym.values())
    print(f"  {len(etym):,} words with etymological neighbors, "
          f"{total_etym_edges:,} total edges  ({time.time()-t0:.1f}s)")

    # ---- Update substrate
    sub["relations"]["morphological"] = morpho
    sub["relations"]["etymological"] = etym

    # Update meta
    m = sub["meta"]
    m["morphological"] = len(morpho)
    m["etymological"] = len(etym)
    m["morphological_edges"] = total_morpho_edges
    m["etymological_edges"] = total_etym_edges

    # ---- Save
    print("\nSaving updated substrate ...")
    packed = msgpack.packb(sub)
    Path(SUB_PATH).write_bytes(lzma.compress(packed, preset=9))

    # Also save uncompressed
    full_path = SUB_PATH.replace(".min.bin", ".bin")
    Path(full_path).write_bytes(packed)

    print(f"\nDone.")
    print(f"  morphological:  {len(morpho):,} words, {total_morpho_edges:,} edges")
    print(f"  etymological:   {len(etym):,} words, {total_etym_edges:,} edges")
    print(f"  substrate.min.bin  {Path(SUB_PATH).stat().st_size:,} bytes")

    # ---- Sample
    print("\nSamples:")
    for h in ["compose", "abandon", "philosophy", "heat", "write"]:
        m_neigh = morpho.get(h, [])
        e_neigh = etym.get(h, [])
        print(f"  {h}:")
        print(f"    morphological: {m_neigh[:8]}")
        print(f"    etymological:  {e_neigh[:8]}")


if __name__ == "__main__":
    main()
