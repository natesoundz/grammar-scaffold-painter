"""
painter.py — Structure Painter: grammar-based knowledge instantiation engine.

D(h) = (content, scaffold)

The grammar is the skeleton. The context is the palette.
Grammar + context = paint any knowledge structure that has that relational shape.

The structure already knows what should go there. When you have [___ by ___]
in an authority context, the structure is screaming for agency words.
The grammar constrains the context so tightly that semantically deep words
are the only ones that fit.

Profound = maximum coherence between structure, context, and instantiation.

Usage:
    from painter import StructurePainter
    painter = StructurePainter(records, relations)
    result = painter.paint("to ___ by ___ to ___", context=["command", "authority"], closure=closure)
"""

import re
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scaffold import (
    ScaffoldAnalyzer, _SCAFFOLD_ROLES, _FUNCTION_WORD_RE,
    JOIN_SCAFFOLD_AFFINITY,
)
from dict_algebra import definition_words, INTERSECT


class StructurePainter:
    """Paint knowledge onto structure using grammar as the medium
    and context as the constraint."""

    def __init__(self, records, relations=None, activations=None):
        self.records = records
        self.relations = relations or {}
        self.function_words = set(self.relations.get("function_words", []))
        self.sa = ScaffoldAnalyzer()
        self._func_word_set = set(_SCAFFOLD_ROLES.keys())
        self._activations = activations or {}

    def _content_words(self, text):
        """Extract content words (non-function) from a text string."""
        tokens = re.findall(r'[a-zA-Z]+', text.lower())
        return [t for t in tokens if t not in self._func_word_set and len(t) > 2]

    def _scaffold_roles_from_pattern(self, pattern_str):
        """Parse a pattern string like 'DIRECT CAUSE DIRECT' into role list."""
        return pattern_str.strip().split()

    def _structure_from_template(self, template):
        """Parse a template like 'to ___ by ___ to ___' into (skeleton, slots).
        
        Returns:
            skeleton: list of function words
            slots: list of None placeholders (one per ___)
            roles: list of role names corresponding to each function word
        """
        tokens = template.split()
        skeleton = []
        slot_positions = []  # indices in skeleton where a slot precedes
        slots = []
        slot_idx = 0

        parts = []
        for tok in tokens:
            if tok == "___":
                slots.append([])
                parts.append(("slot", slot_idx))
                slot_idx += 1
            else:
                role = _SCAFFOLD_ROLES.get(tok.lower(), "OTHER")
                parts.append(("word", tok.lower(), role))

        return parts

    def _headword_scaffold_roles(self, headword):
        """Get the set of scaffold roles for a headword's first sense."""
        if headword not in self.records:
            return set()
        senses = self.records[headword].get("senses", [])
        if not senses:
            return set()
        defn = senses[0].get("definition", "")
        return set(self.sa.extract_scaffold(defn))

    def _headword_definition(self, headword):
        """Get first-clause definition of a headword."""
        if headword not in self.records:
            return ""
        senses = self.records[headword].get("senses", [])
        if not senses:
            return ""
        return senses[0].get("definition", "")

    def _context_field(self, context_words):
        """Build a context field set from context word list."""
        field = set()
        for w in context_words:
            w = w.lower()
            if w in self.records:
                field.add(w)
                # Add definition words of context headwords
                dw = definition_words(w, self.records, self.function_words)
                field |= dw
            else:
                field.add(w)
        return field

    def search_by_roles(self, closure, required_roles, context_field=None,
                         activations=None):
        """Find headwords in the closure whose scaffolds contain the required roles.
        
        Args:
            closure: set/list of headwords to search
            required_roles: set of role names that must be present
            context_field: optional set of context words for filtering
            activations: optional dict from state.trace["activations"] for layer info
            
        Returns:
            List of dicts: {headword, scaffold, roles, definition, context_overlap,
                            layer, score}
        """
        results = []
        max_layer = 1
        if activations:
            max_layer = max((a.get("layer", 1) for a in activations.values()), default=1)
        
        for h in closure:
            if h not in self.records or not self.records[h]["senses"]:
                continue
            defn = self.records[h]["senses"][0]["definition"]
            scaffold = self.sa.extract_scaffold(defn)
            if not scaffold:
                continue
            roles = set(scaffold)
            has_all = required_roles.issubset(roles)
            has_any = bool(roles & required_roles)
            
            if not has_any:
                continue
            
            # Context coherence: how many context words appear in definition
            ctx_overlap = set()
            if context_field:
                defn_words = set(re.findall(r'[a-zA-Z]+', defn.lower()))
                ctx_overlap = defn_words & context_field
            
            # Closure proximity: traversal layer (0 = input, higher = deeper)
            layer = 1
            if activations and h in activations:
                layer = activations[h].get("layer", 1)
            # Proximity: 1.0 at layer 0, decays with depth
            proximity = 1.0 / (1.0 + layer)
            
            # Multiplicative intersection: all three must be nonzero
            role_fit = len(roles & required_roles) / len(required_roles) if required_roles else 0
            ctx_fit = len(ctx_overlap) / max(1, len(context_field)) if context_field else 0
            profundity_raw = role_fit * ctx_fit * proximity
            
            results.append({
                "headword": h,
                "scaffold": scaffold,
                "sig": self.sa.scaffold_signature(scaffold),
                "roles": sorted(roles),
                "roles_matched": sorted(roles & required_roles),
                "definition": defn,
                "first_clause": defn.split(";")[0].strip().rstrip("."),
                "context_overlap": sorted(ctx_overlap) if ctx_overlap else [],
                "has_all_roles": has_all,
                "layer": layer,
                "proximity": proximity,
                "role_fit": role_fit,
                "ctx_fit": ctx_fit,
                "profundity_raw": profundity_raw,
                "score": profundity_raw,
            })
        
        results.sort(key=lambda r: r["profundity_raw"], reverse=True)
        return results

    def paint(self, template, context, closure, n_candidates=10):
        """Fill a structure template with words that fit the context.
        
        Grammar guides selection. Context restrains it.
        Result: knowledge rendered as instantiated structure.
        
        Args:
            template: structure string like "to ___ by ___ to ___"
            context: list of context headwords (e.g., ["command", "authority"])
            closure: set/list of reachable headwords
            n_candidates: max candidates per slot
            
        Returns:
            dict with painted structures and candidate analysis
        """
        parts = self._structure_from_template(template)
        context_field = self._context_field(context)
        
        # Determine which roles the template uses
        template_roles = set()
        for part in parts:
            if part[0] == "word":
                template_roles.add(part[2])
        
        # Search closure for headwords matching the template's roles
        activations = self._activations
        candidates = self.search_by_roles(closure, template_roles, context_field,
                                          activations=activations)
        
        # For each candidate, try to instantiate the template
        # by extracting content words from the candidate's definition
        # and placing them into the template's slots
        paintings = []
        
        for cand in candidates[:n_candidates * 2]:
            h = cand["headword"]
            defn = cand["definition"]
            content = self._content_words(defn)
            
            if not content:
                continue
            
            # Fill slots with content words from this definition
            n_slots = sum(1 for p in parts if p[0] == "slot")
            
            # Try different content word assignments
            for offset in range(min(3, max(1, len(content) - n_slots + 1))):
                slot_words = content[offset:offset + n_slots]
                if len(slot_words) < n_slots:
                    continue
                
                # Build the painted string
                painted_parts = []
                si = 0
                for part in parts:
                    if part[0] == "slot":
                        painted_parts.append(slot_words[si])
                        si += 1
                    else:
                        painted_parts.append(part[1])
                
                painted = " ".join(painted_parts)
                
                # Profundity = multiplicative intersection
                # role_fit × ctx_fit × proximity — all must be nonzero
                role_fit = cand["role_fit"]
                ctx_fit = cand["ctx_fit"]
                proximity = cand["proximity"]
                profundity = role_fit * ctx_fit * proximity
                
                paintings.append({
                    "painted": painted,
                    "donor": h,
                    "donor_definition": cand["first_clause"],
                    "donor_scaffold": cand["sig"],
                    "roles_matched": cand["roles_matched"],
                    "context_overlap": cand["context_overlap"],
                    "role_fit": role_fit,
                    "ctx_fit": ctx_fit,
                    "proximity": proximity,
                    "layer": cand["layer"],
                    "profundity": profundity,
                    "content_used": slot_words,
                })
        
        # Sort by profundity (coherence between structure, context, instantiation)
        paintings.sort(key=lambda p: p["profundity"], reverse=True)
        
        # Deduplicate by painted string
        seen = set()
        unique_paintings = []
        for p in paintings:
            if p["painted"] not in seen:
                seen.add(p["painted"])
                unique_paintings.append(p)
        
        return {
            "template": template,
            "context": context,
            "template_roles": sorted(template_roles),
            "candidates_found": len(candidates),
            "paintings": unique_paintings[:n_candidates],
        }

    def paint_across(self, context, closure, templates=None, n_per=5):
        """Paint the same context onto multiple structures.
        
        Same context, different structures = different expressions
        of the same intelligence.
        
        Args:
            context: list of context headwords
            closure: reachable headwords
            templates: list of template strings. If None, uses defaults.
            n_per: max paintings per template
            
        Returns:
            list of paint results, one per template
        """
        if templates is None:
            templates = [
                "the ___ of ___",
                "to ___ by ___ to ___",
                "a ___ in ___ which ___",
                "___ and ___ or ___",
                "to ___ for ___",
                "___ of ___ in ___",
            ]
        
        results = []
        for tmpl in templates:
            result = self.paint(tmpl, context, closure, n_candidates=n_per)
            results.append(result)
        
        return results

    def discover_structures(self, closure, top_n=20):
        """Discover the most common scaffold structures in the closure.
        
        Returns the dominant structural patterns, not the content.
        """
        patterns = Counter()
        examples = defaultdict(list)
        
        for h in closure:
            if h not in self.records or not self.records[h]["senses"]:
                continue
            defn = self.records[h]["senses"][0]["definition"]
            scaffold = self.sa.extract_scaffold(defn)
            if not scaffold:
                continue
            sig = self.sa.scaffold_signature(scaffold)
            patterns[sig] += 1
            if len(examples[sig]) < 3:
                examples[sig].append({
                    "headword": h,
                    "definition": defn[:80],
                    "skeleton": [w for w in scaffold],
                })
        
        top = []
        for sig, count in patterns.most_common(top_n):
            top.append({
                "signature": sig,
                "count": count,
                "pct": 100 * count / max(1, sum(patterns.values())),
                "examples": examples[sig],
            })
        
        return top


def main():
    """CLI: paint structures with context from the substrate."""
    import argparse
    ap = argparse.ArgumentParser(description="Structure Painter")
    ap.add_argument("context", help="Context words (space-separated)")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--frontier-cap", type=int, default=600)
    ap.add_argument("--template", default="", help="Single template to paint")
    ap.add_argument("--across", action="store_true", help="Paint across multiple structures")
    ap.add_argument("--discover", action="store_true", help="Discover dominant structures in closure")
    ap.add_argument("--n", type=int, default=5, help="Number of paintings per template")
    args = ap.parse_args()

    from discover import load_substrate
    from analyzer import analyze

    sub = load_substrate(os.path.join(_HERE, "substrate.min.bin"))
    records = sub["records"]
    relations = sub["relations"]
    function_words = set(relations["function_words"])

    print(f"Loading substrate ... {len(records):,} headwords\n")

    context_words = args.context.lower().split()
    print(f"Context: {context_words}\n")

    # Build closure from context
    state = analyze(
        input_text=args.context,
        context_texts=[],
        sub=sub,
        max_layers=args.layers,
        frontier_cap=args.frontier_cap,
    )
    closure = list(state.closure - function_words)
    activations = state.trace.get("activations", {})
    print(f"Closure: {len(closure)} headwords\n")

    painter = StructurePainter(records, relations, activations=activations)

    if args.discover:
        print("=" * 60)
        print("DOMINANT STRUCTURES IN CLOSURE")
        print("=" * 60)
        structures = painter.discover_structures(closure, top_n=15)
        for s in structures:
            print(f"\n  [{s['count']:>3}] {s['signature']}  ({s['pct']:.1f}%)")
            for ex in s["examples"]:
                print(f"         {ex['headword']}: {ex['definition']}")
        print()

    if args.template:
        print("=" * 60)
        print(f"PAINTING: {args.template}")
        print("=" * 60)
        result = painter.paint(args.template, context_words, closure, n_candidates=args.n)
        print(f"\n  Template roles: {result['template_roles']}")
        print(f"  Candidates found: {result['candidates_found']}\n")
        for i, p in enumerate(result["paintings"][:args.n]):
            print(f"  [{i+1}] {p['painted']}")
            print(f"      donor: {p['donor']} — {p['donor_definition'][:60]}")
            print(f"      scaffold: {p['donor_scaffold']}")
            print(f"      roles matched: {p['roles_matched']}")
            print(f"      context overlap: {p['context_overlap']}")
            print(f"      profundity: {p['profundity']:.4f} = role={p['role_fit']:.2f} × ctx={p['ctx_fit']:.2f} × prox={p['proximity']:.2f} (L{p['layer']})")
            print()
    
    if args.across:
        print("=" * 60)
        print("PAINTING ACROSS STRUCTURES")
        print("=" * 60)
        results = painter.paint_across(context_words, closure, n_per=args.n)
        for result in results:
            print(f"\n--- {result['template']}  (roles: {result['template_roles']}) ---")
            for i, p in enumerate(result["paintings"][:3]):
                print(f"  [{i+1}] {p['painted']}")
                print(f"      from {p['donor']}: {p['donor_definition'][:50]}")
                print(f"      profundity={p['profundity']:.4f} (role={p['role_fit']:.2f} ctx={p['ctx_fit']:.2f} prox={p['proximity']:.2f} L{p['layer']})")

    if not args.template and not args.across and not args.discover:
        # Default: discover + across
        print("=" * 60)
        print("DOMINANT STRUCTURES IN CLOSURE")
        print("=" * 60)
        structures = painter.discover_structures(closure, top_n=10)
        for s in structures[:5]:
            print(f"  [{s['count']:>3}] {s['signature']}  ({s['pct']:.1f}%)")
            for ex in s["examples"][:1]:
                print(f"         {ex['headword']}: {ex['definition']}")
        print()

        print("=" * 60)
        print("PAINTING ACROSS STRUCTURES")
        print("=" * 60)
        results = painter.paint_across(context_words, closure, n_per=args.n)
        for result in results:
            print(f"\n--- {result['template']}  (roles: {result['template_roles']}) ---")
            if result["paintings"]:
                for i, p in enumerate(result["paintings"][:3]):
                    print(f"  [{i+1}] {p['painted']}")
                    print(f"      from {p['donor']}: {p['donor_definition'][:50]}")
                    print(f"      profundity={p['profundity']:.4f} (role={p['role_fit']:.2f} ctx={p['ctx_fit']:.2f} prox={p['proximity']:.2f} L{p['layer']})")
                    if p['context_overlap']:
                        print(f"      ctx_overlap: {p['context_overlap']}")
            else:
                print(f"  (no candidates found)")
        print()


if __name__ == "__main__":
    main()
