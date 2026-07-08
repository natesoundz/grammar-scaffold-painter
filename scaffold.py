"""
scaffold.py — Definition scaffold analysis.

D(h) = (content, scaffold)

Every definition has two layers:
  - content: the nouns/verbs/adjectives that specify semantic neighborhood
  - scaffold: the function-word skeleton that specifies relational structure

The scaffold is the grammar of the definition — the prepositions, articles,
conjunctions, and auxiliary verbs that encode HOW content relates.

Join operators (cause_effect, elaboration, sequence, ...) are themselves
scaffold templates. When the system selects terms under a join, it is
projecting the same semantic neighborhood through a grammatical frame.

Scaffold compatibility measures whether a term's internal scaffold
admits the join operator's external scaffold.

Key claim:
  Content selects objects. Grammar selects worlds.
  The physics is in the grammar, not the vocabulary.
"""

import re

# ---------------------------------------------------------------- scaffold roles

# Map raw function words to relational roles.
# This is the vocabulary of scaffold structure.
_SCAFFOLD_ROLES = {
    # composition
    "of":      "COMPOSE",
    "with":    "COMPOSE",
    # containment / location
    "in":      "CONTAIN",
    "on":      "CONTAIN",
    "at":      "CONTAIN",
    "under":   "CONTAIN",
    "within":  "CONTAIN",
    # causation
    "by":      "CAUSE",
    "from":    "CAUSE",
    "through": "CAUSE",
    "to":      "DIRECT",
    # identity / equivalence
    "is":      "IDENTIFY",
    "are":     "IDENTIFY",
    "was":     "IDENTIFY",
    "were":    "IDENTIFY",
    "be":      "IDENTIFY",
    "being":   "IDENTIFY",
    # specification
    "which":   "SPECIFY",
    "that":    "SPECIFY",
    "who":     "SPECIFY",
    "whose":   "SPECIFY",
    # contrast
    "but":     "CONTRAST",
    "yet":     "CONTRAST",
    "however": "CONTRAST",
    "not":     "NEGATE",
    "no":      "NEGATE",
    "without": "NEGATE",
    # sequence
    "then":    "SEQUENCE",
    "first":   "SEQUENCE",
    "next":    "SEQUENCE",
    "finally": "SEQUENCE",
    # condition
    "if":      "CONDITION",
    "when":    "CONDITION",
    "while":   "CONDITION",
    "unless":  "CONDITION",
    # quantity
    "a":       "ARTICLE",
    "an":      "ARTICLE",
    "the":     "ARTICLE",
    "any":     "QUANTIFY",
    "all":     "QUANTIFY",
    "some":    "QUANTIFY",
    "one":     "QUANTIFY",
    # conjunction
    "and":     "CONJOIN",
    "or":      "CONJOIN",
    "nor":     "CONJOIN",
    # possession
    "for":     "PURPOSE",
    "as":      "ROLE",
}

# Which scaffold roles each join operator prefers.
# These are the "external scaffold" templates.
JOIN_SCAFFOLD_AFFINITY = {
    "cause_effect":          {"CAUSE", "DIRECT", "COMPOSE"},
    "elaboration":           {"SPECIFY", "IDENTIFY", "ARTICLE", "COMPOSE"},
    "sequence":              {"SEQUENCE", "DIRECT", "CAUSE"},
    "condition_consequence": {"CONDITION", "DIRECT", "CAUSE"},
    "contrast":              {"CONTRAST", "NEGATE", "IDENTIFY"},
    "plain":                 set(),
}

# Function words to detect in definitions
_FUNCTION_WORD_RE = re.compile(
    r'\b(of|in|on|at|under|within|by|from|through|to|is|are|was|were|be|being|'
    r'which|that|who|whose|but|yet|however|not|no|without|then|first|next|finally|'
    r'if|when|while|unless|a|an|the|any|all|some|one|and|or|nor|for|as)\b',
    re.IGNORECASE,
)


class ScaffoldAnalyzer:
    """Extract and analyze definition scaffolds."""

    def extract_scaffold(self, definition: str) -> list[str]:
        """Extract the function-word skeleton from a definition.

        "a force in nature which is recognized in various effects"
        → ['ARTICLE', 'CONTAIN', 'SPECIFY', 'IDENTIFY', 'CONTAIN']

        Returns a list of scaffold roles in order of appearance.
        """
        roles = []
        for m in _FUNCTION_WORD_RE.finditer(definition):
            word = m.group(1).lower()
            role = _SCAFFOLD_ROLES.get(word)
            if role:
                roles.append(role)
        return roles

    def scaffold_compatibility(self, scaffold: list[str], join_operator: str) -> float:
        """How compatible is this scaffold with the given join operator?

        Measures the fraction of scaffold roles that appear in the join's
        affinity set. Returns 0.0 for empty scaffolds or unknown joins.

        A compatibility of 1.0 means every scaffold role matches the join.
        0.0 means no scaffold role matches.
        """
        if not scaffold:
            return 0.0
        affinity = JOIN_SCAFFOLD_AFFINITY.get(join_operator, set())
        if not affinity:
            return 0.0
        matches = sum(1 for r in scaffold if r in affinity)
        return matches / len(scaffold)

    def validate_selection(self, selected_terms: list[tuple], records,
                           join_type: str) -> dict:
        """Validate scaffold compatibility across a selection.

        selected_terms: list of (headword, tier, layer, via) tuples
        records: substrate records dict
        join_type: the join operator used

        Returns:
        {
            "per_term": {headword: {scaffold, compatibility, roles}},
            "mean_compatibility": float,
            "compatible_count": int,
            "total": int,
            "join_type": str,
            "verdict": "high" | "medium" | "low",
        }
        """
        per_term = {}
        compat_values = []

        for entry in selected_terms:
            h = entry[0] if isinstance(entry, tuple) else entry
            if h not in records:
                continue
            senses = records[h].get("senses", [])
            if not senses:
                continue
            definition = senses[0].get("definition", "")
            scaffold = self.extract_scaffold(definition)
            compat = self.scaffold_compatibility(scaffold, join_type)
            per_term[h] = {
                "scaffold": scaffold,
                "compatibility": compat,
                "roles": list(set(scaffold)),
            }
            compat_values.append(compat)

        if not compat_values:
            return {
                "per_term": {},
                "mean_compatibility": 0.0,
                "compatible_count": 0,
                "total": 0,
                "join_type": join_type,
                "verdict": "unknown",
            }

        mean_compat = sum(compat_values) / len(compat_values)
        compatible = sum(1 for c in compat_values if c >= 0.5)

        if mean_compat >= 0.75:
            verdict = "high"
        elif mean_compat >= 0.5:
            verdict = "medium"
        else:
            verdict = "low"

        return {
            "per_term": per_term,
            "mean_compatibility": mean_compat,
            "compatible_count": compatible,
            "total": len(compat_values),
            "join_type": join_type,
            "verdict": verdict,
        }

    def scaffold_signature(self, scaffold: list[str]) -> str:
        """Render a scaffold as a compact signature string.

        ['ARTICLE', 'CONTAIN', 'SPECIFY', 'IDENTIFY', 'CONTAIN']
        → "ART·CON·SPC·IDN·CON"
        """
        abbrev = {
            "COMPOSE": "CMP",
            "CONTAIN": "CON",
            "CAUSE":   "CAU",
            "DIRECT":  "DIR",
            "IDENTIFY":"IDN",
            "SPECIFY": "SPC",
            "CONTRAST": "CTN",
            "NEGATE":  "NEG",
            "SEQUENCE": "SEQ",
            "CONDITION":"CND",
            "ARTICLE": "ART",
            "QUANTIFY":"QNT",
            "CONJOIN": "CNJ",
            "PURPOSE": "PUR",
            "ROLE":    "ROL",
        }
        return "·".join(abbrev.get(r, r[:3]) for r in scaffold)
