"""
analyzer.py — Full analysis pipeline.

    Text  →  CommunicativeFrame  →  ActivationFields  →  ListenerState

Discourse is established first. The communicative frame is fixed before
any dictionary activation occurs. Only then does the substrate become
active.

Context is the accumulated collection of constraints. Every stage
contributes another constraint. The complete collection is the context.

Stage 1  Establish communicative frame (discourse before content).
Stage 2  Activate input field and context field over the substrate.
Stage 3  Run structural traversal (discover outward through definitions,
         crossrefs, oppositions).
Stage 4  Apply discrete constraints → closure.
         Closure = Activated Entries ∩ Accumulated Constraints.
"""

import re

from discover import discover, resolve, words
from dict_algebra import FILTER, CONSTRAIN
from listener_state import CommunicativeFrame, ActivationFields, ListenerState

# ---------------------------------------------------------------- module-level constants

_ACTION_RE = re.compile(r"^\s*to\s+\w", re.I)

_INSTRUCTION_RE = re.compile(
    r"\b(what should i|what's my best|what is my best|how do i|how can i|"
    r"what next|next move|next step|what do i do)\b", re.I)
_YES_NO_RE = re.compile(r"^\s*(is|are|was|were|does|did|do|can|will|would|should)\b", re.I)

_ADJACENCY = {
    "instruction":    "assertion",
    "explanation":    "explanation",
    "narrative":      "narrative",
    "counterfactual": "explanation",
    "hypothesis":     "assertion",
    "assertion":      "explanation",
}

_FUNCTION_STARTS = frozenset({
    "the", "a", "an", "it", "he", "she", "they", "you", "i", "we",
    "this", "that", "these", "those",
})


# ---------------------------------------------------------------- discourse classification

def classify_speech_act(text: str) -> str:
    """Heuristic, stdlib-only. Order matters: most distinctive signal first."""
    t = text.strip()
    low = t.lower()
    if t.endswith("?"):
        return "question"
    if re.search(r"\b(if|unless)\b.*\b(would|could|might|had)\b", low) or \
       re.search(r"\b(would|could|might|had)\b.*\b(if|unless)\b", low):
        return "counterfactual"
    if re.search(r"\b(may|might|perhaps|possibly|could it be)\b", low):
        return "hypothesis"
    if re.search(r"\b(because|since|therefore|thus|hence)\b", low):
        return "explanation"
    if re.search(r"\b(then|after that|meanwhile|yesterday|earlier|later)\b", low):
        return "narrative"
    first = low.split()[0].strip(".,!") if low.split() else ""
    if first and first not in _FUNCTION_STARTS:
        return "instruction"
    return "assertion"


def required_response_type(input_text: str, speech_act: str) -> str:
    if speech_act != "question":
        return _ADJACENCY[speech_act]
    if _INSTRUCTION_RE.search(input_text.lower()):
        return "instruction"
    if _YES_NO_RE.search(input_text.strip()):
        return "assertion"
    return "explanation"


# ---------------------------------------------------------------- domain constraint

def _active_domains(input_heads, records):
    """Discrete set of domain tags present in the input headwords' senses."""
    doms = set()
    for h in input_heads:
        for s in records[h]["senses"]:
            if s["domain"]:
                doms.add(s["domain"].strip().lower())
    return doms


# ---------------------------------------------------------------- main entry point

def analyze(input_text, context_texts, sub,
            max_layers=6, frontier_cap=400,
            intent="", goal="", setting="", audience="",
            checkpoint="listener_checkpoint.json") -> ListenerState:
    """
    Full analysis pipeline. Returns a ListenerState.

    Takes an already-loaded substrate dict (sub) rather than a path, so
    the caller can load once and share it with the generator.
    """
    records = sub["records"]
    function_words = set(sub["relations"]["function_words"])

    # ---- Stage 1: Communicative frame — discourse before content
    speech_act = classify_speech_act(input_text)
    response_type = required_response_type(input_text, speech_act)

    frame = CommunicativeFrame(
        speech_act=speech_act,
        response_type=response_type,
        intent=intent,
        goal=goal,
        setting=setting,
        audience=audience,
    )

    # ---- Stage 2: Activation fields
    input_field = set()
    for surface in words(input_text):
        head = resolve(surface, records)
        if head and head not in function_words:
            input_field.add(head)

    context_field = set()
    for text in context_texts:
        for surface in words(text):
            head = resolve(surface, records)
            if head and head not in function_words:
                context_field.add(head)

    fields = ActivationFields(
        input_field=input_field,
        context_field=context_field,
    )

    # ---- Stage 3: Structural traversal (outward through substrate)
    trace = discover(
        input_text, context_texts, sub,
        max_layers, 0, frontier_cap, checkpoint,
    )

    # ---- Stage 4: Discrete constraints → closure
    #  Closure = Activated Entries ∩ Accumulated Constraints
    input_heads = [h for h, a in trace["activations"].items() if a["via"] == "input"]
    allowed_domains = _active_domains(input_heads, records)

    shaped = FILTER(
        set(trace["activations"]),
        lambda h: (response_type != "instruction"
                   or any(_ACTION_RE.match(s["definition"])
                          for s in records[h]["senses"])),
    )
    closure = CONSTRAIN(shaped, allowed_domains, records)

    return ListenerState(
        frame=frame,
        fields=fields,
        closure=closure,
        allowed_domains=allowed_domains,
        trace=trace,
    )
