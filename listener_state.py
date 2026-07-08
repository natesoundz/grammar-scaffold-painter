"""
listener_state.py — The central invariant of the compiled listener architecture.

The listener state is the object that separates analysis from generation.
Everything before it is analysis. Everything after it is realization.

  Training:   Content  →  Listener State
  Inference:  Listener State  →  Content

This symmetry is the defining property of the architecture.
"""

from dataclasses import dataclass, field


@dataclass
class CommunicativeFrame:
    """Epilinguistic constraints established before any dictionary activation.

    Discourse is always first. The frame is fixed before any content word
    is touched. These are constraints on interpretation, not content.
    """
    speech_act: str = ""      # question / assertion / instruction / explanation / narrative / counterfactual / hypothesis
    response_type: str = ""   # adjacency constraint: what the listener must produce
    intent: str = ""          # caller-supplied: what the speaker is trying to accomplish
    goal: str = ""            # caller-supplied: what outcome they seek
    setting: str = ""         # caller-supplied: domain context
    audience: str = ""        # caller-supplied: who the listener is addressing


@dataclass
class ActivationFields:
    """The three coexisting activation fields over the dictionary substrate.

    Input field:   literal language supplied by the speaker.
    Context field: headwords activated by the supplied communicative context.
    Output field:  the successful listener realization (populated during training).

    Each field occupies the same substrate coordinate system.
    The substrate does not change. Only which regions are illuminated changes.
    """
    input_field: set = field(default_factory=set)
    context_field: set = field(default_factory=set)
    output_field: set = field(default_factory=set)


@dataclass
class ListenerState:
    """The central invariant. Constructed by the analyzer; consumed by the generator.

    Everything before this object is analysis.
    Everything after this object is realization.

    The closure is: Activated Dictionary Entries ∩ Accumulated Constraints.
    It is the set algebra answer to "what does this communicative situation
    illuminate in the substrate?" — not a ranked list, a set.
    """
    frame: CommunicativeFrame
    fields: ActivationFields
    closure: set          # structural closure under EXPAND ∩ constraints (Layers 0-2, unbounded)
    allowed_domains: set  # discrete domain membership constraint
    trace: dict           # full auditable traversal record (every activation, every parent edge)

    def display(self):
        f = self.frame
        print("=" * 60)
        print("LISTENER STATE")
        print("=" * 60)
        print(f"  SPEECH ACT       {f.speech_act}")
        print(f"  RESPONSE TYPE    {f.response_type}"
              f"  (adjacency constraint, fixed before content)")
        print(f"  INTENT           {f.intent or '(none supplied)'}")
        print(f"  GOAL             {f.goal or '(none supplied)'}")
        print(f"  SETTING          {f.setting or '(none supplied)'}")
        print(f"  AUDIENCE         {f.audience or '(none supplied)'}")
        print(f"  ALLOWED DOMAINS  {self.allowed_domains or '(none tagged in input)'}")
        print(f"  INPUT FIELD      {len(self.fields.input_field)} headwords"
              f"  {sorted(self.fields.input_field)}")
        print(f"  CONTEXT FIELD    {len(self.fields.context_field)} headwords"
              f"  {sorted(self.fields.context_field) or '(none)'}")
        print(f"  VISITED          {self.trace['visited_total']:,} substrate headwords traversed")
        print(f"  CLOSURE          {len(self.closure):,}"
              f"  (structural, unbounded — Layers 0-2)")
        print("=" * 60)
