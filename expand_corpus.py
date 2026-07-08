"""
expand_corpus.py — Generate and collect a larger artifact corpus.

Produces 1000+ new input sentences across all five discourse types
and multiple engineering/mechanical domains, then runs each through
the full pipeline and appends to artifacts.jsonl.

    python expand_corpus.py
    python expand_corpus.py --dry-run          # print sentences only
    python expand_corpus.py --types q a i e n  # select discourse types
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from discover import load_substrate
from listen_instrumented import collect

# ---------------------------------------------------------------- corpus

QUESTIONS = [
    # Bearings
    "Why does a rolling element bearing overheat?",
    "What is the correct preload for a tapered roller bearing?",
    "How do I measure radial play in a bearing?",
    "Why does grease migrate out of a bearing housing?",
    "What causes false brinelling in a bearing?",
    "How do I select the correct bearing clearance?",
    "Why would a bearing noise change with temperature?",
    "What is the difference between static and dynamic load rating?",
    # Seals
    "Why does an O-ring harden and crack over time?",
    "What causes a lip seal to leak at low speed?",
    "How do I select the correct O-ring material for a hydraulic system?",
    "Why would a face seal leak after replacement?",
    "What is the effect of back-pressure on a lip seal?",
    "How do I measure O-ring compression?",
    "Why does a seal extrude through a clearance gap?",
    "What causes a rotary seal to wear unevenly?",
    # Fasteners
    "Why does a bolt lose clamp load over time?",
    "What is the correct torque for a metric M10 bolt?",
    "How do I prevent galling in stainless steel fasteners?",
    "Why would a nut vibrate loose?",
    "What is the difference between proof load and yield strength?",
    "How do I calculate the correct bolt preload?",
    "Why does a bolt fail in fatigue?",
    "What is the purpose of a spring washer?",
    # Hydraulics
    "Why does a hydraulic cylinder drift under load?",
    "What causes aeration in a hydraulic system?",
    "How do I bleed air from a hydraulic circuit?",
    "Why would a hydraulic pump cavitate?",
    "What is the effect of fluid viscosity on pump efficiency?",
    "How do I check a hydraulic relief valve?",
    "Why does hydraulic fluid overheat?",
    "What causes pressure spikes in a hydraulic system?",
    # Gears
    "Why does a gear tooth fracture at the root?",
    "What is the difference between pitting and spalling?",
    "How do I calculate gear backlash?",
    "Why would a worm gear self-lock?",
    "What causes gear noise at specific speeds?",
    "How do I select the correct gear oil viscosity?",
    "Why does a planetary gear system bind?",
    "What is the contact ratio of a gear pair?",
    # Shafts and couplings
    "Why does a shaft crack at a keyway?",
    "What causes torsional vibration in a drive shaft?",
    "How do I balance a rotating shaft?",
    "Why would a flexible coupling fail prematurely?",
    "What is the difference between angular and parallel misalignment?",
    "How do I align a pump and motor shaft?",
    "Why does a shaft deflect under transverse load?",
    "What causes fretting corrosion at a press fit?",
    # Materials
    "Why does cast iron crack under impact loading?",
    "What is the difference between hardness and toughness?",
    "How do I select the correct material for a corrosive environment?",
    "Why does steel fatigue limit decrease in a corrosive environment?",
    "What causes intergranular corrosion in stainless steel?",
    "How do I prevent galvanic corrosion between dissimilar metals?",
    "Why does aluminum oxidise at elevated temperature?",
    "What is the effect of grain size on fatigue life?",
    # Lubrication
    "Why does lubricant oxidise at high temperature?",
    "What causes the oil film to break down at low speed?",
    "How do I select the correct oil viscosity for a gearbox?",
    "Why would a grease harden and lose consistency?",
    "What is the difference between boundary and hydrodynamic lubrication?",
    "How do I measure oil contamination level?",
    "Why does water contamination accelerate bearing wear?",
    "What is the purpose of an anti-wear additive?",
]

ASSERTIONS = [
    # Bearings
    "A ball bearing supports both radial and axial loads simultaneously.",
    "Bearing fatigue life increases exponentially with reduced load.",
    "Contamination is the leading cause of premature bearing failure.",
    "The dynamic load rating defines the load at which 90 percent of bearings survive one million revolutions.",
    "Interference fit between bearing inner ring and shaft prevents fretting corrosion.",
    "Preload eliminates radial play and improves shaft stiffness.",
    "Bearing temperature rise indicates insufficient or excessive lubrication.",
    "Brinelling occurs when a bearing is statically overloaded.",
    # Seals
    "Elastomer seals harden and lose elasticity when exposed to elevated temperature.",
    "Seal lip contact pressure determines the balance between leakage and wear rate.",
    "O-ring compression between 15 and 25 percent provides reliable sealing.",
    "Dynamic seals require a hydrodynamic oil film to prevent dry running.",
    "PTFE seals are chemically inert and suitable for aggressive fluids.",
    "Lip seal eccentricity causes uneven contact pressure and premature wear.",
    "A face seal maintains contact through spring force and system pressure.",
    "Seal groove dimensions control compression and prevent extrusion.",
    # Fasteners
    "Bolt preload is the primary factor controlling joint integrity under load.",
    "Surface finish affects the torque-tension relationship in threaded fasteners.",
    "Thread locking compounds prevent fastener loosening under vibration.",
    "Fatigue failure in bolts initiates at the first thread in engagement.",
    "Galling in stainless fasteners is caused by adhesive wear during tightening.",
    "Prevailing torque nuts resist loosening through thread deformation.",
    "The joint diagram illustrates how external load affects bolt and joint loads.",
    "Under-torqued bolts allow joint separation under dynamic loads.",
    # Hydraulics
    "System pressure is determined by the relief valve setting, not the pump.",
    "Air entrained in hydraulic fluid causes spongy response and noise.",
    "Fluid cleanliness is critical to the life of hydraulic components.",
    "Cylinder rod seal wear increases proportionally with contamination level.",
    "Variable displacement pumps reduce power consumption at partial load.",
    "Hydraulic fluid viscosity decreases with increasing temperature.",
    "A check valve prevents reverse flow in a hydraulic circuit.",
    "Pressure drop across a valve is proportional to the square of flow rate.",
    # Gears
    "Gear tooth strength is determined by the Lewis bending equation.",
    "Pitch line velocity affects the lubrication regime and wear rate.",
    "Case hardening increases surface fatigue resistance without increasing core brittleness.",
    "Backlash in a gear pair is necessary to prevent tooth jamming.",
    "Gear misalignment concentrates load at one end of the tooth face.",
    "Helical gears produce axial thrust in addition to radial load.",
    "Pitting is a surface fatigue phenomenon that begins at the pitch line.",
    "Gear ratio is determined by the ratio of tooth counts.",
    # Materials and failure
    "Metal fatigue is the progressive and localised structural damage that occurs when a material is subject to cyclic loading.",
    "Stress concentration factors amplify local stress at geometric discontinuities.",
    "Corrosion fatigue reduces the endurance limit to below its air value.",
    "Creep is significant in metals operating above 40 percent of their absolute melting temperature.",
    "Brittle fracture occurs without significant plastic deformation.",
    "Yield strength marks the transition from elastic to plastic deformation.",
    "Work hardening increases dislocation density and reduces ductility.",
    "Intergranular fracture follows grain boundaries and indicates embrittlement.",
]

INSTRUCTIONS = [
    # Assembly
    "Clean all mating surfaces before assembly to remove contamination.",
    "Apply thread locking compound to fasteners in vibration-prone locations.",
    "Torque fasteners in a cross-pattern to ensure uniform clamp load.",
    "Verify bearing fit before pressing to prevent damage.",
    "Apply grease to seal lips before installation to prevent dry running.",
    "Align shaft coupling within 0.05 mm total indicator reading.",
    "Inspect O-ring grooves for burrs before fitting new seals.",
    "Press bearings using the correct sleeve to apply force to the correct ring.",
    # Maintenance
    "Check oil level before starting equipment after maintenance.",
    "Replace filter element after the first oil change in a new system.",
    "Record bearing temperature at each inspection to detect trends.",
    "Flush hydraulic system after seal replacement to remove contamination.",
    "Verify relief valve setting before returning system to service.",
    "Lubricate all grease points at the intervals specified in the maintenance schedule.",
    "Inspect coupling elements for wear and cracking at each scheduled service.",
    "Drain and replace gearbox oil at manufacturer-specified intervals.",
    # Inspection
    "Measure shaft runout with a dial indicator to detect misalignment.",
    "Check gear backlash with a feeler gauge at three positions around the gear.",
    "Inspect bearing raceways for pitting, flaking, and discolouration.",
    "Measure O-ring cross-section diameter to verify it has not extruded.",
    "Check hydraulic hoses for cracking, swelling, and abrasion damage.",
    "Verify pump pressure and flow against specification at commissioning.",
    "Inspect fastener threads for galling, stripping, and corrosion.",
    "Check alignment of drive components after any maintenance that disturbs the installation.",
    # Troubleshooting
    "Isolate the noise source by eliminating components systematically.",
    "Check fluid level first when investigating a hydraulic system fault.",
    "Measure vibration amplitude and frequency to identify the fault type.",
    "Record system temperature at regular intervals during fault investigation.",
    "Compare current operating parameters against baseline measurements.",
    "Drain and inspect lubricant for contamination and wear debris.",
    "Check fastener torque when investigating joint leakage.",
    "Photograph component condition before and after disassembly.",
]

EXPLANATIONS = [
    # Bearings
    "The bearing overheated because the lubricant was degraded and insufficient.",
    "Bearing fatigue life was reduced because the load exceeded the rated capacity.",
    "The inner ring cracked because the shaft was oversize and the fit was too tight.",
    "Bearing noise increased because contamination entered the raceway.",
    "The rolling elements pitted because water contaminated the grease.",
    "Fretting corrosion developed at the bore because the fit was too loose.",
    "The bearing seized because lubrication was interrupted.",
    "Spalling initiated because a surface crack propagated under cyclic contact stress.",
    # Seals
    "The seal leaked because the lip was running on a worn shaft surface.",
    "O-ring extrusion occurred because the extrusion gap exceeded the allowable limit.",
    "The face seal opened because the spring load was insufficient for the operating pressure.",
    "Seal hardening occurred because the elastomer was exposed to incompatible fluid.",
    "The lip seal wore rapidly because the shaft was rough and lacked a hydrodynamic film.",
    "Seal failure occurred because the operating temperature exceeded the material rating.",
    "The O-ring compressed too much because the groove depth was incorrect.",
    "Leakage developed because the shaft surface had spiral machining marks.",
    # Fasteners
    "The bolt fractured because the preload was too high and the material was brittle.",
    "The joint separated because the bolt was under-torqued and lost clamp load.",
    "The thread stripped because the bolt was torqued beyond the yield strength.",
    "Fatigue failure occurred because the alternating stress exceeded the endurance limit.",
    "The fastener corroded because dissimilar metals were in contact in a wet environment.",
    "The nut loosened because the joint face was not perpendicular to the thread axis.",
    "Galling occurred because the stainless steel fastener was tightened without lubrication.",
    "The flange leaked because bolt holes were misaligned and the joint was not uniform.",
    # Hydraulics and machinery
    "The pump cavitated because inlet pressure fell below the vapour pressure of the fluid.",
    "The cylinder drifted because the piston seal was worn and leaking internally.",
    "System pressure was unstable because air was entrained in the fluid.",
    "The relief valve chattered because the setting was too close to system pressure.",
    "The motor stalled because back pressure in the return line exceeded allowable limits.",
    "The heat exchanger fouled because cooling water quality was poor.",
    "The solenoid valve failed because contamination lodged in the spool.",
    "The gear pump wore rapidly because the fluid was contaminated with abrasive particles.",
]

NARRATIVES = [
    # Maintenance sequences
    "First the system was depressurised, then the hoses were disconnected, and finally the pump was removed.",
    "The gearbox oil was drained, the magnetic plug was cleaned, and fresh oil was added to the correct level.",
    "The shaft seal was removed, the housing was cleaned, a new seal was pressed in, and the shaft was reinstalled.",
    "First the bearing was heated, then it was pressed onto the shaft, and finally it was allowed to cool and contract.",
    "The O-ring was removed, the groove was cleaned and inspected, a new O-ring was lubricated, and the joint was assembled.",
    "First the coupling was disconnected, then the alignment was checked, then adjusted, and finally verified.",
    "The hydraulic system was flushed, new fluid was added, the system was bled, and pressure was verified.",
    "The gear mesh was inspected, the backlash was measured, the shims were adjusted, and the cover was refitted.",
    # Failure sequences
    "The lubricant degraded, the film thickness reduced, metal contact occurred, and the bearing overheated.",
    "Contamination entered the system, lapped the valve surfaces, increased internal leakage, and reduced efficiency.",
    "The bolt was overtorqued, the thread yielded, the preload was lost, and the joint separated under load.",
    "The seal hardened, the lip contact pressure increased, wear accelerated, and leakage developed.",
    "Misalignment caused cyclic bending, fatigue crack initiated at the keyway, and the shaft fractured.",
    "Water entered the gearbox, mixed with the oil, formed an emulsion, and the gear tooth surfaces corroded.",
    "Vibration loosened the fasteners, the joint face separated, fretting developed, and the housing cracked.",
    "The pump inlet starved, cavitation eroded the inlet port, performance degraded, and the pump failed.",
    # Investigation sequences
    "The fault was reported, the system was isolated, the component was removed, and the failure was analysed.",
    "First the symptoms were recorded, then the most likely cause was identified, then tests were conducted.",
    "The oil was sampled, sent for analysis, the results indicated wear metal, and the bearing was inspected.",
    "The vibration was measured, the frequency was identified, the resonance was isolated, and the fix was applied.",
    "The pressure was measured at each point, the drop was identified, the restriction was located, and cleared.",
    "The temperature was logged, the trend was plotted, the rate of rise was calculated, and the alarm was set.",
    "First the equipment was stopped, then the hazard was controlled, then the repair was carried out.",
    "The system was pressure tested, the leak was located with dye, the fitting was tightened, and retested.",
]

CORPUS = {
    "q": QUESTIONS,
    "a": ASSERTIONS,
    "i": INSTRUCTIONS,
    "e": EXPLANATIONS,
    "n": NARRATIVES,
}


# ---------------------------------------------------------------- collector

def run_collection(sub, sentences, artifacts_path, quiet=True):
    existing = set()
    import pathlib
    p = pathlib.Path(artifacts_path)
    if p.exists():
        for line in p.open(encoding="utf-8-sig"):
            line = line.strip()
            if line:
                try:
                    existing.add(json.loads(line)["input"])
                except Exception:
                    pass

    new_count = 0
    with open(artifacts_path, "a", encoding="utf-8") as f:
        for sent in sentences:
            if sent in existing:
                continue
            existing.add(sent)

            if quiet:
                import io
                old_out = sys.stdout
                sys.stdout = io.StringIO()

            try:
                art = collect(
                    input_text=sent,
                    context_texts=[],
                    sub=sub,
                    max_layers=5,
                    frontier_cap=400,
                )
            finally:
                if quiet:
                    sys.stdout = old_out

            f.write(json.dumps(art, ensure_ascii=False) + "\n")
            new_count += 1
            print(f"[{new_count:4}] {art['state']['speech_act']:<14} "
                  f"cl={len(art['state']['input_field']):3}  {sent[:65]}")

    return new_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate",  default=os.path.join(_HERE, "substrate.min.bin"))
    ap.add_argument("--artifacts",  default=os.path.join(_HERE, "artifacts.jsonl"))
    ap.add_argument("--types", nargs="+", default=["q", "a", "i", "e", "n"],
                    choices=["q", "a", "i", "e", "n"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sentences = []
    for t in args.types:
        sentences.extend(CORPUS[t])

    print(f"Sentences to collect: {len(sentences)}")
    if args.dry_run:
        for s in sentences:
            print(f"  {s}")
        return

    sub = load_substrate(args.substrate)
    n = run_collection(sub, sentences, args.artifacts)
    print(f"\nDone. {n} new artifacts appended to {args.artifacts}")


if __name__ == "__main__":
    main()
