# ACS StereoMolGraph whole-molecule chirality benchmark

This directory vendors the 258-case “Validation enantiomer” CSV published as
supporting information for Papusha and Leonhard, *StereoMolGraph:
Stereochemistry-Aware Molecular and Reaction Graphs*, J. Chem. Inf. Model.
**66** (2026), 3830–3839.

- Article DOI: <https://doi.org/10.1021/acs.jcim.5c02523>
- Dataset DOI: <https://doi.org/10.1021/acs.jcim.5c02523.s002>
- License: CC BY-NC 4.0; this third-party CSV is not covered by SynKit's
  project license.
- Publisher MD5: `17da48f77988c6616e62ab92a5d3453c`
- SHA-256: `b90d64bba99d36f0be2429cad255e7836b244dfc26e7c9b4281b36d9ed51fff0`

The metadata JSON records the complete provenance and license boundary. The
benchmark JSON freezes one SynKit and direct StereoMolGraph run. Reproduce it
with:

```text
conda run -n synkit python \
  Test/Chem/Molecule/benchmark_molecular_chirality.py \
  --stereomolgraph /tmp/StereoMolGraph \
  --output Data/Benchmark/Stereo/ACS-StereoMolGraph/published_chirality_benchmark.json
```

## Interpretation

The published StereoMolGraph column agrees with the manual label on all 258
cases. The original SynKit probe agreed on only 235 cases, exactly matching
the CSV's RDKit-SMILES aggregate boundary. The disagreement had two independent
causes:

1. **Incomplete molecular stereo topology (20 false-achiral cases).** The old
   probe reflected only local descriptors retained by RDKit. RDKit removes the
   local `@` tags for several cages and globally stereogenic frameworks because
   an individual atom is not a conventional local tetrahedral stereocentre.
   StereoMolGraph's published protocol uses `stereo_complete=True`: it places
   provisional orientations on every eligible sp3 topology and lets the
   whole-molecule automorphism decide whether those probes cancel. Adding the
   same molecule-level completion recovers all 20 cases.
2. **Lewis-state identity instead of the publisher's molecular identity (five
   symmetry cases).**
   SynKit's ordinary stereo graph equality intentionally distinguishes charge,
   bond order, lone-pair state, and one Kekulé/resonance form because those are
   important for reactions. That is too strict for this molecular task. It
   blocks mirror automorphisms accepted by the publisher protocol for `VS042`,
   `VS044`, `VS170`, `VS215`, and `VS216`. The molecule classifier instead
   matches element, total hydrogen count, and connectivity. This is equivalent
   to StereoMolGraph's atom-type graph with explicit hydrogens while avoiding
   explicit-H automorphism expansion. It is a declared topology convention,
   not a general claim that every distinct bond-order assignment is chemically
   equivalent; `VS170` specifically depends on that boundary.

The dedicated `synkit.Chem.Molecule.chirality` classifier now agrees with the
manual and published StereoMolGraph labels on 258/258 cases. It is separate
from the Lewis-state identity used elsewhere in SynKit.

The central task-aware report additionally records the publisher's RDKit-SMILES
column at 235/258, InChI at 238/258, and chython at 210/258. Those columns are
immutable published results, not live reruns of later backend versions.

## Efficiency

Two molecule-specific optimizations retain exact authority:

- a safe 1-WL stereo-colour prefilter resolves 129/258 cases without entering
  automorphism search;
- the remaining VF2 search checks every fully mapped local stereo frame during
  expansion instead of waiting for a complete whole-graph mapping.

In the frozen environment (Python 3.11.0, RDKit 2025.09.3), the optimized
SynKit classifier took 1.683 s and the live StereoMolGraph checkout took
1.780 s. The 0.946 SynKit/StereoMolGraph ratio makes SynKit about 5% faster
for this run while preserving 258/258 accuracy. Timing includes backend
conversion and mirror equality but excludes SMILES parsing and native input
preparation. The live checkout at commit
`2189f610f23eaaf992e2e01a12ea4d0532496601` reproduced 254/258 rather than the
published 258/258: `VS246`, `VS247`, `VS248`, and `VS299` changed from achiral
to chiral. That drift is reported separately and does not replace the
publisher labels.

## Removing SMILES stereo flags

Removing all atom and bond stereo flags reduces this classifier to 223/258
(86.43%): 142 true chiral, 81 true achiral, 22 false achiral, and 13 false
chiral. More importantly, the stripped corpus contains six constitutional
SMILES groups that each contain both manually chiral and manually achiral
stereoisomers. No deterministic classifier receiving only those stripped
inputs can distinguish the members. Even an oracle that memorizes the majority
label for every stripped constitution is bounded at 250/258 (96.90%) on this
dataset.

`stereo_complete=True` supplies a provisional orientation for topology probes;
it does not reconstruct erased relative configurations. Stereo-stripped input
should therefore be reported as stereochemically underspecified or as
“potentially chiral,” not as an exact stereoisomer-level chiral/achiral result.

## Sprint 23 configuration-aware assessment

``assess_molecular_chirality`` replaces a forced binary guess for
underspecified input with four outcomes: necessarily chiral, necessarily
achiral, configuration-dependent, or unsupported/incomplete. It enumerates
unique RDKit-supported unassigned tetrahedral and double-bond configurations,
stops as soon as both molecular classes prove configuration dependence, caches
isomer classifications, and fails closed when an explicit ``max_isomers`` cap
cannot prove a one-sided population. ``require_specified=True`` supplies the
corresponding strict binary refusal mode.

With every stereo flag removed from the 258 published inputs and a 256-isomer
cap, 123 rows are necessarily chiral, 65 necessarily achiral, 66
configuration-dependent, and four incomplete. Thus 254/258 receive definitive
set-valued answers without pretending to reconstruct the original isomer; the
manual label lies in the observed completion population for all 258 rows. The
214 unique constitutions required 2,761 classified stereoisomers. A cold pass
took 22.940 s and a repeated cache-warm pass took 1.293 s (0.056 ratio, about
17.7x faster). Seven warmed repetitions of the original specified-input binary
benchmark retained 258/258 accuracy with a 1.678 s median total
(1.535--1.842 s range). Raw evidence is frozen in
``sprint/sprint_23_benchmark.json``.

This is constitutional configuration coverage, not a prediction of which
stereoisomer was synthesized, populated, or experimentally stable. Unresolved
square-planar, TBP, octahedral, cumulene, and atropisomeric input is outside the
enumerator and is reported as unsupported/incomplete.

Here “configuration-dependent” is a proven mixed completion population. The
four incomplete cases are VS226 (1,024 theoretical assignments; six unique
achiral completions seen from the capped search), VS265 (16,777,216; 256
chiral seen), VS266 (4,194,304; 256 chiral seen), and VS268 (33,554,432; 256
chiral seen). Exhausting VS226 at 1,024 resolves it as necessarily achiral;
the frozen common-cap report retains the uniform 256 limit for comparability.

This directory, module, runner, and tests concern molecules only. They do not
import or exercise reaction, rule-extraction, wildcard, or product-generation
code.
