# Live backend disagreement review

This review covers the SynKit/StereoMolGraph disagreements in
`backend_comparison_report.json`. A disagreement is not automatically a SynKit
failure. ACS supplies published global truth. RotA is a typed positive-locus
task and CIP is a local descriptor-assignment task; neither supplies compatible
global binary truth.

The exploratory fixed-orientation cumulene/biaryl enhancement was unsound.
Its ``488/650`` RotA and ``298/300`` CIP claims are retracted. Connectivity may
emit a potential locus, but it cannot provide handedness or a rotational-
stability conclusion.

## Case-level conclusions

| Cases | SynKit status after enhancement | Evidence and conclusion |
| --- | --- | --- |
| VS175, VS177, VS178, VS179, VS183, VS186 | fixed | Isotope is now part of molecular identity, so isotope-defined tetrahedral chirality is no longer collapsed. |
| VS079, VS141, VS144, VS166, VS287 | potential only | Even consecutive-double-bond paths are reported as orientation-unspecified cumulene loci. They do not participate in binary classification without configured evidence. |
| VS023, VS055, VS057, VS073, VS086, VS158 | potential only | Inter-ring aromatic bonds are reported as orientation-unspecified, stability-unassessed atrop-axis candidates. Connectivity alone does not resolve these cases. |
| VS010, VS011 | still unsupported | These opposite helicene records parse to exactly the same canonical SMILES and RDKit exposes no potential stereo locus or chiral tag. A molecule-only topology/helicity model is needed; assigning either handedness from this input would be fabricated. |
| VS180, VS181, VS182, VS187 and stripped VS119 | live SMG limitation | Live StereoMolGraph constructs atoms from element symbols and drops isotope mass. SynKit retains the isotope distinction. |
| VS246, VS247, VS248, VS299 | live SMG regression | The immutable ACS manual and published StereoMolGraph columns label all four achiral. SynKit and RDKit agree; the tested live StereoMolGraph revision changed them to chiral. |

## Accuracy that is scientifically defined

| Dataset/task | SynKit | RDKit | StereoMolGraph |
| --- | --- | --- | --- |
| ACS supplied global binary | 258/258 | 235/258 | 254/258 live; 258/258 published column |
| ACS removed versus original label | 223/258 apparent agreement | 94/258 apparent agreement | 220/258 apparent agreement |
| RotA native axial-locus detection | N/A: no axis detector | N/A: no axis detector | N/A: no axis detector from these SMILES |
| CIP native local descriptor assignment | 155/300 exact label sets (51.67%); incremental Rules 1a/1b/2 | 245/300 exact label sets (81.67%) | N/A: no CIP-label API |

The ACS removed numbers are not recovery accuracy because stereo information
was erased. RDKit CIP micro label recall is 90.18% and precision is 99.82%
over 1,252 reference labels. SynKit recall is 67.33% and precision is 97.80%; its
145 non-exact rows have one reviewed primary cause each: 15 unsupported class,
27 missing orientation evidence, 101 ranking defects, and 2 label-projection
defects. Most missing label sets involve helical,
atropisomeric, extended tetrahedral/cis-trans, pseudoasymmetric, or later
globally stereogenic validation cases.

Exact-record coverage by every stereo-unit tag present in a row is:

| Unit tag | SynKit | RDKit |
| --- | ---: | ---: |
| TH | 124/249 | 214/249 |
| CT | 41/65 | 54/65 |
| HE | 0/2 | 0/2 |
| AT | 0/7 | 0/7 |
| CT4 | 0/5 | 0/5 |
| TH3 | 0/8 | 0/8 |
| TH5 | 0/2 | 0/2 |

Mixed-unit rows contribute to every tag they carry, and exactness requires the
entire expected label set. The zeroes therefore expose unsupported extended,
helical, and atropisomeric assignment rather than parser failure.

RotA is a positive-only axis corpus. Reporting the fraction classified globally
chiral as “accuracy” would conflate a 2D constitution with a specified
atropisomer and would supply no true-negative information.

## Efficiency

The old RotA/CIP timing rows measure the retracted fixed-orientation method and
are not current performance evidence. New timings require a rerun of the sound
potential-locus/configured-descriptor implementation under its native tasks.

RDKit native CIP assignment takes 0.710 s in the frozen report (2.37 ms per
input); eleven independent passes have a 0.682 s median. The reviewed SynKit
native pass takes 10.172 s (33.91 ms/input), including descriptor extraction,
independent ligand evidence, all pairwise comparisons, and typed diagnostics.

## Required implementation follow-up

1. Add a topology-aware helicene locus model and an input representation that
   can retain or derive helicity. VS010/VS011 prove that ordinary SMILES alone
   cannot distinguish opposite `M/P` configurations.
2. Do not use direct binary `stereo_complete` classification for stripped
   input. VS229 proves that a provisional orientation can fall outside the
   valid enumerated completion population. Use the four-state assessment.
3. Extend the four-state enumerator to include candidate cumulene and biaryl
   axes instead of applying one provisional orientation. Binary completion is
   accurate on supplied CIP input but is not a proof over erased assignments.
4. Implement task-specific RotA atom-pair/chain output plus a calibrated
   rotational-stability model. Global mirror classification and a topological
   candidate axis are not substitutes for the native task.
5. Replace the flattened ligand-sphere comparison with hierarchical branch
   comparison, implement Rules 3--5 and extended classes, and correct the three
   reviewed projection/adapter defects. Do not relabel RDKit-derived CIP
   properties as independent SynKit output.
