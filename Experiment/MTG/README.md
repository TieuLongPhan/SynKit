# MTG validation experiment

The Article results figures can be regenerated from the existing compact
reports without rerunning chemistry or native kernels:

```bash
MPLCONFIGDIR=/tmp/mtg-article-mpl python -m Experiment.MTG.article_results_assets
```

This writes four vector PDFs to `paper/mtg/fig/` and a complete plotted-data
record to `paper/mtg/results/article_figure_data.json`. The record pins its ten
source reports, renderer and output figures; it retains the individual timing
rounds, counts and actual source states. Corpus bars overlap, configurations
are not populations, and the scaling comparison excludes common source work
and linear-coefficient derivation. These figures present the completed corpus,
transferase, isotope and scale experiments. The underlying protocols are
unchanged. Matplotlib and NumPy are the renderer's only scientific dependencies.

This experiment exercises native rule composition, occurrence processes, and
mechanistic transition graph derivation with bounded deterministic fixtures.
The output schema is `synkit.mtg-validation/1`.

## Run the validation

From the repository root, run:

```bash
python Experiment/MTG/validation.py --output mtg-validation.json
```

The command exits with status zero only when every contract passes. The report
records the Python and NetworkX versions, input digest, search limits, timing
and memory observations, individual checks, and curated case-study summaries.

The fixture checks finite-graph construction and replay properties. Timing and
memory values are operational measurements; they do not establish chemical
kinetics, energetics, yield, or mechanism preference.

## Render tabular and LaTeX assets

A passing evidence file can be converted to stable paper inputs:

```bash
python Experiment/MTG/paper_assets.py mtg-validation.json \
  --latex-output mtg-results.tex \
  --timing-output mtg-timings.tsv
```

`paper_assets.py` rejects evidence with an unsupported schema or a failing
status. The generated files should be regenerated whenever the validation
fixture, search limits, or runtime environment changes.

## SynEPD construction research

The construction audit uses an explicit external SynEPD input and records its
checksum. It compares ordered single-arrow replay with one common-prestate
group, enumerates all contiguous partitions for records with at most four
arrows, and checks whole-state DPO and occurrence constructions. Source
endpoint representation declarations are retained. The revised construction
report uses schema version 2 and checks finite natural DPO squares. Electron
replay, square certification and chemical interpretation remain distinct.

```bash
python Experiment/MTG/synepd_construction.py ../SynEPD/data/polar.json \
  --output /tmp/mtg-synepd-post-repair-full.json \
  --summary-output Experiment/MTG/evidence/synepd_post_repair_summary.json
python Experiment/MTG/grouping_dag.py ../SynEPD/data/polar.json \
  /tmp/mtg-synepd-post-repair-full.json \
  --output /tmp/mtg-grouping-post-repair-full.json \
  --summary-output Experiment/MTG/evidence/grouping_dag_post_repair_summary.json
python Experiment/MTG/partial_attribute_audit.py \
  --output Experiment/MTG/evidence/partial_attribute_audit.json
```

`grouping_dag.py` counts fixed-order contiguous groupings using passing ordered
prefix states. It verifies every admitted interval against its expected prefix
state and checks exact grouping sets against the exhaustive small-record audit.
It also counts the family remaining when a hydrogen duet bound is imposed at
group boundaries and checks may/must boundary queries. This is a restricted research prototype using a private
replay API. It does not infer elementary steps, certify chemical feasibility,
handle arbitrary arrow reorderings, or establish general categorical concurrency.

`partial_attribute_audit.py` exhausts a one-carrier, two-coordinate label model.
It tests the local union and intersection conditions for natural label squares;
it is not a proof of general graph adhesivity or concurrency.

Generated evidence stays outside upstream source tests. The detailed reports
go to `/tmp`; compact evidence in this experiment is currently ignored by Git.
Current interpretation and draft links are in `paper/mtg/DRAFT_STATUS.md`.
The earlier research notes retain the findings from before the repair.

## Complete selected cases

The three supplied records are proton transfer (1), general SN2 (67) and aldol
addition (475). Each has an ordered and a common-prestate interpretation.
Export their nine natural-square certificates and compare MTG queries with the
equal-information mapped trajectories:

```bash
python Experiment/MTG/verified_cases.py ../SynEPD/data/polar.json \
  --output Experiment/MTG/evidence/verified_cases.json \
  --summary-output Experiment/MTG/evidence/comparison_summary.json
python Experiment/MTG/verify_exported_cases.py \
  Experiment/MTG/evidence/verified_cases.json
```

The verifier rebuilds certificates from disk without accessing SynEPD. It checks
the square equations and stored query equality. Recomputing queries from source
and MTG belongs to the construction command. This selected-case serialization
supports its integer atom carriers and closed electron rules; it is not a
general serializer for arbitrary compound carriers or guarded rules.

An actual MØD native-rule-engine comparison now agrees on the selected states,
including the two independent-observation isotope candidates. The actual
published overlay and supplied-map CAT implementations are now also compared
below, including native KaStor causal-history reduction on the formal N-poset
and inverse-cycle panel. Broader chemical and flux-reduction comparisons remain open.
The paper's comparison note records what was actually run and its limits.

## Partial pattern API

Rule interfaces and contexts now use the endpoint schema with `partial=True`.
Undefined semantic coordinates are absent keys; unchanged coordinates can be
retained. Consumers should inspect label definedness rather than expecting
`interface.schema.node_state == ()`. The square checker compares exact raw
coordinates; supported aromatic phase equivalence in generic graph matching is
not silently promoted into a categorical map.

Every constructed composite retains a sequential execution condition. Colored
span isomorphism compares its carrier span; it does not compare conditions.
Retain each condition-bearing witness when quotienting spans.

## Tests

The lightweight contracts are covered by:

```bash
python -m pytest Test/Benchmark/test_mtg_validation.py \
  Test/Benchmark/test_mtg_paper_assets.py
```

Reversing a guarded rule requires an inverse condition and is refused by default.
`reverse_rule(rule, span_only=True)` explicitly projects a bare reversed carrier
span for structural comparisons; it does not invert its execution domain.

## Isotope observation and actual MØD

```bash
python Experiment/MTG/isotope_case.py \
  --output Experiment/MTG/evidence/isotope_case.json
/home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/mod_history_comparison.py \
  Experiment/MTG/evidence/verified_cases.json \
  --isotope-cases Experiment/MTG/evidence/isotope_case.json \
  --output Experiment/MTG/evidence/mod_history_comparison.json
```

The isotope schema preserves declared mass and isotope-resolved open tokens.
Mass zero is unspecified; labelled hydrogen must be explicit. The independently
sourced qualitative observation scores two supplied reconstructions; their event
groups are not experimentally validated. MØD 1.0.0.7 receives coordinate-incidence
graphs retaining full labels, partial interfaces and anchors. Thirteen transitions
and five full-overlap compositions agree. No speed advantage is claimed.

## Released FlowER example and checkpoint calibration

Pinned external source trees, data archives and weights remain in `/tmp`.
The review package does not bundle them. Acquire official releases 2.0.0 and
1.0.0 and the DOI `10.6084/m9.figshare.28359407.v3`; verify the digests reported in
`paper/mtg/INDEPENDENT_COMPARISONS.md`. The original small model must use release
1.0.0: its pickled settings class otherwise resolves against different code.

```bash
python Experiment/MTG/flower_history_audit.py \
  /tmp/mtg-flower-published/examples/samples/unrecognized_pistachio.pkl \
  --output Experiment/MTG/evidence/flower_example_audit.json
PYTHONPATH=/tmp/mtg-flower-deps python Experiment/MTG/flower_checkpoint_audit.py \
  --flower /tmp/mtg-flower-original \
  --checkpoint /tmp/mtg-flower-published/checkpoints/flower_dataset/best_hyperparam/model.1440000_47.pt \
  --data /tmp/mtg-flower-data.zip \
  --cohort Experiment/MTG/evidence/flower_cpu_cohort.json \
  --output Experiment/MTG/evidence/flower_checkpoint_calibration.json
PYTHONPATH=/tmp/mtg-flower-deps python Experiment/MTG/flower_isotope_beam.py \
  --flower /tmp/mtg-flower-original \
  --checkpoint /tmp/mtg-flower-published/checkpoints/flower_dataset/best_hyperparam/model.1440000_47.pt \
  --protocol Experiment/MTG/evidence/flower_isotope_beam_protocol.json \
  --output Experiment/MTG/evidence/flower_isotope_beam.json
```

`/tmp/mtg-flower-deps` contains the isolated `torchdiffeq==0.2.4` and
`iteround==1.0.4` dependencies; reports record the other installed versions.
The fixed cohort is saved before inference. Raw rounded matrices are retained
separately from decoder outcomes. Teacher forcing, file-order selection and
changed dependency versions are disclosed compatibility limits. These scripts
are research experiments, not upstream source CI or physical mechanism oracles.
The bounded independent-case search does not reach phenyl benzoate; its product
observation is therefore not evaluable.


## Core graph composition and partial-order witnesses

The original theory is audited in `paper/mtg/ORIGINAL_THEORY_AUDIT.md`.
Formal note 09 supplies the core definitions and proofs; SynEPD is a linear
construction case study. These new scripts exercise actual native objects and
independent resource oracles, without adding paper fixtures to source CI:

```bash
python Experiment/MTG/partial_order_construction.py \
  --output Experiment/MTG/evidence/partial_order_construction.json
python Experiment/MTG/same_molecule_parallel.py \
  --output Experiment/MTG/evidence/same_molecule_parallel.json
python Experiment/MTG/poset_inventory.py \
  --output Experiment/MTG/evidence/poset_inventory.json
```

The bound N-poset has four events, five schedules and eight configuration cuts.
The public configuration API directly decodes all eight worlds, checked against
independent active-material unions. All 25 schedule-cut observations agree.
It is generic attributed-graph evidence, not a physical reaction mechanism.
The connected-molecule Lewis-state witness checks four closed applications of
two independent homolyses with equal anchored endpoints. Nonlocal radical
conditions give an explicit refused countercase. It does not assign two
consumers to a whole-material occurrence or claim simultaneous physical rates.
The public additive inventory API is exercised against constrained minimum cut to exhaustive
ideals and all schedule prefixes: 600 coordinate comparisons on 300 seeded
small instances agree. The all-schedule bound is different from the Pareto
family of stock sufficient for some schedule. This applies established maximum
closure, without claiming a new flow algorithm or solving graph-match admission.


The public contracts are documented in `paper/mtg/PARTIAL_ORDER_API.md`.
`OccurrenceMTG.configuration_cut(ideal)` returns the decoded world and active
material/carrier correspondence without selecting a schedule. Boundary
projection reads the empty and full configurations directly. Replay refuses
nonideal sets, incorrect preservation partitions or edge endpoints, and
inconsistent derived electron state. `minimum_initial_stock` accepts integer
resource profiles on a finite DAG and returns exact robust stock with a
positive-bound attaining ideal witness. Source regressions live in `Test/`;
those assertions depend only on source fixtures, not paper or evidence files.

## Open history composition

```bash
python Experiment/MTG/open_history_composition.py \
  --output Experiment/MTG/evidence/open_history_composition.json
```

This command assembles a non-series–parallel history from four separate
fragments using the public `compose_histories` API. Three composition
certificates replay, four local applications validate and all eight worlds
match the directly bound process through explicit origin maps. The graph
example is structural; it has no experimental mechanism or timing claim.
Source regressions also cover admitted associativity/interchange and an
annotation-sensitive guard counterexample to general associative admission.
The operation requires explicit copy isomorphisms and cross-pair evidence,
while general guard congruence and frame locality remain separate premises.

## Actual published catalytic overlay comparison

```bash
git clone https://github.com/JuriKolcak/overlay-graphs.git /tmp/mtg-overlay-graphs
git -C /tmp/mtg-overlay-graphs checkout 56683194d088e60ce033193f3eb9e84ef5a1f89d
/home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/overlay_history_comparison.py \
  --upstream /tmp/mtg-overlay-graphs \
  --output Experiment/MTG/evidence/overlay_history_comparison.json
python Experiment/MTG/overlay_comparison_assets.py \
  Experiment/MTG/evidence/overlay_history_comparison.json \
  --output paper/mtg/fig/overlay_query_witness.pdf
```

The comparison invokes the pinned author implementation without changing its
tracked source. The shared domain includes elements/isotopes, charge, total
bond order, fixed hydrogen counts and supplied original-atom maps. Eight
histories and thirteen steps agree on their projected products and both
isotope-role answers. Two formal closed-electron four-step histories produce
equal actual overlays with different peak-order query answers. Full mapped
trajectories also retain the difference; the witness does not establish a
physical C2 mechanism or superiority to the complete upstream pipeline.
External GPL-3.0 source stays in its own checkout. The report records module
hashes, source states, actual output and separate operational timings.

The partial-order construction also verifies `cooccurrence_configuration` on
all 128 subsets of its seven explicit material copies against eight independent
configuration worlds. This is structural coexistence and a least-ideal witness,
not physical simultaneity; executable prefixes require the locality hypotheses
in formal notes 09 and 11.


## Original construction and complete chemical network case

Formal note 12 proves the original sequence specialization, frame-completion
correspondence and concrete copy-binding multiplicity. The separate finite
audit checks fifty counts and two diagrams whose copy assignments change the
causal order and joint intermediate availability.

```bash
python Experiment/MTG/original_composition_audit.py \
  --output Experiment/MTG/evidence/original_composition_audit.json
python Experiment/MTG/mcsa155_case.py \
  --output Experiment/MTG/evidence/mcsa155_case.json
python Experiment/MTG/mcsa155_network.py \
  --output Experiment/MTG/evidence/mcsa155_network.json
/home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/mcsa155_comparison.py \
  --upstream /tmp/mtg-overlay-graphs \
  --output Experiment/MTG/evidence/mcsa155_comparison.json
python Experiment/MTG/mcsa155_assets.py \
  --output paper/mtg/fig/mcsa155_configuration.pdf
```

Acquire all eighteen external inputs from `mcsa155_protocol.json` at their
stated filenames; the default input directory is `/tmp`. The case command
checks their SHA-256 hashes before construction. Both complete proposals
certify fourteen steps and sixteen worlds. The independent deposited
Asp169–CoA observation distinguishes their equal-boundary histories, and the
actual full overlay gives the same answers. Its actual substrate abstraction
intentionally removes the catalyst anchor. Two-site and serial-reuse native
models check 98 worlds and 160 transitions; real-site independence and kinetics
are not established. Full details are in `paper/mtg/MCSA155_CASE_STUDY.md`.
The drawing adapter is original restricted XML code, not the GPL ChemAxon
converter. Attribution is in `licenses/MCSA-ATTRIBUTION.md`; external source
snapshots remain outside the Git tree and generated reports stay out of CI.

## Published CAT supplied-map comparison

`cat_history_comparison.py` executes the unmodified author Transformation,
GeneratorSet and ProjectedCayleyGraph kernels at the pinned CAT commit.
Native MØD molecular values are exposed through a material-occurrence network
view; supplied atom maps and copy identities are equal to those of the MTG
reference. The isotope cases retain their original isotope schema. Automatic
map inference and native chemical-network generation are not claimed.

```bash
git clone https://github.com/Nojgaard/cat.git /tmp/mtg-cat
git -C /tmp/mtg-cat checkout aeafb3963e72eb0f2b8e873eb812a576a9ab1064
/home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/cat_history_comparison.py \
  --upstream /tmp/mtg-cat \
  --output Experiment/MTG/evidence/cat_history_comparison.json
```

The original formose computation gives six generators, eight states and
fourteen edges, with drawing calls omitted. The frozen protocol includes all
ten selected histories and twenty-seven steps. Every orbit tuple is checked
against every decoded prefix-cut projection: the producer/consumer control
retains exactly 37 of 107. Seventy unrestricted-word tuples are outside these
fixed histories' executable-prefix domain, not physical prediction errors.
Both isotope roles and transferase observation answers already agree before
filtering. Timing records do not establish a speed advantage. The protocol and
report carry input, script, adapter, upstream module and source API checksums.
An altered isotope-input checksum is refused before CAT execution.

See `paper/mtg/formal/13_origin_orbits_and_configuration_control.md` for the
prefix-action and exact liftability proofs, and `licenses/CAT-ATTRIBUTION.md`
for source attribution. The native causal-reduction comparison is recorded below.

## Unanchored incidence conditions and finite completion

`typed_incidence.py` is a finite reference kernel distinct from the anchored
MØD adapter and the production rewrite engine. It implements source/typed
embedding enumerators, nested-condition translation/evaluation, image decoding,
ambient monic DPO and finite risk-pair absence completion for total endpoints.

```bash
python Experiment/MTG/condition_transport_audit.py \
  --output Experiment/MTG/evidence/condition_transport_audit.json
```

The fixed binary-coordinate cohort checks forty partial graphs, 1,600 graph
pairs and 615 monic maps; six condition families give 4,044 comparisons over
231 total hosts. All 692 complements from twenty-one generic spans decode;
154 outputs fail exactly where the source square oracle refuses. Eight malformed
structures refuse. The actual source DPO comparison checks seventeen total-endpoint
rules over 231 hosts: 3,459 admissions and 1,080 refusals, including certificate
replay and a full correspondence for preserved and fresh carriers. Three rules
need absence conditions, rejecting 488 existing-bond matches. Renamed arms,
new atoms and multiple simultaneous risk pairs are included. A context bond
with fresh value 1000 still refuses through the finite completion.

`paper/mtg/formal/14_condition_transport_and_image_closure.md` supplies the
proofs and clarifies why this does not establish arbitrary source composite-span
closure, chemical viability or a general symbolic Shift/Trans implementation.
The JSON records kernel/script and production-source hashes.

## Grounded composition diagrams

`typed_grounded_composition.py` builds the actual overlap and union in the
realized middle world, applies the first span in reverse and the second forward,
and intersects their typed complements. It is independent of `compose_rules`.
It checks representational geometry with an abstract outer resource boundary;
the sequential condition retains the original rule contracts.

```bash
python Experiment/MTG/grounded_composition_audit.py \
  --output Experiment/MTG/evidence/grounded_composition_audit.json
```

The audit includes all injective matches of fifteen independently renamed
rules on fifteen binary total hosts with at most two atoms: 120 first steps
and 1,327 two-step sequences. Every reference outer result preserves supplied
carrier transport. All 268 distinct rule/overlap diagrams match production
L,K,R and both arms, including exact coordinate definedness. Seventeen adjacent
chemical pairs from the exported SynEPD, isotope and transferase histories pass
the same comparison. Generate those three case reports first using the commands
above. Input and source/kernel/script hashes are recorded in the report.

Formal note 15 proves grounded image closure. General ambient overlap enumeration,
symbolic condition compilation, chemical boundary synthesis and physical
validation are outside this audit. The production constructor can refuse a
guard whose required frame is missing from its minimal host.
An explicit three-atom guard case verifies that refusal while the grounded
reference executes on the supplied full host and still refuses without its frame.

## Native causal stories and target-focused compression

`kappa_comparison_protocol.json` freezes released KappaTools v4.1.3.1,
source commit `bafaeb8873cb4e4afd9b22ae312dc93724cfc2fa`, archive and executable
hashes, models and seed cohorts. Download the named Linux archive and extract
its `bin/KaSim` and `bin/KaStor` into a separate directory. Clone the named
tag separately; the script requires its unmodified matching tracked tree.
No upstream source or binaries are bundled here. Our two `.ka` models and
interpreter are original comparison adapters; see `licenses/KAPPA-ATTRIBUTION.md`.

```bash
python Experiment/MTG/kappa_history_comparison.py \
  --upstream /tmp/mtg-kappa-tools --binaries /tmp/mtg-kappa-bin \
  --archive /tmp/mtg-kappa-v4.1.3.1-linux.tar.gz \
  --output Experiment/MTG/evidence/kappa_history_comparison.json
python Experiment/MTG/kappa_story_assets.py
```

Native simulation and none/weak/strong compression are executed in private
temporary directories. A separate interpreter validates emitted concrete
tests/actions; it does not reimplement causal compression. Immutable sites
encode the supplied origin anchors, and free-site tests implement the bound
singleton material ports. There is no sequence controller or physical rate claim.

All 32 base runs cover five schedules. The 96 native stories have the same
N-poset relation and agree with all eight source configuration worlds in
768 comparisons. The separately frozen inverse-cycle extension uses the
same seeds and stops at the first target or sixteen events. Thirty runs reach
the target; seeds 3 and 31 are retained as bounded coverage limits. The full
traces give 290 source certificates. The 30 target causal pasts contain
258 rule events; weak and strong each retain 120. All targets and final worlds
agree. Twenty-eight target traces lose a transient pulse query and change
the preserved-coordinate subobject after reduction. Their original full
native traces retain that answer. This is consistent with goal-focused
compression and establishes no pipeline advantage.

The 90 stories give 498 further primitive certificates; 122 whole-program
contractions and 666 adjacent typed diagrams are checked separately across
original programs and stories. Whole-program checking intersects actual
contexts and replays a flat retained program on this persistent-carrier
profile, rather than recursively expanding every prefix condition. It is
not a general guarded chemical compiler. Native models, traces, stories and
source hashes are retained in the compact JSON report. The vector figure uses
actual story vertex IDs and positive/inhibitory edges for prespecified seed 1;
introductions and the observer are omitted from its displayed rule subgraph.

Formal note 16 gives the proof contracts. Broader independently observed
chemical networks, other reductions and resource-flow evaluation remain open.

## Native resource flux and pointed-marking recovery

`resource_flux_protocol.json` freezes the N-poset, both complete transferase
proposals and the existing two-site/serial-reuse models before native outputs.
Exact downloaded-byte hashes pin the author's Linux fSSA binary, guide and
model archive. No advertised source release/version or seed flag is available.
Acquire the identified external downloads separately; they are not bundled.
Our generated models, adapters and outputs have attribution in
`licenses/FSSA-ATTRIBUTION.md`.

```bash
python Experiment/MTG/resource_flux_comparison.py \
  --binary /tmp/mtg-fSSA_Linux --downloads /tmp \
  --output Experiment/MTG/evidence/resource_flux_comparison.json
python Experiment/MTG/pointed_marking_audit.py \
  --output Experiment/MTG/evidence/pointed_marking_audit.json
```

Each supplied material-copy port is a distinct pointed species and each
event a distinct reaction occurrence. The native simulator performs its
own simulation and producer-bin allocation. Untouched initial frames are
not permitted in the native grammar; the adapter restores them explicitly
from constant shared metadata when decoding graph cuts. With `every 1`,
the native CSV can batch two events, so complete markings are inverted
without inferring native within-block order or times. These corrections
are recorded in `resource_flux_adapter_amendments.json`, preserving the
original cases, rates and run count. Model text and full native CSV/JSON
outputs are retained compactly, with input/script/API/download hashes.

The 160 runs pass 2,368 flux-cell comparisons, 960 recorded-state checks
and 122 structural configuration worlds. Configuration counts per history
are 8, 7, 9, 81 and 17; recorded coverage is only 4, 4, 5, 40 and 9.
The recovered source configurations agree on two-site/reuse thioester
coexistence (true/false). Eight two-site runs record the joint state and
none of the reuse runs does. These are formal construction/coverage counts,
not physical probabilities. Unit rates do not estimate chemistry or validate
site independence; supplied graph values and origins are shared input.

Formal note 17 proves the resource projection, compatibility with valid port
gluing, and a linear incidence-kernel inverse of complete pointed markings.
History validation and graph decoding are separate costs. The independent
N-poset audit checks all 128 binary markings, accepting eight legal cuts and
refusing 120 inconsistent markings. It also checks two configurations with
equal ordinary state inventories, showing why pointed input is stronger.
The inverse implementation is currently an experiment kernel, not a public
source query API. Agreement with native resource flow does not establish
a complete-pipeline or performance advantage.

## Endpoint overlap completeness

Formal note 18 and the supplement characterize total common-world overlaps
before sequential admission. An independent subset/permutation oracle constructs
tagged union graphs and compares the complete concrete mapping sets with source
enumeration: 2,817 graph pairs, 34,885 candidates and 14,011 compatible unions
agree, including every shared bond in the intersection. The through-two-atom
225-pair catalog is exhaustive; the three-atom panels are six-probe comparisons
in each direction. One-sided frame bonds are compatible; conflicting shared
bond values are not. This accompanies a source search repair and three regressions
for frame admission, value conflict and retained composition refusal.

```bash
python Experiment/MTG/overlap_union_audit.py \
  --output Experiment/MTG/evidence/overlap_union_audit.json
```

This checks raw endpoint geometry, not guard synthesis, inferred copy bindings,
complete sequential admission or physical feasibility.

## Local-coordinate histories on connected worlds

`local_coordinate_audit.py` exercises the public
`build_local_coordinate_history` constructor with a connected N-poset, two
and three same-molecule homolyses, and all ten previous selected chemical
interpretations/candidates. Its frozen protocol retains all selected cases.
The complete panel has 13 supported histories, 36 event occurrences,
23 extensions, 57 worlds and 92 prefix checks. Native MØD 1.0.0.7 agrees on
69 admitted applications and 19 refused synthetic prefixes.

```bash
python Experiment/MTG/local_coordinate_audit.py --mod \
  --output Experiment/MTG/evidence/local_coordinate_audit.json
```

The source profile preserves all atoms, requires exact raw matches and
excludes arbitrary guards/open environmental state. Aromatic rules use
conservative complete-world reads. Versioned coordinates admit shared reads
without multiple consumers of one molecule. Direct ideal projections agree
with actual whole-world source execution across every tested schedule.
The first strict report and native frame failure are retained; amendments
explain the wider raw-aromatic profile and restoration of native unused
input components. No physical timing, maximal independence, general unfolding
or runtime superiority is claimed. See formal note 19 and the public API note.

The coordinate observation audit is separate from source CI and retains the
full source-history validation contract. Reproduce it with:

```sh
python Experiment/MTG/observation_projection_audit.py --output Experiment/MTG/evidence/observation_projection_audit.json
```

The frozen protocol covers 14 histories and 65 scopes, including hidden
preconditions, all N-poset state subsets, same-molecule bond subsets and the
existing externally anchored isotope/enzyme questions. Formal note 20 and
`build_observation_projection` describe exact fibers and executable lifts.
No upstream experiment fixture is required by the new source tests.


## Coordinate realization after mixed-network composition

The public `realize_coordinate_history` transports the original material-bound
matches to persistent atom IDs and retains the original port order, including
read-only precedence. It refuses uncovered access dependencies and differing
canonical prefix worlds. Formal note 21 states the source profile and the
reconstruction/locality hypotheses.

```sh
python Experiment/MTG/observation_network_audit.py --output Experiment/MTG/evidence/observation_network_audit.json
```

`observation_network_protocol.json` fixes both complete two-turnover MCSA155
models. Every subset of sixteen events is tested for idealhood. Bond presence,
carbonyl-containing thioester motif and whole-material coexistence are distinct
queries. Every full transition uses actual primitive execution; all observation
fibers and every reduced-extension lift are checked. Full parallel schedules
are not enumerated. Single-replicate timings include every public validation
call and are descriptive local stages, with no external speedup claim.

`observation_network_runtime_amendments.json` records the interrupted initial
run and its profiling diagnosis. LLG now caches only tuple-backed graph indexes;
label dictionaries remain fresh and authoritative tuple replacement invalidates
the indexes. Every certificate and observation validation still executes. The
before/after profiles are diagnostics, not controlled benchmark measurements.
Seven source regressions cover the cache contract; source CI remains independent
of the experiment inputs. External frameworks already answer the selected
questions correctly and remain the stronger comparative baseline obligation.

## Observation interfaces before order composition

Run `/home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/order_interface_audit.py`
from the repository root. The frozen protocol checks every five-event
index-forward DAG presentation and every scope, both brackets on the complete
four-event three-fragment panel, seven negative controls, and the four pinned
complete enzyme observation orders/fibers. Outputs go to
`evidence/order_interface_audit.json`. Chemical world evidence is reused from
the pinned complete mixed audit; no new primitive execution panel is claimed.

`order_interface.py` is experimental and independent of source CI. Summaries
keep all connection endpoints until composition, preserving hidden paths and
cycle detection. They require disjoint original event namespaces and an already
validated bound model. They do not reconstruct atom maps or source programs.

## Strong symbolic observation reference

Run `/home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/symbolic_observation_audit.py`
from the repository root. It requires native `z3-solver` (the executed version
is 5.1.0.0, native Z3 5.1.0). The experimental adapter is outside source CI.
The complete report includes five alternating prepared-query pairs per
question, all chemical truth-table checks, actual distinct witness-prefix
execution, synthetic compiler controls and the actual mapped solver-library
hash. The Linux provenance check uses `/proc/self/maps`.

The full reference retains the original event order and compiles only queried
coordinates. The writer-order variant uses the identical solver/compiler and
source data. Neither enumerates worlds to answer a predicate. Both predicate
polarities are checked; unknown/timeouts are errors. The source construction
and realization are shared measured stages, and duplicate witness validation
is explicitly shared. The initial stage JSON is historical, with superseded
adapter hashes; it is not a controlled second final-pipeline replicate.

## Unwitnessed overlap geometry

Run `/home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/overlap_geometry_audit.py`
from the repository root. The frozen `overlap_geometry_protocol.json` and
formal note 23 specify the geometric profile. The independent incidence oracle,
production composer and reconstructed source witnesses agree on all 799 maps:
563 positives, 203 coordinate conflicts and 33 structural obstructions.
Every positive recovers its exact overlap and full outer span; all 17 frozen
chemical pairs pass. Controls cover the four obstructions, large values,
defined-None, spectator guards and transient bond absence. This experimental
classifier does not remove public conditions, infer material bindings or
establish physical chemistry or unrestricted guarded synthesis.

## Finite transient frame conditions

Run `PYTHONHASHSEED=0 /home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/frame_condition_audit.py`.
The experimental compiler, frozen protocol and explicit profile amendment
are documented in formal note 24. The complete operational reference passes
15,885 frames and 28,957 admitted primitive applications. Six transient
absence conjuncts have necessity witnesses. Eight complete chemical words
retain original schemas/closed contracts; two aromatic isotope words are
explicit raw projections. Original conditions and aromatic source words
refuse; signed event tokens remain retained even when their net delta is zero.
Historical raw-stage protocol/report hashes are superseded evidence, not
current source gates or a performance replicate. Source CI remains unchanged.

## Read-supported finite condition composition

`compositional_condition_compiler.py` extends the frozen full-world compiler in
a separate experiment. It supports only the exact finite absence-condition
class, extracts actual initial reads, retains declared material carriers and
keeps original event metadata. Run:

```bash
PYTHONHASHSEED=0 /home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/compositional_condition_audit.py
```

The protocol/runtime amendment and compact report specify small-frame versus
chemical-boundary coverage. All 33,241 operational comparisons, 626 anchored
regroupings and 8 explicit unsupported-input refusals pass. General guards,
aromatic original-schema admission and external advantage are unestablished.

## Planned admission cost reference

```bash
PYTHONHASHSEED=0 /home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/compiled_admission_audit.py
```

The frozen protocol compares direct replay, full-world finite compilation and
read-supported finite compilation on the same four complete source words and
27 accepted/refused probes. All 243 timed outcomes agree; raw timings include
36 preparations, query/certificate costs and common source validation. No
uniform advantage or external-tool timing comparison is claimed. The compact
report and `paper/mtg/research/ADMISSION_COST_REFERENCE.md` give the cost boundary.

## Finite guarded coordinate histories and ideal composites

```bash
PYTHONHASHSEED=0 /home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/conditioned_network_audit.py
```

The separate fixed-atom experiment retains actual original guards and adds
presence reads. Its ideal compiler synthesizes spans from accesses/assignments,
with complete original-source replay still a cost. The frozen protocol covers
64 guard presentations, all 688 ideal cuts, 607 nonempty spans/guards, 3,279
frame comparisons, 361 full small/chemical extensions and all 64 enabled
mixed-network diamonds. Ten refusals and 208 admitted outside-order cases keep
the sufficient trace-order limitation explicit. Original port models and all
10 chemical candidates remain, with aromatic isotope cases explicitly projected.
No maximal/physical order, general guard compiler or new native external-tool
comparison is claimed. The source API and Test suite remain unchanged.


The frozen solvent-oxygen case uses attributed RCSB ATP connectivity and the
full-text Walker–Ferretti–Stadtman observation. Reproduce with:

```bash
PYTHONHASHSEED=0 /home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/selenophosphate_isotope_case.py
```

Its report and scope are in `evidence/selenophosphate_isotope_case.json` and
`paper/mtg/research/SOLVENT_OXYGEN_CASE.md`. MØD is an external native dependency;
no binaries or primary-paper copies are archived. Source/Test files stay fixed.


The catalytic-intermediate continuation has its own frozen protocol and
ABSTRACT connectivity projection. Reproduce with:

```bash
PYTHONHASHSEED=0 /home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/phosphoenzyme_refinement.py
```

`evidence/phosphoenzyme_refinement.json` retains 54 PIX cycles, both branching
panels, 202 cuts, 156 actual native primitives and 54 native compositions.
The unresolved enzyme port is not a mapped chemical residue. Its explicit
retention control withdraws generalization of the preceding coarse pool bound.
See `paper/mtg/research/PHOSPHOENZYME_REFINEMENT.md` for the full construction,
negative-exchange control and limits. No public source/Test changes are added.


Connectivity-dependent component observations use separate experimental
observers and preserve every catalytic port. Reproduce the exact-cut/direct
comparison and the strong independent-factor reference sequentially:

```bash
PYTHONHASHSEED=0 /home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/component_observation_audit.py
PYTHONHASHSEED=0 /home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/component_factorization_audit.py
```

The frozen protocols and encoding amendment retain the initial timeout and
all original copy sizes/label controls. The reports distinguish unknown native
answers from false, inventory values from counts of ideals, and measured local
representation cost from full-source validation and external-tool advantage.
See `paper/mtg/research/COMPONENT_OBSERVATION.md` and formal note 27. These
experiments add no public kernel or source tests.


## Exact observer additivity boundary

`observer_additivity_audit.py` independently compares enabled-diamond
cancellation with the unique principal-ideal ledger on complete cut tables.
It checks all 104,846 binary tables on the 51 distinct closures of forward
pairs with zero through four events, and 9,435 ternary tables through three
events. All classifications agree. It additionally classifies 104 cuts from
the earlier source-validated component report, retaining every phosphate fork
and the nonadditive synthetic control. No new chemical execution is counted.

```bash
/home/lolo/.conda/envs/synkit/bin/python Experiment/MTG/observer_additivity_audit.py
```

The frozen protocol precedes execution. The compact report records hashes.
Formal note 28 contains the proof, complete-material additivity corollary and
primary AtTTM3 substrate-read warning. Classical valuation theory is credited;
this is not a generic novelty, measured chemical or timing claim.


## Protected component-boundary composition

`component_boundary.py` composes declared component-count observations through
shared exposed atoms and supplied port connections. Private atoms and existing
fragment edges remain protected. Boundary weights count shared phosphorus once;
closed-component counts add. Source admission, full carrier provenance and
event dependencies remain outside the count kernel and are retained.

```bash
/home/lolo/.conda/envs/synkit/bin/python -m Experiment.MTG.component_boundary_audit
```

The independent literal-DFS audit passes 51,809 disjoint joins (103,618 inventory
checks before/after closure), 512 shared-port joins and 512 three-fragment bracket
comparisons. Five unsupported-input refusals and count-only, isotope, carbon,
admission and deletion controls remain. These are abstract graph checks, with
no new chemical execution or measured performance claim. Formal note 29 contains
the proof and boundary-method prior attribution.
