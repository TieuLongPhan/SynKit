# V21 targeted repair — 10 September 2026

## Outcome

Line 21609 (minimal mode) is repaired at the unchanged 60-second limit, 16 workers,
4 GiB worker address-space limit and 1,000,000 retained-record cap. Three fresh
runs completed search, classification and JSON output in 9.451, 10.275 and 9.036
seconds. The proven minimum is 10.0; all 72 weighted product representatives and
the complete class/frequency outputs agree with the previously completed
reference-CD shell at distance 10.

Line 40548 (reference-CD mode) remains incomplete. Its exact output is too large
for the current cumulative retained-record protocol. A separate 180-second,
10,000,000-cap diagnostic already retained 6,098,851 distinct exact classes before
timing out (7,779,321 worker-local records). The 1,000,000 cap cannot accommodate
even this partial distinct-class set. A 900-second, 50,000,000-cap diagnostic then
raised MemoryError after 269.172 seconds; child peak RSS was 3,848,336 KiB under
the unchanged 4 GiB address-space limit. Increasing just the count/time allowance
is therefore not a verified solution. These expanded-budget diagnostics are not
original-protocol completions. There is no successful full result for this case.

## Implementation and proof

Changed only synkit/Chem/Mapper/native_analysis.py in production. When the
profile-bound probe exhausts its shell without a witness, a bounded native proof
attempt now searches the attainable lower shells before the Python optimizer
fallback. The seed is reference-free. The proof attempt receives at most 15
seconds or half the remaining deadline, whichever is smaller.

For the validated symmetric half-integer, zero-diagonal inputs, let g be the gcd
of all doubled upper-triangle entries of both graphs. Each doubled distance is
a sum of absolute differences. Because |x-y| is congruent to x+y modulo 2g
when both are multiples of g, all possible distances have a fixed residue and
spacing g in original units. Thus only those shells need checking. A witnessed
shell is minimum only after every smaller attainable shell above the certified
profile lower bound has been exhausted. If no lower witness is found, the
reference-free seed itself attains the proved minimum. Interrupted shells never
certify a minimum; they fall back to the existing optimizer with the remaining
deadline. Complete shell enumeration and output follow as before.

For 21609, the bound is 6, spacing is 1, shells 6, 7, 8 and 9 are empty, and the
blind seed has distance 10. A diagnostic with a different native ordering also
exhausted these lower shells. No reference mapping was used to obtain the proof.

## Verification

- 124 tests passed: the existing native/two-sided suite plus 19 new checks.
- New checks compare against exhaustive weighted permutations, verify the integer
  parity lattice, and test interrupted searches and expired deadlines.
- 32 previously completed minimum tasks selected from the V20 fallback cohort
  all matched exact counts, class maps, coordinate frequencies, reference-class
  results and minimum values. All finished below 60 seconds; maximum 8.611 s.
- All three fresh 21609 repeats match the complete historical distance-10 shell.
- Focused Ruff correctness checks pass.
- Source manifests for regression/repeat runs confirm unchanged frozen source.
- No full 10,000-reaction rerun was performed after this change. Historical
  19,998/20,000 campaign results remain unchanged, with one newly repaired task
  verified separately. One task is still unresolved.

## Artifacts and next requirement

Detailed records, source snapshot, commands, diagnostics and audit script:
 /tmp/synister_v21_20260910/
Summary: /tmp/synister_v21_20260910/final_summary.json
Audit: /tmp/synister_v21_20260910/audit.py
Tests: /tmp/synister_v21_20260910/tests.log

Resolving 40548 requires a separately defined storage/budget protocol, such as
bounded-memory external storage for exact classes plus an adequate total-record
allowance, or a proved compressed aggregation method. It cannot be reported as
passing the existing cumulative one-million-record protocol by silently raising
the cap or discarding classes. Such a redesign has not been implemented here.
