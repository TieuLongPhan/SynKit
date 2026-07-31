# Hydrogen-extension comparison

This experiment uses the 109-reaction `hydrogen.pkl.gz` corpus from the
PartialAAMs analysis. It runs sequentially with one process and forces common
numeric libraries to one thread.

```bash
bash Experiment/Lewis/hydrogen_expand/run_comparison.sh
```

Use `--limit 3 --repetitions 1` for a pilot. Results are written under
`Experiment/Lewis/Runs` by default.

The compared names have different contracts:

- `hextend_legacy` enumerates hydrogen-extension classes using SynKit's
  historical same-permutation behavior.
- `hextend_new` enumerates provenance-aware hydrogen-transfer classes.
- `gm`, `rb1`, and `rb2` are PartialAAM capability controls. The runner
  materializes the transferring hydrogens and clears their maps before calling
  these methods. It records failures because partial-AAM completion does not
  generally support an unmapped atom that participates in the reaction center.

The downloaded analysis scripts' **Method A** and **Method B** are not aliases
for RB1 and RB2. Method A classifies every hydrogen permutation using full ITS
isomorphism. Method B first finds automorphisms of the heavy-atom ITS and then
uses those mappings as anchors for co-extension. Their three scripts vary the
Method B isomorphism backend: native GranMapache stable extension,
anchor-relabeling with GranMapache, and anchor-relabeling with NetworkX.

The aggregate JSON keeps these contracts explicit so capability failures are
not presented as successful runtime measurements.
