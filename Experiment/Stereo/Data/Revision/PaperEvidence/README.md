# Paper evidence and historical audit

The active paper is `paper/stereo_graph`; its one final package is `paper/stereo_graph/submission`.

This directory contains the unchanged chemical selection manifests, RDKit representation fixtures/report, independent witness-verification receipt and hash-pinned environment records used by the integrated experiments. The numerical paper manifest is `paper/stereo_graph/results/manifest.json`; its 37 selected reports retain their original bytes.

`archived_paper_revision.tar.gz` preserves the complete tracked paper before consolidation, including the detailed revision plan, proof/chemical reviews, historical report selections, release checks and earlier generated packages. `archive.json` identifies the source commit, archive digest and scope. Original member paths beginning `paper/stereo_graph/revision/` are historical paths inside that archive; they are not active dependencies of the consolidated paper. The underlying reviewed source remains reachable through Git at the recorded commit.

The full clean scientific execution remains in `../CleanReleaseFull/execution`, including all 90 files and both the initial dependency failure and successful continuation. Source commits and absolute paths recorded by historical executions retain their original meaning; consolidation does not rewrite those raw provenance fields or claim that an old execution was performed from the current workspace.

The current integrated tests and document/source rebuild are recorded separately in `paper/stereo_graph/results/integration.json` and `paper/stereo_graph/submission/verification.json`. The workspace also contains unrelated user work, which is preserved rather than included in the stereo integration commit.

`archived_workspace_artifacts.tar.gz` additionally preserves the historical ignored experiment reports, old paper graphics/build outputs and stereo sprint records from the removed review workspace. `workspace_artifact_archive.json` lists every original path and byte digest. These historical artifacts are kept out of active benchmark paths and are not promoted into the current numerical selection.

`prior_main_paper_delta.tar.gz` retains the 16 distinct older paper files from the main workspace that are not already preserved byte-for-byte in the current paper or the review archives. Its matching JSON inventory records the original paths and digests.
