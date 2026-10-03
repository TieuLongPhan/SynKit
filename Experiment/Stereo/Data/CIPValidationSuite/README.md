# CIP Validation Suite registration

The current release provides the original publisher supporting archive under CC BY-NC 4.0 in `../Revision/CIPPublished`, with author attribution, the publisher metadata and checksums. The licence applies to the data separately from the MIT software licence. Run `python -m Experiment.Stereo.cip_sources` to verify the prepared inputs.

Two bibliographic punctuation repairs in VS297/VS298 reproduce the previously evaluated SMILES bytes. No SMILES, identifier, expected label, stereo-family or sequence-rule field changes. The 300 original publisher 3D structures are unchanged. The later GitHub coordinate file differs at VS095 and in those bibliographic fields; its separate accepted checksum remains available for historical replay, with no publisher licence asserted for that amendment.

The audited GitHub revision `6b9f9db46dadc6749da8234b05164e1e0fb413b9` had no explicit redistribution grant. That observation is retained in `metadata.json`; it is distinct from the grant on the publisher archive. Historical reports and their input hashes remain unchanged.

This suite evaluates local CIP descriptor assignment. It must remain separate from whole-molecule chirality accuracy; whole-molecule runs on its inputs remain non-accuracy diagnostics.
