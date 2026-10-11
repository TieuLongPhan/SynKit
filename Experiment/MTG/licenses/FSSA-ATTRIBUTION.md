# Native fSSA comparison attribution

The resource-flow foundation and native simulator are by Ozan Kahramanoğulları:
[Quantifying and comparing causal patterns in stochastic chemical reaction
networks](https://doi.org/10.1038/s41540-025-00625-8), npj Systems Biology and
Applications 12, 7 (2026). Downloads are from the author's
[FluxTools page](https://ozan-k.com/fluxCompare.html).

The local review package contains original comparison adapters, models
generated from our supplied bound histories, their native outputs, protocols
and proofs. It contains no downloaded native binary, author CRN model archive,
external notebook source or guide PDF. The downloaded files' exact hashes and
URLs are recorded for independent acquisition. No redistribution license for
those downloads is inferred from their public availability.

The fSSA simulator performs all reaction simulation and producer-bin allocation.
The Python adapter only checks the emitted resource cells and markings. The
shared graph metadata uses the separately attributed M-CSA case, with its
drawing/deposit restrictions in `MCSA-ATTRIBUTION.md`.

Open-net composition prior art is John C. Baez and Jade Master,
[Open Petri nets](https://doi.org/10.1017/S0960129520000043), Mathematical
Structures in Computer Science 30(3), 314–341 (2020). Its general composition
framework is credited; the original source-histories remain graph-decorated
and their bindings/conditions are separately checked.
