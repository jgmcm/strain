The committee evaluated all 23,731 candidate frames. The figures and numbers on
this page cover the 18,583 pure-carbon frames that the selection scores (step
5). Over these frames, the median largest per-atom force spread is 0.77 eV/Å
and the median energy spread is 17 meV/atom.

For sheets, the force spread grows with MD temperature, from a median of
0.67 eV/Å at 300 K to 0.94 eV/Å at 2000 K. Unrelaxed starting structures have
the largest spread (median 0.93 eV/Å) and relaxed minima the smallest (0.58).
Among defect classes, pristine sheets agree best (median 0.60 eV/Å),
Stone–Wales defects and divacancies come next (0.71), and classes with carbon
adatoms disagree most: Frenkel pairs (0.95), SW + adatom (0.99), adatom pairs
(1.03) and V1 + adatom (1.08).

Curvature drives the energy spread. Every C60 frame has an energy spread near
51 meV/atom, whatever its geometry, so the members disagree about the energy of
the cage relative to flat graphene rather than about local details. The smallest
nanotube, (3,3), with a diameter of about 4 Å, shows the same effect most
strongly among the tubes.

Two families of models separate clearly. Averaged over the scored frames, the
models trained on MPtrj (MACE-MP-0, MP-0b2, MP-0b3 and MPA-0) give force
magnitudes of 1.88–2.22 eV/Å, and GRACE-1L-OAM gives 2.07 eV/Å. The models
trained on OMat24 (MACE-OMAT-0, MATPES-0 and MH, MatterSim, PET-OMat-XS and the
DPA `Omat24` head) give 2.38–2.56 eV/Å for the same structures. This matches
the systematic softening reported for MPtrj-trained universal potentials, and
the gap between the two families is a large part of the committee spread. The
r²SCAN heads give forces very close to the PBE heads of the same architecture.

Measured as the RMS force difference from the mean of the other members,
MACE-MP-0b2 large (0.30 eV/Å) and the two MatterSim models (0.32 eV/Å) sit
closest to the committee. MACE-MP-0b3 medium (0.66 eV/Å) and MACE-MP-0b2 small
(0.64 eV/Å) sit furthest from it. These are the two models that reconstructed
the vacancy in the [model comparison](../../comp_models/notes.qmd).
