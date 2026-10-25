Over the 23,731 candidate frames, the median largest per-atom force spread is
0.78 eV/Å and the median energy spread is 16 meV/atom.

For sheets, the force spread grows with MD temperature, from a median of
0.69 eV/Å at 300 K to 0.94 eV/Å at 2000 K. Unrelaxed starting structures have
the largest spread (median 1.01 eV/Å) and relaxed minima the smallest (0.62).
Among defect classes, pristine sheets agree best (median 0.60 eV/Å),
Stone–Wales defects and divacancies come next (0.71), and adatom pairs (0.99),
Frenkel pairs (0.95) and C2 dimers (0.93) disagree most.

Curvature drives the energy spread. Every C60 frame has an energy spread near
52 meV/atom, whatever its geometry, so the members disagree about the energy of
the cage relative to flat graphene rather than about local details. The smallest
nanotube, (3,3), with a diameter of about 4 Å, shows the same effect most
strongly among the tubes.

Two families of models separate clearly. Averaged over the pool, the models
trained on MPtrj (MACE-MP-0, MP-0b2, MP-0b3 and MPA-0) give force magnitudes of
1.86–2.20 eV/Å, and GRACE-1L-OAM gives 2.05 eV/Å. The models trained on OMat24
(MACE-OMAT-0, MATPES-0 and MH, MatterSim, PET-OMat-XS and the DPA `Omat24`
head) give 2.35–2.53 eV/Å for the same structures. This matches the systematic
softening reported for MPtrj-trained universal potentials, and the gap between
the two families is a large part of the committee spread. The r²SCAN heads give
forces very close to the PBE heads of the same architecture.

Measured as the RMS force difference from the mean of the other members,
MACE-MP-0b2 large (0.30 eV/Å) and the two MatterSim models (0.32 eV/Å) sit
closest to the committee. MACE-MP-0b3 medium (0.65 eV/Å) and MACE-MP-0b2 small
(0.64 eV/Å) sit furthest from it. These are the two models that reconstructed
the vacancy in the [model comparison](../../comp_models/notes.qmd).
