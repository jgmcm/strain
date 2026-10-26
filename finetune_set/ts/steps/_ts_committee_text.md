The committee evaluated 6,313 frames from the 1,056 pure-carbon bands: 4,201
transition-state frames (1,056 climbing images, 2,089 neighbouring images and
1,056 displaced copies) and the 2,112 end states. The evaluation took 34 min and
used at most 6.9 GB of memory.

Over the transition-state frames, the median largest per-atom force spread is
0.94 eV/Å (0.90 eV/Å for the climbing images alone), against 0.77 eV/Å for the
pure-carbon frames of the first set. The median barrier spread is 0.39 eV.

The barrier spread is largest for a second bond rotation next to a Stone–Wales
defect (median 0.67 eV), Stone–Wales formation (0.53 eV), Frenkel
recombination (0.51 eV), and divacancy hops and rotations at a divacancy
(0.48 eV). The members agree best on the V1 reconstruction switch (0.21 eV), V1 hops
(0.27 eV) and carbon adatom hops (0.27–0.29 eV). The force spread ranks the
processes differently. It is largest for the adatom processes (median
1.1–1.25 eV/Å) and smallest for V1 hops (0.76 eV/Å).

The members split into the same two families as in the first set. Relative to
the median barrier of the 18 PBE-level members, the MPtrj-trained MACE models
give lower barriers, from −0.03 eV (MACE-MPA-0) to −0.39 eV (MACE-MP-0b3
medium) at the median. The OMat24-trained MACE-OMAT-0 medium and MACE-MH models,
MatterSim 5M and DPA-3.1-3M give higher barriers, +0.21 to +0.31 eV. The r²SCAN
heads are higher still, +0.27 to +0.45 eV. GRACE-1L-OAM, PET-OMat-XS, MatterSim
1M and the generator sit within 0.1 eV of the median. The MPtrj-trained models,
which give softer forces in the first set, also give lower barriers. For
Stone–Wales formation on sheets, the PBE-level members' median barriers range
from 6.3 eV (MACE-MP-0b3 medium) to 9.3 eV (DPA-3.1-3M); DFT studies report
about 9–10 eV.

The generator sits near the committee median overall, but not for carbon adatom
hops. It gives them a median barrier of 0.94 eV on sheets, while the PBE-level
median on the same geometries is 0.42 eV (see [TS 2](02_neb.qmd)).

Measured as the RMS force difference from the mean of the other members over
the transition-state frames, MACE-MP-0b2 large (0.28 eV/Å) and the two
MatterSim models (0.29 and 0.30 eV/Å) sit closest to the committee. DPA-3.1-3M
(0.51 eV/Å), MACE-MP-0b2 small (0.50 eV/Å) and MACE-MP-0b3 medium (0.46 eV/Å)
sit furthest from it.
