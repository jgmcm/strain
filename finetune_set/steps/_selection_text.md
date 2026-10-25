Frames scoring at least 2.14 form the selection pool of 9,493 frames. The
running median of the pick distance reaches the MD-snapshot distance (0.30) at
pick 1,507, inside the 1,000–2,000 range, so the stopping rule set the final
count. The selected frames have a median largest force spread of 1.03 eV/Å and a
median energy spread of 22 meV/atom, compared with 0.78 eV/Å and 16 meV/atom
over all candidates.

The selection holds 595 MD frames (40 %), 521 starting structures (35 %), 253
relaxation steps (17 %) and 138 relaxed minima (9 %). Sheets make up 68 % of
it, nanotubes 17 % and C60 15 %. A preview run without the per-stage limits
selected 62 % unrelaxed frames and 3 % relaxed minima, because a starting
structure and its first relaxation steps score similarly; the limits make each
configuration contribute different kinds of frame.

The highest scores go to strongly curved carbon: of the 100 highest-scoring
selected frames, 45 are C60 and 34 are the (3,3) nanotube. The (3,3) tube
contributes 49 frames in total. If such extreme curvature is outside the
intended use, these frames can be removed by filtering on `framework_detail`
before labelling.
