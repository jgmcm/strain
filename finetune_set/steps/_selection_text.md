The configuration builder also placed H, N, O and F adatoms, which were not
meant to be part of the set. The selection drops the 5,148 candidate frames
(22 %) that contain those atoms and scores the remaining 18,583 pure-carbon
frames. Frames scoring at least 2.13 form the selection pool of 7,433 frames.
The running median of the pick distance reaches the MD-snapshot distance (0.36)
at pick 1,059, inside the 1,000–2,000 range, so the stopping rule set the final
count. The selected frames have a median largest force spread of 1.02 eV/Å and a
median energy spread of 23 meV/atom, compared with 0.77 eV/Å and 17 meV/atom
over all scored frames.

The selection holds 445 MD frames (42 %), 353 starting structures (33 %), 179
relaxation steps (17 %) and 82 relaxed minima (8 %). Sheets make up 63 % of it,
nanotubes 19 % and C60 18 %. A preview run without the per-stage limits
selected 58 % unrelaxed frames and 4 % relaxed minima, because a starting
structure and its first relaxation steps score similarly; the limits make each
configuration contribute different kinds of frame.

The highest scores go to strongly curved carbon: of the 100 highest-scoring
selected frames, 46 are C60 and 33 are the (3,3) nanotube. The (3,3) tube
contributes 41 frames in total. If such extreme curvature is outside the
intended use, these frames can be removed by filtering on `framework_detail`
before labelling.

An earlier version of this set, with 1,507 frames, included 373 frames with
H, N, O or F atoms. It is in the git history (commit `72db2ff`).
