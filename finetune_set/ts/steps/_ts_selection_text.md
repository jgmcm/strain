The pool holds the 528 climbing images that score at least 2.06. The running
median of the pick distance reaches the neighbouring-image distance (0.43) after
189 climbing images. That is below the 250 bands needed for 500 frames, so the
lower bound set the count: 250 bands and 500 frames (32,272 atoms). Beyond about
190 bands, a new climbing image resembles one already chosen about as closely
as two adjacent images of one band do. The bands in the set are ordered by
`selection_order`, so keeping only those with `selection_order` below 189 gives
the set at the stopping point.

The selected bands have a median barrier spread of 0.58 eV and a median
climbing-image force spread of 1.01 eV/Å, against 0.39 eV and 0.90 eV/Å over all
bands. Besides its climbing image, each band contributes a neighbouring image
(137 bands) or a displaced copy (113 bands).

Bond rotations and Stone–Wales formation make up 153 of the 250 bands (61 %),
vacancy migration and coalescence 51, Frenkel recombination 26 and carbon adatom
hops 20. No V1 reconstruction switch is selected, because its barrier spread is
the smallest of all processes. Sheets give 204 bands, nanotubes 41 and C60 5.
The C60 bands have the largest median barrier spread (0.53 eV), but C60 is not
strained or bent, so the same process gives the same climbing image anywhere on
the cage. The descriptor search treats these as repeats. They are also the
picks with near-zero distance at the end of the selection curve.
