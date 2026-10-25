| Generator | Device | Configurations | Relaxations converged | Median FIRE steps | MD runs | Frames written | Compute (min) | Stopped runs |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DPA-3.1-3M Omat24 | cuda | 130 | 95 % | 65 | 65 | 2371 | 89 | none |
| GRACE-1L-OAM | cuda | 350 | 92 % | 64 | 162 | 6068 | 33 | none |
| MACE-MPA-0 medium | cuda | 324 | 98 % | 60 | 160 | 5844 | 97 | none |
| MatterSim v1.0.0-5M | cuda | 238 | 92 % | 64 | 131 | 4666 | 82 | none |
| PET-OMat-XS v1.0.0 | cpu | 258 | 94 % | 66 | 133 | 4835 | 63 | none |

After filtering, the pool holds **23,731 frames** (1,826,653 atoms) out of 23,784 sampled.

**Removed frames**

| Stage and reason | Frames |
|---|---:|
| initial: detached atom | 5 |
| initial: generator force above 30 eV/A | 2 |
| md: detached atom | 30 |
| relax: detached atom | 15 |
| relaxed: detached atom | 1 |

**Kept frames by stage**

| Stage | Frames |
|---|---:|
| initial | 1293 |
| md | 16896 |
| relax | 4243 |
| relaxed | 1299 |

**Kept frames by framework**

| Framework | Frames |
|---|---:|
| fullerene | 2116 |
| nanotube | 3646 |
| sheet | 17969 |

**MD frames by temperature**

| Temperature (K) | Frames |
|---:|---:|
| 300 | 4302 |
| 600 | 3354 |
| 1000 | 3520 |
| 1500 | 2938 |
| 2000 | 2782 |
