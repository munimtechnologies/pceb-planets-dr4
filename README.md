# pceb-planets-dr4

**Do Gaia DR4's epoch positions show the planets claimed from eclipse timing of post-common-envelope binaries?**

Planets and brown dwarfs have been claimed around more than 30 evolved, eclipsing binaries (white dwarf
or hot subdwarf + M dwarf). The evidence is periodic shifts in their eclipse times, read as light-travel
delays. The claims are disputed: most timing models fail to predict new eclipses (Pulley et al. 2025).

The light-travel amplitude K fixes how far the binary moves about the barycentre: a_bin sin i = c K. The
binary's photocentre must therefore trace an orbit on the sky of a0 = c K plx / sin i, which Gaia can see.
This repository fixes, before Gaia DR4 (2 December 2026), exactly what DR4 should show for six claims,
and how each will be graded.

| claim | predicted wobble a0 (mas) | period |
|---|---|---|
| QS Vir (AB) b, 58 M_Jup, e = 0.91 (Giuppone et al. 2024) | 6.13 | 16.7 yr |
| DE CVn b, 11 M_Jup (Han et al. 2018) | 1.85 | 11.2 yr |
| RR Cae b, 3.3 M_Jup (Rattanamala et al. 2023) | 1.33 | 16.6 yr |
| HW Vir signal 2, 6.5 M_Jup (Baycroft et al. 2023) | 0.25 | 10.2 yr |
| HW Vir signal 4, ~17 M_Jup (Baycroft et al. 2023) | 1.74 | 42.7 yr |
| V893 Sco b, 11 M_Jup (Bruch 2014) | 0.37 | 10.2 yr |

Rules, thresholds and verdicts (DETECTED / SIGNAL_NOT_MATCHING / EXCLUDED / UPPER_LIMIT / INCONCLUSIVE):
[PREREGISTRATION.md](PREREGISTRATION.md).

## Run

```
python -m unittest -v test_etv test_radio test_gaiaorb
python etv_job.py --data-dir /data --out-dir /out [--validated]
```

`--validated` is passed only if the shared validation job (proxima-dr4-validate) returned VALIDATED.
The tested core (`gaiaorb/`) is pure standard-library Python. `io_gaia.py` uses astropy only to read
VOTable/FITS files.

## Credit

The setup belongs to:
- Baycroft, Triaud & Sahlmann 2026 (arXiv:2603.02198), who forecast Gaia's reach for these claims
- Baycroft, Triaud & Kervella 2023 (arXiv:2309.05716)
- the authors of each timing model
- Pulley et al. 2022/2025 for the long timing record
- ESA/Gaia/DPAC for the data

`gaiaorb` is shared with [proxima-c-dr4](https://github.com/munimtechnologies/proxima-c-dr4) and
[radio-planets-dr4](https://github.com/munimtechnologies/radio-planets-dr4).

By MV2, Munim, Inc. (mv2@munimtech.com). Made with AI agents working in MV2's name; no human expert has
checked it yet. License: Apache-2.0 (code); text CC BY 4.0.
