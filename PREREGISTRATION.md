# Pre-registration: do Gaia DR4's epoch positions show the planets claimed from eclipse timing of post-common-envelope binaries?

Project: pceb-planets-dr4, by MV2, Munim, Inc. (mv2@munimtech.com). Apache-2.0 (code), CC BY 4.0 (text).
Made with AI agents working in MV2's name. No human expert has checked this yet.

**Fixed:** 2026-11-22, ten days before Gaia DR4 (2026-12-02, 12:00 CET), when the first epoch astrometry
for these stars becomes public. Nothing below may change after the release, except the I/O (source
lookup, download URLs, file parsing). Any amendment is dated and committed before the release, or it is
reported as a post-hoc change.

## 1. The question and why it matters

Over 30 circumbinary planets and brown dwarfs have been claimed around post-common-envelope binaries
(PCEBs). The evidence is periodic shifts in eclipse times, read as light-travel-time (LTT) delays as the
binary orbits an unseen companion. The claims are contested:
- Pulley et al. 2025 (MNRAS 544, 24; arXiv:2507.06748) found that 15 of 18 recent models fail to predict
  new eclipse times.
- Magnetic (Applegate-type) mechanisms are the main alternative.

An LTT fit measures the binary's orbit about the barycentre along the line of sight, a_bin sin i = c K.
That same orbit must move the binary's photocentre on the sky, because the companion is dark. Gaia should
therefore see an astrometric orbit of semi-major axis a0 = c K plx / sin i, with the period, eccentricity,
argument of periastron and phase of the LTT fit. This is a test that does not use the eclipse times.
- Baycroft, Triaud & Kervella 2023 (arXiv:2309.05716) used Hipparcos-Gaia proper-motion anomalies for
  HW Vir and wrote that full epoch astrometry would "help resolve" whether the companions exist.
- Baycroft, Triaud & Sahlmann 2026 (MNRAS, arXiv:2603.02198) forecast that Gaia **DR5** (about 2030)
  will be sensitive to 3-11 of 32 PCEB planet candidates, with QS Vir the most strongly constrained.
- Nobody has written down, before DR4, what DR4 itself should show for each claim. This file does that.

**The wow, if it works:** the first direct astrometric confirmation of a planet or brown dwarf found by
eclipse timing of an evolved binary, or a clean retirement of the strongest such claims years before DR5.
Either outcome bears on the long dispute over whether these timing signals are planets at all.

## 2. The claims tested (all values fixed now; code: gaiaorb/etv.py, CLAIMS and STARS)

K is a_bin sin i / c in every source used. a0 is the coplanar prediction (i' = binary inclination) using
the Gaia DR3 parallax. Mass is the coplanar companion mass from that a0. Coverage is the DR4 span
(~2000 d) divided by P.

| key | system (DR3 source_id) | claim and source | K (s) | P | e | a0 pred (mas) | mass (M_Jup) | DR4 coverage |
|---|---|---|---|---|---|---|---|---|
| qsvir | QS Vir (3612227169936143360) | Giuppone et al. 2024, A&A 683, A129, Table 1, linear ephemeris | 148.66 (-38.5/+145) | 16.69 yr | 0.91 | 6.13 | 57.7 | 0.33 |
| decvn | DE CVn (1550299304833675392) | Han et al. 2018, ApJ 868, 53, circular LTT | 28.08 +- 5.01 | 11.22 yr | 0 | 1.85 | 11.2 | 0.49 |
| rrcae | RR Cae (4788741548375134336) | Rattanamala et al. 2023, MNRAS 523, 5086, Table 8, one-object model | 14.0 +- 1.0 | 16.6 yr | 0 | 1.33 | 3.3 | 0.33 |
| hwvir_s2 | HW Vir (3675067076961979264) | Baycroft et al. 2023, Table 5, signal 2 | 21.2 (+1.5/-1.2) | 3710 d | 0.089 | 0.249 | 6.5 | 0.54 |
| hwvir_s4 | HW Vir (same) | Baycroft et al. 2023, Table 5, signal 4 (the authors' favoured companion) | 148 +- 45 | 15600 +- 3500 d | 0.687 | 1.74 | ~17 | 0.13 |
| v893sco | V893 Sco (6039131391540808832) | Bruch 2014, A&A 566, A101 (as tabulated by Souza & Baptista 2024, arXiv:2408.07850) | 22 +- 3 | 10.2 yr | 0 (assumed) | 0.374 | 10.8 | 0.54 |

**Phases.**
- DE CVn and RR Cae: LTT = K sin(2 pi (t - T_ref)/P + phi). DE CVn: T_ref = BJD 2452784.554043,
  phi = 135.55 +- 12.5 deg. RR Cae: T_ref = BJD 2451523.0485, phi = -3.06 +- 0.08 rad.
- QS Vir: periastron at BJD 2453849.66 +- 27 d, omega = 215.95 +- 16.6 deg.
- HW Vir: s2 periastron at BJD 2442500 +- 365 d, omega = 1.74 +- 0.49 rad; s4 periastron at BJD 2433100
  +- 130 d, omega = 1.77 +- 0.73 rad.
- V893 Sco: no phase published; the phase is drawn uniformly.
- We map each LTT form to the delay z/c = (a_bin sin i / c)(sin omega X + cos omega Y), which is the
  line-of-sight part of the same Thiele-Innes orbit Gaia uses (unit tests check this against each
  paper's formula).
- The sign convention of omega does not matter, because the node angle Omega is drawn uniformly: flipping
  omega by 180 deg is the same set of sky orbits.

**Binary inclinations** (for a0 and the injected orbits): DE CVn 86.0, RR Cae 82.9, QS Vir 75.5,
HW Vir 80.6, V893 Sco 72.5 deg. Any non-coplanar orbit has a larger a0 than the coplanar one, by 1 / sin i'.

**Not tested, and why.**
- HW Vir signals 1 (2612 d, 0.09 mas) and 3 (8400 d): DR4's 5.5-yr span cannot separate their periods
  from signals 2 and 4.
- NN Ser, NY Vir, NSVS 14256825, HU Aqr, V470 Cam and the other candidates in Baycroft et al. 2026,
  Table 4: their predicted DR4 signals are below about 3x the per-transit noise.
- V471 Tau: already disfavoured by direct imaging (Hardy et al. 2015) and by Hipparcos-Gaia.
- These may be added only by a dated amendment before the release.

**Context known now (Gaia DR3, 34 months).** RUWE is 1.95 for QS Vir, 1.84 for RR Cae, 1.47 for DE CVn,
1.14 for HW Vir. Raised RUWE fits a long-period companion. It is also expected from the binaries' own
photocentre motion over their 2-9 hour orbits, which is about 0.1-0.25 mas for these white-dwarf + M-dwarf
pairs. DR3 alone cannot tell these apart. DR4's epoch data can, because the short-period part averages
into the jitter term while an LTT companion gives a smooth curve over years.

## 3. Data and validation

- **Data:** Gaia DR4 epoch astrometry (per-CCD along-scan positions) for the six DR3 sources above,
  fetched through the Gaia archive DataLink. The DR4 source is chosen by radio.select_source: parallax
  within 10% of DR3, G within 0.75 mag, nearest to the DR3 position propagated to J2017.5, within 2".
- **Per-transit normal points:** as in proxima-c-dr4 (gaiaorb/epochs.py).
- **Validation:** the fitter, time conventions and 5-parameter model are shared with proxima-c-dr4 and
  radio-planets-dr4. The verdicts count only if `proxima-dr4-validate-v1` returns VALIDATED (Gaia BH3
  against ESA's fit, DR3 orbits of HD 114762 and Gaia-4, quasars, injection recovery on the DR4
  pre-release). Otherwise every verdict is INCONCLUSIVE.
- **Unit tests** (`test_etv.py`) cover:
  - the LTT-to-astrometry mapping against each paper's formula
  - the predicted a0 and masses against the papers (DE CVn 11.2 vs ~11.5 M_Jup; QS Vir 57.7 vs 57.71 M_Jup)
  - exact recovery of an injected orbit at the published ephemeris
  - detection of the DE CVn orbit and exclusion of an absent RR Cae companion in synthetic data

## 4. The test (the same for every claim)

- **Base model:** the 5-parameter single-star model. An LTT companion must show up as curvature, or a
  partial or full orbit, that a linear proper motion cannot absorb. No acceleration terms are added to the
  base. Radial velocity is set to 0; the perspective term is below 0.02 mas over DR4 for all six.
- **Noise:** one jitter term in quadrature, so that the base fit has reduced chi2 = 1. This absorbs the
  binaries' short-period photocentre motion. Transits with |residual| > 5 sigma are dropped once.
- **Statistic:** S = max over a (P, e, Tp) grid of delta chi2 (base + Thiele-Innes orbit vs base).
  - P box: claimed P +- max(3 sigma_P, 10% of P).
  - 9 periods (15 for hwvir_s4), 12 periastron phases for each e > 0.
  - e grids: (0, 0.2, 0.4), but (0, 0.5, 0.8, 0.91) for QS Vir and (0, 0.35, 0.7) for hwvir_s4.
  - Grid sizes: 225, 333 (QS Vir) and 375 (hwvir_s4) points.
- **Nulls:** 2,000 Gaussian draws and 2,000 permutations of the whitened base residuals. The threshold
  is the larger 99.9% quantile (FAP 0.001 per claim).
- **Injections into the real epochs:** 200 per cell, at 0.05, 0.1, 0.25, 0.5, 1 and 2 times the
  predicted a0. In both families, i = binary inclination and Omega is uniform.
  - "etv": P, e, omega and phase drawn from the published values and errors. P is clipped to the box and
    e to [0, e_max] (e_max 0.5; 0.97 for QS Vir; 0.6 and 0.9 for HW Vir s2 and s4).
  - "anyphase": the same, with a uniform phase. This allows for the fact that the timing models
    mis-predict new eclipses.
- **Reported for each claim:**
  - the a0 (and equivalent K) at 50% and 90% detection and at 95% exclusion
  - the best orbit in the box (refined)
  - the "ephemeris fit": a0 and Omega with every other element fixed at the published values
  - the periodogram of S over P

## 5. Verdicts (per claim; gaiaorb/etv.py, verdict)

- **INCONCLUSIVE:** validation is not VALIDATED, there are fewer than 30 transits, or the source or
  epochs are missing.
- **DETECTED:** S > threshold, and the best-fit a0 lies in the claim's band. The band runs from
  a0_pred x (1 - 3 sigma_K/K) up to a0_pred x (1 + 3 sigma_K/K) x 2 (allowing sin i' down to 0.5).
- **SIGNAL_NOT_MATCHING:** S > threshold with a0 outside the band (for example a much smaller wobble,
  or a different companion).
- **EXCLUDED:** at least 95% of the injections at the predicted a0 give S above the observed S, in both
  families. A companion that matches the claim would almost always have produced a larger signal than
  the one Gaia shows.
- **UPPER_LIMIT:** otherwise. The 95% exclusion a0 and K are reported.

A DETECTED verdict is a candidate astrometric confirmation, not a final one. Partial orbits leave P, e
and a0 correlated, and a real signal must be checked against the binary's own photocentre motion and
against DPAC's own non-single-star solution for the source.

## 6. Limits stated in advance

- **Model assumptions.** The predictions assume each LTT model as published. Most of these models
  already mis-predict new eclipse times, which is why the "anyphase" family exists.
- **An exclusion retires the companion, not the timing signal.** EXCLUDED means no companion with that K
  and period moves the binary as claimed. It does not explain the eclipse-time changes.
- **Long periods.** For P much longer than DR4's span (HW Vir s4), the test sees only curvature, and a
  null result is weaker.
- **Unchecked values.** V893 Sco's K is taken as a semi-amplitude (as Souza & Baptista use it in the LTT
  mass function). If it is a full amplitude, the prediction halves, and the 0.5x cell applies.

## 7. Credit

The setup belongs to:
- Baycroft, Triaud & Sahlmann (2026) and Baycroft, Triaud & Kervella (2023)
- the LTT model authors: Giuppone et al. (2024), Han et al. (2018), Rattanamala et al. (2023), Bruch (2014)
- Pulley et al. (2022, 2025) for the timing record
- ESA/Gaia/DPAC for the data
