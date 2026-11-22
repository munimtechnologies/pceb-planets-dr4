"""gaiaorb: a small, pure-Python fitter for Gaia DR4 along-scan epoch astrometry.

By MV2, Munim, Inc. (mv2@munimtech.com). Apache-2.0.
Made with AI agents working in MV2's name. Shared by proxima-c-dr4,
radio-planets-dr4 and pceb-planets-dr4.

Modules (standard library only):
  linalg  - weighted least squares, Cholesky, Gram-Schmidt
  kepler  - Kepler's equation, Thiele-Innes <-> Campbell, masses
  epochs  - Gaia epoch-astrometry rows -> per-transit normal points
  model   - 5-parameter AL model with exact perspective acceleration,
            orbit columns, grid statistic, orbit fitting
  prereg  - the pre-registered Proxima c test (nulls, injections, verdict)
  validate- validation gates on the DR4 pre-release sources
  radio   - the pre-registered test of the two VLBA radio-astrometry planets
            (TVLM 513-46546 b, GJ 896 Ab); 5/7/9-term base models
  etv     - the pre-registered test of companions claimed from eclipse
            timing of post-common-envelope binaries (QS Vir, DE CVn, RR Cae,
            HW Vir, V893 Sco)
"""
__version__ = "0.3.0"
