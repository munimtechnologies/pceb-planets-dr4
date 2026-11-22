"""Tests for gaiaorb.radio (radio-planets-dr4). Standard library only.

MV2 / Munim, Inc. Apache-2.0. Made with AI agents working in MV2's name.
Run: python -m unittest -v test_radio
"""
import math
import random
import unittest

from gaiaorb import kepler, linalg, radio

K = math.pi / (180.0 * 3600.0 * 1000.0)
GJ = radio.TARGETS["gj896a"]
TV = radio.TARGETS["tvlm513"]

SMALL = dict(P_step=5.0, e_grid=(0.0, 0.3), n_phase=4, n_null=100, scales=(0.25, 1.0), n_inj=10)


def angdiff(a, b):
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def synth(n=90, seed=1, plx=159.66, pm=(578.0, -59.8), acc=(0.887, 0.14), jerk=(0.02, -0.01),
          sigma=0.05, orbit=None, span=5.4, noise=True):
    """Synthetic per-transit along-scan points: 9-term single-star motion
    (exact w = L / (1 + plx fr k) form, radial term zero) plus an optional orbit."""
    rng = random.Random(seed)
    ti = None
    if orbit:
        ti = kepler.campbell_to_ti(orbit["a0"], orbit["inc"], orbit["omega"], orbit["Omega"])
    pts = []
    for k in range(n):
        t = -0.5 * span + span * (k + 0.5 * rng.random()) / n
        psi = rng.uniform(-math.pi, math.pi)
        s, c = math.sin(psi), math.cos(psi)
        f, fr = rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)
        L = (0.3 * s - 0.2 * c + plx * f + pm[0] * t * s + pm[1] * t * c
             + 0.5 * t * t * (acc[0] * s + acc[1] * c) + t ** 3 / 6.0 * (jerk[0] * s + jerk[1] * c))
        w = L / (1.0 + plx * fr * K)
        if orbit:
            X, Y = kepler.xy(t * kepler.DAYS_PER_YEAR, orbit["P"], orbit["e"], orbit["Tp"])
            w += kepler.along_scan(ti, X, Y, s, c)
        if noise:
            w += rng.gauss(0.0, sigma)
        pts.append(dict(t=t, w=w, sig=sigma, s=s, c=c, f=f, fr=fr, n=9, infl=1.0))
    return pts


class TestBase(unittest.TestCase):
    def test_base_cols_exact_9p(self):
        pts = synth(noise=False)
        cols, y = radio.base_cols(pts, 0.0, 9)
        self.assertEqual(len(cols), 9)
        fit = linalg.wls(cols, y, [p["sig"] for p in pts])
        want = (0.3, -0.2, 159.66, 578.0, -59.8, 0.887, 0.14, 0.02, -0.01)
        for got, w in zip(fit["beta"], want):
            self.assertAlmostEqual(got, w, delta=1e-6)
        self.assertLess(max(abs(r) for r in fit["resid"]), 1e-6)
        cols5, y5 = radio.base_cols(pts, 0.0, 5)
        self.assertEqual(len(cols5), 5)
        fit5 = linalg.wls(cols5, y5, [p["sig"] for p in pts])
        self.assertGreater(max(abs(r) for r in fit5["resid"]), 0.05)
        self.assertEqual(len(radio.base_cols(pts, 0.0, 7)[0]), 7)
        with self.assertRaises(ValueError):
            radio.base_cols(pts, 0.0, 6)

    def test_gridstatn_equals_chi2_difference(self):
        orb = radio.published_orbit(GJ)
        pts = synth(n=70, seed=4, orbit=orb)
        sig = [p["sig"] for p in pts]
        grid = [(orb["P"], orb["e"], orb["Tp"]), (275.0, 0.0, 0.0), (299.0, 0.6, 100.0)]
        gs = radio.GridStatN(pts, sig, grid, 0.0, 9)
        cb, y = radio.base_cols(pts, 0.0, 9)
        f0 = linalg.wls(cb, y, sig)
        vals = gs.values([r / s for r, s in zip(f0["resid"], sig)])
        for (P, e, Tp), v in zip(grid, vals):
            f1 = linalg.wls(cb + radio.model.orbit_cols(pts, P, e, Tp), y, sig)
            self.assertAlmostEqual(v, f0["chi2"] - f1["chi2"], delta=1e-6 * max(1.0, v))
        self.assertEqual(max(range(3), key=lambda g: vals[g]), 0)

    def test_clean_jitter_and_params(self):
        pts = synth(n=200, seed=2, sigma=0.05)
        for p in pts:
            p["sig"] = 0.025
        cl = radio.clean(pts, 0.0, 9, clip=None)
        self.assertAlmostEqual(cl["chi2"], 200 - 9, delta=0.01)
        self.assertTrue(0.03 < cl["eps"] < 0.06)
        self.assertAlmostEqual(cl["params"][2][0], 159.66, delta=0.05)


class TestOrbits(unittest.TestCase):
    def test_published_masses(self):
        self.assertAlmostEqual(radio.mass_mjup(0.145, 220.0, TV), 0.39, delta=0.02)
        self.assertAlmostEqual(radio.mass_mjup(0.51, 284.39, GJ), 2.28, delta=0.05)

    def test_tp_days(self):
        self.assertAlmostEqual(radio.tp_days(GJ), 2455702.65 - 2457936.875, places=6)
        self.assertAlmostEqual(radio.tp_days(TV), -305.875, places=6)

    def test_draw_orbit(self):
        rng = random.Random(7)
        cfg = dict(radio.DEFAULT_CFG, **GJ["grid"])
        for _ in range(200):
            o = radio.draw_orbit(rng, "pub", 0.5, GJ, cfg)
            self.assertTrue(270.0 <= o["P"] <= 300.0)
            self.assertTrue(0.0 <= o["e"] <= 0.8)
            self.assertTrue(0.0 <= o["inc"] <= 180.0)
            self.assertAlmostEqual(o["a0"], 0.255, places=12)
            q = radio.draw_orbit(rng, "iso", 1.0, GJ, cfg)
            self.assertTrue(0.0 <= q["Tp"] < q["P"])
        with self.assertRaises(ValueError):
            radio.draw_orbit(rng, "edge", 1.0, GJ, cfg)
        cfg_t = dict(radio.DEFAULT_CFG, **TV["grid"])
        self.assertEqual(radio.draw_orbit(rng, "pub", 1.0, TV, cfg_t)["e"], 0.0)

    def test_fit_at_exact(self):
        orb = radio.published_orbit(GJ)
        pts = synth(n=80, seed=5, orbit=orb, noise=False)
        sig = [p["sig"] for p in pts]
        fit = radio.fit_at(pts, sig, 0.0, 9, orb["P"], orb["e"], orb["Tp"])
        self.assertAlmostEqual(fit["a0_mas"], 0.51, delta=1e-6)
        self.assertAlmostEqual(fit["inc_deg"], 69.20, delta=1e-4)
        self.assertLess(angdiff(fit["Omega_deg"], 45.62), 1e-4)
        self.assertLess(angdiff(fit["omega_deg"], 353.11), 1e-4)
        self.assertLess(fit["chi2"], 1e-6)

    def test_template_sign(self):
        orb = radio.published_orbit(GJ)
        pts = synth(n=80, seed=6, orbit=orb, noise=False)
        sig = [p["sig"] for p in pts]
        t1 = radio.template_fit(pts, sig, 0.0, 9, GJ)
        self.assertAlmostEqual(t1["k"], 1.0, delta=1e-6)
        flipped = dict(orb, omega=orb["omega"] + 180.0)
        pts2 = synth(n=80, seed=6, orbit=flipped, noise=False)
        self.assertAlmostEqual(radio.template_fit(pts2, sig, 0.0, 9, GJ)["k"], -1.0, delta=1e-6)
        pts3 = synth(n=80, seed=6, noise=False)
        self.assertAlmostEqual(radio.template_fit(pts3, sig, 0.0, 9, GJ)["k"], 0.0, delta=1e-6)


class TestSelect(unittest.TestCase):
    @staticmethod
    def row(sid, star, epoch=2017.5, plx=None, g=None, dra_arcsec=0.0):
        ra, dec = radio.propagate(star, epoch)
        ra += dra_arcsec / 3600.0 / math.cos(math.radians(dec))
        return dict(source_id=str(sid), ra=repr(ra), dec=repr(dec),
                    parallax=repr(star["dr3_plx"] if plx is None else plx),
                    phot_g_mean_mag=repr(star["dr3_g"] if g is None else g))

    def test_picks_eq_peg_a_not_b(self):
        eqpb = dict(GJ, dr3_ra=352.9716753030632, dr3_dec=19.937313746866927, dr3_pmra=552.3494119624912,
                    dr3_pmdec=20.27460819805265, dr3_plx=159.90846046252173, dr3_g=10.828941)
        rows = [self.row(22, eqpb), self.row(11, GJ, dra_arcsec=0.05),
                self.row(33, GJ, plx=1.51, g=12.15, dra_arcsec=0.01)]
        sid, r, sep = radio.select_source(rows, GJ)
        self.assertEqual(sid, 11)
        self.assertAlmostEqual(sep, 0.05, delta=1e-3)
        # EQ Peg B is about 5.6 arcsec away
        ra_a, dec_a = radio.propagate(GJ)
        ra_b, dec_b = radio.propagate(eqpb)
        self.assertTrue(4.0 < radio.sep_arcsec(ra_a, dec_a, ra_b, dec_b) < 8.0)

    def test_none_when_far_or_wrong(self):
        self.assertEqual(radio.select_source([self.row(1, GJ, dra_arcsec=3.0)], GJ), (None, None, None))
        self.assertEqual(radio.select_source([self.row(1, GJ, g=12.0)], GJ), (None, None, None))
        self.assertEqual(radio.select_source([dict(source_id="1", ra="", dec="x", parallax="1")], GJ),
                         (None, None, None))

    def test_propagate(self):
        ra, dec = radio.propagate(TV, 2016.0)
        self.assertEqual((ra, dec), (TV["dr3_ra"], TV["dr3_dec"]))
        ra, dec = radio.propagate(GJ, 2017.5)
        self.assertAlmostEqual((dec - GJ["dr3_dec"]) * 3.6e6, -59.768604898823476 * 1.5, delta=1e-6)


class TestRun(unittest.TestCase):
    def test_detects_published_gj896_orbit(self):
        pts = synth(n=90, seed=11, orbit=radio.published_orbit(GJ))
        res = radio.run_target(pts, GJ, SMALL, log=lambda m: None)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_grid"], 7 * 5)
        self.assertGreater(res["S_obs"], res["threshold"])
        self.assertEqual(radio.verdict(res, True, GJ), "DETECTED")
        self.assertLess(abs(res["best_fit"]["a0_mas"] - 0.51), 0.2)
        self.assertTrue(270.0 <= res["best_fit"]["P"] <= 300.0)
        self.assertAlmostEqual(res["template"]["k"], 1.0, delta=0.35)
        self.assertEqual(radio.verdict(res, False, GJ), "INCONCLUSIVE")

    def test_no_planet_is_excluded(self):
        pts = synth(n=90, seed=12)
        res = radio.run_target(pts, GJ, SMALL, log=lambda m: None)
        self.assertEqual(res["status"], "ok")
        at1 = [r for r in res["injections"] if r["scale"] == 1.0]
        self.assertEqual(len(at1), 2)
        self.assertTrue(all(r["frac_above_observed"] == 1.0 for r in at1))
        # with 100 nulls the threshold is the null maximum, so pure noise
        # crosses it about 1% of the time; the verdict is checked when it does not
        if res["S_obs"] <= res["threshold"]:
            self.assertEqual(radio.verdict(res, True, GJ), "EXCLUDED")
        self.assertEqual(len(res["periodogram"]), 7)

    def test_verdict_rules(self):
        def res(S, thr, a0, fr_pub, fr_iso):
            return dict(status="ok", S_obs=S, threshold=thr, best_fit=dict(a0_mas=a0), injections=[
                dict(family="pub", scale=0.5, frac_above_observed=0.1),
                dict(family="pub", scale=1.0, frac_above_observed=fr_pub),
                dict(family="iso", scale=1.0, frac_above_observed=fr_iso)])
        self.assertEqual(radio.verdict(res(50.0, 30.0, 0.4, 1.0, 1.0), True, GJ), "DETECTED")
        self.assertEqual(radio.verdict(res(50.0, 30.0, 1.5, 1.0, 1.0), True, GJ), "SIGNAL_NOT_MATCHING")
        self.assertEqual(radio.verdict(res(50.0, 30.0, 0.03, 1.0, 1.0), True, GJ), "SIGNAL_NOT_MATCHING")
        self.assertEqual(radio.verdict(res(10.0, 30.0, 0.4, 0.95, 0.97), True, GJ), "EXCLUDED")
        self.assertEqual(radio.verdict(res(10.0, 30.0, 0.4, 0.95, 0.90), True, GJ), "UPPER_LIMIT")
        self.assertEqual(radio.verdict(res(10.0, 30.0, 0.4, 1.0, 1.0), False, GJ), "INCONCLUSIVE")
        self.assertEqual(radio.verdict(res(50.0, 30.0, 0.03, 1.0, 1.0), True, TV), "DETECTED")

    def test_tvlm_path_runs(self):
        pts = synth(n=70, seed=13, plx=93.17, pm=(-43.1, -65.1), acc=(0.0, 0.0), jerk=(0.0, 0.0), sigma=0.25)
        res = radio.run_target(pts, TV, SMALL, log=lambda m: None)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_grid"], 15 * 5)
        self.assertIn(radio.verdict(res, True, TV), ("UPPER_LIMIT", "EXCLUDED", "DETECTED", "SIGNAL_NOT_MATCHING"))
        self.assertIn("a0_95_exclusion_pub", res["limits"])

    def test_too_few_transits(self):
        res = radio.run_target(synth(n=20, seed=14), GJ, SMALL, log=lambda m: None)
        self.assertEqual(res["status"], "too_few_transits")
        self.assertEqual(radio.verdict(res, True, GJ), "INCONCLUSIVE")

    def test_helpers(self):
        self.assertEqual(len(radio.make_grid(dict(radio.DEFAULT_CFG, **GJ["grid"]))), 31 * 49)
        self.assertEqual(len(radio.make_grid(dict(radio.DEFAULT_CFG, **TV["grid"]))), 36 * 37)
        self.assertEqual(radio.quantile(list(range(1, 1001)), 0.999), 999)
        self.assertAlmostEqual(radio.crossing([1.0, 2.0, 4.0], [0.1, 0.5, 1.0], 0.9)[0], 3.6)
        self.assertEqual(radio.crossing([1.0, 2.0], [0.1, 0.2], 0.95), (None, "above_grid"))


if __name__ == "__main__":
    unittest.main()
