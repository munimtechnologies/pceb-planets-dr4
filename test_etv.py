"""Tests for gaiaorb.etv (pceb-planets-dr4). Standard library only.

MV2 / Munim, Inc. Apache-2.0. Made with AI agents working in MV2's name.
Run: python -m unittest -v test_etv
"""
import math
import random
import unittest

from gaiaorb import etv, kepler, radio

K = math.pi / (180.0 * 3600.0 * 1000.0)
DE = etv.CLAIMS["decvn"]
RR = etv.CLAIMS["rrcae"]
QS = etv.CLAIMS["qsvir"]
HW2 = etv.CLAIMS["hwvir_s2"]
HW4 = etv.CLAIMS["hwvir_s4"]
VS = etv.CLAIMS["v893sco"]

SMALL = dict(n_P=3, n_phase=4, n_null=100, scales=(0.25, 1.0), n_inj=10)


def angdiff(a, b):
    d = (a - b) % 360.0
    return min(d, 360.0 - d)


def synth(n=90, seed=1, plx=32.77, pm=(-202.3, -175.7), sigma=0.05, orbit=None, span=5.4, noise=True):
    """Synthetic per-transit along-scan points: 5-term single-star motion
    (w = L / (1 + plx fr k), radial term zero) plus an optional orbit."""
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
        L = 0.3 * s - 0.2 * c + plx * f + pm[0] * t * s + pm[1] * t * c
        w = L / (1.0 + plx * fr * K)
        if orbit:
            X, Y = kepler.xy(t * kepler.DAYS_PER_YEAR, orbit["P"], orbit["e"], orbit["Tp"])
            w += kepler.along_scan(ti, X, Y, s, c)
        if noise:
            w += rng.gauss(0.0, sigma)
        pts.append(dict(t=t, w=w, sig=sigma, s=s, c=c, f=f, fr=fr, n=9, infl=1.0))
    return pts


def central_orbit(claim, a0, Omega):
    return dict(P=claim["P"], e=claim["e"], Tp=etv.central_tp_days(claim), a0=a0, inc=claim["inc_bin"],
                omega=claim["omega"], Omega=Omega)


def solve_E(M, e):
    lo, hi = M - e - 1e-12, M + e + 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid - e * math.sin(mid) - M < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


class TestPredictions(unittest.TestCase):
    def test_light_second(self):
        self.assertAlmostEqual(etv.C_AU_PER_S, 1.0 / 499.004784, delta=1e-11)

    def test_a0_pred(self):
        self.assertAlmostEqual(etv.a0_pred(DE), 1.84869, delta=2e-3)
        self.assertAlmostEqual(etv.a0_pred(QS), 6.12819, delta=3e-3)
        self.assertAlmostEqual(etv.a0_pred(RR), 1.33447, delta=2e-3)
        self.assertAlmostEqual(etv.a0_pred(HW2), 0.248594, delta=1e-3)

    def test_k_roundtrip(self):
        for c in etv.CLAIMS.values():
            self.assertAlmostEqual(etv.k_from_a0(etv.a0_pred(c), c), c["K_s"], delta=1e-9 * c["K_s"])

    def test_masses_match_papers(self):
        # Han et al. 2018: M3 ~ 0.011 Msun = 11.5 M_Jup; Giuppone et al. 2024: 57.71 M_Jup
        self.assertAlmostEqual(etv.mass_mjup(etv.a0_pred(DE), DE["P"], DE), 11.24, delta=0.2)
        self.assertAlmostEqual(etv.mass_mjup(etv.a0_pred(QS), QS["P"], QS), 57.7, delta=1.0)

    def test_band(self):
        lo, hi = etv.a0_band(DE)
        self.assertAlmostEqual(lo, 0.859168, delta=1e-3)
        self.assertAlmostEqual(hi, 5.676437, delta=2e-3)
        lo, hi = etv.a0_band(QS)
        self.assertEqual(lo, etv.a0_pred(QS) * (1.0 - 3.0 * 38.52 / 148.66))

    def test_central_tp(self):
        self.assertAlmostEqual(etv.central_tp_days(QS) + etv.J2017_5_JD, 2453849.66, places=6)
        want = 2452784.554043 - 135.55 / 360.0 * DE["P"]
        self.assertAlmostEqual(etv.central_tp_days(DE) + etv.J2017_5_JD, want, places=6)
        self.assertIsNone(etv.central_tp_days(VS))

    def test_los_shape_matches_paper_forms(self):
        for t in (2450100.0, 2452000.0, 2455000.0, 2458000.0, 2459500.0):
            want = math.sin(2.0 * math.pi * (t - 2452784.554043) / DE["P"] + math.radians(135.55))
            self.assertAlmostEqual(etv.los_shape(DE, t), want, delta=1e-9)
            want = math.sin(2.0 * math.pi * (t - 2451523.0485) / RR["P"] - 3.06)
            self.assertAlmostEqual(etv.los_shape(RR, t), want, delta=1e-9)
            e, w = 0.91, math.radians(215.95)
            E = solve_E(2.0 * math.pi * (t - 2453849.66) / QS["P"], e)
            want = math.sin(w) * (math.cos(E) - e) + math.cos(w) * math.sqrt(1.0 - e * e) * math.sin(E)
            self.assertAlmostEqual(etv.los_shape(QS, t), want, delta=1e-9)

    def test_eph_basis_identity(self):
        rng = random.Random(3)
        for _ in range(20):
            a, inc, om, Om = rng.uniform(0.1, 5), rng.uniform(0, 180), rng.uniform(0, 360), rng.uniform(0, 360)
            T1, T2 = etv.eph_basis(inc, om)
            ti = kepler.campbell_to_ti(a, inc, om, Om)
            for j in range(4):
                want = a * math.cos(math.radians(Om)) * T1[j] + a * math.sin(math.radians(Om)) * T2[j]
                self.assertAlmostEqual(ti[j], want, delta=1e-12)

    def test_cfg_and_grid(self):
        cfg = etv.claim_cfg(DE)
        self.assertAlmostEqual(cfg["P_min"], DE["P"] - 409.8105, delta=1e-6)
        self.assertAlmostEqual(cfg["P_step"], 2 * 409.8105 / 8, delta=1e-6)
        self.assertEqual(len(radio.make_grid(cfg)), 9 * 25)
        self.assertEqual(len(radio.make_grid(etv.claim_cfg(QS))), 9 * 37)
        c4 = etv.claim_cfg(HW4)
        self.assertAlmostEqual(c4["P_min"], 5100.0, delta=1e-6)
        self.assertAlmostEqual(c4["P_max"], 26100.0, delta=1e-6)
        self.assertEqual(len(radio.make_grid(c4)), 15 * 25)
        self.assertEqual(len(radio.make_grid(etv.claim_cfg(DE, SMALL))), 3 * 9)

    def test_tables_consistent(self):
        for key in etv.ORDER:
            self.assertIn(key, etv.CLAIMS)
            self.assertIn(etv.CLAIMS[key]["star"], etv.STARS)
        self.assertEqual(set(etv.ORDER), set(etv.CLAIMS))


class TestDraws(unittest.TestCase):
    def test_draw_orbit(self):
        rng = random.Random(7)
        cfg = etv.claim_cfg(QS)
        for _ in range(200):
            o = etv.draw_orbit(rng, "etv", 0.5, QS, cfg)
            self.assertTrue(cfg["P_min"] <= o["P"] <= cfg["P_max"])
            self.assertTrue(0.0 <= o["e"] <= 0.97)
            self.assertEqual(o["inc"], 75.5)
            self.assertTrue(0.0 <= o["Omega"] < 360.0)
            self.assertAlmostEqual(o["a0"], 0.5 * etv.a0_pred(QS), places=12)
            q = etv.draw_orbit(rng, "anyphase", 1.0, QS, cfg)
            self.assertTrue(0.0 <= q["Tp"] < q["P"])
        cd = etv.claim_cfg(DE)
        self.assertEqual(etv.draw_orbit(rng, "etv", 1.0, DE, cd)["e"], 0.0)
        cv = etv.claim_cfg(VS)
        for _ in range(20):
            o = etv.draw_orbit(rng, "etv", 1.0, VS, cv)
            self.assertTrue(0.0 <= o["Tp"] < o["P"])
        with self.assertRaises(ValueError):
            etv.draw_orbit(rng, "iso", 1.0, DE, cd)


class TestEphemerisFit(unittest.TestCase):
    def test_exact_circular(self):
        pts = synth(n=80, seed=5, orbit=central_orbit(DE, 1.2, 40.0), noise=False)
        sig = [p["sig"] for p in pts]
        f = etv.eph_fit(pts, sig, 0.0, DE)
        self.assertAlmostEqual(f["a0_mas"], 1.2, delta=1e-5)
        self.assertLess(angdiff(f["Omega_deg"], 40.0), 1e-3)
        self.assertGreater(f["dchi2"], 0.0)
        self.assertAlmostEqual(f["ratio_to_pred"], 1.2 / etv.a0_pred(DE), delta=1e-5)

    def test_exact_eccentric(self):
        pts = synth(n=80, seed=6, plx=19.9, pm=(41.9, 17.9), orbit=central_orbit(QS, 6.0, 200.0), noise=False)
        sig = [p["sig"] for p in pts]
        f = etv.eph_fit(pts, sig, 0.0, QS)
        self.assertAlmostEqual(f["a0_mas"], 6.0, delta=1e-5)
        self.assertLess(angdiff(f["Omega_deg"], 200.0), 1e-3)

    def test_none_without_phase(self):
        pts = synth(n=40, seed=7, plx=8.09, pm=(-52.9, -58.6))
        self.assertIsNone(etv.eph_fit(pts, [p["sig"] for p in pts], 0.0, VS))


class TestRun(unittest.TestCase):
    def test_detects_claimed_de_cvn_orbit(self):
        a0 = etv.a0_pred(DE)
        pts = synth(n=90, seed=21, orbit=central_orbit(DE, a0, 30.0))
        res = etv.run_claim(pts, DE, SMALL, log=lambda m: None)
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["n_grid"], 27)
        self.assertGreater(res["S_obs"], res["threshold"])
        self.assertIn(etv.verdict(res, True, DE), ("DETECTED", "SIGNAL_NOT_MATCHING"))
        self.assertLess(abs(res["ephemeris_fit"]["a0_mas"] - a0), 0.2)
        self.assertEqual(etv.verdict(res, False, DE), "INCONCLUSIVE")
        self.assertEqual(len(res["injections"]), 4)

    def test_absent_rr_cae_companion_is_excluded(self):
        pts = synth(n=90, seed=22, plx=47.2, pm=(30.0, -533.9), sigma=0.02)
        res = etv.run_claim(pts, RR, SMALL, log=lambda m: None)
        self.assertEqual(res["status"], "ok")
        at1 = [r for r in res["injections"] if r["scale"] == 1.0]
        self.assertEqual(sorted(r["family"] for r in at1), ["anyphase", "etv"])
        self.assertTrue(all(r["frac_above_observed"] == 1.0 for r in at1))
        # with 100 nulls the threshold is the null maximum, so pure noise
        # crosses it about 1% of the time; the verdict is checked when it does not
        if res["S_obs"] <= res["threshold"]:
            self.assertEqual(etv.verdict(res, True, RR), "EXCLUDED")
        self.assertEqual(len(res["periodogram"]), 3)
        self.assertIn("a0_95_exclusion_etv", res["limits"])

    def test_verdict_rules(self):
        def res(S, thr, a0, fr_etv, fr_any, status="ok"):
            return dict(status=status, S_obs=S, threshold=thr, best_fit=dict(a0_mas=a0), injections=[
                dict(family="etv", scale=0.5, frac_above_observed=0.1),
                dict(family="etv", scale=1.0, frac_above_observed=fr_etv),
                dict(family="anyphase", scale=1.0, frac_above_observed=fr_any)])
        self.assertEqual(etv.verdict(res(50.0, 30.0, 1.8, 1.0, 1.0), True, DE), "DETECTED")
        self.assertEqual(etv.verdict(res(50.0, 30.0, 6.0, 1.0, 1.0), True, DE), "SIGNAL_NOT_MATCHING")
        self.assertEqual(etv.verdict(res(50.0, 30.0, 0.5, 1.0, 1.0), True, DE), "SIGNAL_NOT_MATCHING")
        self.assertEqual(etv.verdict(res(10.0, 30.0, 0.4, 0.95, 0.97), True, DE), "EXCLUDED")
        self.assertEqual(etv.verdict(res(10.0, 30.0, 0.4, 0.95, 0.90), True, DE), "UPPER_LIMIT")
        self.assertEqual(etv.verdict(res(10.0, 30.0, 0.4, 1.0, 1.0), False, DE), "INCONCLUSIVE")
        self.assertEqual(etv.verdict(res(10.0, 30.0, 0.4, 1.0, 1.0, "no_epochs"), True, DE), "INCONCLUSIVE")

    def test_too_few_transits(self):
        res = etv.run_claim(synth(n=20, seed=14), DE, SMALL, log=lambda m: None)
        self.assertEqual(res["status"], "too_few_transits")
        self.assertEqual(etv.verdict(res, True, DE), "INCONCLUSIVE")


class TestSelect(unittest.TestCase):
    def test_select_qs_vir(self):
        star = etv.STARS["qsvir"]
        ra, dec = radio.propagate(star, 2017.5)
        near = dict(source_id="7", ra=repr(ra), dec=repr(dec), parallax=repr(star["dr3_plx"]),
                    phot_g_mean_mag=repr(star["dr3_g"]))
        far = dict(near, source_id="8", dec=repr(dec + 3.0 / 3600.0))
        sid, row, sep = radio.select_source([far, near], star)
        self.assertEqual(sid, 7)
        self.assertLess(sep, 1e-6)
        self.assertEqual(radio.select_source([far], star), (None, None, None))


if __name__ == "__main__":
    unittest.main()
