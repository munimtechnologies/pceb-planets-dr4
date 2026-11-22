"""Pre-registered Gaia DR4 test of circumbinary companions claimed from the
eclipse-timing variations (ETVs) of post-common-envelope binaries (PCEBs).
Rules: PREREGISTRATION.md (fixed 2026-11-22, before Gaia DR4).

Part of pceb-planets-dr4 (MV2, Munim, Inc.). Apache-2.0. Made with AI agents
working in MV2's name. Pure standard library.

Physics. A light-travel-time (LTT) fit to eclipse times measures the
binary's orbit about the system barycentre along the line of sight:
a_bin sin i' = c K. The same orbit moves the binary's photocentre on the
sky (the companion is dark), so Gaia should see an astrometric orbit of
angular semi-major axis
    a0 = c K plx / sin i'      (mas, with c K in au and plx in mas),
with the period, eccentricity, argument of periastron and phase of the LTT
fit. Only the position angle of the node (Omega) is unknown. We take the
companion coplanar with the eclipsing binary (i' = i_bin, close to 90 deg),
which gives the smallest wobble any orbit with that LTT amplitude can have
(to within 1 / sin i_bin <= 3.4 %). Any other inclination gives a larger a0.

Statistic S = max over a (P, e, Tp) grid inside each claim's period box of
delta chi^2 (5-parameter single star + orbit vs 5-parameter single star),
with one jitter term making the base fit's reduced chi^2 equal 1. S is
calibrated with a Gaussian null and a permutation null; sensitivity comes
from injecting the claimed orbit into the real epochs.
"""
import math
import random

from . import kepler, model, radio
from .linalg import wls

J2017_5_JD = radio.J2017_5_JD
C_AU_PER_S = 299792.458 / 149597870.7          # au per light-second
M_SUN_IN_MJUP = radio.M_SUN_IN_MJUP
YR = kepler.DAYS_PER_YEAR

# Gaia DR3 astrometry (epoch J2016.0), fetched 2026-11-22 from the Gaia TAP.
STARS = dict(
    decvn=dict(key="decvn", name="DE CVn", dr3_source_id=1550299304833675392,
               dr3_ra=201.7206797854293, dr3_dec=45.545520529192935,
               dr3_pmra=-202.30935727001676, dr3_pmdec=-175.7416802851047,
               dr3_plx=32.77275975184601, dr3_g=11.962571, dr3_ruwe=1.47, rv_kms=0.0),
    rrcae=dict(key="rrcae", name="RR Cae", dr3_source_id=4788741548375134336,
               dr3_ra=65.27338195414822, dr3_dec=-48.65433424724288,
               dr3_pmra=30.03001127448013, dr3_pmdec=-533.8860768572599,
               dr3_plx=47.20000835664457, dr3_g=13.643824, dr3_ruwe=1.84, rv_kms=0.0),
    qsvir=dict(key="qsvir", name="QS Vir", dr3_source_id=3612227169936143360,
               dr3_ra=207.4668699579405, dr3_dec=-13.226865062987894,
               dr3_pmra=41.877266047226826, dr3_pmdec=17.869107172296168,
               dr3_plx=19.915200746290253, dr3_g=13.607791, dr3_ruwe=1.95, rv_kms=0.0),
    hwvir=dict(key="hwvir", name="HW Vir", dr3_source_id=3675067076961979264,
               dr3_ra=191.08436809847007, dr3_dec=-8.67141649181095,
               dr3_pmra=9.008625314267656, dr3_pmdec=-15.829436055868877,
               dr3_plx=5.77283066187233, dr3_g=10.590079, dr3_ruwe=1.14, rv_kms=0.0),
    v893sco=dict(key="v893sco", name="V893 Sco", dr3_source_id=6039131391540808832,
                 dr3_ra=243.81215079788774, dr3_dec=-28.625847343745612,
                 dr3_pmra=-52.902242772509375, dr3_pmdec=-58.565604309334,
                 dr3_plx=8.091250910040184, dr3_g=14.560813, dr3_ruwe=None, rv_kms=0.0),
)

# The claims. P and sigma in days; angles in degrees; K in light-seconds
# (K = a_bin sin i' / c in every source below). phase is one of
#   ("phi", T_ref_jd, phi_deg, phi_sig_deg): LTT = K sin(2 pi (t - T_ref) / P + phi)
#   ("tp", Tp_jd, Tp_sig_days): time of periastron, LTT = K [sin w X + cos w Y]
#   None: no usable phase was published.
CLAIMS = dict(
    decvn=dict(
        key="decvn", star="decvn", claim="DE CVn b (11.2 yr giant planet)",
        ref="Han et al. 2018, ApJ 868, 53 (arXiv:1811.08625), Table 3 (circular LTT fit)",
        K_s=28.08, K_lo_sig=5.01, K_hi_sig=5.01,
        P=11.22 * YR, P_sig=0.36 * YR, e=0.0, e_sig=0.0, omega=0.0, omega_sig=0.0,
        phase=("phi", 2452784.554043, 135.55, 12.5), inc_bin=86.0, m_bin=0.92,
        e_grid=(0.0, 0.2, 0.4), e_max=0.5, n_P=9),
    rrcae=dict(
        key="rrcae", star="rrcae", claim="RR Cae b (16.6 yr giant planet, one-companion model)",
        ref="Rattanamala et al. 2023, MNRAS 523, 5086 (arXiv:2109.09397), Table 8, one-object model",
        K_s=14.0, K_lo_sig=1.0, K_hi_sig=1.0,
        P=16.6 * YR, P_sig=0.2 * YR, e=0.0, e_sig=0.0, omega=0.0, omega_sig=0.0,
        phase=("phi", 2451523.0485, math.degrees(-3.06), math.degrees(0.08)), inc_bin=82.9, m_bin=0.621,
        e_grid=(0.0, 0.2, 0.4), e_max=0.5, n_P=9),
    qsvir=dict(
        key="qsvir", star="qsvir", claim="QS Vir (AB) b (16.7 yr, ~58 M_Jup, e = 0.91)",
        ref="Giuppone et al. 2024, A&A 683, A129 (arXiv:2401.14897), Table 1, linear-ephemeris model "
            "(grid-search errors)",
        K_s=148.66, K_lo_sig=38.52, K_hi_sig=145.44,
        P=16.69 * YR, P_sig=0.445 * YR, e=0.91, e_sig=0.12, omega=215.95, omega_sig=16.6,
        phase=("tp", 2400000.5 + 53849.16, 27.1), inc_bin=75.5, m_bin=1.21,
        e_grid=(0.0, 0.5, 0.8, 0.91), e_max=0.97, n_P=9),
    hwvir_s2=dict(
        key="hwvir_s2", star="hwvir", claim="HW Vir signal 2 (3710 d, 6.3 M_Jup)",
        ref="Baycroft, Triaud & Kervella 2023, MNRAS 526, 2241 (arXiv:2309.05716), Table 5, signal 2",
        K_s=21.2, K_lo_sig=1.2, K_hi_sig=1.5,
        P=3710.0, P_sig=67.0, e=0.089, e_sig=0.037, omega=math.degrees(1.74), omega_sig=math.degrees(0.49),
        phase=("tp", 2442500.0, 365.0), inc_bin=80.6, m_bin=0.546,
        e_grid=(0.0, 0.2, 0.4), e_max=0.6, n_P=9),
    hwvir_s4=dict(
        key="hwvir_s4", star="hwvir", claim="HW Vir signal 4 (15600 d, 17 M_Jup, e = 0.69; the authors' favoured companion)",
        ref="Baycroft, Triaud & Kervella 2023, MNRAS 526, 2241 (arXiv:2309.05716), Table 5, signal 4",
        K_s=148.0, K_lo_sig=45.0, K_hi_sig=45.0,
        P=15600.0, P_sig=3500.0, e=0.6867, e_sig=0.013, omega=math.degrees(1.77), omega_sig=math.degrees(0.73),
        phase=("tp", 2433100.0, 130.0), inc_bin=80.6, m_bin=0.546,
        e_grid=(0.0, 0.35, 0.7), e_max=0.9, n_P=15),
    v893sco=dict(
        key="v893sco", star="v893sco", claim="V893 Sco b (10.2 yr giant planet)",
        ref="Bruch 2014, A&A 566, A101; values as tabulated by Souza & Baptista 2024 (arXiv:2408.07850), Table 1",
        K_s=22.0, K_lo_sig=3.0, K_hi_sig=3.0,
        P=10.2 * YR, P_sig=0.22 * YR, e=0.0, e_sig=0.0, omega=0.0, omega_sig=0.0,
        phase=None, inc_bin=72.5, m_bin=1.065,
        e_grid=(0.0, 0.2, 0.4), e_max=0.5, n_P=9),
)
ORDER = ("qsvir", "decvn", "rrcae", "hwvir_s2", "hwvir_s4", "v893sco")

DEFAULT_CFG = dict(
    n_null=2000, fap=0.001, scales=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0), n_inj=200,
    families=("etv", "anyphase"), clip=5.0, min_transits=30, seed=20261202, n_phase=12,
)
BAND_INCLINATION_FACTOR = 2.0   # DETECTED allows sin i' down to 0.5


# ---------------------------------------------------------------- predictions
def star_of(claim):
    return STARS[claim["star"]]


def a0_pred(claim):
    """Predicted photocentre semi-major axis (mas) for the coplanar orbit."""
    return claim["K_s"] * C_AU_PER_S * star_of(claim)["dr3_plx"] / math.sin(math.radians(claim["inc_bin"]))


def k_from_a0(a0_mas, claim):
    """Inverse of a0_pred: the LTT amplitude K (s) that a coplanar orbit of
    semi-major axis a0_mas would produce."""
    return a0_mas * math.sin(math.radians(claim["inc_bin"])) / (C_AU_PER_S * star_of(claim)["dr3_plx"])


def a0_band(claim):
    """(lo, hi) mas: K within -3 / +3 sigma, and sin i' >= 1 / BAND_INCLINATION_FACTOR."""
    a = a0_pred(claim)
    lo = a * max(0.0, 1.0 - 3.0 * claim["K_lo_sig"] / claim["K_s"])
    hi = a * (1.0 + 3.0 * claim["K_hi_sig"] / claim["K_s"]) * BAND_INCLINATION_FACTOR
    return lo, hi


def mass_mjup(a0_mas, P_days, claim):
    return kepler.companion_mass(a0_mas, star_of(claim)["dr3_plx"], P_days, claim["m_bin"]) * M_SUN_IN_MJUP


def central_tp_days(claim):
    """Periastron (or zero-phase) time in days from J2017.5; None if no phase."""
    ph = claim["phase"]
    if ph is None:
        return None
    if ph[0] == "tp":
        return ph[1] - J2017_5_JD
    if ph[0] == "phi":
        return ph[1] - ph[2] / 360.0 * claim["P"] - J2017_5_JD
    raise ValueError(ph[0])


def los_shape(claim, t_jd):
    """sin w X + cos w Y at t_jd for the central elements: the LTT delay in
    units of K (up to a constant)."""
    X, Y = kepler.xy(t_jd - J2017_5_JD, claim["P"], claim["e"], central_tp_days(claim))
    w = math.radians(claim["omega"])
    return math.sin(w) * X + math.cos(w) * Y


def claim_cfg(claim, cfg=None):
    full = dict(DEFAULT_CFG)
    full.update(e_grid=claim["e_grid"], e_max=claim["e_max"], n_P=claim["n_P"])
    full.update(cfg or {})
    half = max(3.0 * claim["P_sig"], 0.1 * claim["P"])
    full["P_min"] = claim["P"] - half
    full["P_max"] = claim["P"] + half
    full["P_step"] = (full["P_max"] - full["P_min"]) / (full["n_P"] - 1)
    return full


# ---------------------------------------------------------------- injections
def draw_orbit(rng, family, scale, claim, cfg):
    """'etv': every element drawn from the published value and error, the
    phase from the LTT ephemeris (uniform if none was published); 'anyphase':
    the same with a uniform phase. Both: i = i_bin, Omega uniform,
    a0 = scale x a0_pred."""
    P = min(cfg["P_max"], max(cfg["P_min"], rng.gauss(claim["P"], claim["P_sig"])))
    e = claim["e"]
    if claim["e_sig"] > 0:
        e = min(cfg["e_max"], max(0.0, rng.gauss(claim["e"], claim["e_sig"])))
    om = claim["omega"] + (rng.gauss(0.0, claim["omega_sig"]) if claim["omega_sig"] > 0 else 0.0)
    Om = rng.uniform(0.0, 360.0)
    ph = claim["phase"]
    if family == "etv":
        if ph is None:
            Tp = rng.uniform(0.0, P)
        elif ph[0] == "tp":
            Tp = ph[1] + rng.gauss(0.0, ph[2]) - J2017_5_JD
        else:
            phi = ph[2] + rng.gauss(0.0, ph[3])
            Tp = ph[1] - phi / 360.0 * P - J2017_5_JD
    elif family == "anyphase":
        Tp = rng.uniform(0.0, P)
    else:
        raise ValueError(family)
    return dict(P=P, e=e, Tp=Tp, a0=scale * a0_pred(claim), inc=claim["inc_bin"], omega=om, Omega=Om)


def signal(points, orb):
    return radio.signal(points, orb)


# ---------------------------------------------------------------- ephemeris fit
def eph_basis(inc, omega):
    """Thiele-Innes vectors T1, T2 with campbell_to_ti(a, inc, omega, W) =
    a cos W T1 + a sin W T2 (exact identity)."""
    return kepler.campbell_to_ti(1.0, inc, omega, 0.0), kepler.campbell_to_ti(1.0, inc, omega, 90.0)


def eph_fit(pts, sig, mu_r, claim):
    """Diagnostic: with P, e, omega, phase and i fixed at the published
    values, the only free orbit parameters are u = a0 cos W, v = a0 sin W.
    Returns the linear fit of (u, v) next to the 5 base terms, or None when
    no phase was published."""
    tp = central_tp_days(claim)
    if tp is None:
        return None
    T1, T2 = eph_basis(claim["inc_bin"], claim["omega"])
    s1 = model.orbit_signal(pts, T1, claim["P"], claim["e"], tp)
    s2 = model.orbit_signal(pts, T2, claim["P"], claim["e"], tp)
    cb, y = radio.base_cols(pts, mu_r, 5)
    f0 = wls(cb, y, sig)
    f1 = wls(cb + [s1, s2], y, sig)
    u, v = f1["beta"][5], f1["beta"][6]
    cuu, cvv, cuv = f1["cov"][5][5], f1["cov"][6][6], f1["cov"][5][6]
    a = math.hypot(u, v)
    var = (u * u * cuu + v * v * cvv + 2.0 * u * v * cuv) / (a * a) if a > 0 else cuu
    pred = a0_pred(claim)
    return dict(a0_mas=a, a0_err_mas=math.sqrt(max(0.0, var)), Omega_deg=math.degrees(math.atan2(v, u)) % 360.0,
                dchi2=f0["chi2"] - f1["chi2"], a0_pred_mas=pred, ratio_to_pred=a / pred,
                K_equiv_s=k_from_a0(a, claim))


# ---------------------------------------------------------------- the test
def run_claim(points, claim, cfg=None, log=print):
    cfg = claim_cfg(claim, cfg)
    star = star_of(claim)
    rng = random.Random(cfg["seed"])
    pred = a0_pred(claim)
    res = dict(target=claim["key"], star=star["key"], claim=claim["claim"], published=dict(
        K_s=claim["K_s"], P_days=claim["P"], e=claim["e"], omega_deg=claim["omega"], inc_bin_deg=claim["inc_bin"],
        a0_pred_mas=pred, a0_band=list(a0_band(claim)), mass_mjup_coplanar=mass_mjup(pred, claim["P"], claim),
        ref=claim["ref"]),
        cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()})
    if len(points) < cfg["min_transits"]:
        res.update(status="too_few_transits", n_transits=len(points))
        return res
    mu_r = model.mu_r_from_rv(star["rv_kms"], star["dr3_plx"])
    cl = radio.clean(points, mu_r, 5, cfg["clip"])
    pts, sig = cl["points"], cl["sig"]
    log("%s base 5p: N=%d clipped=%d jitter=%.4f mas plx=%.4f+-%.4f" % (
        claim["key"], cl["n"], cl["n_clipped"], cl["eps"], cl["params"][2][0], cl["params"][2][1]))

    grid = radio.make_grid(cfg)
    gs = radio.GridStatN(pts, sig, grid, mu_r, 5)
    rw = [r / s for r, s in zip(cl["fit"]["resid"], sig)]
    vals = gs.values(rw)
    g_obs = max(range(len(vals)), key=lambda g: vals[g])
    S_obs = vals[g_obs]
    per_P = {}
    for (P, e, Tp), v in zip(grid, vals):
        per_P[P] = max(per_P.get(P, 0.0), v)
    log("%s S_obs=%.3f at P=%.1f e=%.2f (grid %d)" % (claim["key"], S_obs, grid[g_obs][0], grid[g_obs][1], len(grid)))

    N = len(pts)
    null_g = [gs.scan([rng.gauss(0.0, 1.0) for _ in range(N)])[0] for _ in range(cfg["n_null"])]
    null_p = []
    for _ in range(cfg["n_null"]):
        perm = list(rw)
        rng.shuffle(perm)
        null_p.append(gs.scan(perm)[0])
    q = 1.0 - cfg["fap"]
    thr_g, thr_p = radio.quantile(null_g, q), radio.quantile(null_p, q)
    thr = max(thr_g, thr_p)
    p_g = (1 + sum(1 for v in null_g if v >= S_obs)) / float(1 + len(null_g))
    p_p = (1 + sum(1 for v in null_p if v >= S_obs)) / float(1 + len(null_p))
    log("%s thresholds gauss=%.3f perm=%.3f p=%.4g/%.4g" % (claim["key"], thr_g, thr_p, p_g, p_p))

    inj_rows = []
    table = {}
    for fam in cfg["families"]:
        for sc in cfg["scales"]:
            n_det = n_exc = 0
            for _ in range(cfg["n_inj"]):
                orb = draw_orbit(rng, fam, sc, claim, cfg)
                s = signal(pts, orb)
                S = gs.scan([r + si / sg for r, si, sg in zip(rw, s, sig)])[0]
                n_det += S > thr
                n_exc += S > S_obs
            fd, fe = n_det / float(cfg["n_inj"]), n_exc / float(cfg["n_inj"])
            table[(fam, sc)] = (fd, fe)
            a0 = sc * pred
            inj_rows.append(dict(family=fam, scale=sc, a0_mas=a0, K_s=sc * claim["K_s"],
                                 mass_mjup_at_P=mass_mjup(a0, claim["P"], claim),
                                 n=cfg["n_inj"], frac_detected=fd, frac_above_observed=fe))
            log("%s inj %s x%.2f (a0 %.3f mas) det=%.3f exc=%.3f" % (claim["key"], fam, sc, a0, fd, fe))
    limits = {}
    a0s = [sc * pred for sc in cfg["scales"]]
    for fam in cfg["families"]:
        det = [table[(fam, sc)][0] for sc in cfg["scales"]]
        exc = [table[(fam, sc)][1] for sc in cfg["scales"]]
        for name, fr, lev in (("a0_50_detect", det, 0.5), ("a0_90_detect", det, 0.9),
                              ("a0_95_exclusion", exc, 0.95)):
            v, flag = radio.crossing(a0s, fr, lev)
            limits["%s_%s" % (name, fam)] = dict(
                a0_mas=v, flag=flag, K_s=k_from_a0(v, claim) if v else None,
                mass_mjup_at_P=mass_mjup(v, claim["P"], claim) if v else None)

    best = radio.refine(pts, sig, mu_r, 5, cfg, grid[g_obs])
    best["mass_mjup"] = mass_mjup(best["a0_mas"], best["P"], claim) if best["a0_mas"] > 0 else 0.0
    eph = eph_fit(pts, sig, mu_r, claim)
    log("%s best P=%.1f e=%.2f a0=%.4f+-%.4f mas; ephemeris fit %s" % (
        claim["key"], best["P"], best["e"], best["a0_mas"], best["a0_err_mas"],
        "a0=%.4f+-%.4f (pred %.4f)" % (eph["a0_mas"], eph["a0_err_mas"], pred) if eph else "n/a"))
    res.update(
        status="ok", n_transits=N, n_clipped=cl["n_clipped"], jitter_mas=cl["eps"],
        median_sigma_mas=sorted(sig)[N // 2], span_years=pts[-1]["t"] - pts[0]["t"],
        base_params=cl["params"], chi2_base=cl["chi2"], n_grid=len(grid),
        S_obs=S_obs, grid_best=dict(P=grid[g_obs][0], e=grid[g_obs][1], Tp=grid[g_obs][2]),
        threshold=thr, threshold_gauss=thr_g, threshold_perm=thr_p, p_gauss=p_g, p_perm=p_p,
        null_quantiles=dict(gauss={str(qq): radio.quantile(null_g, qq) for qq in (0.5, 0.9, 0.99, 0.999)},
                            perm={str(qq): radio.quantile(null_p, qq) for qq in (0.5, 0.9, 0.99, 0.999)}),
        injections=inj_rows, limits=limits, best_fit=best, ephemeris_fit=eph,
        periodogram=[dict(P=P, S=per_P[P]) for P in sorted(per_P)])
    return res


def verdict(res, validation_ok, claim):
    """Pre-registered verdict (PREREGISTRATION.md, section 5)."""
    if not validation_ok or res.get("status") != "ok":
        return "INCONCLUSIVE"
    if res["S_obs"] > res["threshold"]:
        lo, hi = a0_band(claim)
        return "DETECTED" if lo <= res["best_fit"]["a0_mas"] <= hi else "SIGNAL_NOT_MATCHING"
    at1 = [r for r in res["injections"] if abs(r["scale"] - 1.0) < 1e-9]
    fams = set(r["family"] for r in at1)
    if at1 and fams >= {"etv", "anyphase"} and all(r["frac_above_observed"] >= 0.95 for r in at1):
        return "EXCLUDED"
    return "UPPER_LIMIT"
