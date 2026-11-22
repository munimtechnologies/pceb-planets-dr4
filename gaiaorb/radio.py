"""Pre-registered Gaia DR4 test of the two VLBA radio-astrometry planets:
TVLM 513-46546 b (Curiel et al. 2020) and GJ 896 Ab = EQ Peg Ab (Curiel et
al. 2022). Rules: PREREGISTRATION.md (fixed 2026-11-22, before Gaia DR4).

Part of radio-planets-dr4 (MV2, Munim, Inc.). Apache-2.0. Made with AI
agents working in MV2's name. Pure standard library.

Statistic S = max over a (P, e, Tp) grid inside each planet's published
period box of delta chi^2 (base + orbit vs base). The base model is the
5-parameter single-star model (perspective acceleration included) plus, for
GJ 896 A, acceleration and jerk terms for its 229-yr binary orbit (9 terms).
One jitter term makes the base fit's reduced chi^2 equal 1. S is calibrated
by a Gaussian null and a permutation null; sensitivity comes from injecting
the published orbit (and scaled versions) into the real epochs.
"""
import math
import random

from . import kepler, model
from .linalg import extend_basis, wls

J2017_5_JD = 2457936.875
M_SUN_IN_MJUP = 1047.5655

TARGETS = dict(
    tvlm513=dict(
        key="tvlm513", name="TVLM 513-46546", planet="TVLM 513-46546 b",
        ref=("Curiel et al. 2020, AJ 160, 97 (arXiv:2008.01595): Table 4, VLBA_combined fit "
             "(P, T0, Omega, i); a0 = 0.145 +- 0.019 mas is the paper's adopted weighted value"),
        dr3_source_id=1262763648230973440,
        dr3_ra=225.2839023288742, dr3_dec=22.83363768365745,
        dr3_pmra=-43.11951859553303, dr3_pmdec=-65.13802373033424,
        dr3_plx=93.16551240283296, dr3_g=16.509804,
        rv_kms=0.0, mass_msun=0.07, n_poly=5,
        P=220.0, P_sig=5.0, e=0.0, e_sig=0.0, omega=0.0, omega_sig=0.0,
        Omega=139.0, Omega_sig=39.0, inc=71.0, inc_sig=38.0,
        Tp_jd=2457631.0, Tp_sig=22.0, a0=0.145, a0_sig=0.019,
        # widest published fit: VLBA_new 0.17 +- 0.10 mas -> +3 sigma = 0.47 mas
        a0_band=(0.0, 0.47),
        grid=dict(P_min=190.0, P_max=260.0, P_step=2.0, e_grid=(0.0, 0.1, 0.2, 0.3), n_phase=12)),
    gj896a=dict(
        key="gj896a", name="GJ 896 A (EQ Peg A)", planet="GJ 896 Ab (EQ Peg Ab)",
        ref=("Curiel et al. 2022, AJ 164, 93 (arXiv:2208.14553): Table 3, full combined fit "
             "(T0_Ab, P_Ab, e_Ab, omega_A, Omega_Ab, i_Ab, a_A)"),
        dr3_source_id=2824770686019003904,
        dr3_ra=352.97012375450475, dr3_dec=19.93699282660293,
        dr3_pmra=578.0086593589436, dr3_pmdec=-59.768604898823476,
        dr3_plx=159.66339203721853, dr3_g=9.043998,
        rv_kms=0.0, mass_msun=0.436, n_poly=9,
        P=284.39, P_sig=1.47, e=0.35, e_sig=0.19, omega=353.11, omega_sig=11.81,
        Omega=45.62, Omega_sig=9.60, inc=69.20, inc_sig=25.61,
        Tp_jd=2455702.65, Tp_sig=17.26, a0=0.51, a0_sig=0.15,
        a0_band=(0.06, 0.96),   # 0.51 +- 3 x 0.15 mas
        grid=dict(P_min=270.0, P_max=300.0, P_step=1.0, e_grid=(0.0, 0.15, 0.3, 0.45, 0.6), n_phase=12)),
)

DEFAULT_CFG = dict(
    n_null=2000, fap=0.001, scales=(0.25, 0.5, 0.75, 1.0, 1.5, 2.0), n_inj=200,
    families=("pub", "iso"), clip=5.0, min_transits=30, seed=20261202, e_max=0.8,
)


# ---------------------------------------------------------------- base model
def base_cols(points, mu_r=0.0, n_poly=5):
    """5 single-star columns (model.five_cols), plus 0.5 t^2 (7 terms) and
    t^3 / 6 (9 terms) along-scan columns, as in Gaia's 7p/9p solutions."""
    if n_poly not in (5, 7, 9):
        raise ValueError("n_poly must be 5, 7 or 9")
    cols, y = model.five_cols(points, mu_r)
    if n_poly >= 7:
        cols.append([0.5 * p["t"] ** 2 * p["s"] for p in points])
        cols.append([0.5 * p["t"] ** 2 * p["c"] for p in points])
    if n_poly >= 9:
        cols.append([p["t"] ** 3 / 6.0 * p["s"] for p in points])
        cols.append([p["t"] ** 3 / 6.0 * p["c"] for p in points])
    return cols, y


def clean(points, mu_r=0.0, n_poly=5, clip=5.0):
    """Base fit with one jitter term and one pass of |r|/sigma > clip rejection
    (same rules as model.clean_5p, with n_poly base terms)."""
    pts = list(points)
    n_clipped = 0
    for rnd in range(2):
        cols, y = base_cols(pts, mu_r, n_poly)
        base = [p["sig"] for p in pts]
        eps = model.fit_jitter(cols, y, base, n_poly)
        sig = model._sig(base, eps)
        fit = wls(cols, y, sig)
        if rnd == 1 or clip is None:
            break
        keep = [i for i in range(len(pts)) if abs(fit["resid"][i]) <= clip * sig[i]]
        if len(keep) == len(pts):
            break
        n_clipped = len(pts) - len(keep)
        pts = [pts[i] for i in keep]
    params = [(fit["beta"][a], math.sqrt(fit["cov"][a][a])) for a in range(n_poly)]
    return dict(points=pts, sig=sig, eps=eps, fit=fit, params=params, n_clipped=n_clipped,
                chi2=fit["chi2"], n=len(pts), mu_r=mu_r, n_poly=n_poly)


class GridStatN(model.GridStat):
    """model.GridStat with an n_poly-term base model."""

    def __init__(self, points, sig, grid, mu_r=0.0, n_poly=5):
        self.points = points
        self.N = len(points)
        self.wt = [1.0 / s for s in sig]
        cols, _ = base_cols(points, mu_r, n_poly)
        self.V = extend_basis([], [[c[i] * self.wt[i] for i in range(self.N)] for c in cols])
        self.grid = list(grid)
        self.U = []
        for (P, e, Tp) in self.grid:
            oc = model.orbit_cols(points, P, e, Tp)
            self.U.append(extend_basis(self.V, [[c[i] * self.wt[i] for i in range(self.N)]
                                                for c in oc]))


def make_grid(cfg):
    grid = []
    n = int(round((cfg["P_max"] - cfg["P_min"]) / cfg["P_step"]))
    for k in range(n + 1):
        P = cfg["P_min"] + k * cfg["P_step"]
        for e in cfg["e_grid"]:
            if e == 0.0:
                grid.append((P, 0.0, 0.0))
            else:
                for j in range(cfg["n_phase"]):
                    grid.append((P, e, j * P / cfg["n_phase"]))
    return grid


def quantile(vals, q):
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(math.ceil(q * len(s))) - 1))
    return s[k]


def crossing(xs, fracs, level):
    """Smallest x at which fracs first reaches `level` (linear interpolation);
    (value, 'interp' | 'below_grid' | 'above_grid')."""
    for k, (x, f) in enumerate(zip(xs, fracs)):
        if f >= level:
            if k == 0:
                return xs[0], "below_grid"
            x0, f0 = xs[k - 1], fracs[k - 1]
            return x0 + (level - f0) * (x - x0) / (f - f0), "interp"
    return None, "above_grid"


# ---------------------------------------------------------------- orbits
def tp_days(star):
    """Published periastron (or node) time in days from J2017.5."""
    return star["Tp_jd"] - J2017_5_JD


def published_orbit(star, a0=None):
    return dict(P=star["P"], e=star["e"], Tp=tp_days(star), a0=star["a0"] if a0 is None else a0,
                inc=star["inc"], omega=star["omega"], Omega=star["Omega"])


def draw_orbit(rng, family, scale, star, cfg):
    """'pub': every element drawn from the published value and error (Tp
    propagated with the drawn period, as the model phase is 2 pi (t - Tp)/P);
    'iso': published P and e, random orientation and phase. a0 = scale x the
    published a0."""
    P = min(cfg["P_max"], max(cfg["P_min"], rng.gauss(star["P"], star["P_sig"])))
    e = min(cfg["e_max"], max(0.0, rng.gauss(star["e"], star["e_sig"])))
    a0 = scale * star["a0"]
    if family == "pub":
        inc = min(180.0, max(0.0, rng.gauss(star["inc"], star["inc_sig"])))
        Om = rng.gauss(star["Omega"], star["Omega_sig"])
        om = rng.gauss(star["omega"], star["omega_sig"])
        Tp = tp_days(star) + rng.gauss(0.0, star["Tp_sig"])
    elif family == "iso":
        inc = math.degrees(math.acos(rng.uniform(-1.0, 1.0)))
        Om = rng.uniform(0.0, 360.0)
        om = rng.uniform(0.0, 360.0)
        Tp = rng.uniform(0.0, P)
    else:
        raise ValueError(family)
    return dict(P=P, e=e, Tp=Tp, a0=a0, inc=inc, omega=om, Omega=Om)


def signal(points, orb):
    ti = kepler.campbell_to_ti(orb["a0"], orb["inc"], orb["omega"], orb["Omega"])
    return model.orbit_signal(points, ti, orb["P"], orb["e"], orb["Tp"])


def mass_mjup(a0_mas, P_days, star):
    return kepler.companion_mass(a0_mas, star["dr3_plx"], P_days, star["mass_msun"]) * M_SUN_IN_MJUP


# ---------------------------------------------------------------- fits
def fit_at(pts, sig, mu_r, n_poly, P, e, Tp):
    """Linear fit (base + Thiele-Innes) at fixed (P, e, Tp)."""
    cb, y = base_cols(pts, mu_r, n_poly)
    fit = wls(cb + model.orbit_cols(pts, P, e, Tp), y, sig)
    b = fit["beta"]
    ti = tuple(b[n_poly:n_poly + 4])
    a0, inc, om, Om = kepler.ti_to_campbell(*ti)
    cov = [row[n_poly:n_poly + 4] for row in fit["cov"][n_poly:n_poly + 4]]
    J = []
    for j in range(4):
        h = 1e-6 * max(1e-3, abs(ti[j]) + a0)
        tt = list(ti)
        tt[j] += h
        J.append((kepler.ti_to_campbell(*tt)[0] - a0) / h)
    var_a = sum(J[p] * cov[p][q] * J[q] for p in range(4) for q in range(4))
    return dict(P=P, e=e, Tp=Tp, chi2=fit["chi2"], ti=dict(zip(model.PARAMS_TI, ti)),
                a0_mas=a0, a0_err_mas=math.sqrt(max(0.0, var_a)), inc_deg=inc,
                omega_deg=om, Omega_deg=Om)


def refine(pts, sig, mu_r, n_poly, cfg, start):
    """Nelder-Mead in (P, e, phase) from a grid point, P kept in the box."""
    cb, y = base_cols(pts, mu_r, n_poly)
    chi2_base = wls(cb, y, sig)["chi2"]

    def unpack(x):
        P = x[0]
        if not (cfg["P_min"] <= P <= cfg["P_max"]):
            return None
        return P, min(abs(x[1]), cfg["e_max"]), (x[2] % 1.0) * P

    def obj(x):
        u = unpack(x)
        if u is None:
            return 1e300
        return -(chi2_base - fit_at(pts, sig, mu_r, n_poly, *u)["chi2"])

    P0, e0, Tp0 = start
    x, val = model.nelder_mead(obj, [P0, e0 if e0 > 0 else 0.05, Tp0 / P0],
                               [max(1.0, 0.5 * cfg["P_step"]), 0.05, 0.05], max_iter=300)
    out = fit_at(pts, sig, mu_r, n_poly, *unpack(x))
    out["dchi2"] = -val
    return out


def template_fit(pts, sig, mu_r, n_poly, star):
    """Diagnostic: amplitude k of the fixed published orbit (k = 1: as
    published; k = -1: same orbit with omega + 180 deg; k = 0: absent)."""
    tmpl = signal(pts, published_orbit(star))
    cb, y = base_cols(pts, mu_r, n_poly)
    f0 = wls(cb, y, sig)
    f1 = wls(cb + [tmpl], y, sig)
    return dict(k=f1["beta"][n_poly], k_err=math.sqrt(f1["cov"][n_poly][n_poly]),
                dchi2=f0["chi2"] - f1["chi2"])


# ---------------------------------------------------------------- the test
def run_target(points, star, cfg=None, log=print):
    full = dict(DEFAULT_CFG)
    full.update(star["grid"])
    full.update(cfg or {})
    cfg = full
    rng = random.Random(cfg["seed"])
    n_poly = star["n_poly"]
    res = dict(target=star["key"], planet=star["planet"], published=dict(
        P=star["P"], e=star["e"], a0_mas=star["a0"], a0_band=list(star["a0_band"]), ref=star["ref"]),
        cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()})
    if len(points) < cfg["min_transits"]:
        res.update(status="too_few_transits", n_transits=len(points))
        return res
    mu_r = model.mu_r_from_rv(star["rv_kms"], star["dr3_plx"])
    cl = clean(points, mu_r, n_poly, cfg["clip"])
    pts, sig = cl["points"], cl["sig"]
    log("%s base %dp: N=%d clipped=%d jitter=%.4f mas plx=%.4f+-%.4f" % (
        star["key"], n_poly, cl["n"], cl["n_clipped"], cl["eps"], cl["params"][2][0], cl["params"][2][1]))

    grid = make_grid(cfg)
    gs = GridStatN(pts, sig, grid, mu_r, n_poly)
    rw = [r / s for r, s in zip(cl["fit"]["resid"], sig)]
    vals = gs.values(rw)
    g_obs = max(range(len(vals)), key=lambda g: vals[g])
    S_obs = vals[g_obs]
    per_P = {}
    for (P, e, Tp), v in zip(grid, vals):
        per_P[P] = max(per_P.get(P, 0.0), v)
    log("%s S_obs=%.3f at P=%.1f e=%.2f (grid %d)" % (star["key"], S_obs, grid[g_obs][0], grid[g_obs][1],
                                                     len(grid)))

    N = len(pts)
    null_g = [gs.scan([rng.gauss(0.0, 1.0) for _ in range(N)])[0] for _ in range(cfg["n_null"])]
    null_p = []
    for _ in range(cfg["n_null"]):
        perm = list(rw)
        rng.shuffle(perm)
        null_p.append(gs.scan(perm)[0])
    q = 1.0 - cfg["fap"]
    thr_g, thr_p = quantile(null_g, q), quantile(null_p, q)
    thr = max(thr_g, thr_p)
    p_g = (1 + sum(1 for v in null_g if v >= S_obs)) / float(1 + len(null_g))
    p_p = (1 + sum(1 for v in null_p if v >= S_obs)) / float(1 + len(null_p))
    log("%s thresholds gauss=%.3f perm=%.3f p=%.4g/%.4g" % (star["key"], thr_g, thr_p, p_g, p_p))

    inj_rows = []
    table = {}
    for fam in cfg["families"]:
        for sc in cfg["scales"]:
            n_det = n_exc = 0
            for _ in range(cfg["n_inj"]):
                orb = draw_orbit(rng, fam, sc, star, cfg)
                s = signal(pts, orb)
                S = gs.scan([r + si / sg for r, si, sg in zip(rw, s, sig)])[0]
                n_det += S > thr
                n_exc += S > S_obs
            fd, fe = n_det / float(cfg["n_inj"]), n_exc / float(cfg["n_inj"])
            table[(fam, sc)] = (fd, fe)
            a0 = sc * star["a0"]
            inj_rows.append(dict(family=fam, scale=sc, a0_mas=a0, mass_mjup_at_P=mass_mjup(a0, star["P"], star),
                                 n=cfg["n_inj"], frac_detected=fd, frac_above_observed=fe))
            log("%s inj %s x%.2f (a0 %.3f mas) det=%.3f exc=%.3f" % (star["key"], fam, sc, a0, fd, fe))
    limits = {}
    a0s = [sc * star["a0"] for sc in cfg["scales"]]
    for fam in cfg["families"]:
        det = [table[(fam, sc)][0] for sc in cfg["scales"]]
        exc = [table[(fam, sc)][1] for sc in cfg["scales"]]
        for name, fr, lev in (("a0_50_detect", det, 0.5), ("a0_90_detect", det, 0.9),
                              ("a0_95_exclusion", exc, 0.95)):
            v, flag = crossing(a0s, fr, lev)
            limits["%s_%s" % (name, fam)] = dict(
                a0_mas=v, flag=flag, mass_mjup_at_P=mass_mjup(v, star["P"], star) if v else None)

    best = refine(pts, sig, mu_r, n_poly, cfg, grid[g_obs])
    best["mass_mjup"] = mass_mjup(best["a0_mas"], best["P"], star) if best["a0_mas"] > 0 else 0.0
    tmpl = template_fit(pts, sig, mu_r, n_poly, star)
    log("%s best P=%.2f e=%.2f a0=%.4f+-%.4f mas; template k=%.3f+-%.3f" % (
        star["key"], best["P"], best["e"], best["a0_mas"], best["a0_err_mas"], tmpl["k"], tmpl["k_err"]))
    res.update(
        status="ok", n_transits=N, n_clipped=cl["n_clipped"], jitter_mas=cl["eps"],
        median_sigma_mas=sorted(p["sig"] for p in pts)[N // 2], span_years=pts[-1]["t"] - pts[0]["t"],
        base_params=cl["params"], chi2_base=cl["chi2"], n_grid=len(grid),
        S_obs=S_obs, grid_best=dict(P=grid[g_obs][0], e=grid[g_obs][1], Tp=grid[g_obs][2]),
        threshold=thr, threshold_gauss=thr_g, threshold_perm=thr_p, p_gauss=p_g, p_perm=p_p,
        null_quantiles=dict(gauss={str(qq): quantile(null_g, qq) for qq in (0.5, 0.9, 0.99, 0.999)},
                            perm={str(qq): quantile(null_p, qq) for qq in (0.5, 0.9, 0.99, 0.999)}),
        injections=inj_rows, limits=limits, best_fit=best, template=tmpl,
        periodogram=[dict(P=P, S=per_P[P]) for P in sorted(per_P)])
    return res


def verdict(res, validation_ok, star):
    """Pre-registered verdict (PREREGISTRATION.md, section 5)."""
    if not validation_ok or res.get("status") != "ok":
        return "INCONCLUSIVE"
    if res["S_obs"] > res["threshold"]:
        lo, hi = star["a0_band"]
        return "DETECTED" if lo <= res["best_fit"]["a0_mas"] <= hi else "SIGNAL_NOT_MATCHING"
    at1 = [r for r in res["injections"] if abs(r["scale"] - 1.0) < 1e-9]
    fams = set(r["family"] for r in at1)
    if at1 and fams >= {"pub", "iso"} and all(r["frac_above_observed"] >= 0.95 for r in at1):
        return "EXCLUDED"
    return "UPPER_LIMIT"


# ---------------------------------------------------------------- source lookup
def propagate(star, epoch=2017.5):
    """DR3 (J2016.0) position moved linearly to `epoch` (deg)."""
    dt = epoch - 2016.0
    dec = star["dr3_dec"] + star["dr3_pmdec"] * dt / 3.6e6
    ra = star["dr3_ra"] + star["dr3_pmra"] * dt / 3.6e6 / math.cos(math.radians(star["dr3_dec"]))
    return ra, dec


def sep_arcsec(ra1, dec1, ra2, dec2):
    """Haversine separation in arcsec (accurate at sub-arcsecond scales)."""
    r1, d1, r2, d2 = (math.radians(v) for v in (ra1, dec1, ra2, dec2))
    h = math.sin(0.5 * (d2 - d1)) ** 2 + math.cos(d1) * math.cos(d2) * math.sin(0.5 * (r2 - r1)) ** 2
    return math.degrees(2.0 * math.asin(min(1.0, math.sqrt(h)))) * 3600.0


def select_source(rows, star, epoch=2017.5, max_sep=2.0):
    """Pick the DR4 row for the target: parallax within 10% of DR3, G within
    0.75 mag, nearest to the propagated DR3 position, closer than max_sep
    arcsec. Returns (source_id, row, sep) or (None, None, None)."""
    ra0, dec0 = propagate(star, epoch)
    best = None
    for r in rows:
        try:
            ra, dec, plx = float(r["ra"]), float(r["dec"]), float(r["parallax"])
        except (KeyError, TypeError, ValueError):
            continue
        if abs(plx - star["dr3_plx"]) > 0.1 * star["dr3_plx"]:
            continue
        g = r.get("phot_g_mean_mag")
        try:
            if g not in (None, "") and abs(float(g) - star["dr3_g"]) > 0.75:
                continue
        except ValueError:
            pass
        s = sep_arcsec(ra, dec, ra0, dec0)
        if s <= max_sep and (best is None or s < best[2]):
            best = (int(r["source_id"]), r, s)
    return best if best else (None, None, None)
