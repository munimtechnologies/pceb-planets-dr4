#!/usr/bin/env python3
"""The pre-registered Gaia DR4 test of the circumbinary companions claimed
from eclipse timing of post-common-envelope binaries (QS Vir, DE CVn,
RR Cae, HW Vir, V893 Sco). To be run on or after the Gaia DR4 release
(2026-12-02).

MV2 / Munim, Inc. Apache-2.0. Made with AI agents working in MV2's name.
Rules: PREREGISTRATION.md. Only the I/O (source lookup, download URLs and
file format) may be adapted after the release; the analysis may not.

Usage:
  python etv_job.py --data-dir /data --out-dir /out [--validated]
        [--claims qsvir,decvn,...] [--source-id STAR=N ...] [--epochs-file STAR=PATH ...]
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import io_gaia  # noqa: E402
from gaiaorb import epochs, etv, radio  # noqa: E402

T0 = time.time()
VERSION = "0.1.0"


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def resolve_source(star):
    ra, dec = radio.propagate(star, 2017.5)
    for table in ("gaiadr4.gaia_source", "gaiadr4.gaia_source_lite"):
        q = ("SELECT source_id, ra, dec, parallax, phot_g_mean_mag FROM %s WHERE "
             "1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', %.7f, %.7f, 0.003))" % (table, ra, dec))
        try:
            rows = io_gaia.tap_csv(q)
            log("%s %s: %d rows" % (star["key"], table, len(rows)))
            sid, row, sep = radio.select_source(rows, star)
            if sid is not None:
                return sid, dict(row, sep_arcsec=sep, table=table)
        except Exception as e:
            log("TAP %s failed: %s" % (table, e))
    return star["dr3_source_id"], dict(note="DR4 lookup failed; using the DR3 source_id")


def kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def load_points(star, args, sids, files):
    key = star["key"]
    if key in sids:
        sid, src = int(sids[key]), dict(note="given on the command line")
    else:
        sid, src = resolve_source(star)
    log("%s source_id %s" % (key, sid))
    paths = []
    if key in files:
        paths = [files[key]]
    else:
        for url in io_gaia.datalink_urls(sid):
            try:
                blob = io_gaia.fetch(url)
                if len(blob) < 200:
                    log("short response from %s: %r" % (url, blob[:200]))
                    continue
                paths = io_gaia.save_epochs(blob, args.data_dir, "%s-epochs" % key)
                log("epochs from %s -> %s" % (url, paths))
                if paths:
                    break
            except Exception as e:
                log("fetch failed %s: %s" % (url, e))
    if not paths:
        return sid, src, None, None
    rows = []
    for p in paths:
        rows.extend(io_gaia.read_table_rows(p))
    by = epochs.rows_by_source(rows)
    mine = by.get(sid) or (list(by.values())[0] if len(by) == 1 else [])
    points, counts = epochs.normal_points(mine, inflate=True)
    log("%s rows %d -> points %d (%s)" % (key, len(mine), len(points), counts))
    return sid, src, points, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--out-dir", default="/out")
    ap.add_argument("--validated", action="store_true",
                    help="set only if the proxima-dr4-validate verdict was VALIDATED")
    ap.add_argument("--claims", default=",".join(etv.ORDER))
    ap.add_argument("--source-id", action="append", help="STAR=N")
    ap.add_argument("--epochs-file", action="append", help="STAR=PATH")
    args = ap.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    sids, files = kv(args.source_id), kv(args.epochs_file)
    summary = dict(project="pceb-planets-dr4", version=VERSION, validated_flag=bool(args.validated),
                   verdicts={}, a0_pred_mas={}, a0_95_exclusion_mas={})
    cache = {}
    for key in [k.strip() for k in args.claims.split(",") if k.strip()]:
        claim = etv.CLAIMS[key]
        star = etv.STARS[claim["star"]]
        res = dict(project="pceb-planets-dr4", version=VERSION, target=key, star=star["key"],
                   validated_flag=bool(args.validated))
        try:
            if star["key"] not in cache:
                cache[star["key"]] = load_points(star, args, sids, files)
                sid, src, points, counts = cache[star["key"]]
                if points:
                    write_csv(os.path.join(args.out_dir, "points_%s.csv" % star["key"]), points)
            sid, src, points, counts = cache[star["key"]]
            res.update(source_id=sid, source=src, counts=counts)
            if not points:
                res.update(status="no_epochs", verdict="INCONCLUSIVE")
            else:
                try:
                    res.update(etv.run_claim(points, claim, log=log))
                except Exception:
                    res.update(status="error", error=traceback.format_exc())
                res["verdict"] = etv.verdict(res, args.validated, claim)
        except Exception:
            res.update(status="error", error=traceback.format_exc(), verdict="INCONCLUSIVE")
        with open(os.path.join(args.out_dir, "results_%s.json" % key), "w") as f:
            json.dump(jsonable(res), f, indent=1)
        if res.get("status") == "ok":
            write_csv(os.path.join(args.out_dir, "injections_%s.csv" % key), res["injections"])
            write_csv(os.path.join(args.out_dir, "periodogram_%s.csv" % key), res["periodogram"])
            summary["a0_95_exclusion_mas"][key] = {
                fam: res["limits"]["a0_95_exclusion_%s" % fam]["a0_mas"] for fam in ("etv", "anyphase")}
        summary["verdicts"][key] = res.get("verdict")
        summary["a0_pred_mas"][key] = etv.a0_pred(claim)
        log("VERDICT %s %s" % (key, res.get("verdict")))
    summary["runtime_s"] = time.time() - T0
    with open(os.path.join(args.out_dir, "summary_etv.json"), "w") as f:
        json.dump(jsonable(summary), f, indent=1)


if __name__ == "__main__":
    main()
