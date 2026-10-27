#!/usr/bin/env python3
"""
Compare the tracking strategies (direct / sequential / hybrid) on every rally:
runs part1 for each (strategy, rally), then the Appendix A checks
(tests/evaluate_sanity.py), then writes a comparison table.

    uv run python compare_strategies.py                       # everything, 2 jobs at a time
    uv run python compare_strategies.py --rallies rally_03 --strategies direct hybrid
    uv run python compare_strategies.py --dry-run             # only list what would run
    uv run python compare_strategies.py --table-only          # table from existing reports

Resumable: a part1 run counts as done when its results directory has tracking_log.json
(part1 writes it last), and a check when its sanity_report.json is newer than that log.
Stop it with Ctrl+C at any point and run it again later; finished work is skipped.
Each job is a separate low-priority process with single-threaded BLAS, at most --jobs
at a time (more makes the machine unusable: one part1 + checks is already heavy).

Outputs: <results-prefix>_<strategy>/<rally>/ (part1), <reports>/<strategy>/<rally>/
(checks), and <reports>/comparison.md + comparison.json.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
STRATEGIES = ("direct", "sequential", "hybrid")


def run(cmd, log_path):
    """Run cmd at low priority with single-threaded BLAS; output to log_path. True on success."""
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    preexec = (lambda: os.nice(15)) if hasattr(os, "nice") else None
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                preexec_fn=preexec)
        try:
            return proc.wait() == 0
        except BaseException:                      # Ctrl+C: don't leave the child running
            proc.terminate()
            proc.wait()
            raise


def paths(args, strategy, rally):
    results = os.path.join(args.results_prefix + "_" + strategy, rally)
    report = os.path.join(args.reports, strategy, rally)
    return results, report


def part1_done(results):
    return os.path.isfile(os.path.join(results, "tracking_log.json"))


def sanity_done(results, report):
    rep = os.path.join(report, "sanity_report.json")
    return (part1_done(results) and os.path.isfile(rep)
            and os.path.getmtime(rep) >= os.path.getmtime(os.path.join(results, "tracking_log.json")))


def job(args, strategy, rally):
    """part1 then the checks for one (strategy, rally), skipping what is already done."""
    results, report = paths(args, strategy, rally)
    os.makedirs(args.logs, exist_ok=True)
    if not part1_done(results):
        ok = run([sys.executable, "part1.py", os.path.join(args.ref, rally),
                  os.path.join(args.datasets, rally), os.path.join(args.features, rally),
                  results, "--strategy", strategy],
                 os.path.join(args.logs, f"part1_{strategy}_{rally}.log"))
        if not ok:
            return f"part1 {strategy} {rally}: FAILED (see {args.logs})"
    if not sanity_done(results, report):
        ok = run([sys.executable, os.path.join("tests", "evaluate_sanity.py"),
                  "--results", results, "--images", os.path.join(args.datasets, rally),
                  "--features", os.path.join(args.features, rally), "--ref", os.path.join(args.ref, rally),
                  "--out", report],
                 os.path.join(args.logs, f"sanity_{strategy}_{rally}.log"))
        if not ok:
            return f"checks {strategy} {rally}: FAILED (see {args.logs})"
    return f"{strategy} {rally}: done"


def fmt(v, digits=2):
    return "–" if v is None else f"{v:.{digits}f}"


def table(args):
    """Comparison table (markdown + json) from the existing reports."""
    rows = []
    for rally in args.rallies:
        for strategy in args.strategies:
            results, report = paths(args, strategy, rally)
            rep_path = os.path.join(report, "sanity_report.json")
            if not (part1_done(results) and os.path.isfile(rep_path)):
                continue
            rep = json.load(open(rep_path))
            log = json.load(open(os.path.join(results, "tracking_log.json")))
            cyc = rep.get("cycle_by_baseline", {})
            base = sorted(cyc, key=int)
            rows.append({
                "rally": rally, "strategy": strategy, "frames": len(log),
                "no_court": sum(e["method"] == "no_court" for e in log),
                "line_px": rep.get("line_reprojection_error_px"),
                "cycle_near_px": cyc[base[0]]["track"] if base else None,
                "cycle_far_px": cyc[base[-1]]["track"] if base else None,
                "fb_round_trip_px": rep.get("forward_backward_drift_px"),
                "chain_vs_track_px": rep.get("chain_vs_track_px"),
                "velocity_px": rep.get("temporal_velocity_px_per_frame"),
                "unexplained_spikes": len(rep.get("unexplained_spike_frames", [])),
            })
    if not rows:
        print("[compare] no finished reports yet")
        return

    head = ("| Rally | Strategy | No-court / frames | Line err px | Cycle px (nearest / farthest baseline) "
            "| FB round trip px | Chain vs output px | Median velocity px/f | Unexplained spikes |")
    lines = [head, "|" + "---|" * 9]
    for r in rows:
        lines.append(f"| {r['rally']} | {r['strategy']} | {r['no_court']} / {r['frames']} | {fmt(r['line_px'])} "
                     f"| {fmt(r['cycle_near_px'])} / {fmt(r['cycle_far_px'])} | {fmt(r['fb_round_trip_px'])} "
                     f"| {fmt(r['chain_vs_track_px'])} | {fmt(r['velocity_px'])} | {r['unexplained_spikes']} |")
    md = "\n".join(lines)
    print("\n" + md)
    os.makedirs(args.reports, exist_ok=True)
    with open(os.path.join(args.reports, "comparison.md"), "w") as f:
        f.write("# Tracking strategy comparison\n\n" + md + "\n\n"
                "Forward-backward and chain-vs-output use frame-to-frame composition, which is the "
                "same for every strategy on a rally; they show the drift that direct/hybrid avoid.\n")
    json.dump(rows, open(os.path.join(args.reports, "comparison.json"), "w"), indent=2)
    print(f"\n[compare] -> {os.path.join(args.reports, 'comparison.md')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rallies", nargs="+", default=["rally_01", "rally_02", "rally_03", "rally_04"])
    ap.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    ap.add_argument("--jobs", type=int, default=2, help="parallel jobs (keep it low)")
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--ref", default="ref")
    ap.add_argument("--features", default="features", help="SIFT features from main1.py, one dir per rally")
    ap.add_argument("--results-prefix", default="results_v2")
    ap.add_argument("--reports", default=os.path.join("sanity_reports", "comparison"))
    ap.add_argument("--logs", default=os.path.join("sanity_reports", "comparison", "logs"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()

    pending = []
    for strategy in args.strategies:
        for rally in args.rallies:
            results, report = paths(args, strategy, rally)
            for need, what in ((os.path.isdir(os.path.join(args.features, rally)), "features"),
                               (os.path.isdir(os.path.join(args.ref, rally)), "ref")):
                if not need:
                    raise SystemExit(f"Missing {what} for {rally}: run main1.py / extract_reference.py first")
            todo = [s for s, done in (("part1", part1_done(results)), ("checks", sanity_done(results, report)))
                    if not done]
            if todo:
                pending.append((strategy, rally, todo))

    print(f"[compare] {len(pending)} of {len(args.strategies) * len(args.rallies)} (strategy, rally) "
          f"pairs need work")
    for strategy, rally, todo in pending:
        print(f"    {strategy:<10} {rally}: {' + '.join(todo)}")
    if args.dry_run or args.table_only or not pending:
        table(args)
        return

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futures = [ex.submit(job, args, s, r) for s, r, _ in pending]
            for k, fut in enumerate(as_completed(futures), 1):
                print(f"[compare] {k}/{len(futures)} {fut.result()}", flush=True)
    except KeyboardInterrupt:
        print("\n[compare] interrupted; finished runs are kept, run again to resume")
        os._exit(130)                              # don't wait for queued jobs
    table(args)


if __name__ == "__main__":
    main()
