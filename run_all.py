#!/usr/bin/env python3
"""Run the whole pipeline end to end.

    python3 run_all.py            # full run (roughly an hour on a laptop)
    python3 run_all.py --quick    # smoke test of every stage (a few minutes)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

FULL = [
    (["-m", "pytest", "tests/", "-q"], []),
    (["phase1_nonidentifiability.py"], []),
    (["phase2_bayesian_pomdp.py"], ["--n-agents", "20", "--n-episodes", "2000",
                                    "--log-every", "10"]),
    (["phase2_bayesian_pomdp.py"], ["--only-kappa", "--n-agents", "12",
                                    "--n-episodes", "2000", "--log-every", "10"]),
    (["phase3_interventions.py"], ["--n-agents", "25"]),
    (["phase4_hysteresis_phasediagram.py"], ["--n-agents", "16", "--sweep-n", "11",
                                             "--sweep-seeds", "3",
                                             "--parts", "A,A2,B,C,D"]),
    (["phase5_predictions.py"], []),
    (["make_figures.py"], ["4", "5", "8"]),
    (["make_submission_figures.py"], []),
    (["make_numbers.py"], []),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    for script, extra in FULL:
        if script[0] == "-m":
            arguments = []
        elif args.quick:
            arguments = [] if script[0].startswith("make_") else ["--quick"]
        else:
            arguments = extra
        cmd = [sys.executable] + script + arguments
        print(f"\n{'=' * 72}\n$ {' '.join(cmd)}\n{'=' * 72}", flush=True)
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            print(f"\n[run_all] FAILED: {' '.join(cmd)}")
            return r.returncode
    print(f"\n[run_all] complete in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
