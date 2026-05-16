"""
compute_test_acc.py
Aggregate experiment metrics from completed seed checkpoints and emit a
LaTeX-friendly mean ± std table per experiment directory.

Most numbers are read from `<dir>/summary.json` (already aggregated across
seeds at training time); the only pickle access is to retrieve held-out
test-set accuracy from each `seed_N.pkl`, which is not in summary.json.

Usage
-----
# All paper-exp subdirectories
python scripts/compute_test_acc.py

# A specific run
python scripts/compute_test_acc.py --dir paper-exp/main-soft0.3-qwen3-30b-a3b-instruct-2507

# Per-seed numbers (verify the std yourself)
python scripts/compute_test_acc.py --dir <...> --per-seed

# Dump everything to CSV
python scripts/compute_test_acc.py --csv all.csv

# Filter seeds (e.g. include only 0, 1)
python scripts/compute_test_acc.py --dir <...> --seeds 0 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

# Ensure the project root is importable so that pickles referencing
# `env.SeedResult`, `data_gen.UserPersona`, agent classes, etc. deserialise.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import pickle
import re
from glob import glob
from typing import Dict, List, Optional

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Run inside `conda run -n tool-call ...`.", file=sys.stderr)
    sys.exit(1)


AGENT_ORDER = [
    "Random",
    "ZeroShot-LLM",
    "Freq-Greedy",
    "Pure-Bandit",
    "InContext-Memory",
    "Profile-Memory",
    "Bandit+CoT",
    "Freq+Override",
    "Bandit+Override",
]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt(mean, std, decimals: int = 1) -> str:
    """LaTeX-friendly cell. With std: '1.7 ($\\pm$ 0.2)'. Without std: '1.7'."""
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ($\\pm$ {std:.{decimals}f})"


# ---------------------------------------------------------------------------
# Reading the data
# ---------------------------------------------------------------------------


def _find_seed_pkls(ckpt_dir: str, allowed_seeds: Optional[List[int]]) -> List[tuple]:
    """Return [(seed_number, path), ...] for finalized seed_N.pkl files."""
    out = []
    for path in glob(os.path.join(ckpt_dir, "seed_*.pkl")):
        base = os.path.basename(path)
        if "_progress" in base:
            continue
        m = re.match(r"seed_(\d+)\.pkl$", base)
        if not m:
            continue
        seed_num = int(m.group(1))
        if allowed_seeds is not None and seed_num not in allowed_seeds:
            continue
        out.append((seed_num, path))
    out.sort()
    return out


def _load_test_results(pkl_path: str) -> Optional[dict]:
    """Open one seed pickle and return only its test_results dict (or None)."""
    try:
        with open(pkl_path, "rb") as f:
            sr = pickle.load(f)
        return sr.test_results
    except Exception as e:
        print(f"  [warn] could not load {pkl_path}: {e}", file=sys.stderr)
        return None


def aggregate_one_dir(dir_path: str, allowed_seeds: Optional[List[int]] = None) -> dict:
    """
    Aggregate metrics for a single experiment directory.

    Pulls mean/std for regret, online OOD acc, and preference-recovery metrics
    directly from `summary.json`; loads each `seed_N.pkl` only to extract its
    `test_results`. No benchmark JSON is required.
    """
    summary_path = os.path.join(dir_path, "summary.json")
    if not os.path.isfile(summary_path):
        return {"error": f"no summary.json in {dir_path}"}

    with open(summary_path, "r") as f:
        summary = json.load(f)

    cfg = summary.get("config", {})
    agents_in_summary = list(summary.get("agents", {}).keys())

    # Detect soft mode from the recovery field schema
    soft_mode = False
    for a, m in summary["agents"].items():
        rec = m.get("preference_recovery_rate", {})
        if "cosine_sim" in rec or "kl_divergence" in rec or "spearman" in rec:
            soft_mode = True
            break

    # Pull per-seed test_results (the only thing summary.json doesn't have)
    ckpt_dir = os.path.join(dir_path, "checkpoints")
    seed_pkls = _find_seed_pkls(ckpt_dir, allowed_seeds) if os.path.isdir(ckpt_dir) else []
    if allowed_seeds is not None:
        # also restrict summary numbers below
        pass

    per_seed_test: Dict[int, dict] = {}
    for seed_num, path in seed_pkls:
        tr = _load_test_results(path)
        if tr:
            per_seed_test[seed_num] = tr

    if not per_seed_test:
        print(f"  [warn] no test_results found in {ckpt_dir}", file=sys.stderr)

    # Stable agent order
    agents_present = set(agents_in_summary)
    ordered = [a for a in AGENT_ORDER if a in agents_present]
    extras = sorted(agents_present - set(ordered))
    ordered = ordered + extras

    rows = []
    for agent in ordered:
        m = summary["agents"].get(agent, {})

        # Regret (per-user average), already mean ± std in summary.json
        regret = m.get("final_cumulative_regret", {})
        regret_mean = regret.get("mean")
        regret_std = regret.get("std")
        per_seed_regret = regret.get("per_seed", [])
        if allowed_seeds is not None and per_seed_regret and "per_seed" in regret:
            # Recompute mean/std from the restricted seed set if user asked to filter
            full_seeds = cfg.get("seeds", list(range(len(per_seed_regret))))
            kept = [per_seed_regret[i] for i, s in enumerate(full_seeds) if s in allowed_seeds]
            if kept:
                regret_mean = float(np.mean(kept))
                regret_std = float(np.std(kept))
                per_seed_regret = kept

        # Online OOD acc (late rounds), mean ± std in summary.json (no per-seed)
        ood_late = m.get("ood_accuracy_late", {})
        ood_late_mean = (ood_late.get("mean") or 0.0) * 100 if ood_late.get("mean") is not None else None
        ood_late_std = (ood_late.get("std") or 0.0) * 100 if ood_late.get("std") is not None else None

        # Test accuracy from per-seed test_results.
        # Use ddof=1 (unbiased sample std) to match summary.json's convention.
        accs, oods = [], []
        for seed_num, tr in per_seed_test.items():
            if agent in tr:
                accs.append(tr[agent]["accuracy"])
                oods.append(tr[agent]["ood_accuracy"])
        acc_mean = float(np.mean(accs)) * 100 if accs else None
        acc_std = (float(np.std(accs, ddof=1)) * 100
                   if len(accs) > 1 else (0.0 if accs else None))
        test_ood_mean = float(np.mean(oods)) * 100 if oods else None
        test_ood_std = (float(np.std(oods, ddof=1)) * 100
                        if len(oods) > 1 else (0.0 if oods else None))

        # Preference recovery (already aggregated across seeds in summary.json)
        rec = m.get("preference_recovery_rate", {})

        row = {
            "agent": agent,
            "n_seeds_summary": len(per_seed_regret),
            "n_seeds_test": len(accs),
            "regret_mean": regret_mean,
            "regret_std": regret_std,
            "online_ood_mean": ood_late_mean,
            "online_ood_std": ood_late_std,
            "acc_mean": acc_mean,
            "acc_std": acc_std,
            "ood_mean": test_ood_mean,
            "ood_std": test_ood_std,
            "per_seed_regret": per_seed_regret,
            "per_seed_acc": [a * 100 for a in accs],
            "per_seed_ood": [o * 100 for o in oods],
            "per_seed_seeds": sorted(per_seed_test.keys()),
        }

        # Recovery columns differ by regime (and have no per-seed std in summary)
        if soft_mode:
            row["cosine_sim_mean"] = rec.get("cosine_sim")
            row["kl_divergence_mean"] = rec.get("kl_divergence")
            row["spearman_mean"] = rec.get("spearman")
            row["cosine_sim_std"] = None
            row["kl_divergence_std"] = None
            row["spearman_std"] = None
        else:
            recv = rec.get("recovery_rate")
            row["recovery_mean"] = recv * 100 if recv is not None else None
            row["recovery_std"] = None

        rows.append(row)

    return {
        "dir": dir_path,
        "soft_mode": soft_mode,
        "config": cfg,
        "seeds_summary": cfg.get("seeds"),
        "seeds_test": sorted(per_seed_test.keys()),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_table(agg: dict, per_seed: bool = False) -> None:
    print(f"\n=== {agg['dir']} ===")
    mode = "soft" if agg.get("soft_mode") else "one-hot"
    print(f"    mode: {mode}    seeds (summary)={agg.get('seeds_summary')}    "
          f"seeds (test)={agg.get('seeds_test')}")

    rows = agg["rows"]
    # Columns: (header, key_mean, key_std, decimals)
    cols = [
        ("Regret",   "regret_mean",   "regret_std",   1),
        ("Acc",      "acc_mean",      "acc_std",      1),
        ("OOD",      "ood_mean",      "ood_std",      1),
    ]
    if agg.get("soft_mode"):
        cols += [
            ("Cosine",   "cosine_sim_mean",    "cosine_sim_std",    3),
            ("KL",       "kl_divergence_mean", "kl_divergence_std", 3),
            ("Spearman", "spearman_mean",      "spearman_std",      3),
        ]
    else:
        cols += [("Recovery", "recovery_mean", "recovery_std", 1)]

    name_w = max(len(r["agent"]) for r in rows + [{"agent": "Agent"}])
    cells = [[_fmt(r.get(km), r.get(ks), decimals=dec) for _, km, ks, dec in cols]
             for r in rows]
    col_w = [
        max(len(h), max(len(cells[i][j]) for i in range(len(cells))))
        for j, (h, *_) in enumerate(cols)
    ]

    header = f"  {'Agent':<{name_w}}  " + "  ".join(
        f"{h:^{col_w[j]}}" for j, (h, *_) in enumerate(cols)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r, row_cells in zip(rows, cells):
        line = f"  {r['agent']:<{name_w}}  " + "  ".join(
            f"{row_cells[j]:>{col_w[j]}}" for j in range(len(cols))
        )
        print(line)

    if per_seed:
        print("\n  Per-seed breakdown:")
        for r in rows:
            print(f"    {r['agent']:<{name_w}}")
            n = max(len(r["per_seed_regret"]), len(r["per_seed_acc"]))
            seeds = r["per_seed_seeds"] or list(range(n))
            for i in range(n):
                parts = []
                if i < len(r["per_seed_regret"]):
                    parts.append(f"regret={r['per_seed_regret'][i]:7.1f}")
                if i < len(r["per_seed_acc"]):
                    parts.append(f"acc={r['per_seed_acc'][i]:5.1f}%")
                if i < len(r["per_seed_ood"]):
                    parts.append(f"ood={r['per_seed_ood'][i]:5.1f}%")
                s = seeds[i] if i < len(seeds) else i
                print(f"      seed {s}: " + "  ".join(parts))


def write_csv(aggs: List[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "experiment", "mode", "agent",
            "regret_mean", "regret_std",
            "acc_mean", "acc_std",
            "ood_mean", "ood_std",
            "online_ood_mean", "online_ood_std",
            "recovery_mean",
            "cosine_sim_mean", "kl_divergence_mean", "spearman_mean",
            "per_seed_seeds", "per_seed_regret", "per_seed_acc", "per_seed_ood",
        ])
        for agg in aggs:
            if "error" in agg:
                continue
            exp_name = os.path.basename(agg["dir"].rstrip("/"))
            mode = "soft" if agg.get("soft_mode") else "one-hot"
            for r in agg["rows"]:
                w.writerow([
                    exp_name, mode, r["agent"],
                    r.get("regret_mean"), r.get("regret_std"),
                    r.get("acc_mean"), r.get("acc_std"),
                    r.get("ood_mean"), r.get("ood_std"),
                    r.get("online_ood_mean"), r.get("online_ood_std"),
                    r.get("recovery_mean"),
                    r.get("cosine_sim_mean"),
                    r.get("kl_divergence_mean"),
                    r.get("spearman_mean"),
                    ";".join(str(s) for s in r["per_seed_seeds"]),
                    ";".join(f"{x:.4f}" for x in r["per_seed_regret"]),
                    ";".join(f"{x:.4f}" for x in r["per_seed_acc"]),
                    ";".join(f"{x:.4f}" for x in r["per_seed_ood"]),
                ])
    print(f"\nCSV written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", action="append", default=None,
                   help="One experiment directory. May be repeated. "
                        "If omitted, scans all subdirs of `paper-exp/`.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Optional seed whitelist (e.g. --seeds 0 1 2).")
    p.add_argument("--per-seed", action="store_true",
                   help="Print per-seed numbers in addition to mean±std.")
    p.add_argument("--csv", default=None,
                   help="Write aggregated results to this CSV file.")
    args = p.parse_args()

    if args.dir:
        dirs = args.dir
    else:
        if not os.path.isdir("paper-exp"):
            print("ERROR: no `paper-exp/` directory in cwd, and no --dir specified.",
                  file=sys.stderr)
            sys.exit(1)
        dirs = sorted(os.path.join("paper-exp", d) for d in os.listdir("paper-exp")
                      if os.path.isdir(os.path.join("paper-exp", d)))

    aggs = []
    for d in dirs:
        agg = aggregate_one_dir(d, allowed_seeds=args.seeds)
        if "error" in agg:
            print(f"\n=== {d} ===\n  SKIP: {agg['error']}")
            continue
        print_table(agg, per_seed=args.per_seed)
        aggs.append(agg)

    if args.csv:
        write_csv(aggs, args.csv)


if __name__ == "__main__":
    main()
