"""Driver: run algorithm variants on cheetah for 3 seeds each, collate results.

Run from the autoresearch directory:
    python run_dg_seeds.py
"""

import os
import re
import subprocess
import sys
import time
from statistics import mean, stdev

SEEDS = [42, 43, 44, 45, 46]  # Full 5 seeds
TASK = ""
CONFIGS = [
    ("sac_n8",       {"CHEETAH_ONLY": "1", "N_CRITICS": "8"}),                                    # SAC N=8
    ("spg32_duel_n8", {"CHEETAH_ONLY": "1", "N_CRITICS": "8", "SPG": "1", "SPG_SAMPLES": "32", "DUELING": "1"}),  # SPG32+Dueling+N=8
]
PYTHON = sys.executable
LOG_DIR = f"/tmp/n8_sweep"
os.makedirs(LOG_DIR, exist_ok=True)

EVAL_RE = re.compile(r"(?:cheetah|quadruped)_return:\s+([-\d\.]+)")
TIME_RE = re.compile(r"training_seconds:\s+([-\d\.]+)")
STEP_RE = re.compile(r"\[(?:cheetah|quadruped)\] eval=([-\d\.]+)")


def run_one(config_name: str, overrides: dict, seed: int) -> dict:
    log_path = os.path.join(LOG_DIR, f"{config_name}_s{seed}.log")
    env = os.environ.copy()
    task_env = {"TASK": TASK} if TASK else {"CHEETAH_ONLY": "1"}
    env.update({**overrides, **task_env, "SEED": str(seed)})
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(
            [PYTHON, "-u", "train_dmc.py"],
            env=env, stdout=f, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.time() - t0
    # Copy per-step metrics CSV for post-hoc analysis
    csv_name = f"{TASK}_metrics.csv" if TASK else "cheetah_metrics.csv"
    csv_src = os.path.join(os.path.dirname(__file__), "runs", csv_name)
    csv_dst = os.path.join(LOG_DIR, f"{config_name}_s{seed}_metrics.csv")
    if os.path.exists(csv_src):
        import shutil
        shutil.copy2(csv_src, csv_dst)
    with open(log_path) as f:
        text = f.read()
    m_ret = EVAL_RE.search(text)
    m_time = TIME_RE.search(text)
    return {
        "config": config_name, "seed": seed,
        "return": float(m_ret.group(1)) if m_ret else float("nan"),
        "train_s": float(m_time.group(1)) if m_time else float("nan"),
        "wall_s": elapsed,
        "exit_code": proc.returncode,
        "log": log_path,
    }


def summarize(rows):
    print(f"\n{'config':>10s}  {'seed':>5s}  {'return':>8s}  {'train_s':>8s}  {'wall_s':>8s}  exit")
    print("-" * 60)
    for r in rows:
        print(f"{r['config']:>10s}  {r['seed']:>5d}  {r['return']:>8.2f}  {r['train_s']:>8.1f}  {r['wall_s']:>8.1f}  {r['exit_code']}")
    print()
    for cfg, _ in CONFIGS:
        xs = [r["return"] for r in rows if r["config"] == cfg]
        ts = [r["train_s"] for r in rows if r["config"] == cfg]
        if len(xs) >= 2:
            t_avg = mean(ts) if ts else 0
            print(f"{cfg}: mean={mean(xs):.2f}  std={stdev(xs):.2f}  avg_time={t_avg:.0f}s  n={len(xs)}")


def main():
    rows = []
    total = len(CONFIGS) * len(SEEDS)
    i = 0
    for seed in SEEDS:
        for name, overrides in CONFIGS:
            i += 1
            print(f"[{i}/{total}] {name} seed={seed} ...", flush=True)
            r = run_one(name, overrides, seed)
            print(f"    return={r['return']:.2f} train_s={r['train_s']:.1f} exit={r['exit_code']}", flush=True)
            rows.append(r)
    summarize(rows)
    # Dump tsv
    out_path = os.path.join(LOG_DIR, "summary.tsv")
    with open(out_path, "w") as f:
        f.write("config\tseed\treturn\ttrain_s\twall_s\texit\n")
        for r in rows:
            f.write(f"{r['config']}\t{r['seed']}\t{r['return']:.2f}\t{r['train_s']:.1f}\t{r['wall_s']:.1f}\t{r['exit_code']}\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
