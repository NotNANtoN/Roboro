"""Driver: run DG-on vs DG-off on cheetah for 3 seeds each, collate results.

Run from the autoresearch directory:
    python run_dg_seeds.py
"""

import os
import re
import subprocess
import sys
import time
from statistics import mean, stdev

SEEDS = [42, 43, 44]
# Configs are (name, env_overrides_dict).
# Baseline SAC is well-established: mean=339.92, std=9.42
# Dueling SAC (β=0.01) is well-established: mean=367.03, std=23.65
CONFIGS = [
    ("dueling_b001",  {"DUELING": "1", "DUELING_BETA": "0.001"}),
    ("dueling_b01",   {"DUELING": "1", "DUELING_BETA": "0.1"}),
]
PYTHON = sys.executable
LOG_DIR = "/tmp/dueling_beta_seeds"
os.makedirs(LOG_DIR, exist_ok=True)

EVAL_RE = re.compile(r"cheetah_return:\s+([-\d\.]+)")
TIME_RE = re.compile(r"training_seconds:\s+([-\d\.]+)")
STEP_RE = re.compile(r"\[cheetah\] eval=([-\d\.]+)")


def run_one(config_name: str, overrides: dict, seed: int) -> dict:
    log_path = os.path.join(LOG_DIR, f"{config_name}_s{seed}.log")
    env = os.environ.copy()
    env.update({**overrides, "SEED": str(seed), "CHEETAH_ONLY": "1"})
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(
            [PYTHON, "-u", "train_dmc.py"],
            env=env, stdout=f, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.time() - t0
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
        if len(xs) >= 2:
            print(f"{cfg:>10s}: mean={mean(xs):.2f}  std={stdev(xs):.2f}  n={len(xs)}")


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
