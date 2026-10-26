"""
TestPilot-RL web server.

  python server.py          ->  http://127.0.0.1:5000

Serves:
  /                 the dashboard (frontend/)
  /results/...      your existing results CSVs + figures
  /api/health       liveness check
  /api/evaluate     POST a CI-log CSV, get model/baseline performance back as JSON
"""
import ast
import io
import os
import sys
import time
import traceback
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.simulator import RealWorldCIAdapter
from src.q_learning import LinearQLearningAgent, DuelingDQNAgent
from src.train import train_agent, build_canonical_ci_history
from src.evaluate import run_benchmark_comparison
from src.metrics import aggregate_experiment_results, compute_paired_cycle_comparisons
from src.data_loader import get_dataset_summary, split_ci_cycles

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads

FRONTEND_DIR = os.path.join(ROOT, "frontend")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")
EVAL_SEEDS = [42, 101, 777]
MAX_UPLOAD_CYCLES_EVALUATED = 400   # keeps pretrained evaluation responsive on huge logs


# ───────────────────────── helpers ─────────────────────────
class UserError(Exception):
    pass


def parse_results(val: Any) -> List[int]:
    if isinstance(val, list):
        return [int(x) for x in val]
    if isinstance(val, str):
        v = val.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                p = ast.literal_eval(v)
                if isinstance(p, list):
                    return [int(x) for x in p]
            except Exception:
                pass
    return []


def read_ci_csv(raw: bytes) -> pd.DataFrame:
    """Same cleaning as src.data_loader.load_ci_dataset, but for an arbitrary uploaded file."""
    text = raw.decode("utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text.strip() else ""
    sep = ";" if first.count(";") >= first.count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in ["Id", "Name", "Duration", "Verdict", "Cycle"] if c not in df.columns]
    if missing:
        raise UserError(
            f"Missing required column(s): {', '.join(missing)}. Expected: "
            "Id;Name;Duration;CalcPrio;LastRun;LastResults;Verdict;Cycle"
        )
    df["Name"] = df["Name"].astype(str)
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce").fillna(1.0).clip(lower=0.01)
    df["Verdict"] = pd.to_numeric(df["Verdict"], errors="coerce").fillna(0).astype(int)
    df["Cycle"] = pd.to_numeric(df["Cycle"], errors="coerce").fillna(0).astype(int)
    if "LastResults" in df.columns:
        df["LastResults"] = df["LastResults"].apply(parse_results)
    else:
        df["LastResults"] = [[] for _ in range(len(df))]
    if len(df) == 0:
        raise UserError("The uploaded file has no rows.")
    return df


def load_pretrained():
    """Fresh copies of the saved models (they adapt online during evaluation, so never reuse)."""
    lin_path = os.path.join(MODELS_DIR, "q_table.npy")
    dqn_path = os.path.join(MODELS_DIR, "dueling_dqn.npy")
    if not (os.path.exists(lin_path) and os.path.exists(dqn_path)):
        raise UserError("Pretrained models not found in models/. Run `python run_experiments.py` first, or use retrain mode.")
    lin = LinearQLearningAgent(seed=42)
    lin.load(lin_path)
    dqn = DuelingDQNAgent(seed=42)
    dqn.load(dqn_path)
    for a in (lin, dqn):
        a.epsilon = a.epsilon_min
    return lin, dqn


def clean(obj):
    """Make DataFrame records JSON-safe (NaN/inf -> null, numpy -> python)."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    return obj


def evaluate_dataset(df: pd.DataFrame, budget_ratio: float, train_ratio: float, episodes: int, mode: str) -> Dict[str, Any]:
    n_cycles = df["Cycle"].nunique()
    if n_cycles < 4:
        raise UserError(f"Need at least 4 CI cycles to split into train/test; found {n_cycles}.")

    train_cycles, test_cycles = split_ci_cycles(df, train_ratio=train_ratio)
    all_test_builds = RealWorldCIAdapter.load_all_builds_from_df(
        df, test_cycles, budget_ratio=budget_ratio, include_zero_failure_cycles=True
    )
    test_builds = [b for b in all_test_builds if b.num_hidden_bugs > 0]
    excluded = len(all_test_builds) - len(test_builds)
    if not test_builds:
        raise UserError("No held-out cycle contains a failing test, so BDR/APFD are undefined. Try a smaller train split.")
    truncated = False
    if len(test_builds) > MAX_UPLOAD_CYCLES_EVALUATED:
        test_builds = test_builds[-MAX_UPLOAD_CYCLES_EVALUATED:]
        truncated = True

    history = build_canonical_ci_history(df, train_cycles)

    if mode == "retrain":
        train_builds = RealWorldCIAdapter.load_all_builds_from_df(
            df, train_cycles, budget_ratio=budget_ratio, include_zero_failure_cycles=False
        )
        if not train_builds:
            raise UserError("No training cycle contains a failing test, cannot retrain.")
        common = dict(train_builds=train_builds, num_episodes=episodes, test_history=history, seed=42,
                      w_b=3.0, w_c=0.0, w_e=0.25, shaping_coef=0.0, gamma=0.95, epsilon_min=0.02,
                      verbose_interval=max(1, episodes))
        lin, _ = train_agent(use_linear=True, agent_type="linear", alpha=0.01, **common)
        dqn, _ = train_agent(agent_type="dueling_dqn", alpha=0.005, **common)
        seeds = [42]
    else:
        lin, dqn = load_pretrained()
        seeds = EVAL_SEEDS

    raw = run_benchmark_comparison(
        lin, test_builds, seeds=seeds, test_history=history,
        extra_agents=[("Dueling DQN (Online RL)", dqn)],
    )
    summary = aggregate_experiment_results(raw)
    try:
        paired = compute_paired_cycle_comparisons(raw, reference_strategy="Dueling DQN (Online RL)")
    except Exception:
        paired = pd.DataFrame()

    return {
        "dataset": get_dataset_summary(df),
        "split": {
            "train_cycles": len(train_cycles),
            "test_cycles": len(test_cycles),
            "evaluated_cycles": len(test_builds),
            "excluded_zero_fail": excluded,
            "truncated_to_last": MAX_UPLOAD_CYCLES_EVALUATED if truncated else None,
        },
        "mode": mode,
        "summary": summary.to_dict("records"),
        "paired": paired.to_dict("records"),
        "per_cycle": raw.to_dict("records"),
    }


# ───────────────────────── routes ─────────────────────────
@app.get("/api/health")
def health():
    return jsonify(status="ok", models=sorted(os.listdir(MODELS_DIR)) if os.path.isdir(MODELS_DIR) else [])


@app.post("/api/evaluate")
def api_evaluate():
    t0 = time.time()
    try:
        f = request.files.get("file")
        if f is None:
            raise UserError("No file uploaded.")
        budget = min(max(float(request.form.get("budget_ratio", 0.45)), 0.05), 1.0)
        split = min(max(float(request.form.get("train_ratio", 0.70)), 0.3), 0.9)
        episodes = min(max(int(float(request.form.get("episodes", 400))), 10), 3000)
        mode = request.form.get("mode", "pretrained")
        if mode not in ("pretrained", "retrain"):
            raise UserError("mode must be 'pretrained' or 'retrain'.")
        df = read_ci_csv(f.read())
        out = evaluate_dataset(df, budget, split, episodes, mode)
        out["elapsed_sec"] = round(time.time() - t0, 1)
        return jsonify(clean(out))
    except UserError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:  # noqa
        traceback.print_exc()
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.get("/results/<path:p>")
def results(p):
    return send_from_directory(RESULTS_DIR, p)


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:p>")
def frontend(p):
    return send_from_directory(FRONTEND_DIR, p)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  TestPilot-RL dashboard -> http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
