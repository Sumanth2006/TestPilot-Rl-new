# TestPilot-RL: Intelligent Software Testing Using Reinforcement Learning

**TestPilot-RL** is a Reinforcement Learning framework and interactive analytics dashboard for budget-constrained software test case selection and prioritization in Continuous Integration (CI) environments. It formulates regression test execution under a strict runtime budget as a sequential Markov Decision Process (MDP) and learns cost-aware, coverage-diverse, and fault-sensitive policies from scratch with zero ground-truth data leakage.


---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dependencies & System Requirements](#dependencies--system-requirements)
4. [Installation Steps](#installation-steps)
5. [Execution Commands](#execution-commands)
6. [Step-by-Step Reproduction Instructions](#step-by-step-reproduction-instructions)
7. [Methodology & RL Architectures](#methodology--rl-architectures)
8. [Benchmark Results Summary](#benchmark-results-summary)

---

## Project Overview

In large-scale CI pipelines, executing the entire regression test suite on every commit or build cycle is often prohibitively expensive. **TestPilot-RL** trains autonomous RL agents to dynamically select and order test cases under a strict execution cost budget (default: 45% of full suite execution cost) to maximize **Bug Detection Rate (BDR)** and **Normalized Average Percentage of Faults Detected (NAPFD)** while maintaining high **Code Feature Coverage**.

Key capabilities include:
- **Strict Information Boundary**: Ground-truth defect mappings of unexecuted tests are strictly hidden inside the simulator. All 12 state-action features (`\phi(s, a)`) are constructed exclusively from prior builds/cycles (`0 .. i-1`) and the observed outcomes of tests already executed within the current build episode.
- **Dueling Advantage Linear Q-Learning & Dueling Double DQN**: Employs candidate-centered Dueling Advantage updates with Double Q-learning targets and Prioritized Experience Replay (PER) to separate baseline state-value estimation $V(s)$ from action-discriminative cost-rate advantages $A(s, a)$.
- **Within-Episode Bayesian Posterior & Redundancy Suppression**: Dynamically updates candidate test priorities within an episode based on observed pass/fail outcomes, suppressing redundant sibling tests whose covered features have already revealed a fault and boosting co-failing subsystem tests.
- **Density-Adaptive Cost Scaling**: Adapts cost sensitivity $c_a^{\beta(s)}$ to the build's observable fault density and remaining budget ratio.
- **Interactive Flask Web Dashboard**: Includes a full-stack web dashboard (`server.py` + `frontend/`) to visualize experimental CSVs, training curves, and ablation charts, and to upload custom CI-log CSVs for live benchmarking via `/api/evaluate`.

---

## Repository Structure

```text
TestPilot-Rl-new/
│
├── .gitignore                         # Git ignore rules
├── README.md                          # Project documentation, setup, and reproduction guide
├── pytest.ini                         # Pytest configuration
├── requirements.txt                   # Python package dependencies
├── run.sh                             # One-step Linux/macOS environment setup & dashboard launcher
├── run.bat                            # One-step Windows environment setup & dashboard launcher
├── run_experiments.py                 # Master orchestrator for all 8 experimental suites & figures
├── server.py                          # Flask backend server & REST API for the interactive dashboard
│
├── data/
│   ├── iofrol.csv                     # ABB IOF/ROL industrial CI dataset (RETECS benchmark)
│   └── paintcontrol.csv               # ABB Paint Control industrial CI dataset (RETECS benchmark)
│
├── frontend/
│   ├── index.html                     # Interactive analytics dashboard UI
│   ├── app.js                         # Dashboard logic, Chart.js rendering & CSV upload handler
│   ├── styles.css                     # Dashboard styling
│   └── vendor/
│       ├── chart.umd.min.js           # Bundled Chart.js library
│       └── papaparse.min.js           # Bundled PapaParse CSV parser
│
├── models/
│   ├── dueling_dqn.npy                # Trained Dueling Double DQN model weights
│   └── q_table.npy                    # Trained Linear Q-Learning model weights
│
├── results/
│   ├── summary.csv                    # Main synthetic benchmark summary (mean, std, 95% CI)
│   ├── paired_synthetic_comparison.csv# Paired build-by-build differences and win/tie/loss counts
│   ├── raw_results.csv                # Raw per-build evaluation logs on synthetic benchmark
│   ├── real_world_iofrol_summary.csv  # ABB IOF/ROL summary metrics
│   ├── real_world_iofrol_paired.csv   # ABB IOF/ROL paired cycle comparisons
│   ├── real_world_iofrol_raw.csv      # ABB IOF/ROL raw per-cycle evaluation logs
│   ├── real_world_paintcontrol_summary.csv # ABB Paint Control summary metrics
│   ├── real_world_paintcontrol_paired.csv  # ABB Paint Control paired cycle comparisons
│   ├── real_world_paintcontrol_raw.csv     # ABB Paint Control raw per-cycle evaluation logs
│   ├── budget_sensitivity.csv         # Performance across 20%, 35%, 50%, 65%, and 80% budgets
│   ├── reward_ablation.csv            # Reward weight ablation study results
│   ├── state_ablation.csv             # State-action feature group ablation study results
│   ├── agent_ablation.csv             # Architecture comparison (Dueling DQN, Linear Q, LinUCB, Tabular Q)
│   ├── training_history.csv           # Episode-by-episode training reward, BDR, and TD error logs
│   └── figures/                       # Publication-ready 300 DPI plots (.png)
│       ├── agent_ablation.png
│       ├── baseline_comparison.png
│       ├── budget_sensitivity.png
│       ├── real_world_iofrol.png
│       ├── real_world_paintcontrol.png
│       ├── reward_ablation.png
│       └── training_curve.png
│
├── src/
│   ├── __init__.py                    # Package initializer
│   ├── baselines.py                   # Random, Greedy (Coverage/Cost, Failure/Cost, Recency/Cost)
│   ├── data_loader.py                 # CI dataset loader, parser, and chronological cycle splitter
│   ├── environment.py                 # Gym-compatible TestingEnvironment with budget & action masking
│   ├── evaluate.py                    # Standalone CLI and API for evaluating policies vs. baselines
│   ├── metrics.py                     # BDR, Coverage, Efficiency, APFD, NAPFD & paired CI statistics
│   ├── q_learning.py                  # DuelingDQNAgent, LinearQLearningAgent, ContextualBandit, TabularQ
│   ├── simulator.py                   # Synthetic BuildGenerator & RealWorldCIAdapter
│   └── train.py                       # Standalone CLI and API for training RL agents
│
└── tests/
    └── test_testpilot.py              # Unit test suite verifying constraints, leakage, and metrics
```

---

## Dependencies & System Requirements

- **Python**: `>= 3.9` (tested with Python 3.10, 3.11, 3.12, and 3.13)
- **Operating System**: Linux, macOS, or Windows
- **Hardware**: Standard CPU (no GPU required; all RL agents—including Dueling Double DQN and Prioritized Experience Replay—are implemented in vectorized Pure NumPy)

### Required Python Packages (`requirements.txt`)

```text
flask>=3.0
numpy>=1.26
pandas>=2.0
scipy
matplotlib
seaborn
pytest
```

| Package | Version Constraint | Purpose |
| :--- | :--- | :--- |
| `flask` | `>= 3.0` | Backend web server (`server.py`) and `/api/evaluate` REST endpoint |
| `numpy` | `>= 1.26` | Vectorized Q-learning, Dueling DQN, Prioritized Experience Replay, and RNG |
| `pandas` | `>= 2.0` | CI log parsing, experimental result aggregation, and CSV export |
| `scipy` | `latest` | Statistical analysis and confidence interval utilities |
| `matplotlib` | `latest` | Generating publication-grade 300 DPI benchmark and ablation figures |
| `seaborn` | `latest` | Statistical visualization styling and grouped bar plots |
| `pytest` | `latest` | Running the automated unit test suite (`tests/test_testpilot.py`) |

---

## Installation Steps

### Option A: Quick One-Step Setup & Launch
The repository includes automated setup scripts that create a `.venv` virtual environment, install all dependencies from `requirements.txt`, and start the web server:

- **Linux / macOS**:
  ```bash
  git clone https://github.com/Sumanth2006/TestPilot-Rl-new.git
  cd TestPilot-Rl-new
  chmod +x run.sh
  ./run.sh
  ```
- **Windows (Command Prompt / PowerShell)**:
  ```cmd
  git clone https://github.com/Sumanth2006/TestPilot-Rl-new.git
  cd TestPilot-Rl-new
  run.bat
  ```

### Option B: Manual Virtual Environment Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Sumanth2006/TestPilot-Rl-new.git
   cd TestPilot-Rl-new
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv .venv
   # On Linux / macOS:
   source .venv/bin/activate
   # On Windows (Command Prompt):
   .venv\Scripts\activate
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies from `requirements.txt`**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Dataset Availability**:
   The industrial CI datasets (`data/paintcontrol.csv` and `data/iofrol.csv`) are pre-bundled in `data/`. If missing, `src/data_loader.py` automatically downloads them from the RETECS repository or generates a schema-exact offline fallback if internet access is unavailable.

---

## Execution Commands

Run all commands from the project root directory (`TestPilot-Rl-new/`).

### 1. Launch the Interactive Web Dashboard (`server.py`)
```bash
python server.py
```
- Opens the web dashboard at **`http://127.0.0.1:5000`** (or set a custom port via `PORT=8080 python server.py`).
- **Available Endpoints**:
  - `GET /`: Serves the interactive analytics frontend (`frontend/index.html`).
  - `GET /results/<path:filename>`: Serves generated CSV tables and PNG figures from `results/`.
  - `GET /api/health`: Health check endpoint returning `{"status": "ok"}`.
  - `POST /api/evaluate`: Accepts an uploaded CI-log CSV file and returns live benchmark evaluation metrics as JSON.

### 2. Run the Complete Experimental Suite (All 8 Experiments + Figures)
```bash
python run_experiments.py
```

### 3. Run the Automated Unit Test Suite
```bash
python -m pytest tests/ -v
```

### 4. Train an Individual RL Agent (`src/train.py`)
Train **Linear Q-Learning** on the synthetic benchmark:
```bash
python -m src.train \
    --dataset synthetic \
    --agent_type linear \
    --episodes 450 \
    --seed 42 \
    --save_path models/q_table.npy \
    --history_path results/training_history.csv
```

Train **Dueling Double DQN** on real-world **ABB Paint Control** or **ABB IOF/ROL**:
```bash
python -m src.train \
    --dataset paintcontrol \
    --agent_type dueling_dqn \
    --episodes 300 \
    --seed 42 \
    --save_path models/dueling_dqn.npy
```
```bash
python -m src.train \
    --dataset iofrol \
    --agent_type linear \
    --episodes 300 \
    --seed 42 \
    --save_path models/q_table.npy
```
*(Supported `--dataset` choices: `synthetic`, `paintcontrol`, `iofrol`. Supported `--agent_type` choices: `linear`, `dueling_dqn`, `bandit`, `tabular`.)*

### 5. Evaluate a Trained Model Against All Baselines (`src/evaluate.py`)
Evaluate on the held-out synthetic benchmark (120 builds across 3 evaluation seeds):
```bash
python -m src.evaluate \
    --dataset synthetic \
    --model_path models/q_table.npy \
    --num_test_builds 120 \
    --raw_output results/raw_results.csv \
    --summary_output results/summary.csv
```

Evaluate on real-world industrial CI datasets (`paintcontrol` or `iofrol`):
```bash
python -m src.evaluate \
    --dataset iofrol \
    --model_path models/q_table.npy \
    --raw_output results/real_world_iofrol_raw.csv \
    --summary_output results/real_world_iofrol_summary.csv
```
```bash
python -m src.evaluate \
    --dataset paintcontrol \
    --model_path models/q_table.npy \
    --raw_output results/real_world_paintcontrol_raw.csv \
    --summary_output results/real_world_paintcontrol_summary.csv
```

---

## Step-by-Step Reproduction Instructions

Follow these exact steps to reproduce all models, CSV tables, statistical comparisons, and figures from scratch:

1. **Step 1 — Verify Environment & Unit Tests**:
   ```bash
   python -m pytest tests/ -v
   ```
   Confirm all 4 unit test suites pass:
   - `test_environment_budget_and_no_repeat_constraints`: Verifies strict budget limits and legal-action masking.
   - `test_strict_information_boundary_no_leakage_on_initial_build`: Verifies zero ground-truth leakage on standalone builds.
   - `test_q_learning_updates_from_near_zero_initialization`: Verifies near-zero initialization and TD weight updates across `LinearQLearningAgent`, `DuelingDQNAgent`, and `TabularQLearningAgent`.
   - `test_apfd_and_napfd_hand_calculated`: Verifies full-suite APFD and NAPFD mathematical exactness.

2. **Step 2 — Run the Master Experiment Pipeline**:
   ```bash
   python run_experiments.py
   ```
   This script deterministically executes all 8 experimental suites across seeds `[42, 101, 777]`:
   - **Experiment `[1/8]` Multi-Seed Training**: Trains `LinearQLearningAgent` and `DuelingDQNAgent` across 80 synthetic training builds for 450 episodes per seed, saving trained models to `models/q_table.npy` and `models/dueling_dqn.npy` and training curves to `results/training_history.csv`.
   - **Experiment `[2/8]` Main Baseline Comparison**: Evaluates the trained RL agents against `Random`, `Greedy (Coverage/Cost)`, `Greedy (Failure/Cost)`, and `Greedy (Recency/Cost)` across 120 held-out synthetic builds (`360` total evaluations per strategy), writing `results/raw_results.csv`, `results/summary.csv`, and `results/paired_synthetic_comparison.csv`.
   - **Experiment `[3/8]` Budget Sensitivity Analysis**: Evaluates all strategies across 5 budget levels (`20%`, `35%`, `50%`, `65%`, `80%` of full suite cost) using leak-free training histories, writing `results/budget_sensitivity.csv`.
   - **Experiment `[4/8]` Reward Function Ablation**: Trains and evaluates `Balanced`, `Bug-Only`, `Coverage-Only`, and `Cost-Penalized` reward configurations, writing `results/reward_ablation.csv`.
   - **Experiment `[5/8]` State Representation Ablation**: Evaluates `Full State (12 Features)`, `No Recency Features`, `No Coverage Features`, and `Cost & Budget Only`, writing `results/state_ablation.csv`.
   - **Experiment `[6/8]` Agent Architecture Ablation**: Compares `Dueling DQN (Online RL)`, `Linear Q-Learning (ours)`, `Contextual Bandit (LinUCB)`, and `Tabular Q-Learning`, writing `results/agent_ablation.csv`.
   - **Experiment `[7/8]` Real-World Industrial CI Evaluation**: Chronologically splits **ABB Paint Control** (70% train / 30% test = 89 defect-containing held-out cycles) and **ABB IOF/ROL** (70% train / 30% test = 63 defect-containing held-out cycles), trains across seeds `[42, 101, 777]`, and exports `results/real_world_paintcontrol_*.csv` and `results/real_world_iofrol_*.csv`.
   - **Experiment `[8/8]` Publication Figure Generation**: Generates all 300 DPI PNG figures in `results/figures/`.

3. **Step 3 — Inspect Results in the Interactive Dashboard**:
   ```bash
   python server.py
   ```
   Navigate to `http://127.0.0.1:5000` in your browser to inspect all tables, charts, and live CI log evaluations.

---

## Methodology & RL Architectures

### 12 Causal State-Action Features (`\phi(s, a)`)
For each legal candidate test $a \in \mathcal{A}(s)$, `extract_batch_test_features` in `src/q_learning.py` computes 9 action-discriminative cost-rate features and 3 episode-state context features:
- **Action-Discriminative Cost-Rate Features (`0..8`)**:
  0. `long_fail_per_cost`: Blended static training + online causal failure rate per effective cost $c_a^{\beta(s)}$
  1. `recent_fail_per_cost`: Exponentially weighted recency failure rate + inverse-lag bonus per effective cost
  2. `effective_fail_per_cost`: Within-episode posterior failure probability (incorporating sibling redundancy suppression and co-failure locality) per effective cost
  3. `last_verdict_fail_per_cost`: Most recent prior verdict indicator + co-failure boost per effective cost
  4. `new_cov_norm`: Marginal uncovered feature ratio normalized by test cost
  5. `cov_per_cost`: Marginal uncovered features per unit execution cost
  6. `uncovered_risk_per_cost`: Interaction of posterior fault probability and uncovered feature ratio per effective cost
  7. `inv_cost`: Inverse execution cost ($1 / c_a$)
  8. `cost_ratio`: Candidate test cost relative to remaining episode budget ($c_a / B_{\text{rem}}$)
- **State-Context Features (`9..11`)**:
  9. `budget_remaining`: Fraction of total build budget remaining ($B_{\text{rem}} / B_{\text{total}}$)
  10. `coverage_progress`: Fraction of total system features covered so far in the episode
  11. `recent_success`: Indicator of whether the last 3 executed tests revealed at least one new bug

---

## Benchmark Results Summary

All results below are aggregated across seeds `[42, 101, 777]` under a 45% execution budget (`budget_ratio=0.45`):

### 1. Synthetic Benchmark (120 Held-Out Builds, 360 Evaluations)

| Strategy | BDR Mean (±95% CI) | Unique Bugs | Coverage Rate | Bugs / Cost | APFD Mean | NAPFD Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random | 0.5555 (±0.0203) | 3.5306 | 0.9500 | 0.1294 | 0.4413 | 0.4486 |
| Greedy (Coverage/Cost) | 0.5332 (±0.0272) | 3.3750 | 1.0000 | 0.1242 | 0.4346 | 0.4424 |
| Greedy (Failure/Cost) | 0.8082 (±0.0249) | 5.1583 | 0.9153 | 0.1889 | 0.5792 | 0.5824 |
| Greedy (Recency/Cost) | 0.9712 (±0.0107) | 6.1917 | 0.9215 | 0.2264 | 0.8024 | 0.8029 |
| **Q-Learning (Linear)** | **0.9869 (±0.0100)** | **6.3083** | **0.9972** | **0.2329** | **0.8702** | **0.8705** |
| **Dueling DQN (Online RL)** | **0.9873 (±0.0099)** | **6.3111** | **0.9970** | **0.2321** | **0.8715** | **0.8717** |

### 2. Real-World ABB IOF/ROL Benchmark (63 Held-Out Defect Cycles, 189 Evaluations)

| Strategy | BDR Mean (±95% CI) | Unique Bugs | Coverage Proxy | Bugs / Cost | APFD Mean | NAPFD Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random | 0.4740 (±0.0383) | 11.3280 | 0.9083 | 0.0987 | 0.3564 | 0.3618 |
| Greedy (Coverage/Cost) | 0.5169 (±0.0660) | 11.4921 | 0.9450 | 0.1009 | 0.3825 | 0.3869 |
| Greedy (Failure/Cost) | 0.7010 (±0.0560) | 14.4286 | 0.8974 | 0.1300 | 0.5477 | 0.5519 |
| Greedy (Recency/Cost) | 0.7141 (±0.0534) | 15.3492 | 0.9101 | 0.1360 | 0.5597 | 0.5637 |
| **Q-Learning (Linear)** | **0.7441 (±0.0513)** | **16.2381** | **0.9189** | **0.1429** | **0.5876** | **0.5914** |
| **Dueling DQN (Online RL)** | **0.7454 (±0.0486)** | **16.1587** | **0.9185** | **0.1425** | **0.5889** | **0.5927** |

### 3. Real-World ABB Paint Control Benchmark (89 Held-Out Defect Cycles, 267 Evaluations)

| Strategy | BDR Mean (±95% CI) | Unique Bugs | Coverage Proxy | Bugs / Cost | APFD Mean | NAPFD Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random | 0.4331 (±0.0266) | 3.0487 | 0.9800 | 0.0244 | 0.3299 | 0.3330 |
| Greedy (Coverage/Cost) | 0.5112 (±0.0831) | 2.9663 | 0.9918 | 0.0288 | 0.4071 | 0.4090 |
| Greedy (Failure/Cost) | 0.7654 (±0.0440) | 4.8764 | 0.9843 | 0.0420 | 0.5772 | 0.5790 |
| Greedy (Recency/Cost) | 0.8969 (±0.0397) | 5.0449 | 0.9723 | 0.0417 | 0.8280 | 0.8290 |
| **Q-Learning (Linear)** | **0.9223 (±0.0316)** | **5.6105** | **0.9838** | **0.0464** | **0.8415** | **0.8424** |
| **Dueling DQN (Online RL)** | **0.9098 (±0.0345)** | **5.5131** | **0.9833** | **0.0455** | **0.8406** | **0.8415** |
