# TestPilot-RL: Intelligent Software Testing Using Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Software%20Testing%20RL-green)](.)

> **Q-learning with linear function approximation adaptively prioritizes test cases under budget constraints. It edges out the best greedy baseline on bug detection rate on the synthetic benchmark and on IOF/ROL, but not on PaintControl (see results below).**

---

## 📋 Project Overview

TestPilot-RL frames software test case prioritization as a **Reinforcement Learning problem**: a Q-learning agent learns to select the best tests to execute within a limited budget, balancing bug detection, code coverage, and execution cost.

### Key Features

- **Linear Function Approximation Q-Learning** with per-test observable feature vectors
- **Prioritized Experience Replay (PER)** for efficient learning from high-value transitions
- **Double Q-Learning** to reduce overestimation bias
- **Reward Shaping** toward failure-rate oracle, annealed during training
- **Legal Action Masking** — no test is ever run twice, no budget violation
- **Real-world evaluation** on 2 ABB industrial CI datasets (PaintControl & IOF/ROL)

---

## 📊 Results

### Synthetic Benchmark (120 held-out builds × 3 seeds = 360 evaluations)

| Strategy | BDR ↑ | APFD ↑ | Bugs/Cost ↑ | Coverage |
|---|---|---|---|---|
| **Q-Learning (ours)** | **0.810** 🥇 | 0.482 | **0.134** 🥇 | 0.980 |
| Greedy (Failure/Cost) | 0.802 | **0.492** 🥇 | 0.132 | 0.976 |
| Greedy (Coverage/Cost) | 0.698 | 0.398 | 0.112 | **0.998** |
| Random | 0.680 | 0.366 | 0.109 | 0.954 |

> **Q-Learning beats Greedy (Failure/Cost) on Bug Detection Rate (+0.8pp) and Bugs/Cost (+1.7%), but Greedy (Failure/Cost) has slightly higher APFD. The BDR gap is small relative to the std (~0.15).**

### Real-World CI: IOF/ROL Industrial Dataset (63 held-out CI cycles with defects)

| Strategy | BDR ↑ | APFD ↑ | Coverage |
|---|---|---|---|
| **Q-Learning (ours)** | **0.576** 🥇 | **0.269** 🥇 | 0.874 |
| Greedy (Failure/Cost) | 0.532 | 0.244 | 0.873 |
| Greedy (Coverage/Cost) | 0.499 | 0.239 | **0.924** |
| Random | 0.474 | 0.222 | 0.849 |

> **Q-Learning beats all baselines on the IOF/ROL industrial dataset (+4.5pp BDR over the best greedy, 0.576 vs 0.532).**

### Real-World CI: Paint Control Industrial Dataset (69 held-out CI cycles with defects)

| Strategy | BDR ↑ | APFD ↑ | Coverage |
|---|---|---|---|
| Greedy (Coverage/Cost) | **0.686** 🥇 | **0.337** 🥇 | **0.990** |
| Random | 0.460 | 0.228 | 0.958 |
| Q-Learning (ours) | 0.456 | 0.181 | 0.965 |
| Greedy (Failure/Cost) | 0.418 | 0.166 | 0.973 |

> On PaintControl, failure-rate history is weakly predictive (sparse/volatile failures), so coverage-greedy wins. Q-Learning beats Greedy (Failure/Cost) (0.456 vs 0.418) but is on par with Random (0.460) on BDR and below it on APFD.

### Agent Ablation: Linear vs Tabular Q-Learning

| Agent | BDR ↑ | APFD ↑ |
|---|---|---|
| **Linear Q-Learning (ours)** | **0.820** 🥇 | **0.484** 🥇 |
| Tabular Q-Learning (legacy) | 0.802 | 0.427 |

> Linear function approximation improves BDR by **+1.8pp** and APFD by **+5.7pp** over the tabular baseline.

---

## 🏗️ Architecture

```
TestPilot-RL/
├── data/                      # Downloaded real-world CI datasets
│   ├── paintcontrol.csv       # ABB Paint Control (352 CI cycles)
│   └── iofrol.csv             # ABB IOF/ROL (320 CI cycles)
├── src/
│   ├── q_learning.py          # LinearQLearningAgent (primary) + TabularQLearningAgent
│   ├── environment.py         # Gym-like RL environment with budget masking
│   ├── simulator.py           # Synthetic build generator + RealWorldCIAdapter
│   ├── baselines.py           # Random, Greedy (Coverage/Cost), Greedy (Failure/Cost)
│   ├── train.py               # Training pipeline with PER and epsilon decay
│   ├── evaluate.py            # Multi-seed evaluation framework
│   ├── run_experiments.py     # Master experiment runner (7 experiments)
│   ├── metrics.py             # BDR, APFD, coverage, efficiency metrics
│   └── data_loader.py         # Dataset download and parsing
├── models/
│   └── q_table.npy            # Trained model checkpoint
├── results/
│   ├── summary.csv            # Synthetic benchmark results
│   ├── real_world_paintcontrol_summary.csv
│   ├── real_world_iofrol_summary.csv
│   ├── training_history.csv
│   ├── budget_sensitivity.csv
│   ├── reward_ablation.csv
│   ├── agent_ablation.csv
│   ├── *_raw.csv              # Per-evaluation raw results
│   └── figures/               # All publication-ready plots
└── tests/
    ├── test_environment.py    # Environment unit tests
    └── test_q_learning.py     # Agent unit tests (12 tests total)
```

---

## 🧠 The RL Formulation

### State Space
Compact 3-tuple: `(budget_bin, coverage_bin, history_bin)` → 75 discrete states

### Action Space
Select one unexecuted test `t ∈ {0, ..., N-1}` with `cost(t) ≤ remaining_budget`

### Feature Vector (per-test, observable at selection time)
```
φ(s, a) = [
  fail_rate(a),          # historical failure rate of test a
  fail_rate(a) / cost,   # core greedy signal — normalized
  new_coverage / total,  # marginal feature coverage gain
  new_coverage / cost,   # coverage efficiency
  cost / remaining_budget,  # affordability ratio
  remaining_budget / total_budget,  # budget pressure
  covered_features / total_features,  # episode progress
  recent_bug_momentum,   # 1 if last 3 tests found bugs
]
```

### Q-Function
```
Q(s, a) = w · φ(s, a)    (linear function approximation)
```
Learned via semi-gradient Q-learning with PER batch updates.

### Reward Function
```
r_t = w_b · B_t + w_c · C_t − w_e · E_t
```
Where:
- `B_t` = new bugs found at step t
- `C_t` = new features covered / total features
- `E_t` = execution cost / total budget
- Weights: `w_b=3.0, w_c=0.8, w_e=0.2`

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run All Experiments (reproduces all results)
```bash
python -m src.run_experiments
```
This runs 7 experiments (~15-20 minutes):
1. Training curve
2. Synthetic baseline comparison
3. Budget sensitivity analysis
4. Reward function ablation
5. Agent ablation (Linear vs Tabular)
6. Real-world PaintControl evaluation
7. Real-world IOF/ROL evaluation

### 3. Train Only
```bash
# Train linear agent on synthetic data
python -m src.train --episodes 2000 --dataset synthetic

# Train on real-world Paint Control dataset
python -m src.train --episodes 1500 --dataset paintcontrol

# Train on IOF/ROL dataset
python -m src.train --episodes 1500 --dataset iofrol
```

### 4. Evaluate Only
```bash
python -m src.evaluate --dataset synthetic --num_test_builds 120
```

### 5. Run Unit Tests
```bash
python -m pytest tests/ -v
```

---

## 📈 Key Experimental Findings

### Finding 1: Linear FA beats Tabular Q-Learning
The tabular agent can't generalize: action ID `5` on build A means something different than action ID `5` on build B. The linear agent encodes per-test *observable signals* (fail_rate/cost, coverage efficiency) that transfer across all builds.

### Finding 2: Q-Learning beats Greedy on Synthetic (+0.8pp BDR)
By learning from 2000 episodes with reward shaping, the agent internalizes the failure-rate/cost scoring of the greedy heuristic and adds budget-awareness, coverage momentum, and recency — producing higher bug detection under the same budget.

### Finding 3: Q-Learning beats ALL baselines on IOF/ROL (+4.5pp BDR)
On the IOF/ROL industrial dataset (0.576 vs 0.532 for the best greedy), the agent's ability to generalize via feature vectors is most visible — it surpasses both greedy strategies and Random.

### Finding 4: PaintControl favors Coverage Greedy
When failure history is sparse or volatile (PaintControl has highly variable verdicts per cycle), coverage-per-cost dominates. Q-Learning beats Greedy (Failure/Cost) here but does not beat Random on BDR (0.456 vs 0.460).

### Finding 5: Budget Sensitivity (synthetic, BDR)

| Budget | Random | Greedy (Cov/Cost) | Greedy (Fail/Cost) | Q-Learning |
|---|---|---|---|---|
| 20% | 0.384 | 0.455 | **0.550** | 0.540 |
| 35% | 0.604 | 0.609 | 0.710 | **0.721** |
| 50% | 0.776 | 0.738 | 0.829 | **0.842** |
| 65% | 0.841 | 0.848 | **0.902** | 0.891 |
| 80% | 0.936 | **0.945** | 0.938 | 0.938 |

Q-Learning is not uniformly better. It wins at 35% and 50% budgets, loses narrowly to Greedy (Failure/Cost) at 20% and 65%, and ties it at 80% (where all strategies converge and Greedy (Coverage/Cost) is marginally highest).

### Finding 6: Reward weights barely matter
The three reward-function variants (Balanced, Bug-Centric, Coverage-Centric) all reach BDR 0.817–0.821, so on this benchmark the result is insensitive to the reward weights. The shaping term toward the failure-rate oracle likely dominates the learned policy. This is a genuine result, not a bug.

---

## 📚 Datasets

### ABB Paint Control (`paintcontrol.csv`)
- **Source**: [RETECS by Spieker et al. (ISSTA 2017)](https://github.com/mregorova/RETECS)
- **Size**: 352 CI cycles in the raw file (89 unique test names)
- **Split**: only cycles with at least one defect are used: 159 train / 69 test (70/30 split)
- **Description**: Real ABB industrial CI test results, paint control software system

### ABB IOF/ROL (`iofrol.csv`)
- **Source**: [RETECS by Spieker et al. (ISSTA 2017)](https://github.com/mregorova/RETECS)
- **Size**: 320 CI cycles in the raw file (1941 unique test names)
- **Split**: only cycles with at least one defect are used: 145 train / 63 test (70/30 split)
- **Description**: Real ABB industrial CI test results, IOF/ROL control system

### Citation
```bibtex
@inproceedings{spieker2017reinforcement,
  title={Reinforcement learning for automatic test case prioritization and selection in continuous integration},
  author={Spieker, Helge and Gotlieb, Arnaud and Marijan, Dusica and Mossige, Morten},
  booktitle={Proceedings of the 26th ACM SIGSOFT International Symposium on Software Testing and Analysis},
  pages={12--22},
  year={2017}
}
```

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `alpha` | 0.008 | Learning rate |
| `gamma` | 0.97 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.02 | Minimum exploration rate |
| `epsilon_decay` | 0.997 | Per-episode epsilon decay |
| `batch_size` | 64 | PER mini-batch size |
| `replay_capacity` | 30,000 | Replay buffer size |
| `shaping_coef` | 2.5 | Reward shaping coefficient |
| `target_update_freq` | 200 | Target network sync frequency |
| `episodes` (synthetic) | 2,000 | Training episodes |
| `episodes` (real-world) | 1,500 | Training episodes |
| `budget_ratio` | 0.45 | Testing budget (45% of full suite cost) |

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🔗 References

- Spieker et al. (2017). *Reinforcement learning for automatic test case prioritization and selection in continuous integration.* ISSTA 2017.
- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Schaul et al. (2016). *Prioritized experience replay.* ICLR 2016.
