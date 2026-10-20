# TestPilot-RL: Intelligent Software Testing Using Reinforcement Learning

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

TestPilot-RL frames test-case prioritization as a reinforcement-learning problem. A Q-learning agent learns which tests to execute under a limited testing budget, balancing bug detection, feature coverage, and execution cost.

> **Result summary:** Performance depends on the dataset. Q-learning has a very small BDR edge over the best greedy baseline on the synthetic benchmark, performs best on several IOF/ROL metrics, and does **not** outperform the coverage-per-cost greedy baseline or Random on PaintControl BDR and APFD. These results are not a universal win for reinforcement learning.

## Project overview

### Key features

- Linear function-approximation Q-learning with per-test observable feature vectors
- Prioritized Experience Replay (PER)
- Double Q-learning to reduce overestimation bias
- Reward shaping toward a failure-rate oracle
- Legal-action masking to avoid re-running tests and exceeding the budget
- Evaluation on a synthetic benchmark and two industrial continuous-integration (CI) datasets: PaintControl and IOF/ROL
- Baseline comparisons, budget sensitivity, reward ablation, and agent ablation

## Experimental results

All values are computed from the generated CSVs in `results/` (one seed/config; regenerate with `python -m src.run_experiments`). BDR is bug detection rate and APFD is average percentage of faults detected. Higher is better for BDR, APFD and bugs/cost. Coverage is reported separately.

### Synthetic benchmark

120 held-out builds × 3 seeds = 360 evaluations.

| Strategy | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage |
|---|---:|---:|---:|---:|
| Q-learning (linear) | 0.8044 | 0.4681 | 0.1317 | 0.9951 |
| Greedy (failure/cost) | 0.8018 | 0.4915 | 0.1318 | 0.9757 |
| Greedy (coverage/cost) | 0.6977 | 0.3983 | 0.1125 | 0.9979 |
| Random | 0.6795 | 0.3663 | 0.1093 | 0.9537 |

**Interpretation:** Q-learning has the highest BDR by only 0.26 percentage points over greedy failure/cost. Greedy failure/cost has slightly higher APFD and bugs/cost, while Q-learning has higher coverage. The BDR difference is small and should not be read as a decisive win.

### Real-world CI: IOF/ROL

63 held-out CI cycles with defects; 189 evaluations.

| Strategy | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage |
|---|---:|---:|---:|---:|
| Q-learning (linear) | 0.5785 | 0.2719 | 0.1319 | 0.8910 |
| Greedy (failure/cost) | 0.5315 | 0.2437 | 0.1226 | 0.8730 |
| Greedy (coverage/cost) | 0.4984 | 0.2289 | 0.1016 | 0.9175 |
| Random | 0.4743 | 0.2217 | 0.0998 | 0.8487 |

**Interpretation:** Q-learning leads on BDR (+4.7 percentage points over greedy failure/cost), APFD, and bugs/cost. Greedy coverage/cost has the highest coverage. These are descriptive means; no significance test has been run.

### Real-world CI: PaintControl

69 held-out CI cycles with defects; 207 evaluations.

| Strategy | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage |
|---|---:|---:|---:|---:|
| Greedy (coverage/cost) | 0.6946 | 0.3564 | 0.0395 | 0.9865 |
| Random | 0.4603 | 0.2275 | 0.0244 | 0.9578 |
| Q-learning (linear) | 0.4297 | 0.1643 | 0.0382 | 0.9845 |
| Greedy (failure/cost) | 0.4177 | 0.1657 | 0.0371 | 0.9816 |

**Interpretation:** Greedy coverage/cost is clearly the strongest strategy on BDR and APFD. Q-learning does not beat Random on BDR (0.4297 vs. 0.4603) or APFD (0.1643 vs. 0.2275). Greedy coverage/cost leads on BDR and APFD, and also has the highest bugs/cost (0.0395). Q-learning has high coverage but is not the strongest overall strategy on this dataset.

### Budget sensitivity (synthetic, BDR)

| Budget | Random | Greedy (coverage/cost) | Greedy (failure/cost) | Q-learning |
|---|---:|---:|---:|---:|
| 20% | 0.384 | 0.455 | **0.550** | 0.542 |
| 35% | 0.604 | 0.609 | **0.710** | 0.706 |
| 50% | 0.776 | 0.738 | **0.829** | 0.812 |
| 65% | 0.841 | 0.848 | **0.902** | 0.899 |
| 80% | 0.936 | **0.945** | 0.938 | 0.943 |

**Interpretation:** Q-learning is not uniformly better. Greedy failure/cost leads at 20%, 35%, 50%, and 65% budgets; greedy coverage/cost leads at 80%. Q-learning is close to the leading strategy at several budgets but does not lead at any of these budget settings.

### Reward-function ablation

| Variant | BDR ↑ |
|---|---:|
| Balanced (w_b=3.0, w_c=0.8, w_e=0.2) | 0.8124 |
| Bug-centric (w_b=5.0, w_c=0.0, w_e=0.0) | 0.8010 |
| Coverage-centric (w_b=1.0, w_c=3.0, w_e=0.2) | 0.8108 |

**Interpretation:** The BDR means range from 0.8010 to 0.8124 (about 1.14 percentage points). The balanced variant is highest in this run, but the differences are descriptive and do not establish statistical significance. The effect of the shaping term has not been isolated here.

### Agent ablation: linear vs. tabular Q-learning

Means over 60 held-out builds per variant.

| Agent | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage |
|---|---:|---:|---:|---:|
| Linear Q-learning | 0.8124 | 0.4794 | 0.1311 | 0.9944 |
| Tabular Q-learning (legacy) | 0.7880 | 0.4468 | 0.1263 | 0.9972 |

**Interpretation:** The linear agent has higher mean BDR, APFD, and bugs/cost; the tabular agent has higher mean coverage. The tabular agent cannot transfer what it learns across builds, because an action index refers to a different test in each build. The linear agent scores tests by observable features that carry over between builds. Treat these as descriptive comparisons.

## Project structure

```text
TestPilot-RL/
├── data/
│   ├── paintcontrol.csv         # PaintControl CI dataset
│   └── iofrol.csv               # IOF/ROL CI dataset
├── src/
│   ├── q_learning.py            # Linear and tabular Q-learning agents
│   ├── environment.py           # Budget-constrained testing environment
│   ├── simulator.py             # Synthetic generator and real-world adapter
│   ├── baselines.py             # Random and greedy baselines
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation framework
│   ├── run_experiments.py       # Master runner for the seven experiment stages
│   ├── metrics.py               # BDR, APFD, coverage, efficiency metrics
│   └── data_loader.py           # Dataset loading and parsing
├── models/
│   └── q_table.npy              # Model artifact written by the experiment run
├── results/
│   ├── summary.csv
│   ├── real_world_paintcontrol_summary.csv
│   ├── real_world_iofrol_summary.csv
│   ├── training_history.csv
│   ├── budget_sensitivity.csv
│   ├── reward_ablation.csv
│   ├── agent_ablation.csv
│   ├── *_raw.csv                # Per-evaluation results
│   └── figures/                 # Generated experiment plots
└── tests/
    ├── test_environment.py
    └── test_q_learning.py       # 12 unit tests in total
```

## Reinforcement-learning formulation

### State and actions

The environment represents the testing process using a compact state derived from budget, coverage progress, and history. The action is to select an unexecuted test whose cost fits within the remaining budget.

### Per-test feature vector

The linear agent uses observable features available at selection time:

- Historical failure rate
- Failure rate normalized by execution cost
- Marginal feature coverage and coverage per cost
- Test cost relative to remaining budget
- Remaining budget relative to total budget
- Fraction of features already covered
- Recent bug-detection momentum (whether the last 3 tests found bugs)

The linear Q-function is

$$Q(s,a) = w^\top \phi(s,a)$$

where $\phi(s,a)$ is the feature vector for state $s$ and test action $a$, and $w$ is learned with semi-gradient Q-learning and PER batch updates.

### Reward

$$r_t = w_b B_t + w_c C_t - w_e E_t$$

where $B_t$ is the number of new bugs found, $C_t$ is the fraction of features newly covered, and $E_t$ is the execution cost divided by the total budget. The default weights are $w_b=3.0$, $w_c=0.8$, $w_e=0.2$.

### Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate (`alpha`) | 0.008 |
| Discount (`gamma`) | 0.97 |
| Epsilon start / min / decay | 1.0 / 0.02 / 0.997 per episode |
| PER batch size / replay capacity | 64 / 30,000 |
| Shaping coefficient | 2.5 |
| Target update frequency | 200 |
| Training episodes (synthetic / real-world) | 2,000 / 1,500 |
| Testing budget (`budget_ratio`) | 0.45 of full-suite cost |

## Quick start

```bash
pip install -r requirements.txt

# Reproduce every CSV and figure
python -m src.run_experiments

# Train a model separately
python -m src.train --episodes 2000 --dataset synthetic
python -m src.train --episodes 1500 --dataset paintcontrol
python -m src.train --episodes 1500 --dataset iofrol

# Evaluate separately (see --help for options)
python -m src.evaluate --dataset synthetic --num_test_builds 120

# Run tests
python -m pytest tests/ -v
```

## Main findings and limitations

1. **Synthetic benchmark:** Q-learning has a marginally higher BDR; greedy failure/cost is better on APFD and marginally higher on bugs/cost.
2. **IOF/ROL:** Q-learning has the best BDR (+4.7 percentage points), APFD, and bugs/cost.
3. **PaintControl:** Greedy coverage/cost is best on BDR, APFD, and bugs/cost; Q-learning does not beat Random on BDR or APFD.
4. **Budget sensitivity:** Greedy failure/cost leads at 20–65% budgets, while greedy coverage/cost leads at 80%; no single strategy wins at every budget.
5. **Reward weights:** BDR ranges from 0.8010 to 0.8124 across the three reward variants; differences are descriptive.
6. **Agent ablation:** Linear Q-learning has higher mean BDR, APFD, and bugs/cost; tabular Q-learning has slightly higher coverage.
7. **Statistics:** All tables are descriptive means from a single configuration. Paired per-cycle comparisons or confidence intervals are needed before claiming significance.

## Datasets

The two real-world CI datasets were collected at ABB Robotics Norway and published with Spieker et al. (2017). We did not collect or modify them; they are included in `data/` for reproducibility.

| Dataset | Raw size | Used in our split |
|---|---|---|
| PaintControl | 352 CI cycles, 89 unique tests | 159 train / 69 test cycles |
| IOF/ROL | 320 CI cycles, 1,941 unique tests | 145 train / 63 test cycles |

Only cycles with at least one defect are used for the real-world train/test split (70/30 on cycles). The agent, environment, features, reward, baselines, experiments and analysis in this repository are our own implementation.

### Reference

Spieker, H., Gotlieb, A., Marijan, D., and Mossige, M. (2017). Reinforcement learning for automatic test case prioritization and selection in continuous integration. *Proceedings of the 26th ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA)*. DOI: [10.1145/3092703.3092709](https://doi.org/10.1145/3092703.3092709). Original data: <https://bitbucket.org/helges/atcs-data>.

## License

MIT License. See `LICENSE`. The datasets belong to their original authors; see the reference above.
