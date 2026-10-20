# TestPilot-RL: Intelligent Software Testing Using Reinforcement Learning

**Python 3.9+** · **MIT License** · **Research project**

TestPilot-RL frames test-case prioritization as a reinforcement-learning problem. A Q-learning agent learns which tests to execute under a limited testing budget, balancing bug detection, feature coverage, and execution cost.

> **Result summary:** Performance depends on the dataset. Q-learning is competitive on the synthetic benchmark, performs best on the reported IOF/ROL metrics, and does **not** outperform the coverage-per-cost greedy baseline on PaintControl. These results should not be presented as a universal win for reinforcement learning.

## Project overview

### Key features

- Linear function-approximation Q-learning with per-test observable feature vectors
- Prioritized Experience Replay (PER)
- Double Q-learning to reduce overestimation bias
- Reward shaping toward a failure-rate oracle
- Legal-action masking to avoid re-running tests and exceeding the budget
- Evaluation on two industrial continuous-integration (CI) datasets: PaintControl and IOF/ROL
- Baseline comparisons, budget sensitivity, reward ablation, and agent ablation

## Latest experimental results

The values below are transcribed from the latest generated summary CSVs for the completed experiment run. BDR means bug detection rate; APFD means average percentage of faults detected. Higher is better for both metrics and for bugs per cost. Coverage is reported separately.

### Synthetic benchmark

**120 held-out builds × 3 seeds = 360 evaluations**

| Strategy | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage |
|---|---:|---:|---:|---:|
| Q-learning (linear) | 0.8044 | 0.4681 | 0.1317 | 0.9951 |
| Greedy (failure/cost) | 0.8018 | 0.4915 | 0.1318 | 0.9757 |
| Greedy (coverage/cost) | 0.6977 | 0.3983 | 0.1125 | 0.9979 |
| Random | 0.6795 | 0.3663 | 0.1093 | 0.9537 |

**Interpretation:** Q-learning has a slightly higher BDR than greedy failure/cost (0.8044 vs. 0.8018), but greedy failure/cost has marginally higher bugs/cost and higher APFD. The BDR difference is small; do not claim that Q-learning wins every metric.

### Real-world CI: IOF/ROL

**63 held-out CI cycles with defects; 189 evaluations in the summary**

| Strategy | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage |
|---|---:|---:|---:|---:|
| Q-learning (linear) | 0.5785 | 0.2719 | 0.1319 | 0.8910 |
| Greedy (failure/cost) | 0.5315 | 0.2437 | 0.1226 | 0.8635 |
| Greedy (coverage/cost) | 0.4984 | 0.2289 | 0.1016 | 0.9175 |
| Random | 0.4743 | 0.2217 | 0.0998 | 0.8490 |

**Interpretation:** Q-learning leads the reported IOF/ROL strategies on BDR, APFD, and bugs/cost. Greedy coverage/cost has the highest coverage. The summary results alone do not establish statistical significance; a paired analysis on the same cycles is needed before making a significance claim.

### Real-world CI: PaintControl

**69 held-out CI cycles with defects; 207 evaluations in the summary**

| Strategy | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage |
|---|---:|---:|---:|---:|
| Greedy (coverage/cost) | 0.6946 | 0.3564 | 0.0395 | 0.9865 |
| Random | 0.4603 | 0.2275 | 0.0244 | 0.9414 |
| Q-learning (linear) | 0.4297 | 0.1643 | 0.0382 | 0.9845 |
| Greedy (failure/cost) | 0.4177 | 0.1657 | 0.0371 | 0.9816 |

**Interpretation:** Coverage/cost greedy is the strongest strategy on PaintControl BDR and APFD. Q-learning does not beat Random on BDR or APFD in this dataset. This is an important limitation, and the reason for the result should be investigated rather than assumed.

### Agent ablation: linear vs. tabular Q-learning

Means over 60 held-out builds per variant, calculated from the latest `agent_ablation.csv`.

| Agent | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage |
|---|---:|---:|---:|---:|
| Linear Q-learning | 0.8124 | 0.4794 | 0.1311 | 0.9944 |
| Tabular Q-learning (legacy) | 0.7880 | 0.4468 | 0.1263 | 0.9972 |

**Interpretation:** The linear agent has higher mean BDR, APFD, and bugs/cost in this ablation; the tabular agent has slightly higher mean coverage. Treat these as descriptive comparisons unless a statistical test is performed.

## Architecture

```text
TestPilot-RL/
├── data/                       # Real-world CI datasets
│   ├── paintcontrol.csv
│   └── iofrol.csv
├── src/
│   ├── q_learning.py            # Linear and tabular Q-learning agents
│   ├── environment.py           # Budget-constrained testing environment
│   ├── simulator.py             # Synthetic generator and real-world adapter
│   ├── baselines.py             # Random and greedy baselines
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation framework
│   ├── run_experiments.py       # Master runner for seven experiment stages
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
```

File names can change as the experiment pipeline evolves; check the current repository if a listed path is absent.

## Reinforcement-learning formulation

### State and actions

The environment represents the testing process using a compact state derived from budget, coverage progress, and history. The action is to select an unexecuted test whose cost fits within the remaining budget.

### Per-test feature vector

The linear agent uses observable features available at selection time, including:

- Historical failure rate
- Failure rate normalized by execution cost
- Marginal feature coverage and coverage per cost
- Test cost relative to remaining budget
- Remaining budget relative to total budget
- Fraction of features already covered
- Recent bug-detection momentum

The linear Q-function has the form:

\[
Q(s,a)=w^\top \phi(s,a)
\]

where \(\phi(s,a)\) is the feature vector for state \(s\) and test action \(a\), and \(w\) is learned during training.

### Reward

The reward combines newly detected bugs, newly covered features, and execution cost:

\[
r_t = w_b B_t + w_c C_t - w_e E_t
\]

where \(B_t\) is the new-bug signal, \(C_t\) is the coverage gain, and \(E_t\) is the normalized execution cost. The implementation and its current configuration are the source of truth for exact weights and training details.

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run all experiment stages

```bash
python -m src.run_experiments
```

The master runner executes the training and evaluation stages, including synthetic baseline comparison, budget sensitivity, reward ablation, linear-versus-tabular ablation, and the two real-world datasets. Runtime depends on your machine.

### 3. Train a model separately

```bash
# Synthetic dataset
python -m src.train --episodes 2000 --dataset synthetic

# PaintControl
python -m src.train --episodes 1500 --dataset paintcontrol

# IOF/ROL
python -m src.train --episodes 1500 --dataset iofrol
```

### 4. Evaluate separately

```bash
python -m src.evaluate --dataset synthetic --num_test_builds 120
```

Use `python -m src.evaluate --help` to check the options supported by the current version.

### 5. Run tests

```bash
python -m pytest tests/ -v
```

## Main findings and limitations

1. **Synthetic benchmark:** Q-learning's BDR is slightly above greedy failure/cost, but the greedy baseline is slightly better on APFD and bugs/cost.
2. **IOF/ROL:** Q-learning has the best reported BDR, APFD, and bugs/cost among the listed strategies.
3. **PaintControl:** Greedy coverage/cost performs best on BDR and APFD. Q-learning does not outperform Random on those two metrics.
4. **Agent ablation:** The linear agent's reported means exceed the tabular agent on BDR, APFD, and bugs/cost, while tabular coverage is slightly higher.
5. **No universal winner:** The results vary by dataset and metric. Avoid stating that Q-learning beats every baseline across all settings.
6. **Statistical caution:** These summary tables are descriptive. Use paired per-cycle/per-build comparisons and confidence intervals or suitable statistical tests before asserting significance.

## Datasets

The project uses two industrial CI datasets associated with the RETECS work by Spieker et al. (ISSTA 2017):

- **PaintControl:** the raw dataset is reported as 352 CI cycles; the project split used 159 training cycles and 69 held-out cycles with defects.
- **IOF/ROL:** the raw dataset is reported as 320 CI cycles; the project split used 145 training cycles and 63 held-out cycles with defects.

Only cycles with at least one defect are used for the reported real-world training/test split. Consult the loader and experiment logs for the exact filtering and split procedure used in a given run.

### Reference

Spieker, H., Gotlieb, A., Marijan, D., and Mossige, M. (2017). “Reinforcement learning for automatic test case prioritization and selection in continuous integration.” *Proceedings of the 26th ACM SIGSOFT International Symposium on Software Testing and Analysis*, 12–22.

## License

MIT License. See `LICENSE`.
