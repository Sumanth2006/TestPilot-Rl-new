# TestPilot-RL: Intelligent Software Testing Using Reinforcement Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Research](https://img.shields.io/badge/Research-Software%20Testing%20RL-green)](.)

> TestPilot-RL studies whether a reinforcement-learning agent can prioritize software tests under a limited execution budget. Results are mixed across datasets: Q-learning is competitive on the synthetic benchmark and performs best on the reported IOF/ROL metrics, while Greedy Coverage/Cost is the strongest strategy on PaintControl.

## Project Overview

TestPilot-RL frames software test-case prioritization as a reinforcement-learning problem. The agent learns which test to execute next while balancing bug discovery, coverage, and execution cost. It is evaluated against Random, Greedy Coverage/Cost, and Greedy Failure/Cost strategies.

### Key Features

- Linear Q-learning with per-test observable feature vectors
- Prioritized Experience Replay (PER)
- Double Q-learning
- Reward shaping during training
- Legal action masking to avoid rerunning tests and exceeding the available budget
- Synthetic benchmark and two real-world continuous-integration datasets: PaintControl and IOF/ROL
- Experiment runner for baseline comparisons, budget sensitivity, reward ablation, and agent ablation

## Latest Experimental Results

The tables below reflect the committed summary CSVs. BDR means bug detection rate; APFD measures how early faults are detected, with higher values generally preferred. Values are means across the reported evaluations. The CSV files also contain standard deviations and 95% confidence-interval half-widths.

### Synthetic benchmark

120 held-out builds × 3 seeds = 360 evaluations.

| Strategy | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage ↑ |
|---|---:|---:|---:|---:|
| Random | 0.6795 | 0.3663 | 0.1093 | 0.9537 |
| Greedy (Coverage/Cost) | 0.6977 | 0.3983 | 0.1125 | **0.9979** |
| Greedy (Failure/Cost) | 0.8018 | **0.4915** | **0.1318** | 0.9757 |
| Q-Learning | **0.8044** | 0.4681 | 0.1317 | 0.9951 |

**Interpretation:** Q-learning's BDR is slightly higher than Greedy Failure/Cost (0.8044 vs. 0.8018, a 0.26 percentage-point difference). Greedy Failure/Cost has higher APFD and a marginally higher bugs-per-cost mean. These aggregate results do not establish that the BDR difference is statistically significant.

### Real-world CI: PaintControl

207 evaluations in the generated summary.

| Strategy | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage ↑ |
|---|---:|---:|---:|---:|
| Random | 0.4603 | 0.2275 | 0.0244 | 0.9414 |
| Greedy (Coverage/Cost) | **0.6946** | **0.3564** | **0.0395** | **0.9865** |
| Greedy (Failure/Cost) | 0.4177 | 0.1657 | 0.0371 | 0.9816 |
| Q-Learning | 0.4297 | 0.1643 | 0.0382 | 0.9845 |

**Interpretation:** Greedy Coverage/Cost is the strongest strategy on PaintControl for BDR and APFD. Q-learning is below Random on BDR and APFD in this evaluation. This is a key limitation and a target for further investigation; the results alone do not establish why the gap occurs.

### Real-world CI: IOF/ROL

189 evaluations in the generated summary.

| Strategy | BDR ↑ | APFD ↑ | Bugs/cost ↑ | Coverage ↑ |
|---|---:|---:|---:|---:|
| Random | 0.4743 | 0.2217 | 0.0998 | 0.8490 |
| Greedy (Coverage/Cost) | 0.4984 | 0.2289 | 0.1016 | **0.9175** |
| Greedy (Failure/Cost) | 0.5315 | 0.2437 | 0.1226 | 0.8635 |
| Q-Learning | **0.5785** | **0.2719** | **0.1319** | 0.8910 |

**Interpretation:** Q-learning has the highest reported BDR, APFD, and bugs-per-cost mean on IOF/ROL. Greedy Coverage/Cost has the highest coverage mean. Statistical significance should be assessed with an appropriate paired analysis before making stronger claims.

### Agent ablation: linear vs. tabular Q-learning

The committed `agent_ablation.csv` contains per-build rows for 60 held-out builds per variant. The means below are calculated from those rows.

| Agent | BDR ↑ | APFD ↑ | Bugs/cost ↑ |
|---|---:|---:|---:|
| Linear Q-Learning | **0.8124** | **0.4794** | **0.1311** |
| Tabular Q-Learning (legacy) | 0.7880 | 0.4468 | 0.1263 |

**Interpretation:** In this ablation output, the linear agent has higher mean BDR, APFD, and bugs-per-cost than the tabular variant. Treat this as an observed result for this evaluation setup, not proof of universal superiority.

## Architecture

```text
TestPilot-RL/
├── data/                       # Real-world CI datasets (downloaded separately)
├── src/
│   ├── q_learning.py            # Linear and tabular Q-learning agents
│   ├── environment.py           # RL environment and budget/action constraints
│   ├── simulator.py             # Synthetic builds and real-world CI adapter
│   ├── baselines.py              # Random and greedy baselines
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation framework
│   ├── run_experiments.py        # Master experiment runner
│   ├── metrics.py                # Evaluation metrics
│   └── data_loader.py            # Dataset loading and parsing
├── models/
│   ├── q_table.npy               # Saved model artifact
│   └── synthetic_model.npy       # Synthetic training model artifact
├── results/
│   ├── summary.csv
│   ├── real_world_paintcontrol_summary.csv
│   ├── real_world_iofrol_summary.csv
│   ├── training_history.csv
│   ├── synthetic_training_history.csv
│   ├── budget_sensitivity.csv
│   ├── reward_ablation.csv
│   ├── agent_ablation.csv
│   ├── *_raw.csv                 # Per-evaluation outputs
│   └── figures/                  # Generated experiment figures
└── tests/
```

## Reinforcement-Learning Formulation

### State

The project README's compact state description is `(budget_bin, coverage_bin, history_bin)`, with 75 discrete states.

### Actions

Select an unexecuted test that can be run within the remaining budget. Legal-action masking prevents selecting a test more than once or exceeding the budget.

### Per-test feature vector

The linear agent uses observable signals associated with each candidate test, including:

```text
fail_rate(test)
fail_rate(test) / cost(test)
marginal_coverage_gain / total_features
marginal_coverage_gain / cost(test)
cost(test) / remaining_budget
remaining_budget / total_budget
covered_features / total_features
recent_bug_momentum
```

The linear Q-function is represented as:

```text
Q(s, a) = w · φ(s, a)
```

The training implementation uses Q-learning updates and prioritized replay. See `src/q_learning.py` and `src/train.py` for the implementation details and current parameter values.

### Reward

The intended reward balances newly detected bugs, coverage gain, and execution cost. The exact implementation and any reward-shaping terms are defined in the training/environment code; consult those files when reproducing or modifying the objective.

## Quick Start

Run these commands from the repository root.

### 1. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 2. Run the test suite

```powershell
python -m pytest tests/ -v
```

### 3. Run all experiments

```powershell
python -m src.run_experiments
```

The experiment runner executes the configured training and evaluation stages, including synthetic baseline comparison, budget sensitivity, reward ablation, agent ablation, and real-world PaintControl and IOF/ROL evaluations. Runtime depends on the machine and configuration.

### 4. Train a model separately

```powershell
# Synthetic benchmark
python -m src.train --episodes 2000 --dataset synthetic

# PaintControl
python -m src.train --episodes 1500 --dataset paintcontrol

# IOF/ROL
python -m src.train --episodes 1500 --dataset iofrol
```

### 5. Evaluate separately

```powershell
python -m src.evaluate --dataset synthetic --num_test_builds 120
```

Check each command's `--help` output if options differ in your checked-out version:

```powershell
python -m src.train --help
python -m src.evaluate --help
python -m src.run_experiments --help
```

## Key Findings and Limitations

1. **Results vary by dataset.** Q-learning is competitive on synthetic data, leads on the reported IOF/ROL BDR/APFD/bugs-per-cost metrics, and underperforms Greedy Coverage/Cost on PaintControl BDR and APFD.
2. **Coverage and fault detection are not interchangeable.** A strategy can achieve high coverage without detecting faults as early or as frequently as another strategy.
3. **The synthetic BDR margin is small.** Q-learning's mean BDR is 0.8044 compared with 0.8018 for Greedy Failure/Cost; avoid presenting this as a decisive win without statistical testing.
4. **More robust statistical analysis is needed.** Where policies are evaluated on shared builds or seeds, use paired comparisons and report uncertainty rather than relying only on aggregate means.
5. **PaintControl needs further investigation.** Inspect the data split, observable features, reward, evaluation procedure, and generalization before attributing the result to a specific cause.

## Datasets

The project uses the ABB Paint Control and IOF/ROL continuous-integration datasets from the RETECS work.

- **Source repository:** [RETECS](https://github.com/mregorova/RETECS)
- **Paper:** Spieker, H., Gotlieb, A., Marijan, D., and Mossige, M. (2017). *Reinforcement learning for automatic test case prioritization and selection in continuous integration.* Proceedings of the 26th ACM SIGSOFT International Symposium on Software Testing and Analysis, pp. 12–22.

The README's earlier dataset description reports 352 raw PaintControl cycles and 320 raw IOF/ROL cycles. The real-world summary files report 207 and 189 evaluations respectively; these are evaluation counts, not raw dataset sizes.

### BibTeX

```bibtex
@inproceedings{spieker2017reinforcement,
  title={Reinforcement learning for automatic test case prioritization and selection in continuous integration},
  author={Spieker, Helge and Gotlieb, Arnaud and Marijan, Dusica and Mossige, Morten},
  booktitle={Proceedings of the 26th ACM SIGSOFT International Symposium on Software Testing and Analysis},
  pages={12--22},
  year={2017}
}
```

## Results and Reproducibility

- Summary tables: `results/summary.csv`, `results/real_world_paintcontrol_summary.csv`, and `results/real_world_iofrol_summary.csv`
- Per-evaluation outputs: `results/*_raw.csv`
- Training history: `results/training_history.csv` and `results/synthetic_training_history.csv`
- Ablation and sensitivity outputs: `results/agent_ablation.csv`, `results/reward_ablation.csv`, and `results/budget_sensitivity.csv`
- Figures: `results/figures/`

For a reproducible comparison, keep the code version, dataset version, seeds, command-line options, and generated outputs together. Re-running the experiment runner may overwrite result files, so back up outputs you want to preserve.

## References

- Spieker et al. (2017). *Reinforcement learning for automatic test case prioritization and selection in continuous integration.* ISSTA 2017.
- Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
- Schaul et al. (2016). *Prioritized experience replay.* ICLR 2016.

## License

MIT License. See [LICENSE](LICENSE).
