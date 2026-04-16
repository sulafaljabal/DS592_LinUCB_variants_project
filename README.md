# Weighted Linear Bandits for Non-Stationary Environments
## DS 592 Project — Jelle Hendriks & Sulaf Al Jabal

Reproducing and extending the results from Russac et al. (NeurIPS 2019):
"Weighted Linear Bandits for Non-Stationary Environments"

### Project Structure

```
project/
├── algorithms/
│   ├── __init__.py
│   ├── base.py            # Abstract base class for all algorithms
│   ├── linucb.py          # Standard LinUCB (stationary baseline)
│   ├── d_linucb.py        # Discounted LinUCB (Russac et al. 2019)
│   ├── sw_linucb.py       # Sliding Window LinUCB (Cheung et al. 2019)
│   └── d_linucb_dynamic.py # dLinUCB variant from Cheung et al.
├── environments/
│   ├── __init__.py
│   ├── base.py            # Abstract environment class
│   ├── stationary.py      # Stationary linear bandit (sanity check)
│   └── nonstationary.py   # Non-stationary with pluggable drift functions
├── drift_functions/
│   ├── __init__.py
│   ├── base.py            # Abstract drift function interface
│   ├── abrupt.py          # Sharp breakpoint drift (from the paper)
│   ├── linear.py          # Linear drift (monotone increase/decrease)
│   ├── sinusoidal.py      # Sinusoidal / periodic drift
│   └── piecewise.py       # Piecewise linear drift
├── experiments/
│   ├── reproduce_paper.py # Reproduce Figure 1 from Russac et al.
│   ├── smooth_drift.py    # Run experiments with smooth drift functions
│   └── sensitivity.py     # Hyperparameter sensitivity (gamma, tau, etc.)
├── utils/
│   ├── __init__.py
│   ├── plotting.py        # Shared plotting utilities
│   └── metrics.py         # Regret computation, averaging, etc.
├── plots/                 # Output directory for generated figures
├── config.py              # Shared experiment parameters
├── run_all.py             # Master script to run all experiments
└── README.md
```

### Algorithms
- **LinUCB**: Standard optimistic linear bandit (Ch. 19 of Lattimore & Szepesvári)
- **D-LinUCB**: Discounted LinUCB with exponential weights γ^{-t}
- **SW-LinUCB**: Sliding window LinUCB with window size τ
- **dLinUCB**: Dynamic LinUCB from Cheung et al. (2019)

### References
- Russac, Vernade, Cappé. "Weighted Linear Bandits for Non-Stationary Environments" (NeurIPS 2019)
- Cheung, Simchi-Levi, Zhu. "Learning to Optimize under Non-Stationarity" (AISTATS 2019)
- Lattimore & Szepesvári. "Bandit Algorithms" — Chapters 19 and 31
