# Weighted Linear Bandits for Non-Stationary Environments
## DS 592 Project — Jelle Hendriks & Sulaf Al Jabal

Reproducing and extending the results from Russac et al. (NeurIPS 2019):
"Weighted Linear Bandits for Non-Stationary Environments"

---

### Project Structure

```
project/
├── algorithms/
│   ├── __init__.py
│   ├── base.py              # Abstract base class for all algorithms
│   ├── linucb.py            # Standard LinUCB (stationary baseline)
│   ├── d_linucb.py          # Discounted LinUCB (Russac et al. 2019)
│   └── sw_linucb.py         # Sliding Window LinUCB (Cheung et al. 2019)
├── environments/
│   ├── __init__.py
│   └── nonstationary.py     # Non-stationary env with pluggable drift + simulation engine
├── drift_functions/
│   ├── __init__.py
│   ├── base.py              # Abstract drift function interface
│   ├── abrupt.py            # Sharp breakpoint drift (from the paper)
│   ├── linear.py            # Linear drift (monotone increase/decrease)
│   ├── sinusoidal.py        # Sinusoidal / periodic drift
│   └── piecewise.py         # Piecewise linear drift
├── experiments/
│   ├── reproduce_paper.py   # Reproduce Figure 1 from Russac et al.
│   ├── smooth_drift.py      # Run experiments with smooth drift functions
│   └── sensitivity.py       # Hyperparameter sensitivity (gamma, tau, etc.)
├── utils/
│   ├── __init__.py
│   └── plotting.py          # Shared plotting utilities
├── plots/                   # Output directory for generated figures
├── config.py                # Shared parameters + theoretical γ/τ helpers
├── run_all.py               # Master script to run all experiments
├── .gitignore
└── README.md
```

---

### Algorithms

- **LinUCB** — Standard optimistic linear bandit (Ch. 19 of Lattimore & Szepesvári). Uses all historical data equally. Serves as a baseline that cannot adapt to non-stationarity.
- **D-LinUCB** — Discounted LinUCB with exponential weights (Russac et al. 2019). Uses a discount factor γ to exponentially downweight old observations, allowing it to track a drifting θ*.
- **SW-LinUCB** — Sliding Window LinUCB (Cheung et al. 2019). Only uses the last τ observations, providing a hard cutoff instead of smooth discounting.

---

### How to Run

Run all experiments from the project root:
```bash
python run_all.py
```

Or run individual experiments:
```bash
python -m experiments.reproduce_paper
python -m experiments.smooth_drift
python -m experiments.sensitivity
```

Plots are saved to `plots/`.

---

### Implementation Details & Changes from Initial Scaffold

This codebase was scaffolded and then iteratively refined to match the author's
reference implementation ([YRussac/WeightedLinearBandits](https://github.com/YRussac/WeightedLinearBandits))
as closely as possible. The following sections document each change and why it matters.

#### 1. Exact β_t from Theorem 2 (D-LinUCB)

The confidence width β_t controls how much the algorithm explores. The initial
scaffold used a generic log-determinant formula from stationary LinUCB analysis:

```
β = σ * sqrt(2*log(1/δ) + log(det(V)/λ^d)) + sqrt(λ)*S
```

This was replaced with the **exact expression from Theorem 2** of the paper:

```
β_t = sqrt(λ)*S + σ * sqrt(2*log(1/δ) + d*log(1 + (1 - γ^{2t}) / (d*λ*(1 - γ²))))
```

The key difference is the `d*log(...)` term, which captures the **effective dimension
under discounting**. As t grows, γ^{2t} → 0 and the term stabilizes — meaning
the confidence width doesn't grow unboundedly like in the stationary case.
The algorithm also tracks `gamma2_t = γ^{2t}` multiplicatively each step,
exactly as the author does.

**LinUCB** was similarly updated to use the author's closed-form β_t:
```
β_t = sqrt(λ)*S + σ * sqrt(2*log(1/δ) + d*log(1 + t/(λ*d)))
```

#### 2. The V_sq Sandwich Matrix (D-LinUCB)

The initial scaffold only tracked one design matrix V. The author's implementation
tracks **two**:

- `V` — discounted by γ per step: `V = γ*V + a*a^T + (1-γ)*λ*I`
- `V_sq` — discounted by γ² per step: `V_sq = γ²*V_sq + a*a^T + (1-γ²)*λ*I`

The UCB bonus is then computed using the **sandwich form**:
```
bonus = β_t * sqrt(a^T  V^{-1}  V_sq  V^{-1}  a)
```

instead of just `sqrt(a^T V^{-1} a)`. This comes from the variance of the
weighted least-squares estimator: the weights appear squared in the variance
(hence γ² in V_sq), while the estimator itself uses γ (hence V). Using only
V^{-1} underestimates the variance and leads to insufficient exploration.

#### 3. Sherman-Morrison Matching the Author

The author's code handles matrix inversion differently per algorithm:

- **LinUCB**: Supports Sherman-Morrison (SM) via a `sm` flag. When `sm=True`,
  V^{-1} is updated incrementally using the rank-1 formula. When `sm=False`,
  `pinv(V)` is recomputed from scratch.
- **D-LinUCB**: SM is **disabled** (`sm=False`). The discount step `V = γ*V + ...`
  means it's not a simple rank-1 update of the previous V, so SM doesn't apply.
  Uses `pinv(V)` each step.
- **SW-LinUCB**: SM is also not applicable because dropping the oldest observation
  from the window is not a rank-1 downdate in general. Rebuilds V from the
  window and uses `pinv(V)` each step.

The scaffold was updated to match this exactly. LinUCB defaults to `sm=True`
and uses the author's column-vector formulation:
```python
a_col = a[:, np.newaxis]
const = 1.0 / (1.0 + a @ V_inv @ a)
const2 = V_inv @ a_col
V_inv = V_inv - const * (const2 @ const2.T)
```

#### 4. Theoretically Optimal γ and τ (Corollary 1)

The initial scaffold hardcoded `γ = 0.99` and `τ = 200`. The paper's
**Corollary 1** gives the theoretically optimal values as functions of the
total variation B_T:

```
γ* = 1 - (B_T / (d * T))^{2/3}
τ* = (d * T / B_T)^{2/3}
```

where B_T = Σ ‖θ*_{t+1} - θ*_t‖ is the total drift over the horizon.

`config.py` now provides `compute_optimal_gamma(B_T, d, T)` and
`compute_optimal_tau(B_T, d, T)` helper functions. The experiment scripts
compute B_T from the drift function and use these helpers to set γ and τ
automatically. This means each drift type gets its own theoretically tuned
parameters:

| Drift Type       |   B_T |       γ* |  τ* |
|------------------|------:|---------:|----:|
| Abrupt (paper)   |  4.24 | 0.99500  | 199 |
| Slow Linear      |  0.71 | 0.99849  | 660 |
| Fast Linear      |  2.00 | 0.99697  | 330 |
| Slow Sinusoidal  |  3.14 | 0.99591  | 244 |
| Fast Sinusoidal  | 18.85 | 0.98649  |  74 |

Note: these formulas assume B_T is **known in advance**. In practice this is
unrealistic — the Bandits-over-Bandits (BOB) framework from Cheung et al.
addresses this limitation. This is discussed in our report's Open Problems section.

#### 5. Consistent Tie-Breaking

All three algorithms now use the author's `lexsort` approach for breaking ties
in UCB values, instead of the `np.isclose` + `random.choice` pattern from the
initial scaffold. This adds a random jitter and sorts by (jitter, UCB) to
ensure ties are broken uniformly at random without floating-point edge cases.

#### 6. Alpha Exploration Multiplier

All three algorithms now accept an `alpha` parameter that multiplies the
exploration bonus: `bonus = alpha * beta_t * sqrt(...)`. Setting `alpha=1.0`
(default) gives the theoretical value. In practice, the theoretical β_t is
often conservative, and tuning alpha < 1 can improve empirical performance.
The author's code includes the same parameter.

---

### Differences from the Author's Repo

Our scaffold differs from [YRussac/WeightedLinearBandits](https://github.com/YRussac/WeightedLinearBandits) in structure, not substance:

| Aspect | Author's Repo | Our Scaffold |
|--------|--------------|--------------|
| Layout | Flat (all .py in root) | Packages (algorithms/, environments/, etc.) |
| Actions | `Arm` class with `.features` | Raw NumPy arrays `(n_actions, d)` |
| Drift | Hardcoded in notebooks | Pluggable `DriftFunction` classes |
| Experiments | Jupyter notebooks | Standalone Python scripts |
| Drift types | Abrupt + slowly-varying | Abrupt, linear, sinusoidal, piecewise |
| Algorithms | LinUCB, D-LinUCB, SW-LinUCB, dLinUCB, Random | LinUCB, D-LinUCB, SW-LinUCB |

The core math (β_t, V/V_sq updates, Sherman-Morrison) matches the author's
implementation.

---

### References

- Russac, Vernade, Cappé. "Weighted Linear Bandits for Non-Stationary Environments" (NeurIPS 2019). [arXiv:1909.09146](https://arxiv.org/abs/1909.09146)
- Cheung, Simchi-Levi, Zhu. "Learning to Optimize under Non-Stationarity" (AISTATS 2019). [Paper](http://proceedings.mlr.press/v89/cheung19b/cheung19b.pdf)
- Lattimore & Szepesvári. "Bandit Algorithms" — Chapters 19 (LinUCB) and 31 (Non-stationary bandits). [Free PDF](https://tor-lattimore.com/downloads/book/book.pdf)
- Author's code: [github.com/YRussac/WeightedLinearBandits](https://github.com/YRussac/WeightedLinearBandits)
