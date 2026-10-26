# PrivSynth

An object-oriented Python toolkit that generates privacy-protected synthetic data
and evaluates the balance between privacy and usefulness.

CSC8101 Object-Oriented Programming project — Christine Nakyejjwe (MSc Data Science and Analytics, Uganda Christian University).

## Project structure

```
privsynth/
├── privsynth/
│   ├── exceptions.py        # PrivSynthError and its subclasses
│   ├── data/
│   │   ├── dataset.py       # Dataset: a table + its privacy schema
│   │   └── loaders.py       # DataLoader (abstract), CSVLoader, AdultDataLoader
│   ├── privacy/
│   │   ├── risk.py          # Generalizer, KAnonymityChecker, RiskReport
│   │   ├── budget.py        # PrivacyBudget, BudgetEntry
│   │   └── mechanisms.py    # Laplace, Gaussian, Exponential mechanisms
│   ├── generation/
│   │   ├── encoding.py      # TableEncoder: values <-> integer codes
│   │   ├── base.py          # Synthesizer (abstract)
│   │   ├── marginal.py      # Generator 1: IndependentMarginalSynthesizer
│   │   └── bayesian.py      # Generator 2: BayesianNetworkSynthesizer
│   └── evaluation/          # Week 3: utility and risk metrics
├── scripts/
│   ├── make_sample_data.py  # creates a small fake Adult-format file
│   ├── week1_demo.py        # risk check before/after generalisation
│   └── week2_demo.py        # mechanisms, budget, generator comparison
├── tests/                   # pytest test suite
├── data/                    # put adult.data here (not committed)
├── pyproject.toml
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Data

Experiments use the UCI Adult (Census Income) dataset [1]. Download
`adult.data` from https://archive.ics.uci.edu/dataset/2/adult and place it in
`data/`. Until then, `python scripts/make_sample_data.py` creates
`data/sample_adult.csv`, a randomly generated file in the same format
(no real people).

## Usage

```python
from privsynth import AdultDataLoader, Generalizer, KAnonymityChecker

dataset = AdultDataLoader().load("data/adult.data")
checker = KAnonymityChecker(k=5)
print(checker.evaluate(dataset))

coarser = Generalizer(bin_widths={"age": 10}, rare_threshold=50).generalize(dataset)
print(checker.evaluate(coarser))
```

Generate differentially private synthetic data:

```python
from privsynth import AdultDataLoader, BayesianNetworkSynthesizer

dataset = AdultDataLoader().load("data/adult.data")
synth = BayesianNetworkSynthesizer(epsilon=1.0, bin_edges=AdultDataLoader.BIN_EDGES, random_state=0)
synthetic = synth.fit_sample(dataset, n=5000)

print(synth.network)              # which column depends on which
print(synth.budget.summary())     # every epsilon spent, step by step
synthetic.data.to_csv("synthetic_adult.csv", index=False)
```

Swap in `IndependentMarginalSynthesizer` for the baseline, or pass
`noise="gaussian"` for Gaussian instead of Laplace noise.

Run the demos and the tests:

```bash
python scripts/week1_demo.py
python scripts/week2_demo.py      # about 40 s on the full Adult data
pytest
```

## Progress

- [x] Week 1 — data classes, k-anonymity risk checker, tests
- [x] Week 2 — Laplace/Gaussian/Exponential mechanisms, privacy budget, two generators
- [ ] Week 3 — utility and risk evaluation, experiments
- [ ] Week 4 — Streamlit dashboard, report

## References

[1] D. Dua and C. Graff, "UCI Machine Learning Repository," University of California, Irvine, 2019.
[2] C. Dwork and A. Roth, "The algorithmic foundations of differential privacy," *Found. Trends Theor. Comput. Sci.*, vol. 9, no. 3–4, pp. 211–407, 2014.
[3] B. Balle and Y.-X. Wang, "Improving the Gaussian mechanism for differential privacy: Analytical calibration and optimal denoising," in *Proc. ICML*, 2018, pp. 394–403.
[4] J. Zhang, G. Cormode, C. M. Procopiuc, D. Srivastava, and X. Xiao, "PrivBayes: Private data release via Bayesian networks," *ACM Trans. Database Syst.*, vol. 42, no. 4, pp. 1–41, 2017.
