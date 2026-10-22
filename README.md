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
│   │   └── risk.py          # Generalizer, KAnonymityChecker, RiskReport
│   ├── generation/          # Week 2: synthetic data generators
│   └── evaluation/          # Week 3: utility and risk metrics
├── scripts/
│   ├── make_sample_data.py  # creates a small fake Adult-format file
│   └── week1_demo.py        # risk check before/after generalisation
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

Run the demo and the tests:

```bash
python scripts/week1_demo.py
pytest
```

## Progress

- [x] Week 1 — data classes, k-anonymity risk checker, tests
- [ ] Week 2 — Laplace/Gaussian mechanisms, privacy budget, generators
- [ ] Week 3 — utility and risk evaluation, experiments
- [ ] Week 4 — Streamlit dashboard, report

## References

[1] D. Dua and C. Graff, "UCI Machine Learning Repository," University of California, Irvine, 2019.
