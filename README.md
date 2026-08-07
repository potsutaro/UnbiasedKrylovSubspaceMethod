# Lattice Spectroscopy with Unbiased Krylov subspace method

A Python framework for spectroscopy analysis of lattice correlation functions based on the Unbiased Krylov subspace method (DOI: 10.1103/9891-x33t).

The package provides an end-to-end workflow from raw correlation functions to spectroscopy observables.

---

# Workflow

```text
                   +----------------+
                   |  Raw data      |
                   | (correlators)  |
                   +--------+-------+
                            |
                            |
                            v
                 +---------------------+
                 | Bootstrap Sampling  |
                 | data_bootstrap.py   |
                 +----------+----------+
                            |
                            |
                            v
                 +---------------------+
                 | Correlator          |
                 | Subtraction         |
                 | sweep_subtraction.py|
                 +----------+----------+
                            |
                 +----------+----------+
                 |                     |
                 |                     |
                 v                     v
      +------------------+   +----------------------+
      | Rank Check       |   | Spectroscopy         |
      | EnergyVariance   |   | TGEVP                |
      +--------+---------+   +----------+-----------+
               |                        |
               |                        |
               +------------+-----------+
                            |
                            v
                 +----------------------+
                 | Eigenvalue           |
                 | Extrapolation        |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Final observables    |
                 +----------------------+
```

For validation studies, mock correlators can also be generated before the bootstrap step.

```text
Mock correlators
      |
      v
Bootstrap
      |
      v
Subtraction
      |
      v
TGEVP analysis
```

---

# Directory Structure

```text
project/
│
├── run.py                  # Main entry point
├── config.py               # Configuration classes
│
├── configs/
│   ├── default.yaml
│   ├── mock.yaml
│   └── production.yaml
│
├── analysis/
│   ├── gen_mock_ct.py
│   ├── data_bootstrap.py
│   ├── sweep_subtraction.py
│   ├── tgevp_Spectroscopy.py
│   ├── tgevp_EnergyVariance.py
│   ├── tgevp_EigenvalueExt.py
│   └── ...
│
├── raw_data/
├── correlator_data/
└── diagonalized_data/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourname/LatticeSpectroscopy.git
cd LatticeSpectroscopy
```

Install the required Python packages

```bash
pip install numpy scipy matplotlib pyyaml
```

---

# Configuration

All parameters are specified in a YAML file.

Example:

```yaml
lattice:
  T: 64

statistics:
  stati: 1
  configuration: 500
  bootstrap: 500

analysis:
  size: 21
  dof: 22
  fitini: 3
  fitfin: 6
```

This design allows the entire analysis to be reproduced simply by sharing the configuration file.

---

# Running

General syntax

```bash
python run.py <config.yaml> --task <task>
```

Examples

Generate mock data

```bash
python run.py configs/mock.yaml --task mock
```

Analyze existing data

```bash
python run.py configs/default.yaml --task data
```

Perform spectroscopy

```bash
python run.py configs/default.yaml --task Spectroscopy
```

---

# Available Tasks

| Task             | Description                                         |
| ---------------- | --------------------------------------------------- |
| `mock`           | Generate mock correlators and perform preprocessing |
| `data`           | Bootstrap and subtraction                           |
| `RankCheck`      | Estimate TGEVP variances                            |
| `Spectroscopy`   | Perform spectroscopy analysis                       |
| `EvExt`          | Eigenvalue extrapolation                            |
| `EvExtNoiseless` | Eigenvalue extrapolation without statistical noise  |

---

# Input

The input and output directories are specified in the YAML configuration file.

Typical inputs include

* Raw correlators
* Bootstrap samples
* Correlator matrices

---

# Output

The workflow produces

* Bootstrap correlators
* Subtracted correlators
* Diagonalized correlators
* Spectroscopy observables
* Eigenvalue extrapolation

---

# Extending the Framework

Each analysis module should implement

```python
def main(cfg):
    ...
```

To add a new analysis,

1. Create a new module under `analysis/`.
2. Implement `main(cfg)`.
3. Import the module in `run.py`.
4. Add a new task in the task dispatcher.

---

# Citation

If this software contributes to published work, please cite the corresponding publication.

---

# License

MIT License

