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
                 | Normalization       |
                 | sweep_subtraction.py|
                 +----------+----------+
                            |
                            |
                            v
                +-----------+------------+   
                | Rank Check             |  
                | tgevp_EnergyVariance.py|   
                +-----------+------------+   
                            | 
                            |  
                            v  
                 +----------+-----------+
                 | Spectroscopy         |
                 | tgevp_Spectroscopy.py|
                 +----------+-----------+
                            |
                            |
                            v
                 +----------+---------------------+
                 | Eigenvalue                     |
                 | tgevp_EigenvalueExt.py.        |
                 | or                             |
                 | tgevp_EigenvalueExtNoiseless.py|
                 +----------+---------------------+
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
git clone https://github.com/yourname/UnbiasedKrylovSubspaceMethod.git
cd UnbiasedKrylovSubspaceMethod
```

Install the required Python packages

```bash
pip install numpy scipy matplotlib pyyaml
```

---

# Configuration

All parameters are specified in a YAML file.

Example:
- Temporal length = 64
- Configuration number = 500
- Bootstrap samples = 500
- Use data between t=0~21
- Low-rank approximation with rank=1,2,3
- Extrapolation with rank=2 and 3

```yaml
lattice:
  T: 64

statistics:
  stati: 1
  configuration: 500
  bootstrap: 500

analysis:
  normalization: 1
  init: 0
  size: 21
  dof: 22
  fitini: 3
  fitfin: 6
  svdrankmax: 4

extrapolation:
  rankmax: 3
  rankmin: 2
  msize: 11

mock:
  state: 4
  error: 1e-15

paths:
  corr: correlator_data
  diag: diagonalized_data

files:
  rdata: raw_data/cdata_jk
  cdata: cdata
  ndata: ndata
```

---

### `lattice`

| Parameter | Description              |
| --------- | ------------------------ |
| `T`       | Temporal lattice extent. |

---

### `statistics`

| Parameter       | Description                                           |
| --------------- | ----------------------------------------------------- |
| `stati`         | Statistical analysis mode (Fixed to `1` = bootstrap). |
| `configuration` | Number of gauge configurations.                       |
| `bootstrap`     | Number of bootstrap samples.                          |

---

### `analysis`

| Parameter       | Description                                                              |
| --------------- | ------------------------------------------------------------------------ |
| `normalization` | Time slice used for correlator normalization.                            |
| `init`          | Initial time slice of the correlator (Fixed to 0).                       |
| `size`          | Final time slice of the correlator.                                      |
| `dof`           | Number of degrees of freedom (size+1).                                   |
| `fitini`        | First time slice included in the fit.                                    |
| `fitfin`        | Last time slice included in the fit.                                     |
| `svdrankmax`    | Maximum SVD rank used for truncated-SVD analyses.                        |

---

### `extrapolation`

| Parameter | Description                                       |
| --------- | ------------------------------------------------- |
| `rankmin` | Minimum rank included in the extrapolation.       |
| `rankmax` | Maximum rank included in the extrapolation.       |
| `msize`   | Number of matrix sizes used in the extrapolation. |

---

### `mock`

| Parameter | Description                                               |
| --------- | --------------------------------------------------------- |
| `state`   | Number of states used when generating mock correlators.   |
| `error`   | Statistical uncertainty assigned to the mock correlators. |

---

### `paths`

| Parameter | Description                                                      |
| --------- | ---------------------------------------------------------------- |
| `corr`    | Directory containing correlator data.                            |
| `diag`    | Directory for diagonalized correlators and spectroscopy results. |

---

### `files`

| Parameter | Description                                         |
| --------- | --------------------------------------------------- |
| `rdata`   | Input raw correlator data.                          |
| `cdata`   | Name of the correlator dataset after preprocessing. |
| `ndata`   | Name of the normalized correlator dataset.          |



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

