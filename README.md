# tt2DTM

[![License](https://img.shields.io/pypi/l/tt2DTM.svg?color=green)](https://github.com/jdickerson95/tt2DTM/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tt2DTM.svg?color=green)](https://pypi.org/project/tt2DTM)
[![Python Version](https://img.shields.io/pypi/pyversions/tt2DTM.svg?color=green)](https://python.org)
[![CI](https://github.com/jdickerson95/tt2DTM/actions/workflows/ci.yml/badge.svg)](https://github.com/jdickerson95/tt2DTM/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jdickerson95/tt2DTM/branch/main/graph/badge.svg)](https://codecov.io/gh/jdickerson95/tt2DTM)

Two-dimensional template-matching implemented in Python.

## Installation

The newest released version of the package can be installed from PyPI using pip:

```bash
pip install tt2DTM
```

Or you can alternatively install from source:

```bash
git clone https://github.com/jdickerson95/tt2DTM.git
cd tt2DTM
pip install .
```

### For Developers

To install the package with the necessary development dependencies in an editable configuration, run:

```bash
git clone https://github.com/jdickerson95/tt2DTM.git
cd tt2DTM
pip install -e '.[dev,test]'
```

## Usage

Inputs to the template matching programs are contained within Pydantic model objects which run validation on the input data. These inputs can be set in a Python script like below:

*TODO*: Add example

Alternatively, configurations can be set in a YAML file and loaded into the `MatchTemplateManager` object. See the notebook `examples/01-config_import_export.ipynb` further information on configuration fields and import/export functionality.

```python
from tt2dtm.pydantic_models import MatchTemplateManager

YAML_CONFIG_PATH = "path/to/config.yaml"
ORIENTATION_BATCH_SIZE = 8

def main():
    mtm = MatchTemplateManager.from_yaml(YAML_CONFIG_PATH)
    mtm.run_match_template(ORIENTATION_BATCH_SIZE)

# NOTE: invoking from `if __name__ == "__main__"` is necessary
# for proper multiprocessing/GPU-distribution behavior
if __name__ == "__main__":
    main()
```
