# torchqdynamics
Quantum systems simulation with PyTorch.

This library provides differentiable solvers for the Schrödinger Equation, the Lindblad Master Equation and the Stochastic Master Equation. All the solvers are implemented using PyTorch and can run on GPUs.

## Installation
Clone the repository, install the dependencies and install the repository in editable mode in any Python virtual environment:
```shell
# pip
pip install -e /path/to/torchqdynamics

# conda
conda develop /path/to/torchqdynamics
```

## Usage
:construction: WIP

## Performance
:construction: WIP

## Contribute
Install developer dependencies:
```shell
pip install isort yapf toml
```

Run the following before each commit:
```shell
isort torchqdynamics
yapf -i -r torchqdynamics
```

Alternatively you can use `pre-commit` to run theses automatically before commit:
```shell
pip install pre-commit
pre-commit install
```
