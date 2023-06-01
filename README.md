# DiffNet
DiffNet: A FEM based neural PDE solver package

# Instructions
After cloning the repo, do:
```sh
cd DiffNet
python setup.py develop --user
```

# Examples
Generalizable Poisson Solver:

Download a dataset of randomly generated shapes at https://iastate.box.com/s/u7pbj2eby4ckr23eyx86oksksz8masbe. One dataset used in \emph{Neural PDE Solvers for Irregular Domains}, (https://arxiv.org/abs/2211.03241, Fig. 3a).

```sh
cd DiffNet/examples/poisson/parametric
python IBN_2D.py
```