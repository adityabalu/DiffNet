# DiffNet
DiffNet: A FEM based neural PDE solver package

# Instructions
After cloning the repo, do:
```sh
cd DiffNet
python setup.py develop --user
```

# IBN Examples
Codes for the examples used in the [IBN paper](https://arxiv.org/pdf/2211.03241.pdf) can be found in the `IBN` directory, under the respective sub-directory:
```
IBN
  datasets
  error-analysis
  poisson-2d
    parametric
  poisson-3d
    non-parametric
    parametric
```
The used datasets can be downloaded from [this CyBox location](https://iastate.box.com/s/u7pbj2eby4ckr23eyx86oksksz8masbe).

[//]: <> (One dataset used in \emph{Neural PDE Solvers for Irregular Domains}, (https://arxiv.org/abs/2211.03241, Fig. 3a).)

```sh
cd DiffNet/examples/poisson/parametric
python IBN_2D.py
```
