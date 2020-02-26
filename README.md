<img align="left" src="https://github.com/do-mpc/do-mpc/blob/master/documentation/source/static/dompc_var_02_rtd_blue.svg">

# do-mpc: Robust optimal control toolbox

[![Documentation Status](https://readthedocs.org/projects/do-mpc/badge/?version=latest)](https://do-mpc.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/do-mpc/do-mpc.svg?branch=master)](https://travis-ci.org/do-mpc/do-mpc)

**do-mpc** proposes a new, modularized implementation for optimization based model predictive control (MPC) and moving horizon estimation (MHE). **do-mpc** enables the efficient formulation and solution of control and estimation problems for nonlinear systems, including tools to deal with uncertainty and time discretization. The modular structure of do-mpc contains simulation, estimation and control components that can be easily extended and combined to fit many different applications.

In summary, **do-mpc** offers the following features:

* nonlinear and economic model predictive control
* robust multi-stage model predictive control
* moving horizon state and parameter estimation
* modular design that can be easily extended

The **do-mpc** software is Python based and works therefore on any OS with a Python 3.x distribution. **do-mpc** has been developed at the DYN chair of the TU Dortmund by Sergio Lucia and Alexandru Tatulea. The development is continued at the IOT chair of the TU Berlin by Felix Fiedler and Sergio Lucia.

## Installation instructions
Installation instructions are given [here](https://do-mpc.readthedocs.io/en/latest/installation.html).

## Documentation
Please visit our extensive [documentation](https://do-mpc.readthedocs.io/en/latest/index.html), kindly hosted on readthedocs.

## Citing **do-mpc**
If you use **do-mpc** for published work please cite it as:

S. Lucia, A. Tatulea-Codrean, C. Schoppmeyer, and S. Engell. Rapid development of modular and sustainable nonlinear model predictive control solutions. Control Engineering Practice, 60:51-62, 2017

Please remember to properly cite other software that you might be using too if you use **do-mpc** (e.g. CasADi, IPOPT, ...)
