# Ritz neural-network solvers for Allen–Cahn and semilinear diffusion–reaction problems

Python implementations of **structure-exploiting second-order (Newton-type) training
methods for shallow ReLU networks** used as Ritz/Galerkin approximations of

1. the **semilinear diffusion–reaction equation** (steady state, cubic reaction), and
2. the **Allen–Cahn equation** (time dependent, energy-based time stepping),

in one space dimension.

Instead of training the network with a generic optimizer (SGD, Adam, …), each solver
minimizes a discrete *energy functional* and exploits the special structure of a
one-hidden-layer ReLU network in 1D:

* the **linear parameters** (output weights $c$) are updated by Newton's method whose
  Hessian is tridiagonal in the hat-function basis, so every solve costs $\mathcal O(N)$;
* the **nonlinear parameters** (breakpoints $b$) are updated by a Newton step whose
  Hessian is (up to diagonal scalings) the inverse of a tridiagonal matrix, again solved
  in $\mathcal O(N)$ with the Thomas algorithm (plus Sherman–Morrison for boundary penalties);
* **dead or colliding neurons** are detected and re-inserted at dyadic points of the
  current mesh, so the network never loses expressive power.

---

## Repository layout

```
.
├── AllenCahn/
│   ├── NN/                      # Neural-network (Ritz) time-stepping solvers
│   │   ├── LS.py                # L2 (least-squares) fit of the initial datum
│   │   ├── AC_CSS_DB.py         # Convex-splitting scheme, Dirichlet BCs
│   │   ├── AC_CSS_NB.py         # Convex-splitting scheme, Dirichlet/Neumann BCs
│   │   ├── AC_BDF2_DB.py        # Stabilized IMEX-BDF2, Dirichlet BCs
│   │   ├── AC_BDF2_NB.py        # Stabilized IMEX-BDF2, Dirichlet/Neumann BCs
│   │   ├── test1.py             # Test 1: travelling front (exact solution known)
│   │   └── test2.py             # Test 2: two interfaces + perturbation (spectral reference)
│   ├── SpectralMethods/
│   │   └── IMEX_BDF2_Periodic.py  # Fourier pseudo-spectral reference solver
│   └── README.md                # Details: equations, schemes, API, parameters
├── SemilinearDR/
│   ├── DR3_D.py                 # Steady problem, Dirichlet BCs (penalty)
│   ├── DR3_N.py                 # Steady problem, Dirichlet + Neumann BCs
│   ├── mainDR3.py               # Driver script
│   └── README.md                # Details: equations, manufactured solutions, API
├── requirements.txt
└── README.md
```

Detailed documentation for each part:

* [`AllenCahn/README.md`](AllenCahn/README.md)
* [`SemilinearDR/README.md`](SemilinearDR/README.md)

---

## The network

All solvers use the same one-hidden-layer ReLU network on an interval $[a,b]$:

$$
u_n(x;\,\theta) \;=\; \alpha \;+\; \sum_{i=0}^{N} c_i\,\sigma(x-b_i),
\qquad \sigma(z) = \max(0,z),
$$

with $a = b_0 < b_1 < \dots < b_N < b$. In code, the parameters are stored in an
array `theta` of shape `(N+1, 2)`:

| column | meaning |
|---|---|
| `theta[:, 0]` | breakpoints $b_0,\dots,b_N$ (nonlinear parameters; $b_0=a$ is fixed) |
| `theta[:, 1]` | output weights $c_0,\dots,c_N$ (linear parameters) |

The bias $\alpha = u(a)$ imposes the Dirichlet condition at $x=a$ exactly. $u_n$ is a
continuous piecewise-linear function whose slope on $[b_j, b_{j+1}]$ is
$d_j = \sum_{i\le j} c_i$ (`derivatives_vector`).

## The algorithm (shared by all solvers)

Given an energy functional $J(u)$ (see the sub-READMEs for each problem), every solver
minimizes $J(u_n(\cdot;\theta))$ by alternating

1. **Linear step** – solve for $c$ with $b$ fixed (`gradC`, `solveC`). The Hessian
   with respect to $c$ is assembled in the nodal (hat-function) basis, where it is a
   tridiagonal *stiffness + weighted mass* matrix (weights computed with Simpson /
   Simpson 3/8 / Boole quadrature on each cell), and mapped to the ReLU basis with the
   bidiagonal difference matrix $G$ (`G_matrix`) and $\operatorname{diag}(1/h)$.
   A boundary penalty $\gamma\,(u_n(b)-\beta)^2/2$, when present, adds a rank-one
   term handled with the Sherman–Morrison formula.

2. **Nonlinear step** – Newton update of the breakpoints (`gradB`, `hessianSolver`).
   Writing $\partial J/\partial b_i = c_i\, g_i$ (`gradB` returns $g$), the
   $b$-block of the Hessian has the form
   $$H_{bb} = D\,S\,D + D\,\Lambda,\qquad D=\operatorname{diag}(c),$$
   where $S_{ij} = \int_{\max(b_i,b_j)}^{b} w(x)\,dx$ is a weighted Gram matrix of
   Heaviside functions (its inverse is tridiagonal, `coefficientMatrixInv`) and
   $\Lambda$ is diagonal (the pointwise residual of the reaction term at each
   breakpoint). The step $p=-(SD+\Lambda)^{-1}g$ is therefore computed in
   $\mathcal O(N)$.

3. **Projection / redistribution** (`redistribute_neurons`) – breakpoints are sorted,
   clipped to $[a,b]$, and neurons that are *dead* (|c_i| < `tau_1`), collapsed onto
   another breakpoint, or outside the domain are moved to dyadic points
   ($\tfrac12,\tfrac14,\tfrac34,\dots$) of randomly chosen cells (seeded RNG,
   `default_rng(42)`); the linear parameters are then re-solved.

The residual that is reported is $\|(\nabla_c J,\; c\odot g)\|_2$.

For the Allen–Cahn equation this minimization is performed **once per time step**,
using as energy the variational form of either a first-order convex-splitting scheme
or a second-order stabilized IMEX-BDF2 scheme.

---

## Installation

Python **≥ 3.10** is required (the code uses `match` statements).

```bash
git clone https://github.com/herreraGarroCesar/Ritz_NN_AllenCahn_Semilinear.git
cd Ritz_NN_AllenCahn_Semilinear
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

The scripts import their sibling modules with `from X import *`, so run them **from
inside their folder**:

```bash
# Steady semilinear diffusion–reaction problem
cd SemilinearDR
python mainDR3.py

# Allen–Cahn, test 1 (travelling front, exact solution)
cd AllenCahn/NN
python test1.py

# Allen–Cahn, test 2 (two interfaces, spectral reference solution)
cd AllenCahn/NN
python test2.py

# Pseudo-spectral reference solver on its own
cd AllenCahn/SpectralMethods
python IMEX_BDF2_Periodic.py
```

The Allen–Cahn drivers display an animation (via `plt.pause`) of the network solution,
the reference solution and the breakpoints at every time step, and print the errors.
To run without a display, set `MPLBACKEND=Agg`.

A minimal programmatic example:

```python
import numpy as np
from DR3_N import DR3_N          # run from inside SemilinearDR/

solver = DR3_N(N=20, u_id=2, a=-1, b=1, eps=np.sqrt(1e-3))
solver.Newton(max_itr=15, lr=1, tau_1=1e-8)
print(solver.h1_error[-1])       # relative H1-seminorm error
solver.plot_no_vlines()
```

---

## Notes

* Dimension: all codes are 1D.
* Runtime: the code favours clarity over speed (many Python loops and
  `scipy.integrate.quad` calls). The default Allen–Cahn tests (thousands of time
  steps) can take a long time; reduce `K`, `max_itr` or `N` for a quick look.
* See the *Known issues* sections in the sub-READMEs for details about the
  manufactured right-hand sides.
