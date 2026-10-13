import numpy as np
from numpy import array, zeros, diag, diagflat, dot, linalg
import matplotlib.pyplot as plt
import math
import scipy
from scipy import integrate
import scipy.special as special
import time
import sys
import warnings
from scipy.optimize import minimize_scalar
from scipy.sparse import diags

warnings.filterwarnings('ignore')


class LS:
    def __init__(self, N, a=0.0, b=1.0, u_id=1, eps=1):
        self.N = N
        self.a = a
        self.b = b
        self.u_id = u_id
        self.eps = eps
        self.theta = np.zeros((N+1, 2))
        self.theta[:, 0] = np.linspace(self.a, self.b, self.N+2)[:self.N+1]
        self.itr = 0
        self.alpha = self.u(a)
        self.loss = []
        self.residual = []
        self.L2_u = self.L2_u()
        self.rng = np.random.default_rng(42)
        self.two_adic_integers = [
            1/2, 1/4, 3/4,
            1/8, 3/8, 5/8, 7/8,
            1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16,
            1/32, 3/32, 5/32, 7/32, 9/32, 11/32, 13/32, 15/32,
            17/32, 19/32, 21/32, 23/32, 25/32, 27/32, 29/32, 31/32,
            1/64, 3/64, 5/64, 7/64, 9/64, 11/64, 13/64, 15/64,
            17/64, 19/64, 21/64, 23/64, 25/64, 27/64, 29/64, 31/64,
            33/64, 35/64, 37/64, 39/64, 41/64, 43/64, 45/64, 47/64,
            49/64, 51/64, 53/64, 55/64, 57/64, 59/64, 61/64, 63/64,
            1/128, 3/128, 5/128, 7/128, 9/128, 11/128, 13/128, 15/128,
            17/128, 19/128, 21/128, 23/128, 25/128, 27/128, 29/128, 31/128,
            33/128, 35/128, 37/128, 39/128, 41/128, 43/128, 45/128, 47/128,
            49/128, 51/128, 53/128, 55/128, 57/128, 59/128, 61/128, 63/128,
        ]

    # --- PDE and NN functions ---

    def u(self, x):
        match self.u_id:
            case 1:
                return (np.tanh((x - 0.25) / (np.sqrt(2)*self.eps))
                        - np.tanh((x - 0.75) / (np.sqrt(2)*self.eps))
                        - 1 + 0.1*np.sin(2*np.pi*x))
            case 2:
                return 0.5 - 0.5*np.tanh((x - 0.5) / (2*np.sqrt(2)*self.eps))
            case _:
                return x * (np.exp((-(x - 1/3)**2) / 0.01) - np.exp((-4/9) / 0.01))

    def relu(self, x):
        return np.maximum(0, x)

    def NN_1D(self, theta, x):
        return np.dot(theta[:, 1], self.relu(x - theta[:, 0])) + self.alpha

    def vectorized_NN_1D(self, theta, x):
        return np.vectorize(self.NN_1D, excluded=[0])(theta, x)

    # --- Norms and errors ---

    def L2_u(self):
        integrand = lambda x: self.u(x)**2
        return np.sqrt(integrate.quad(integrand, self.a, self.b,
                                      epsabs=1.49e-10, epsrel=1.49e-10,
                                      limit=500)[0])

    def L2_un(self, theta):
        b   = np.append(theta[:, 0], self.b)
        mid = (b[:-1] + b[1:]) / 2
        h   = np.diff(b)
        f0  = np.vectorize(self.NN_1D, excluded=[0])(theta, b[:-1])
        fm  = np.vectorize(self.NN_1D, excluded=[0])(theta, mid)
        f1  = np.vectorize(self.NN_1D, excluded=[0])(theta, b[1:])
        return np.sum((h / 6) * (f0**2 + 4*fm**2 + f1**2))

    def L2(self, theta):
        integrand = lambda x: (self.u(x) - self.NN_1D(theta, x))**2
        integral, _ = integrate.quad(integrand, self.a, self.b,
                                     epsabs=1.49e-10, epsrel=1.49e-10,
                                     limit=500)
        return np.sqrt(integral)

    # --- Linear solvers ---

    def shermanMorrison(self, Ax, Au, v):
        return Ax - (np.inner(Ax, v) / (1 + np.inner(v, Au))) * Au

    def TDMAsolver(self, a, b, c, d):
        nf = len(d)
        ac, bc, cc, dc = map(np.array, (a, b, c, d))
        for it in range(1, nf):
            mc = ac[it-1] / bc[it-1]
            bc[it] -= mc * cc[it-1]
            dc[it] -= mc * dc[it-1]
        xc = bc
        xc[-1] = dc[-1] / bc[-1]
        for il in range(nf-2, -1, -1):
            xc[il] = (dc[il] - cc[il] * xc[il+1]) / bc[il]
        return xc

    def h_vector(self, b):
        return np.diff(np.append(b, self.b))

    def f_c(self, b):
        b_ext = np.append(b, self.b)
        f_vec = np.zeros(self.N+1)
        for i in range(self.N+1):
            integrand = lambda x, bi=b[i]: self.u(x) * (x - bi)
            integral, _ = integrate.quad(integrand, b[i], self.b,
                                         epsabs=1.49e-10, epsrel=1.49e-10,
                                         limit=500)
            f_vec[i] = (integral
                        - self.alpha * (self.b**2/2 + b[i]**2/2 - b[i]*self.b))
        return f_vec

    def T_inv(self, b, y):
        h = self.h_vector(b)
        diag1     = h[1:] / 6
        diag      = np.empty(self.N+1)
        diag[:-1] = (h[:-1] + h[1:]) / 3
        diag[-1]  = h[-1] / 3
        return self.TDMAsolver(diag1, diag, diag1, y)

    def G_matrix(self):
        return diags([np.ones(self.N+1), -np.ones(self.N)],
                     offsets=[0, -1], format="csr")

    def solveC(self, b, y):
        G     = self.G_matrix()
        h_inv = 1.0 / self.h_vector(b)
        x = G.T @ (h_inv * (G.T @ y))
        x = self.T_inv(b, x)
        return G @ (h_inv * (G @ x))

    # --- Nonlinear parameter routines ---

    def s_vector(self, b):
        return np.diff(np.append(b, self.b))

    def coefficientMatrixInv(self, theta):
        s     = self.s_vector(theta[1:, 0])
        s_inv = 1.0 / s
        main_diag      = np.empty(self.N)
        main_diag[0]   = s_inv[0]
        main_diag[1:]  = s_inv[:-1] + s_inv[1:]
        off_diag       = -s_inv[:-1]
        return diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')

    def tridiag_transform(self, a_lower, a_diag, a_upper, d1, d2):
        m_lower = d1[1:]  * a_lower * d2[:-1]
        m_diag  = 1.0     + d1 * a_diag * d2
        m_upper = d1[:-1] * a_upper * d2[1:]
        return m_lower, m_diag, m_upper

    def hessianSolver(self, theta, y, tau_1):
        A_inv  = self.coefficientMatrixInv(theta)
        c      = theta[1:, 1]
        b      = theta[1:, 0]
        d_inv  = np.where(np.abs(c) > tau_1, 1.0 / c, 0.0)

        un_b   = np.vectorize(self.NN_1D, excluded=[0])(theta, b)
        Lambda = un_b - self.u(b)

        x = d_inv * (A_inv @ y)
        a_diag  = A_inv.diagonal(0)
        a_lower = A_inv.diagonal(-1)
        a_upper = A_inv.diagonal(1)
        diag1, diag, diag2 = self.tridiag_transform(
            a_lower, a_diag, a_upper, d_inv, Lambda)
        return self.TDMAsolver(diag1, diag, diag2, x)

    def gradB(self, theta):
        b    = np.append(theta[1:, 0], self.b)
        grad = np.zeros(self.N)
        for i in range(self.N):
            integrand = lambda x, bi=b[i]: self.u(x) - self.NN_1D(theta, x)
            integral, _ = integrate.quad(integrand, b[i], self.b,
                                         epsabs=1.49e-10, epsrel=1.49e-10,
                                         limit=300)
            grad[i] = integral
        return grad

    # --- Newton solver ---

    def Newton(self, max_itr=100, lr=1, ls=False, tau_1=1e-8,
               stop=True, stopping_tol=1e-9):

        self.theta[:, 1] = self.solveC(self.theta[:, 0],
                                       self.f_c(self.theta[:, 0]))
        self.loss.append(self.L2(self.theta) / self.L2_u)
        self.residual.append(np.linalg.norm(self.gradB(self.theta)))

        print('Newton...')
        print(f'Initial error:    {self.loss[0]}')
        print(f'Initial residual: {self.residual[-1]}')

        gradB = self.gradB(self.theta)
        for i in range(max_itr):
            self.itr += 1
            b   = self.theta[1:, 0]
            pk  = -self.hessianSolver(self.theta, gradB, tau_1)
            lrk = lr

            if ls:
                f   = lambda x: self.ls_L2(b + x*pk)
                res = minimize_scalar(f, bounds=(0, 1), method='bounded',
                                      tol=1e-15)
                lrk = res.x

            self.theta[1:, 0] = b + lrk * pk
            self.theta = self.theta[np.argsort(self.theta[:, 0])]
            self.redistribute_neurons(tau_1=tau_1)
            self.theta = self.theta[np.argsort(self.theta[:, 0])]
            self.theta[:, 1] = self.solveC(self.theta[:, 0],
                                           self.f_c(self.theta[:, 0]))

            current_loss = self.L2(self.theta) / self.L2_u
            self.loss.append(current_loss)
            gradB = self.gradB(self.theta)
            self.residual.append(np.linalg.norm(self.theta[1:, 1] * gradB))

            print(f'Itr: {i+1}  |  lrk: {lrk}  |  '
                  f'L2 err: {current_loss:.3e}  |  Res: {self.residual[-1]:.3e}')

            if stop and self.residual[-1] < stopping_tol:
                print(f'Stopped at itr {i}')
                break

        print(f'Final err:      {self.loss[-1]}')
        print(f'Final residual: {self.residual[-1]}')

    # --- Line search ---

    def ls_L2(self, b):
        if np.any(b < self.a) or np.any(b > self.b):
            return 1e+15
        theta = np.zeros((self.N+1, 2))
        theta[:, 0] = np.append([self.a], b)
        theta[:, 1] = self.theta[:, 1]
        theta = theta[np.argsort(theta[:, 0])]
        return self.L2(theta)

    # --- Neuron redistribution ---

    def redistribute_neurons(self, tau_1):
        self.theta[:, 0] = np.clip(self.theta[:, 0], self.a, None)
        dead = ((self.theta[:, 0] >= self.b - 1e-14)
                | (np.abs(np.roll(self.theta[:, 0], 1) - self.theta[:, 0]) < 1e-8)
                | (np.abs(self.theta[:, 1]) < tau_1))
        indices_to_move = np.where(dead)[0]
        if len(indices_to_move) > 0 and indices_to_move[0] == 0:
            indices_to_move = indices_to_move[1:]

        theta_aux = np.delete(self.theta, indices_to_move, axis=0)
        if len(theta_aux) < 2:
            self.theta[:, 0] = np.linspace(self.a, self.b, self.N+2)[:self.N+1]
            return

        random_count = np.zeros(self.N, dtype=int)
        random_list  = self.rng.integers(0, len(theta_aux) - 1,
                                         size=len(indices_to_move))
        for j, idx in enumerate(indices_to_move):
            k = random_list[j]
            m = random_count[k]
            self.theta[idx, 0] = (theta_aux[k, 0]
                                  + self.two_adic_integers[m]
                                  * (theta_aux[k+1, 0] - theta_aux[k, 0]))
            random_count[k] += 1

    # --- Plotting ---

    def plot(self):
        mesh = np.linspace(self.a, self.b, 2000)
        plt.figure(figsize=(5, 5))
        plt.plot(mesh, self.u(mesh), linewidth=2, label=r'$u$')
        plt.plot(mesh, self.vectorized_NN_1D(self.theta, mesh),
                 linewidth=2, label=r'$u_n$')
        plt.vlines(self.theta[:, 0],
                   ymin=min(self.u(mesh)), ymax=max(self.u(mesh)),
                   linestyles='dotted', colors='gray')
        plt.scatter(self.theta[:, 0],
                    min(self.u(mesh)) * np.ones(len(self.theta[:, 0])),
                    marker='x', color='red', label='Breakpoints')
        plt.legend()
        plt.show()

    def plot_no_vlines(self):
        mesh = np.linspace(self.a, self.b, 2000)
        plt.figure(figsize=(5, 5))
        plt.plot(mesh, self.u(mesh), linewidth=2, label=r'$u$')
        plt.plot(mesh, self.vectorized_NN_1D(self.theta, mesh),
                 linewidth=2, label=r'$u_n$', color='red')
        plt.scatter(self.theta[1:, 0],
                    min(self.vectorized_NN_1D(self.theta, mesh)) * np.ones(self.N),
                    marker='|', color='green', label='Breakpoints')
        plt.legend()
        plt.show()