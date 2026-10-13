import numpy as np
from numpy import array, zeros, diag, diagflat, dot, linalg
import matplotlib.pyplot as plt
import math
import scipy
from scipy import integrate
from scipy.integrate import quad
import time
import platform
import random
import warnings
warnings.filterwarnings('ignore')
from scipy.sparse import diags


class DR3_N:
    def __init__(self, N, a=0, b=1, u_id=1, eps=1):
        self.N = N
        self.a = a
        self.b = b
        self.eps = eps
        self.u_id = u_id
        self.alpha = self.u(a)
        self.beta = self.u_prime(b)
        self.theta = np.zeros((N+1, 2))
        self.theta[:, 0] = np.linspace(self.a, self.b, self.N+2)[:self.N+1]
        self.h1_u = self.H1_seminorm_u()
        self.h1_error = []
        self.residual = []
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
                return x * (np.exp((-(x - 1/3)**2) / 0.01) - np.exp((-4/9) / 0.01))
            case 2:
                return np.tanh((x**2 - 1/4) / self.eps) - np.tanh(3 / (4 * self.eps)) + 1
            case 3:
                return np.tanh((x - 0.5)/(np.sqrt(2)*self.eps))  
            case _:
                return x * (np.exp((-(x - 1/3)**2) / 0.01) - np.exp((-4/9) / 0.01))

    def u_prime(self, x):
        match self.u_id:
            case 1:
                return np.exp(-100*(x - 1/3)**2) * (-200*x**2 + (200/3)*x + 1)
            case 2:
                return (1/np.cosh((x**2 - 1/4) / self.eps))**2 * (2*x / self.eps)
            case 3:
                return (1/np.cosh((x - 0.5)/(np.sqrt(2)*self.eps)))*(1/(np.sqrt(2)*self.eps))
            case _:
                return np.exp(-100*(x - 1/3)**2) * (-200*x**2 + (200/3)*x + 1)

    def f(self, x):
        match self.u_id:
            case 1:
                return (-40000*(x**3 - 2*x**2/3 + 173*x/1800 + 1/300)
                        * np.exp(-100*(x - 1/3)**2) + self.u(x)**3)
            case 2:
                return (-2*(self.eps - 4*x**2 * np.tanh((x**2 - 1/4) / self.eps))
                        * (1/np.cosh((x**2 - 1/4) / self.eps))**2
                        + self.u(x)**3 + self.u(x))
            case 3:
                # Precompute the inner term and the tanh evaluation
                u = (x - 0.5) / (np.sqrt(2) * self.eps)
                tanh_u = np.tanh(u)
              
                # f''(x) = -(1/eps^2) * (1 - tanh^2(u)) * tanh(u)
                return  (1.0 - tanh_u**2) * tanh_u + (self.u(x))**3 + self.u(x)
            case _:
                return (-40000*(x**3 - 2*x**2/3 + 173*x/1800 + 1/300)
                        * np.exp(-100*(x - 1/3)**2) + self.u(x)**3)

    def relu(self, x):
        return np.maximum(0, x)

    def NN_1D(self, theta, x):
        return np.dot(theta[:, 1], self.relu(x - theta[:, 0])) + self.alpha

    def vectorized_NN_1D(self, theta, x):
        return np.vectorize(self.NN_1D, excluded=[0])(theta, x)

    def Heavi(self, x):
        return np.select([x < 0, x == 0, x > 0], [0, 0.5, 1])

    def NN_1D_prime(self, theta, x):
        return np.dot(theta[:, 1], self.Heavi(x - theta[:, 0]))

    # --- Error norms ---

    def H1_seminorm_u(self):
        integrand = lambda x: self.u_prime(x)**2
        if self.u_id == 2:
            pts = np.concatenate([np.linspace(-0.51, -0.49, 300),
                                   np.linspace(-1, 1, 1000),
                                   np.linspace(0.49, 0.51, 300)])
            return np.sqrt(integrate.quad(integrand, self.a, self.b,
                                          limit=2000, points=pts)[0])
        return np.sqrt(integrate.quad(integrand, self.a, self.b)[0])

    def H1_seminorm_un(self, theta):
        h = self.h_vector(theta[:, 0])
        d = self.derivatives_vector(theta)
        return np.sqrt(np.dot(h, d**2))

    def seminorm_error(self, theta):
        d = self.derivatives_vector(theta)
        b = np.append(theta[:, 0], [self.b])
        error = self.H1_seminorm_un(theta)**2 + self.h1_u**2
        error -= 2 * np.dot(d, self.u(b[1:]) - self.u(b[:-1]))
        return np.sqrt(error)

    # --- Helper vectors ---

    def derivatives_vector(self, theta):
        return np.cumsum(theta[:, 1])

    def h_vector(self, b):
        b_ext = np.append(b, self.b)
        return np.diff(b_ext)

    def d_vector(self, b):
        return self.b - b

    # --- Linear solvers ---

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

    def A_inv(self, b, y):
        h = self.h_vector(b)
        h_inv = 1.0 / h
        diag1 = -h_inv[:-1]
        diag  = np.empty(self.N+1)
        diag[0]  = h_inv[0]
        diag[1:] = h_inv[:-1] + h_inv[1:]  # interior + last
        diag[-1] = h_inv[-1]               # overwrite last correctly
        # Reconstruct: diag[i] for i in 1..N = 1/h[i-1] + 1/h[i]
        diag[1:] = h_inv[:-1] + h_inv[1:]
        diag[0]  = h_inv[0]
        return self.TDMAsolver(diag1, diag, diag1, y)

    # --- Quadrature vectors (vectorized) ---

    def _gauss_pts(self, theta, b, h):
        """Return the 5 Boole quadrature nodes for all intervals at once."""
        x0 = b[:-1]
        x1 = b[1:]
        pts = np.stack([x0,
                        x0 + h/4,
                        x0 + h/2,
                        x0 + 3*h/4,
                        x1], axis=1)   # (N+1, 5)
        return pts

    def s0_vector(self, theta):
        b   = np.append(theta[:, 0], self.b)
        h   = self.h_vector(theta[:, 0])
        pts = self._gauss_pts(theta, b, h)          # (N+1, 5)
        un  = np.vectorize(self.NN_1D, excluded=[0])(theta, pts)
        g   = 3 * un**2 + 1
        return (h / 6) * (g[:, 0] + 4*g[:, 2] + g[:, 4])

    def s1_vector(self, theta):
        b   = np.append(theta[:, 0], self.b)
        h   = self.h_vector(theta[:, 0])
        x0  = b[:-1]
        x1  = b[1:]
        p   = np.stack([x0,
                        (2*x0 + x1) / 3,
                        (x0 + 2*x1) / 3,
                        x1], axis=1)              # (N+1, 4), Simpson 3/8
        un  = np.vectorize(self.NN_1D, excluded=[0])(theta, p)
        g   = (p - x0[:, None]) * (3 * un**2 + 1)
        return (1/8) * (g[:, 0] + 3*g[:, 1] + 3*g[:, 2] + g[:, 3])

    def s2_vector(self, theta):
        b   = np.append(theta[:, 0], self.b)
        h   = self.h_vector(theta[:, 0])
        pts = self._gauss_pts(theta, b, h)          # (N+1, 5)
        x0  = b[:-1, None]
        un  = np.vectorize(self.NN_1D, excluded=[0])(theta, pts)
        g   = (pts - x0)**2 * (3 * un**2 + 1)
        return (1 / (90 * h)) * (7*g[:, 0] + 32*g[:, 1] + 12*g[:, 2]
                                  + 32*g[:, 3] + 7*g[:, 4])

    def M1_inv(self, theta, y):
        b    = theta[:, 0]
        h    = self.h_vector(b)
        h_inv = 1.0 / h
        eps2 = self.eps**2
        s0   = self.s0_vector(theta)
        s1   = self.s1_vector(theta)
        s2   = self.s2_vector(theta)

        diag1 = s1[1:] - s2[1:] - eps2 * h_inv[1:]
        diag  = np.empty(self.N+1)
        diag[:-1] = s0[1:] - 2*s1[1:] + s2[:-1] + s2[1:] + eps2*(h_inv[:-1] + h_inv[1:])
        diag[-1]  = s2[-1] + eps2 * h_inv[-1]
        return self.TDMAsolver(diag1, diag, diag1, y)

    def M2_inv(self, b, y):
        h    = self.h_vector(b)
        h_inv = 1.0 / h
        eps2 = self.eps**2

        diag1 = (1/12) * h[1:] - eps2 * h_inv[1:]
        diag  = np.empty(self.N+1)
        diag[:-1] = (1/3) * (h[:-1] + h[1:]) + eps2 * (h_inv[:-1] + h_inv[1:])
        diag[-1]  = (2/3) * h[-1] + eps2 * h_inv[-1]
        return self.TDMAsolver(diag1, diag, diag1, y)

    def G_matrix(self):
        return diags([np.ones(self.N+1), -np.ones(self.N)],
                     offsets=[0, -1], format="csr")

    # --- Linear parameter routines ---

    def init_fc(self, b):
        b_ext = np.append(b, self.b)
        d     = self.d_vector(b)
        eps2  = self.eps**2
        f_c   = np.zeros(self.N+1)
        for i in range(self.N+1):
            if self.u_id == 2:
                pts = np.concatenate([np.linspace(-0.51, -0.49, 200),
                                       np.linspace(-1, 1, 500),
                                       np.linspace(0.49, 0.51, 200)])
                fpts = pts[pts > b[i]]
                integrand = lambda x, bi=b[i]: (self.f(x) - self.alpha) * (x - bi)
                f_c[i] = integrate.quad(integrand, b[i], self.b,
                                        limit=1000, points=fpts)[0]
            else:
                integrand = lambda x, bi=b[i]: (self.f(x) - self.alpha) * (x - bi)
                f_c[i] = integrate.quad(integrand, b[i], self.b)[0]
        f_c += self.beta * eps2 * d
        return f_c

    def initSolverC(self, b, y):
        G     = self.G_matrix()
        h_inv = 1.0 / self.h_vector(b)
        x = G.T @ (h_inv * (G.T @ y))
        x = self.M2_inv(b, x)
        return G @ (h_inv * (G @ x))

    def gradC(self, theta):
        b    = np.append(theta[:, 0], self.b)
        h    = self.h_vector(theta[:, 0])
        c    = theta[:, 1]
        eps2 = self.eps**2
        gradC = eps2 * self.A_inv(b[:-1], c)

        # Boole quadrature nodes for all intervals: (N+1, 5)
        pts = self._gauss_pts(theta, b, h)
        un_pts = np.vectorize(self.NN_1D, excluded=[0])(theta, pts)
        g_pts  = un_pts**3 + un_pts          # g(x) = u^3 + u

        # Upper-triangular accumulation: gradC[i] += sum_{j>=i} integral over [b[j], b[j+1]]
        # Integrand: (x - b[i]) * g(x), Boole weights
        boole_w = np.array([7, 32, 12, 32, 7]) / 90.0
        for i in range(self.N+1):
            bi = b[i]
            for j in range(i, self.N+1):
                nodes  = pts[j]
                g_vals = g_pts[j]
                gradC[i] += h[j] * np.dot(boole_w, (nodes - bi) * g_vals)

        for i in range(self.N+1):
            if self.u_id == 2:
                pts_q = np.concatenate([np.linspace(-0.51, -0.49, 200),
                                         np.linspace(-1, 1, 800),
                                         np.linspace(0.49, 0.51, 200)])
                fpts = pts_q[pts_q > b[i]]
                integrand = lambda x, bi=b[i]: (x - bi) * (self.u(x)**3 + self.u(x))
                gradC[i] -= integrate.quad(integrand, b[i], self.b,
                                           limit=1200, points=fpts)[0]
                gradC[i] -= eps2 * (self.u(self.b) - self.u(b[i]))
            else:
                integrand = lambda x, bi=b[i]: self.f(x) * (x - bi)
                gradC[i] -= integrate.quad(integrand, b[i], self.b)[0]
        return gradC

    def solveC(self, theta, y):
        G     = self.G_matrix()
        h_inv = 1.0 / self.h_vector(theta[:, 0])
        x = G.T @ (h_inv * (G.T @ y))
        x = self.M1_inv(theta, x)
        return G @ (h_inv * (G @ x))

    # --- Nonlinear parameter routines ---

    def gradB(self, theta):
        b    = np.append(theta[:, 0], self.b)
        h    = self.h_vector(theta[:, 0])
        c    = theta[:, 1]
        eps2 = self.eps**2
        d    = self.derivatives_vector(theta)
        gradB = np.zeros(self.N)

        # eps2 term (vectorized)
        gradB -= eps2 * (d[:-1] + c[1:] / 2)

        # Simpson 3/8 nodes for f = u^3 + u over each interval
        x0 = b[:-1]; x1 = b[1:]
        p  = np.stack([x0,
                       (2*x0 + x1) / 3,
                       (x0 + 2*x1) / 3,
                       x1], axis=1)                  # (N+1, 4)
        un_p = np.vectorize(self.NN_1D, excluded=[0])(theta, p)
        fval = un_p**3 + un_p
        s3_8 = (h / 8) * (fval[:, 0] + 3*fval[:, 1]
                           + 3*fval[:, 2] + fval[:, 3])  # (N+1,)

        # Upper-triangular accumulation: gradB[i-1] -= sum_{j>=i} s3_8[j]
        cumsum_rev = np.cumsum(s3_8[::-1])[::-1]   # cumsum from right
        gradB -= cumsum_rev[1:]                     # indices 1..N

        # f(x) integral from b[i+1] to b (vectorized with quad per interval)
        for i in range(self.N):
            if self.u_id == 2:
                pts_q = np.concatenate([np.linspace(-0.51, -0.49, 200),
                                         np.linspace(-1, 1, 500),
                                         np.linspace(0.49, 0.51, 200)])
                fpts = pts_q[pts_q > b[i+1]]
                gradB[i] += integrate.quad(self.f, b[i+1], self.b,
                                           limit=1000, points=fpts)[0]
            else:
                gradB[i] += integrate.quad(self.f, b[i+1], self.b)[0]

        gradB += self.beta * eps2 * np.ones(self.N)
        return gradB

    def coefficientMatrixInv(self, theta):
        s = self.s0_vector(theta)
        s_inv = 1.0 / s[1:]                        # length N
        main_diag = np.empty(self.N)
        main_diag[0]  = s_inv[0]
        main_diag[1:] = s_inv[:-1] + s_inv[1:]
        off_diag = -s_inv[:-1]
        return diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')

    def tridiag_transform(self, a_lower, a_diag, a_upper, d1, d2):
        m_lower = d1[1:]  * a_lower * d2[:-1]
        m_diag  = 1.0     + d1 * a_diag * d2
        m_upper = d1[:-1] * a_upper * d2[1:]
        return m_lower, m_diag, m_upper

    def hessianSolver(self, theta, y, tau_1):
        A_inv  = self.coefficientMatrixInv(theta)
        b      = theta[1:, 0]
        c      = theta[1:, 1]
        d_inv  = np.where(np.abs(c) > tau_1, 1.0 / c, 0.0)

        un_b   = np.vectorize(self.NN_1D, excluded=[0])(theta, b)
        f_b    = self.f(b)
        valid  = (np.abs(c) > tau_1) & np.isfinite(f_b) & (np.abs(f_b) < 1e16)
        Lambda = np.where(valid, un_b**3 + un_b - f_b, 0.0)

        x = d_inv * (A_inv @ y)
        a_diag  = A_inv.diagonal(0)
        a_lower = A_inv.diagonal(-1)
        a_upper = A_inv.diagonal(1)
        diag1, diag, diag2 = self.tridiag_transform(
            a_lower, a_diag, a_upper, d_inv, Lambda)
        return self.TDMAsolver(diag1, diag, diag2, x)

    # --- Newton solver ---

    def Newton(self, max_itr=100, lr=1, tau_1=1e-6):
        self.theta[:, 1] = self.initSolverC(
            self.theta[:, 0], self.init_fc(self.theta[:, 0]))

        for j in range(100):
            gradC = self.gradC(self.theta)
            print(np.linalg.norm(gradC))
            if np.linalg.norm(gradC) < 1e-12:
                break
            self.theta[:, 1] -= self.solveC(self.theta, gradC)

        self.h1_error.append(self.seminorm_error(self.theta) / self.h1_u)
        rate = -np.log(self.h1_error[-1]) / np.log(self.N+1)
        gradC = self.gradC(self.theta)
        gradB = self.gradB(self.theta)
        grad  = np.concatenate((gradC, self.theta[1:, 1] * gradB))
        self.residual.append(np.linalg.norm(grad))

        print('Newton...')
        print(f'Initial residual:     {self.residual[-1]}')
        print(f'Initial seminorm err: {self.h1_error[-1]}')
        print(f'Initial rate:         {rate}')

        for i in range(max_itr):
            gradB = self.gradB(self.theta)
            pk    = -self.hessianSolver(self.theta, gradB, tau_1=tau_1)
            self.theta[1:, 0] += lr * pk

            self.theta = self.theta[np.argsort(self.theta[:, 0])]
            self.redistribute_neurons(tau_1=tau_1)
            self.theta = self.theta[np.argsort(self.theta[:, 0])]

            gradC = self.gradC(self.theta)
            self.theta[:, 1] -= self.solveC(self.theta, gradC)
            gradC = self.gradC(self.theta)

            self.h1_error.append(self.seminorm_error(self.theta) / self.h1_u)
            gradB = self.gradB(self.theta)
            grad  = np.concatenate((gradC, self.theta[1:, 1] * gradB))
            self.residual.append(np.linalg.norm(grad))

            print(f'Itr: {i+1}  |  Err: {self.h1_error[-1]:.3e}  |  Res: {self.residual[-1]:.3e}')

        gradC = self.gradC(self.theta)
        gradB = self.gradB(self.theta)
        grad  = np.concatenate((gradC, self.theta[1:, 1] * gradB))
        self.residual.append(np.linalg.norm(grad))
        rate  = -np.log(self.h1_error[-1]) / np.log(self.N+1)

        print(f'Final residual:    {self.residual[-1]:.3e}')
        print(f'Final seminorm err:{self.h1_error[-1]:.3e}')
        print(f'Final rate:        {rate:.3f}')

    # --- Neuron redistribution ---

    def redistribute_neurons(self, tau_1):
        mask0 = np.roll(self.theta[:, 0] < self.a, 1)
        self.theta[:, 0] = np.clip(self.theta[:, 0], self.a, None)

        mask = np.abs(np.roll(self.theta[:, 0], 1) - self.theta[:, 0]) < 1e-12
        j = 0
        while j < self.N - 1:
            if j+2 < self.N and mask[j+2] and mask[j+1] and mask[j]:
                mask[j+2] = mask[j] = False; j += 2
            elif mask[j] and mask[j+1]:
                mask[j] = False; j += 1
            else:
                j += 1

        dead = ((self.theta[:, 0] >= self.b) | mask | mask0
                | (np.abs(self.theta[:, 1]) < tau_1)
                | np.isnan(self.theta[:, 0]))
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
            q = random_list[j]
            m = random_count[q]
            self.theta[idx, 0] = (theta_aux[q, 0]
                                  + self.two_adic_integers[m]
                                  * (theta_aux[q+1, 0] - theta_aux[q, 0]))
            random_count[q] += 1

        if len(indices_to_move) > 0:
            self.theta = self.theta[np.argsort(self.theta[:, 0])]
            for j in range(50):
                gradC = self.gradC(self.theta)
                if np.linalg.norm(gradC) < 1e-12:
                    break
                self.theta[:, 1] -= self.solveC(self.theta, gradC)

    # --- Plotting ---

    def plot_error(self):
        mesh = np.linspace(0, len(self.errorNewt), len(self.errorNewt))
        plt.figure(figsize=(7.5, 7.5))
        plt.plot(mesh, self.errorNewt, linewidth=2)
        if self.refinement_list:
            plt.vlines(self.refinement_list,
                       ymin=min(self.errorNewt), ymax=max(self.errorNewt),
                       linestyles='dotted', colors='gray')
            plt.scatter(self.refinement_list,
                        min(self.errorNewt) * np.ones(len(self.refinement_list)),
                        marker='x', color='red', label='Refinement')
            plt.legend()
        plt.ylabel('Relative seminorm error')
        plt.yscale('log')
        plt.show()

    def plot_no_vlines(self):
        mesh = np.linspace(self.a, self.b, 2000)
        plt.figure(figsize=(5, 5))
        plt.plot(mesh, self.u(mesh), linewidth=2, label=r'$u$')
        plt.plot(mesh, self.vectorized_NN_1D(self.theta, mesh),
                 linewidth=2, label=r'$u_n$', color='red')
        plt.scatter(self.theta[1:, 0],
                    min(self.vectorized_NN_1D(self.theta, mesh)) * np.ones(self.N),
                    marker='|', color='green', label='Break points')
        plt.legend()
        plt.show()

    def plot(self):
        mesh = np.linspace(self.a, self.b, 2000)
        plt.figure(figsize=(5, 5))
        plt.plot(mesh, self.u(mesh), linewidth=2, label=r'$u$')
        plt.plot(mesh, self.vectorized_NN_1D(self.theta, mesh), linewidth=2, label=r'$u_n$')
        plt.vlines(self.theta[:, 0],
                   ymin=min(self.u(mesh)), ymax=max(self.u(mesh)),
                   linestyles='dotted', colors='gray')
        plt.scatter(self.theta[:, 0], min(self.u(mesh)) * np.ones(self.N+1),
                    marker='x', color='red', label='Break points')
        plt.legend()
        plt.show()