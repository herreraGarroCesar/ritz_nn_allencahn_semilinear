import numpy as np
from numpy import array, zeros, diag, diagflat, dot, linalg
import matplotlib.pyplot as plt
import math
import scipy
from scipy import integrate
import time
import platform
import random
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize_scalar
from scipy.sparse import diags



class AC_CSS_NB:
    def __init__(self, N,  K, a = -1.0, b = 1.0, alpha = -1, beta = -1,gamma = 1e+4, u_id = 1, Tinit = 0,Tfinal = 1, eps = 1, n = 10000):
        # Number of interior mesh points
        self.N = N 
        # Number of time steps
        self.K = K
        self.t = np.linspace(Tinit, Tfinal, num = K+1)
        self.Tinit = Tinit
        self.Tfinal = Tfinal 
        self.dt = self.t[1] - self.t[0]
        # interval of interest [a,b]
        self.a = a
        self.b = b
        # example id (1,2, 3, 4...)
        self.u_id = u_id
        # Pde parameters
        self.eps = eps
        self.alpha = (self.u(a, 0))*np.ones(K + 1)
        self.beta = np.zeros(K + 1)
        # Neural net parametes 
        self.theta = np.zeros((N+1, 2))
        self.C = np.zeros((N+1, self.K+1))
        self.B = np.zeros((N+1, self.K+1))
        # Compute reference solution
        self.n = n
        self.mesh = np.linspace(self.a, self.b, self.n, endpoint=True)
        # For reactivating dead neurons
        self.rng = np.random.default_rng(42)
        # 2-adic list
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



    def u(self, x, t):
        match self.u_id:
            case 1:
                s = 3*self.eps/(np.sqrt(2))
                return 0.5 - 0.5*(np.tanh((x - 0.5 - s*t)/(2*np.sqrt(2)*self.eps)))
            case _:
                s = 3* self.eps/(np.sqrt(2))
                return 0.5 - 0.5*(np.tanh((x - 0.5 - s*t)/(2*np.sqrt(2)*self.eps)))

    def u_prime(self, x, t):
        match self.u_id:
            case 1:
                s = 3*self.eps/(np.sqrt(2))
                return -0.5*(1/(2*np.sqrt(2)*self.eps))*((1/np.cosh((x - 0.5 - s*t)/(2*np.sqrt(2)*self.eps)))**2)
            case _:
                s = 3*self.eps/(np.sqrt(2))
                return -0.5*(1/(2*np.sqrt(2)*self.eps))*((1/np.cosh((x - 0.5 - s*t)/(2*np.sqrt(2)*self.eps)))**2)


    def relu(self, x):
        return np.maximum(0, x)


    def NN_1D(self,theta, x, k):
        c = theta[:, 1] # c0,...,cN = theat_0,....,theta_N
        b = theta[:, 0]  # b0,...,bN = theta_(N+1),...,theta_(2N+1)
        y = self.relu(x-b) 
        return np.dot(c, y) + self.alpha[k] 

    # to be able to put in a vector of x-values, used for plotting
    def vectorized_NN_1D(self, theta, x, k):
        vectorized_NN1D = np.vectorize(self.NN_1D, excluded=[0, 2])
        return vectorized_NN1D(theta, x, k)


    # Heaviside function
    def Heavi(self, x):
        condlist = [x<0, x==0, x>0]
        choicelist = [0, 0.5, 1]
        return np.select(condlist,choicelist)


    # Derivative of the neural network (w.r.t. x)
    def NN_1D_prime(self, theta, x):
        c = theta[:,1]
        b = theta[:,0]  
        Y = self.Heavi(x-b)
        return np.dot(c,Y)

       # This vector contains the derivatives on each subinterval of the NN
    def derivatives_vector(self, theta):
        d = np.zeros(self.N + 1)
        c = theta[:, 1]
        for i in range(self.N+1):
            d[i] = np.sum(c[0:i+1])
        return d

    def h_vector(self, b):
        h = np.zeros(self.N+1)
        h[self.N] = self.b - b[self.N]
        for i in range(1, self.N+1):
            h[i-1] = b[i] - b[i-1]
        return h    



    # Error
    def H1_seminorm_u(self, t):
        s = 3*self.eps/(np.sqrt(2))
        F = lambda x: 0.25*(1/(2*np.sqrt(2)*self.eps))*((np.tanh((x - 0.5 - s*t)/(2*np.sqrt(2)*self.eps))) - (1/3)*(np.tanh((x - 0.5 - s*t)/(2*np.sqrt(2)*self.eps)))**3)
        return np.sqrt(F(self.b) - F(self.a))

    def H1_seminorm_un(self, theta):
        h = self.h_vector(theta[:,0])
        d = self.derivatives_vector(theta)
        b = np.append(theta[:, 0], [self.b])
        h1_un = 0
        for i in range(self.N+1):
            h1_un += h[i]*(d[i]**2)
        return np.sqrt(h1_un)

    def seminorm_error(self, theta, t):
        d = self.derivatives_vector(theta)
        c = theta[:, 1]
        b = np.append(theta[:, 0], [self.b])
        h1_u = self.H1_seminorm_u(t)
        error =  self.H1_seminorm_un(theta)**2 + h1_u**2
        for i in range(self.N+1):
            error -=2*d[i]*(self.u(b[i+1], t) -self.u(b[i], t))
        return np.sqrt(error)/h1_u

        # Square L2 norm of the solution
    def L2_u(self, t):
        integrand = lambda x: (self.u(x, t))**2
        return np.sqrt(integrate.quad(integrand,self.a,self.b, epsabs=1.49e-10, epsrel=1.49e-10, limit=500)[0])

    def L2_error(self, theta, t, k):
        l2_u = self.L2_u(t)
        integrand = lambda x: (self.u(x, t) - self.NN_1D(theta, x, k))**2
        integral, err = integrate.quad(integrand,self.a,self.b, epsabs=1.49e-12, epsrel=1.49e-12, limit=500)
        return np.sqrt(integral)/l2_u      


 



    # https://gist.github.com/cbellei/8ab3ab8551b8dfc8b081c518ccd9ada9
    def TDMAsolver(self, a, b, c, d):
        '''
        TDMA solver, a b c d can be NumPy array type or Python list type.
        refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
        '''
        nf = len(d)  # number of equations
        ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
        for it in range(1, nf):
            mc = ac[it-1]/bc[it-1]
            bc[it] = bc[it] - mc*cc[it-1] 
            dc[it] = dc[it] - mc*dc[it-1]

        xc = bc
        xc[-1] = dc[-1]/bc[-1]

        for il in range(nf-2, -1, -1):
            xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

        return xc


    # This is to solve A^{-1}x = y, where A is the coefficient matrix 
    def A_inv(self, b, y):
        h = self.h_vector(b)
        diag1 = np.zeros(self.N)
        diag = np.zeros(self.N+1)     
        for i in range(self.N):
            # Upper diagonal and lower diagonal
            diag1[i] = - 1/h[i]
            diag[i+1] = 1/h[i] + 1/h[i+1]
        diag[0] = 1/h[0]   
        return self.TDMAsolver(diag1, diag, diag1, y)

    def s0_vector(self, theta, k):
        s0 = np.zeros(self.N+1)
        h = self.h_vector(theta[:, 0])
        b = np.append(theta[:, 0], [self.b])
        f = lambda x: 3*(self.NN_1D(theta, x, k)**2) + 1/self.dt
        for i in range(self.N+1):
            s0[i] = (h[i]/6)*(f(b[i]) + 4*f((b[i+1]+b[i])/2) + f(b[i+1]))
        return s0

    def s1_vector(self, theta, k):
        s1 = np.zeros(self.N+1)
        b = np.append(theta[:, 0], [self.b])   
        for i in range(self.N+1):
            f = lambda x: (x-b[i])*(3*(self.NN_1D(theta, x, k)**2) + 1/self.dt)
            s1[i] = (1/8)*(f(b[i]) + 3*f((2*b[i] + b[i+1])/3) + 3*f((b[i] + 2*b[i+1])/3) + f(b[i+1]))
        return s1

    def s2_vector(self, theta, k):
        s2 = np.zeros(self.N+1)
        h = self.h_vector(theta[:, 0])
        b = np.append(theta[:, 0], [self.b])    
        for i in range(self.N+1):
            f = lambda x: ((x-b[i])**2)*(3*(self.NN_1D(theta, x, k)**2) + 1/self.dt)
            s2[i] = (1/(90*h[i]))*(7*f(b[i]) + 32*f(b[i] + h[i]/4) + 12*f(b[i] + h[i]/2) + 32*f(b[i] + 3*h[i]/4) + 7*f(b[i+1]))
        return s2
 
 


    # This is one component of the mass matrix, b is the mesh and y is a rhs vector 
    def M1_inv(self, theta, y, k):
        b = theta[:, 0]    
        h = self.h_vector(b)
        diag1 = np.zeros(self.N)
        diag = np.zeros(self.N+1)  
        s0 = self.s0_vector(theta, k)
        s1 = self.s1_vector(theta, k)
        s2 = self.s2_vector(theta, k) 
        eps2 = self.eps**2
        for i in range(self.N):
            # Upper diagonal and lower diagonal
            diag1[i] = s1[i+1] - s2[i+1] - eps2/h[i+1]
            # Main diagonal 
            diag[i] = s0[i+1] - 2*s1[i+1] + s2[i] + s2[i+1] + eps2/h[i] + eps2/h[i+1]
        # Especial entries
        diag[self.N] = s2[self.N] + eps2/h[self.N]    
        return self.TDMAsolver(diag1, diag, diag1, y)


    def G_matrix(self):
        main_diag = np.ones(self.N+1)       # diagonal of ones
        sub_diag  = -np.ones(self.N)        # sub-diagonal of -1
        G = diags([main_diag, sub_diag], offsets=[0, -1], format="csr")
        return G

    # Time marching, getting the ci's 
    # Gradient 
    def gradC(self, theta, theta_prev, k):
        gradC = np.zeros(self.N+1)
        c = theta[:, 1]
        h = self.h_vector(theta[:,0])
        hh = self.h_vector(theta_prev[:, 0])
        b = np.append(theta[:, 0], [self.b])
        bb = np.append(theta_prev[:, 0], [self.b])
        gradC += (self.eps**2)*self.A_inv(b, c)
        for i in range(self.N+1):
            f = lambda x: (x-b[i])*(self.NN_1D(theta, x, k))**3 + (1/(self.dt))*(x-b[i])*(self.NN_1D(theta, x, k))
            g = lambda x: (1/(self.dt) + 1)*(x-b[i])*(self.NN_1D(theta_prev, x, k-1))  
            for j in range(i, self.N+1):
                gradC[i] += (h[j]/90)*(7*f(b[j]) + 32*f(b[j] + h[j]/4) + 12*f(b[j] + h[j]/2) + 32*f(b[j] + 3*h[j]/4) + 7*f(b[j+1])) 
            for j in range(self.N+1):    
                if bb[j]<b[i] and bb[j+1]<=b[i]:
                    continue 
                elif bb[j]<=b[i] and b[i] < bb[j+1]:
                    gradC[i] -= ((bb[j+1] - b[i])/6)*(g(b[i]) + 4*g((bb[j+1]+b[i])/2) + g(bb[j+1]))
                else:   
                    gradC[i] -= (hh[j]/6)*(g(bb[j]) + 4*g((bb[j+1]+bb[j])/2) + g(bb[j+1]))
        return gradC



    # y is the rhs vector, b is the mesh 
    def solveC(self, theta, y, k):
        G = self.G_matrix()
        b = theta[:, 0] 
        h = self.h_vector(b)
        h_inv = 1.0/h  
        x = G.T@(h_inv*(G.T@y))
        x = self.M1_inv(theta, x, k)
        return G@(h_inv*(G@x))



    # Non-linear parameters 

   # Gradient
    def gradB(self, theta, theta_prev, k):
        h = self.h_vector(theta[:, 0])
        c = theta[:, 1]  # c0, c1,...,cN 
        b = np.append(theta[:, 0], [self.b])    # b0,...,bN+1     
        bb = np.append(theta_prev[:, 0], [self.b])
        hh = self.h_vector(theta_prev[:, 0])
        gradB = np.zeros(self.N)
        eps2 = self.eps**2
        d = self.derivatives_vector(theta)
        for i in range(self.N):
            gradB[i] -= eps2*(d[i] + c[i+1]/2) 
        f = lambda x: self.NN_1D(theta, x, k)**3 + (1/self.dt)*self.NN_1D(theta, x, k)
        for i in range(1, self.N+1):
            for j in range(i, self.N+1):
                gradB[i-1]-= (h[j]/8)*(f(b[j]) + 3*f((2*b[j] + b[j+1])/3) +  3*f((b[j] + 2*b[j+1])/3) +f(b[j+1]))            
        g = lambda x: (1/self.dt + 1)*self.NN_1D(theta_prev, x, k-1)
        for i in range(1, self.N+1):
            for j in range(self.N+1):    
                if bb[j]<b[i] and bb[j+1]<=b[i]:
                    continue 
                elif bb[j]<=b[i] and b[i] < bb[j+1]:
                    gradB[i-1] += ((bb[j+1] - b[i])/2)*(g(b[i]) + g(bb[j+1]))
                else:   
                    gradB[i-1] += (hh[j]/2)*(g(bb[j]) + g(bb[j+1]))
        return gradB





    # Assume that B is sorted
    def coefficientMatrixInv(self, theta, k):
        s = self.s0_vector(theta, k)
        main_diag = np.zeros(self.N)
        off_diag = np.zeros(self.N-1)
        main_diag[0] = 1/s[1]
        for i in range(1, self.N):
            main_diag[i] = 1/s[i] + 1/s[i+1]
            off_diag[i-1] = -1/s[i]
        A_sparse = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')
        return A_sparse


    

    def tridiag_transform(self, a_lower, a_diag, a_upper, d1, d2):
        n = len(a_diag)
        m_lower = d1[1:] * a_lower * d2[:-1]       # length n-1
        m_diag  = 1.0 + d1 * a_diag * d2           # length n
        m_upper = d1[:-1] * a_upper * d2[1:]       # length n-1
        return m_lower, m_diag, m_upper


    def hessianSolver(self, theta, theta_prev,  y, k, tau_1):
        A_inv = self.coefficientMatrixInv(theta, k)
        Lambda = np.zeros(self.N)
        c = theta[1:, 1]
        b = theta[1:, 0]
        d_inv = np.where((np.abs(c) > tau_1), 1.0 / c, 0.0)
        for i in range(self.N):
            un = self.NN_1D(theta, b[i], k)
            unn = self.NN_1D(theta_prev, b[i], k-1)
            Lambda[i] = (un**3 - unn) + (1/self.dt)*(un - unn)
        x = d_inv*(A_inv@(y))
        a_diag  = A_inv.diagonal(0)   # main diagonal (length n)
        a_lower = A_inv.diagonal(-1)  # sub-diagonal (length n-1)
        a_upper = A_inv.diagonal(1)   # super-diagonal (length n-1)
        diag1, diag, diag2 = self.tridiag_transform(a_lower, a_diag, a_upper, d_inv, Lambda)
        return self.TDMAsolver(diag1, diag, diag2, x)




    # Newton's method
    def Newton(self, theta_init, max_itr = 50, tau_1 = 1e-8, stopping_tolerance = 7e-5, stop = False):

        # First time step
        self.theta = theta_init.copy()
        self.C[:, 0] = self.theta[:, 1]
        self.B[:, 0] = self.theta[:, 0]
        res = 0
        curr_err_h1 = self.seminorm_error(self.theta, 0.0)
        print(f'Initial H1 error: {curr_err_h1:.3e}')

        # Report 
        print('Newton...')
        # Main algorithm
        for k in range(1, self.K+1):
            theta_prev = self.theta.copy()
            self.alpha[k] = self.u(self.a, self.Tinit + self.dt*k)
            # Initial: solving for C and setting B as the previous breakpoints
            for j in range(25):
                gradC = self.gradC(self.theta, theta_prev, k)
                norm_grad = np.linalg.norm(gradC)
                if np.linalg.norm(gradC) < 1e-12:
                    break
                pc = -self.solveC(self.theta, gradC, k)
                self.theta[:, 1] = self.theta[:, 1] + pc 
            gradB = self.gradB(self.theta, theta_prev, k)    
            init_res = np.linalg.norm(self.theta[1:, 1]*gradB)           
            for j in range(max_itr):
                # Newton's step 
                b = self.theta[1:, 0]
                pk = -self.hessianSolver(self.theta, theta_prev,gradB, k, tau_1)
                # Newton step
                self.theta[1:, 0] = b + pk
                # Sorting and redistributing
                sorted_indices = np.argsort(self.theta[:, 0])
                self.theta = self.theta[sorted_indices]                
                self.redistribute_neurons(theta_prev, k, tau_1)
                # Solving for the c's 
                gradC = self.gradC(self.theta, theta_prev, k)
                pc = -self.solveC(self.theta, gradC, k)
                self.theta[:, 1] = self.theta[:, 1] + pc
                # Computing residual 
                gradB = self.gradB(self.theta, theta_prev, k)
                res = np.linalg.norm(self.theta[1:, 1]*gradB)
                if stop and j>1 and  res < stopping_tolerance:
                    print(f'\nStopped at itr {j+1}\n')
                    break


            self.C[:, k] = self.theta[:, 1]
            self.B[:, k] = self.theta[:, 0]
            curr_t = self.Tinit + self.dt*k
            # Plotting animation
            plt.clf()
            plt.plot(self.mesh, self.u(self.mesh, curr_t),linewidth=2,label=r'$u$')
            plt.plot(self.mesh,self.vectorized_NN_1D(self.theta,self.mesh, k),linewidth=2,label= r'$u_n$', color = 'red')
            plt.scatter(self.theta[1:, 0], min(self.vectorized_NN_1D(self.theta,self.mesh, k))*np.ones(len(self.theta[1:, 0])),marker='|', color='green', label='Breakpoints')  
            plt.title(f"t={curr_t:.5f}")
            plt.grid(True)
            plt.legend()
            plt.pause(0.01)

        # Reporting current errors 
            print(f'Current t:{curr_t}')
            curr_err_h1 = self.seminorm_error(self.theta, curr_t)
            curr_err_l2 = self.L2_error(self.theta, curr_t, k)
            normC = np.linalg.norm(self.gradC(self.theta, theta_prev, k))
            normB = np.linalg.norm(self.theta[1:, 1]*self.gradB(self.theta, theta_prev, k))
            print(f'H1 error: {curr_err_h1:.3e}')
            print(f'L2 error: {curr_err_l2:.3e}')
            print(f'Norm of b: {normB:.3e}')
            print(f'Norm of c: {normC:.3e}')




    # Redistributing neurons
    def redistribute_neurons(self, theta_prev, k, tau_1):
        # First we clip
        mask0 = self.theta[:, 0] < self.a
        mask0 = np.roll(mask0, 1)
        self.theta[:,0] = np.clip(self.theta[:, 0], self.a, None)
        mask = np.abs(np.roll(self.theta[:, 0], 1) - self.theta[:, 0]) < 1e-12
        j = 0
        while j < self.N -1:
            if j+2 < self.N and mask[j+2] and mask[j+1] and mask[j]:
                mask[j+2] = 0
                mask[j] = 0
                j +=2 
            elif mask[j] and mask[j+1]:
                mask[j] = 0
                j+=1 
            else:
                j+=1
        # Get the indices of columns that satisfy the condition
        indices_to_move = np.where((self.theta[:, 0] >= self.b) | mask |mask0| (np.abs(self.theta[:, 1]) < tau_1) | (np.isnan(self.theta[:, 0])))[0]
        if len(indices_to_move) > 0 and indices_to_move[0]==0:
            indices_to_move = indices_to_move[1:]
        theta_aux = np.delete(self.theta, indices_to_move, axis=0)
        random_count = np.zeros(self.N, dtype=int)
        if len(theta_aux[:, 0]) < 2:
            self.theta[:, 0] = np.linspace(self.a, self.b, self.N+2)[0: self.N+1]
            return None
        random_list = self.rng.integers(low=0, high=len(theta_aux[:, 0]) - 1,size=len(indices_to_move),dtype=int)
        i = 0
        for j in range(len(indices_to_move)):
            q = random_list[i]
            m = random_count[q] 
            self.theta[indices_to_move[j], 0] = theta_aux[q, 0] + self.two_adic_integers[m]*(theta_aux[q+1, 0]- theta_aux[q, 0])
            random_count[q]+=1
            i+=1
        if len(indices_to_move) > 0:
            sorted_indices = np.argsort(self.theta[:, 0])
            self.theta = self.theta[sorted_indices]
            for j in range(25):
                gradC = self.gradC(self.theta, theta_prev, k)
                norm_grad = np.linalg.norm(gradC)
                if np.linalg.norm(gradC) < 1e-12:
                    break
                pc = -self.solveC(self.theta, gradC, k)
                self.theta[:, 1] = self.theta[:, 1] + pc 



    def plot(self):
        plt.figure(figsize=(5,5))
        plt.plot(self.mesh, self.u(self.mesh, self.Tfinal),linewidth=2,label=r'$u$')
        plt.plot(self.mesh,self.vectorized_NN_1D(self.theta,self.mesh, self.K),linewidth=2,label= r'$u_n$', color = 'red')
        plt.scatter(self.theta[1:, 0], min(self.vectorized_NN_1D(self.theta,self.mesh, self.K))*np.ones(len(self.theta[1:, 0])),marker='|', color='green', label='Breakpoints')       
        plt.legend()
        plt.show()


    def plot_3d(self):
            # Create a meshgrid for the 3D plot
            X, T = np.meshgrid(self.mesh, self.t)
            
            # Initialize the arrays for the Z-axis (solutions)
            Z_ref = np.zeros((self.K + 1, self.n))
            Z_num = np.zeros((self.K + 1, self.n))
            
            # Populate the Z arrays time step by time step
            for k in range(self.K + 1):
                t_val = self.t[k]
                
                # 1. Reference solution
                Z_ref[k, :] = self.u(self.mesh, t_val)
                
                # 2. Numerical solution (reconstructing theta from B and C)
                theta_k = np.zeros((self.N + 1, 2))
                theta_k[:, 0] = self.B[:, k]
                theta_k[:, 1] = self.C[:, k]
                
                Z_num[k, :] = self.vectorized_NN_1D(theta_k, self.mesh, k)

            # --- First Figure: Reference Solution ---
            fig1 = plt.figure(figsize=(8, 6))
            ax1 = fig1.add_subplot(111, projection='3d')
            surf1 = ax1.plot_surface(X, T, Z_ref, cmap='plasma', edgecolor='none', alpha=0.9)
            ax1.set_xlabel('x', fontsize=12)
            ax1.set_ylabel('t', fontsize=12)
            ax1.set_zlabel(r'$u(x, t)$', fontsize=12)
            fig1.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
            fig1.tight_layout()
            
            # --- Second Figure: Numerical Solution ---
            fig2 = plt.figure(figsize=(8, 6))
            ax2 = fig2.add_subplot(111, projection='3d')
            surf2 = ax2.plot_surface(X, T, Z_num, cmap='plasma', edgecolor='none', alpha=0.9)
            ax2.set_xlabel('x', fontsize=12)
            ax2.set_ylabel('t', fontsize=12)
            ax2.set_zlabel(r'$u_n(x, t)$', fontsize=12)
            fig2.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
            fig2.tight_layout()
            
            # Display both figures in separate windows
            plt.show()