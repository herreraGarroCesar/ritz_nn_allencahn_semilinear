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
from scipy.sparse import diags



class DR3_D:
    def __init__(self, N, a = 0, b = 1, u_id = 1, eps = 1, gamma = 1e+4):
        # Number of interior mesh points
        self.N = N 
        # interval of interest [a,b]
        self.a = a
        self.b = b
        self.eps = eps
        self.gamma = gamma
        # example id (1,2, 3, 4...)
        self.u_id = u_id
        # Boundary and penalty parameters
        self.alpha = self.u(a)
        self.beta = self.u(b)
        # Neural net parametes 
        self.theta = np.zeros((N+1, 2))
        self.theta[:, 0] = np.linspace(self.a, self.b, self.N+2)[0: self.N+1]
        # For errors
        self.h1_u = self.H1_seminorm_u()
        self.h1_error = []
        self.residual = []    
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




    def u(self, x):
        match self.u_id:
            case 1:
                return x*(np.exp((-(x-1/3)**2)/0.01) - np.exp((-4/9)/0.01))
            case 2:
                return np.tanh((x**2 - 1/4)/self.eps)-np.tanh(3/(4*self.eps)) +1
            case _:
                return x*(np.exp((-(x-1/3)**2)/0.01) - np.exp((-4/9)/0.01))

        
    def u_prime(self, x):

        match self.u_id:
            case 1:
                return (np.exp(-100*(x-1/3)**2))*(-200*(x**2)+(200/3)*x+1)
            case 2:
                return ((1/np.cosh((x**2 - 1/4)/self.eps))**2)*(2*x/self.eps) 
            case _:
                return (np.exp(-100*(x-1/3)**2))*(-200*(x**2)+(200/3)*x+1)


    def f(self, x): 
        match self.u_id:
            case 1:
                return  -(self.eps**2)*40000*(x**3 -2*(x**2)/3 + 173*x/1800 + 1/300)*np.exp(-100*(x-1/3)**2) + (self.u(x))**3 + self.u(x)
            case 2:
                return -2*(self.eps - 4*(x**2)*np.tanh((x**2 - 1/4)/self.eps))*(1/np.cosh((x**2 -1/4)/self.eps))**2 + (self.u(x))**3 + self.u(x)
            case _:
                return  -(self.eps**2)*40000*(x**3 -2*(x**2)/3 + 173*x/1800 + 1/300)*np.exp(-100*(x-1/3)**2) + (self.u(x))**3 + self.u(x)

    def relu(self, x):
        return np.maximum(0, x)


    def NN_1D(self,theta, x):
        c = theta[:, 1] # c0,...,cN = theat_0,....,theta_N
        b = theta[:, 0]  # b0,...,bN = theta_(N+1),...,theta_(2N+1)
        y = self.relu(x-b) 
        return np.dot(c, y) + self.alpha 

    # to be able to put in a vector of x-values, used for plotting
    def vectorized_NN_1D(self, theta, x):
        vectorized_NN1D = np.vectorize(self.NN_1D,excluded=[0])
        return vectorized_NN1D(theta, x)


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



    ## Relative errors 

    # H1 seminorm of the true solution
    def H1_seminorm_u(self):
        points = np.concatenate([np.linspace(-0.55, -0.48, 150), np.linspace(0.48, 0.55, 150)])
        integrand = lambda x: (self.u_prime(x))**2
        return np.sqrt(scipy.integrate.quad(integrand,self.a,self.b, limit = 400,points = points)[0])



    # H1 seminorm of u_n
    def H1_seminorm_un(self, theta):
        h = self.h_vector(theta[:, 0])
        d = self.derivatives_vector(theta)
        c = theta[:, 1]
        b = np.append(theta[:, 0], [self.b])
        error = 0
        for i in range(self.N+1):
            error += h[i]*(d[i]**2)
        return np.sqrt(error)

    # H1 seminorm error 
    def seminorm_error(self, theta):
        d = self.derivatives_vector(theta)
        c = theta[:, 1]
        b = np.append(theta[:, 0], [self.b])
        error =   self.H1_seminorm_un(theta)**2 + self.h1_u**2
        for i in range(self.N+1):
            error -=2*d[i]*(self.u(b[i+1]) -self.u(b[i]))
        return np.sqrt(error)
    


    # Some subroutines to be used

    # This vector contains the derivatives on each subinterval of the NN
    def derivatives_vector(self, theta):
        d = np.zeros(self.N + 1)
        c = theta[:, 1]
        for i in range(self.N+1):
            d[i] = np.sum(c[0:i+1])
        return d


    # For moving the c's 
    def h_vector(self, b):
        h = np.zeros(self.N+1)
        h[self.N] = self.b - b[self.N]
        for i in range(1, self.N+1):
            h[i-1] = b[i] - b[i-1] 
        return h    


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


    #Evaluation of (A+uvT)^-1(x)
    def shermanMorrison(self, Ax, Au, v):
        return Ax - (np.inner((Ax), v)/(1 + np.inner(v, Au)))*Au


    # Used to construct the rank-1 matrix in the linear system
    def d_vector(self, b):
        d = np.zeros(self.N+1)
        for i in range(self.N+1):
            d[i] = self.b-b[i]
        return d


    # This is to solve A^{-1}x = y, where A is the mass matrix 
    def A_inv(self, b, y):
        h = self.h_vector(b)
        diag1 = np.zeros(self.N)
        diag = np.zeros(self.N+1) 
        for i in range(self.N):
            # Upper diagonal and lower diagonal
            diag1[i] = - 1/h[i]
            diag[i+1] += 1/h[i] + 1/h[i+1]
        diag[0] = 1/h[0]   
        return self.TDMAsolver(diag1, diag, diag1, y)




    # For the contruction of the matrices   

    def s0_vector(self, theta):
        s0 = np.zeros(self.N+1);
        h = self.h_vector(theta[:, 0])
        b = np.append(theta[:, 0], [self.b])
        f = lambda x: 3*(self.NN_1D(theta, x))**2 + 1
        for i in range(self.N+1):
            s0[i] = (h[i]/6)*(f(b[i]) + 4*f((b[i+1]+b[i])/2) + f(b[i+1]))
        return s0

    def s1_vector(self, theta):
        s1 = np.zeros(self.N+1);
        h = self.h_vector(theta[:, 0])
        b = np.append(theta[:, 0], [self.b])      
        for i in range(self.N+1):
            f = lambda x: (x-b[i])*(3*(self.NN_1D(theta, x)**2)  + 1) 
            s1[i] = (1/8)*(f(b[i]) + 3*f((2*b[i] + b[i+1])/3) + 3*f((b[i] + 2*b[i+1])/3) + f(b[i+1]))
        return s1

    def s2_vector(self, theta):
        s2 = np.zeros(self.N+1);
        h = self.h_vector(theta[:, 0])
        b = np.append(theta[:, 0], [self.b])      
        for i in range(self.N+1):
            f = lambda x: ((x-b[i])**2)*(3*(self.NN_1D(theta, x)**2)  + 1) 
            s2[i] = (1/(90*h[i]))*(7*f(b[i]) + 32*f(b[i] + h[i]/4) + 12*f(b[i] + h[i]/2) + 32*f(b[i] + 3*h[i]/4) + 7*f(b[i+1]))
        return s2
 
 


    # This is one component of the mass matrix, b is the mesh and y is a rhs vector 
    def M1_inv(self, theta, y):
        b = theta[:, 0]    
        h = self.h_vector(b)
        diag1 = np.zeros(self.N)
        diag = np.zeros(self.N+1)  
        s0 = self.s0_vector(theta)
        s1 = self.s1_vector(theta)
        s2 = self.s2_vector(theta) 
        eps2 = self.eps**2
        for i in range(self.N):
            # Upper diagonal and lower diagonal
            diag1[i] = s1[i+1] - s2[i+1] - (eps2)/h[i+1]
            # Main diagonal 
            diag[i] = s0[i+1] - 2*s1[i+1] + s2[i] + s2[i+1] + (eps2)/h[i] + (eps2)/h[i+1]
        # Especial entries
        diag[self.N] = s2[self.N] + eps2/h[self.N]    
        return self.TDMAsolver(diag1, diag, diag1, y)


    # This is for the initial
    def M2_inv(self, b, y):
        h = self.h_vector(b)
        diag1 = np.zeros(self.N)
        diag = np.zeros(self.N+1)
        eps2 = self.eps**2     
        for i in range(self.N):
            # Upper diagonal and lower diagonal
            diag1[i] = (1/2)*(1/6)*h[i+1] - eps2/h[i+1]
            # Main diagonal 
            diag[i] = (1/2)*(2/3)*(h[i]+h[i+1]) + (eps2)*(1/h[i] + 1/h[i+1])
        # Especial entries
        diag[self.N] += (2/3)*h[self.N]
        diag[self.N] += eps2/h[self.N]   
        return self.TDMAsolver(diag1, diag, diag1, y)



    def G_matrix(self):
        main_diag = np.ones(self.N+1)       # diagonal of ones
        sub_diag  = -np.ones(self.N)        # sub-diagonal of -1
        G = diags([main_diag, sub_diag], offsets=[0, -1], format="csr")
        return G


    # Linear parameter, initial 
    def init_fc(self, b):
        f_c = np.zeros(self.N+1)
        d = self.d_vector(b)
        h = self.h_vector(b)
        b = np.append(b, [self.b])
        eps2 = self.eps**2
        for i in range(self.N+1):
            if self.u_id ==2:
                points = np.concatenate([np.linspace(-0.55, -0.48, 150), np.linspace(0.48, 0.55, 150)])
                filtered_points = points[(points > b[i])]
                integrand = lambda x: (self.f(x)- self.alpha)*(x-b[i]) 
                f_c[i] += integrate.quad(integrand,b[i],self.b, limit = 400,points = filtered_points)[0]
            else:    
                integrand = lambda x: (self.f(x)- self.alpha)*(x-b[i])
                f_c[i] += integrate.quad(integrand,b[i],self.b)[0]
        return f_c

    # y is the rhs vector, b is the mesh 
    def initSolverC(self, b, y):
        G = self.G_matrix()
        h_inv = 1/self.h_vector(b)
        x = G.T@(h_inv*(G.T@y))
        x = self.M2_inv(b, x)
        return  G@(h_inv*(G@x))



    # Gradient
    def gradC(self, theta):
        gradC = np.zeros(self.N+1)
        d = self.d_vector(theta[:, 0])
        h = self.h_vector(theta[:,0])
        b = np.append(theta[:, 0], [self.b])
        c = theta[:, 1]
        eps2 = self.eps**2
        gradC += eps2*self.A_inv(b, c)
        for i in range(self.N+1):
            f = lambda x: (x-b[i])*(self.NN_1D(theta, x)**3 + self.NN_1D(theta, x))
            for j in range(i, self.N+1):
                gradC[i] += (h[j]/90)*(7*f(b[j]) + 32*f(b[j] + h[j]/4) + 12*f(b[j] + h[j]/2) + 32*f(b[j] + 3*h[j]/4) + 7*f(b[j+1]))
        for i in range(self.N +1):     
            integrand = lambda x: self.f(x)*(x-b[i])
            gradC[i] -= integrate.quad(integrand,b[i],self.b)[0]
        gradC += self.gamma*(self.NN_1D(theta, self.b) - self.beta)*d
        return gradC


    # y is the rhs vector, b is the mesh 
    def solveC(self, theta, y):
        G = self.G_matrix()
        b = theta[:, 0] 
        d = self.d_vector(b)   
        h_inv = 1/self.h_vector(b)
        x = G.T@(h_inv*(G.T@y))
        x = self.M1_inv(theta, x)
        Ax = G@(h_inv*(G@x))
        u = G.T@(h_inv*(G.T@d))
        u = self.M1_inv(theta, u)
        Au = G@(h_inv*(G@u))
        return self.shermanMorrison(Ax, Au, self.gamma*d)




    # Non-linear parameters 

    # Gradient
    def gradB(self, theta):
        h = self.h_vector(theta[:, 0])
        c = theta[:, 1]  # c0, c1,...,cN 
        b = np.append(theta[:, 0], [self.b])    # b0,...,bN+1     
        gradB = np.zeros(self.N)
        eps2 = self.eps**2
        ones = np.ones(self.N)
        d = self.derivatives_vector(theta)
        for i in range(self.N):
            gradB[i] -= eps2*(d[i] + c[i+1]/2) 
        f = lambda x: self.NN_1D(theta, x)**3  + self.NN_1D(theta, x)
        for i in range(1, self.N+1):
            for j in range(i, self.N+1):
                gradB[i-1]-= (h[j]/8)*(f(b[j]) + 3*f((2*b[j] + b[j+1])/3) +  3*f((b[j] + 2*b[j+1])/3) +f(b[j+1]))            
        for i in range(self.N):
            if self.u_id == 2:
                    points = np.concatenate([np.linspace(-0.55, -0.48, 150), np.linspace(0.48, 0.55, 150)])
                    filtered_points = points[( b[i+1] < points)]
                    integrand = lambda x: self.f(x)  
                    gradB[i] += (integrate.quad(integrand,b[i+1],self.b, limit = 400,points = filtered_points)[0])
            else:
                integrand = lambda x: self.f(x)
                gradB[i] += (integrate.quad(integrand,b[i+1],self.b)[0])
        gradB += self.gamma*(self.NN_1D(theta, self.b) - self.beta)*ones
        return gradB



    # Assume that B is sorted
    def coefficientMatrixInv(self, theta):
        s = self.s0_vector(theta)
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
    

    def hessianSolver(self, theta, y,tau_1):
        A_inv = self.coefficientMatrixInv(theta)
        Lambda = np.zeros(self.N)
        I = np.eye(self.N)
        h = self.h_vector(theta[:, 0])
        b = theta[1:, 0]
        c = theta[1:, 1]
        d_inv = np.where((np.abs(c) > tau_1), 1.0 / c, 0.0)
        ones = np.ones(self.N)
        for i in range(self.N):
            if not np.isnan(self.f(b[i])) and not np.isinf(self.f(b[i])) and np.abs(self.f(b[i])) < 1e+16:
                if np.abs(c[i]) > tau_1:
                    Lambda[i] = self.NN_1D(theta, b[i])**3 + self.NN_1D(theta, b[i])-self.f(b[i])            
        x = d_inv*(A_inv@(y))
        u = d_inv*(A_inv@(self.gamma*ones))
        a_diag  = A_inv.diagonal(0)   # main diagonal (length n)
        a_lower = A_inv.diagonal(-1)  # sub-diagonal (length n-1)
        a_upper = A_inv.diagonal(1)   # super-diagonal (length n-1)
        diag1, diag, diag2 = self.tridiag_transform(a_lower, a_diag, a_upper, d_inv, Lambda)
        Ax = self.TDMAsolver(diag1, diag, diag2, x)
        Au = self.TDMAsolver(diag1, diag, diag2, u)
        return self.shermanMorrison(Ax, Au, c)









    # Newton's method
    def Newton(self, max_itr = 100, lr = 1, tau_1 = 1e-5):

        self.theta[:, 1] = self.initSolverC(self.theta[:, 0], self.init_fc(self.theta[:, 0]))
        # Initial c's
        for j in range(50):
            c = self.theta[:,1]
            gradC = self.gradC(self.theta)
            print(np.linalg.norm(gradC))
            if np.linalg.norm(gradC) < 1e-12:
                break
            self.theta[:, 1] = c - self.solveC(self.theta, gradC)


        # Initial error
        self.h1_error.append(self.seminorm_error(self.theta)/self.h1_u)
        rate = -np.log(self.h1_error[-1])/np.log(self.N+1)  
        grad =  np.concatenate((self.gradC(self.theta), self.theta[1:,1]*self.gradB(self.theta)))
        self.residual.append(np.linalg.norm(grad)) 


        # Report 
        print('Newton...')
        print(f'Initial residual: {self.residual[-1]}')
        print(f'Initial seminorm error: {self.h1_error[-1]}')
        print(f'Initial rate: {rate}')

        # Main algorithm
        for i in range(max_itr):
            # Newton's step 
            b = self.theta[:, 0]
            gradB = self.gradB(self.theta)
            pk = -self.hessianSolver(self.theta, gradB, tau_1= tau_1)
            self.theta[1:, 0] = b[1:] + lr*pk
            # Projecting and sorting 
            sorted_indices = np.argsort(self.theta[:, 0])
            self.theta = self.theta[sorted_indices]  
            self.redistribute_neurons(tau_1 = tau_1)
            sorted_indices = np.argsort(self.theta[:, 0])
            self.theta = self.theta[sorted_indices]
            # Initial c's
            c = self.theta[:,1]
            gradC = self.gradC(self.theta)
            self.theta[:, 1] = c - self.solveC(self.theta, gradC)
            gradC = self.gradC(self.theta)



            # Projecting and sorting
            self.h1_error.append(self.seminorm_error(self.theta)/self.h1_u)

            grad =  np.concatenate((self.gradC(self.theta), self.theta[1:,1]*self.gradB(self.theta)))
            self.residual.append(np.linalg.norm(grad)) 

            print(f'Itr: {i+1}')
            print(f'Err: {self.h1_error[-1]:.3e}')
            print(f'Residual: {self.residual[-1]:.3e}')




        # Rate and h1 error     
        grad =  np.concatenate((self.gradC(self.theta), self.theta[1:,1]*self.gradB(self.theta)))
        self.residual.append(np.linalg.norm(grad))     
        rate = -np.log(self.h1_error[-1])/np.log(self.N+1)



        # Report 
        print(f'Final residual: {self.residual[-1]:.3e}')
        print(f'Final seminorm error: {self.h1_error[-1]:.3e}')
        print(f'Final rate: {rate:.3f}')




    # Redistributing neurons
    def redistribute_neurons(self, tau_1):
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
                gradC = self.gradC(self.theta)
                norm_grad = np.linalg.norm(gradC)
                if np.linalg.norm(gradC) < 1e-12:
                    break
                pc = -self.solveC(self.theta, gradC)
                self.theta[:, 1] = self.theta[:, 1] + pc 


    ## For plotting
    def plot_error(self):
        plt.figure(figsize=(7.5, 7.5))
        plt.plot(np.arange(len(self.h1_error)), self.h1_error, linewidth=2)
        plt.xlabel('Newton iteration')
        plt.ylabel('Relative seminorm error')
        plt.yscale('log')
        plt.show()

    def plot_no_vlines(self):
        plt.figure(figsize=(5,5))
        mesh = np.linspace(self.a, self.b, 2000)
        plt.plot(mesh, self.u(mesh),linewidth=2,label=r'$u$')
        plt.plot(mesh,self.vectorized_NN_1D(self.theta,mesh),linewidth=2,label= r'$u_n$', color = 'red')
        plt.scatter(self.theta[1:, 0], min(self.vectorized_NN_1D(self.theta,mesh))*np.ones(len(self.theta[1:, 0])),marker='|', color='green', label='Break points')       
        plt.legend()
        plt.show()




    def plot(self): 
        plt.figure(figsize=(5,5))
        mesh = np.linspace(self.a, self.b, 2000)
        plt.plot(mesh, self.u(mesh),linewidth=2,label=r'$u$')
        plt.plot(mesh,self.vectorized_NN_1D(self.theta,mesh),linewidth=2,label= r'$u_n$')
        plt.vlines(self.theta[:,0],ymin= min(self.u(mesh)),ymax= max(self.u(mesh)), linestyles='dotted',colors='gray')
        plt.scatter(self.theta[:, 0], min(self.u(mesh))*np.ones(self.N+1), marker='x', color='red', label='Break points')       
        plt.legend()
        plt.show()


