from LS import*
from AC_CSS_DB import*
from AC_CSS_NB import*
from AC_BDF2_NB import*


## First test problem
N = 30
u_id = 2

eps = np.sqrt(1e-5)
a, b = 0, 1


myLS = LS(N = N, a  = a, b = b, u_id = u_id, eps = eps)
myLS.Newton(max_itr = 500, ls = False,stop = True, stopping_tol = 1e-6)
#myLS.plot_no_vlines()
theta_init = myLS.theta



theta_init = myLS.theta
myAL = AC_CSS_NB(N = N, K = 10000, a = a, b = b, u_id = u_id, Tfinal = 1.0, eps = eps)
#myAL = AC_BDF2_NB(N = N, K = 100, a = a, b = b, u_id = u_id, Tfinal = 1.0, eps = eps)
myAL.Newton(theta_init = theta_init,max_itr = 50, stop = False, stopping_tolerance = 1e-5)
myAL.plot()
