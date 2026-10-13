from LS import*
from AC_CSS_DB import*
from AC_CSS_NB import*
from AC_BDF2_NB import*
from AC_BDF2_DB import*


## Second test problem
N = 25
# 1 or 2
u_id = 1

eps = np.sqrt(1e-7)
a, b = 0, 1


myLS = LS(N = N, a  = a, b = b, u_id = u_id, eps = eps)
myLS.Newton(max_itr = 500, ls = False,stop = True, stopping_tol = 1e-6)
#myLS.plot_no_vlines()
theta_init = myLS.theta


#K = int(np.round((1/s)*(1/10)*(1/1e-2)))

theta_init = myLS.theta
myAL =  AC_CSS_DB(N = N, K = 150, a = a, b = b, u_id = u_id, Tfinal = 1.5, eps = eps)
#myAL =  AC_BDF2_DB(N = N, K = 250, a = a, b = b, u_id = u_id, Tfinal = 2.5, eps = eps)

myAL.Newton(theta_init = theta_init,max_itr = 50, stop = False)
myAL.plot()
