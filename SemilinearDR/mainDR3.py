from DR3_D import *
from DR3_N import *


eps = np.sqrt(1e-7)

myDR3 = DR3_N(N = 20,u_id = 2, a = -1, b= 1, eps = eps)
myDR3.Newton(max_itr = 0, lr = 1, tau_1 = 1e-8)
myDR3.plot_no_vlines()

#myDR3 = DR3_N(N = 40,u_id = 3, a = 0, b= 1, eps = 1e-4)
#myDR3.Newton(max_itr = 200, lr = 1, tau_1 = 1e-8)
#myDR3.plot_no_vlines()