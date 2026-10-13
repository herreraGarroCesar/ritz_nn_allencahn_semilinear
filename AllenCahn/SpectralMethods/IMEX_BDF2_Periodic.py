import numpy as np
import matplotlib.pyplot as plt

# Parameters
shift = 2.0  # stability shift parameter (>1.6 stable, <1.5 unstable)

# Time parameters
t = 0.0
dt = 1e-3
tmax =  2.5
dt23 = 2 * dt / 3
ntmax = int(tmax / dt)

# Spatial parameters
L = 1.0
n = 508
x = np.linspace(0.0, L, n, endpoint=True)  # no duplicate endpoint
dx = L / (n-1)
# Allen–Cahn parameters
eps = np.sqrt(1e-5)
eps2 = eps**2

# Compute Laplacian operator in Fourier space (fractional if s != 1)
waven = np.fft.fftfreq(n, d=dx)  # gives k / L
laplace = -(2 * np.pi * waven)**2

# Precompute scheme denominator
sch = 1.0 - dt23 * (eps2 * laplace - shift)

# Initial condition: 
#u = x**2 * np.cos(np.pi * x)
#u=   0.1*np.sin(2*np.pi*x)
u = np.tanh((x-0.25)/(np.sqrt(2)*eps)) - np.tanh((x-0.75)/(np.sqrt(2)*eps)) -1   + 0.1*np.sin(2*np.pi*x)
#u =  x * (np.exp((-(x - 1/3)**2) / 0.01) - np.exp((-4/9) / 0.01))
   
# FFT of initial condition
hat_u = np.fft.fft(u)
hat_u2 = hat_u.copy()

# Compute shifted nonlinear term
fu = u**3 - (1 + shift) * u
hat_fu = np.fft.fft(fu)
hat_fu2 = hat_fu.copy()

# Main time-stepping loop
for it in range(ntmax+1):
    if it % 20 == 0:
        print(f"it={it}, t={t:.2f}, max|u|={np.max(np.abs(u)):.3f}")
        plt.clf()
        plt.plot(x, u, label=f"t={t:.2f}")
        plt.grid(True)
        plt.legend()
        plt.pause(0.001)

    # Time update
    t += dt

    # Compute RHS in Fourier space
    hat_rhs = (4 * hat_u - hat_u2) / 3 - dt23 * (2 * hat_fu - hat_fu2)
    hat_u2 = hat_u.copy()
    hat_fu2 = hat_fu.copy()

    # Update solution in Fourier space
    hat_u = hat_rhs / sch

    # Transform back to physical space
    u = np.real(np.fft.ifft(hat_u))
    

    # Compute shifted nonlinear term
    fu = u**3 - (1 + shift) * u
    hat_fu = np.fft.fft(fu)

plt.show()
