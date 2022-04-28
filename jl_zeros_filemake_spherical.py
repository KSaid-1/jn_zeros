#--------------------------------------#
#---Compute zeros of half-integer order Bessel functions Jn------#
#---That's the so called zeros of spherical Bessel function------#
import scipy
from scipy import special as ss
from scipy import optimize
#--------------------------------#
lmax = 15
nmax = 32
#--------------------------------#
def f1(n):
    return (ss.spherical_jn(l,n,derivative=False))
def f1prime(n):
    return (scipy.special.spherical_jn(l,n,derivative=True))
#-------------------------------------------------------------#
for l in range(0,lmax+1):
    for i in range(1,nmax+1):
        x1 = (ss.jn_zeros(l,i)[i-1]) - 2
        x2 = (ss.jn_zeros(l,i)[i-1]) + 2
        x1prime = (ss.jnp_zeros(l,i)[i-1]) - 2
        x2prime = (ss.jnp_zeros(l,i)[i-1]) + 2
        sol = optimize.root_scalar(f1, bracket=[x1, x2], method='brentq')
        solprime = optimize.root_scalar(f1prime, bracket=[x1prime, x2prime], method='brentq')
        print(l,i,sol.root,solprime.root)
#---------------------------------------------------------------------------------------------#
