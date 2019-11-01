"""
initial and gauge conditons for Schwarzschild spacetime in Kerr-Schild coordinate.
"""

from dolfin import *

def get_exact_phi(phi_exact, time):
    phi_exact_exp = Expression("1/x[0]+t*0",t=0.0, degree=10)
    phi_exact_exp.t = time
    phi_exact.interpolate(phi_exact_exp) 

def get_phi(phi):
    #phi_exp = Expression("sin(x[0])", degree=10)
    phi_exp = Expression("1/x[0]", degree=10)
    phi.interpolate(phi_exp)


