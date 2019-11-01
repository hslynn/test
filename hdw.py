"""
help functions
"""

import numpy as np
from dolfin import *
from global_def import *

def get_deri(u_deri, u, bdry_values, i, mark):
    func_space = u.function_space() 
    mesh = func_space.mesh()
    p = TrialFunction(func_space)
    v = TestFunction(func_space)
    n = FacetNormal(mesh)

    term_cell = p*v*dx + u*v.dx(i)*dx

    left_bdry, right_bdry = [Constant(value) for value in bdry_values] 
    if mark == '+':
        term_inner_facet = - n("+")[i]*avg(u)*jump(v)*dS + 0.5*abs(n("+")[i])*jump(u)*jump(v)*dS
        term_boundary = -(0.5*(right_bdry + u)*n[i] + 0.5*(right_bdry - u)*abs(n[i]))*v*ds
    elif mark == '-':
        term_inner_facet = - n("+")[i]*avg(u)*jump(v)*dS - 0.5*abs(n("+")[i])*jump(u)*jump(v)*dS
        term_boundary = -(0.5*(left_bdry + u)*n[i] - 0.5*(left_bdry - u)*abs(n[i]))*v*ds
        
    F = term_cell + term_inner_facet + term_boundary
    a, L = lhs(F), rhs(F)
    solve(a == L, u_deri)

def project_functions(func_forms, func_list):
    for idx in range(len(func_forms)):
        project(func_forms[idx], func_list[idx].function_space(), function=func_list[idx])
        #Func_space = func_list[idx].function_space()
        #U = TrialFunction(func_space)
        #V = TestFunction(func_space)
        #U_form = func_forms[idx]
        #If u_form == 0:
        #    u_form = Constant(0) 
        #F = u*v*dx - u_form*v*dx
        #A, L = lhs(F), rhs(F)
        #Solve(a == L, func_list[idx])

def get_Hhat(deri, Au):
    avg_deri = 0.5*(deri[0]+deri[1]) 
    dif_deri = 0.5*(deri[0]-deri[1]) 

    Hhat = Au*avg_deri - dif_deri

    return Hhat

