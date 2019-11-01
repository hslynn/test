"""
handle boundary conditons
"""

import numpy as np
from dolfin import *
from global_def import *

def get_characteristic_speed(auxi_list):
    lapse = auxi_list[0]
    shift = auxi_list[1]
    gamma11 = auxi_list[4]
        
    #left bdry
    lapse_left = lapse.compute_vertex_values()[0]
    shift_left = shift.compute_vertex_values()[0]
    gamma11_left = gamma11.compute_vertex_values()[0]
    
    v0_left = -(1+paragamma1)*shift_left    
    v_minus_left = -shift_left + lapse_left*gamma11_left**0.5   
    v_plus_left = -shift_left - lapse_left*gamma11_left**0.5   

    #right bdry
    lapse_right = lapse.compute_vertex_values()[-1]
    shift_right = shift.compute_vertex_values()[-1]
    gamma11_right = gamma11.compute_vertex_values()[-1]
    
    v0_right = -(1+paragamma1)*shift_right    
    v_minus_right = -shift_right + lapse_right*gamma11_right**0.5   
    v_plus_right = -shift_right - lapse_right*gamma11_right**0.5   

    return ((v0_left, v_minus_left, v_plus_left),
            (v0_right, v_minus_right, v_plus_right))
    

def get_characteristic_field_values(var_list):
    #field values at left boundary
    left_var_values = [var.compute_vertex_values()[0] for var in var_list]
    g00_left, g01_left, g11_left = left_var_values[:3]
    Pi00_left, Pi01_left, Pi11_left = left_var_values[3:6]
    Phi00_left, Phi01_left, Phi11_left = left_var_values[6:9]

    u0_00_left = g00_left
    u0_01_left = g01_left
    u0_11_left = g11_left
 
    u_minus_00_left = -paragamma2*g00_left + Pi00_left + 1/g11_left**0.5*Phi00_left
    u_minus_01_left = -paragamma2*g01_left + Pi01_left + 1/g11_left**0.5*Phi01_left
    u_minus_11_left = -paragamma2*g11_left + Pi11_left + 1/g11_left**0.5*Phi11_left

    u_plus_00_left = -paragamma2*g00_left + Pi00_left - 1/g11_left**0.5*Phi00_left
    u_plus_01_left = -paragamma2*g01_left + Pi01_left - 1/g11_left**0.5*Phi01_left
    u_plus_11_left = -paragamma2*g11_left + Pi11_left - 1/g11_left**0.5*Phi11_left

    #field values at right boundary
    right_var_values = [var.compute_vertex_values()[-1] for var in var_list]
    g00_right, g01_right, g11_right = right_var_values[:3]
    Pi00_right, Pi01_right, Pi11_right = right_var_values[3:6]
    Phi00_right, Phi01_right, Phi11_right = right_var_values[6:9]

    u0_00_right = g00_right
    u0_01_right = g01_right
    u0_11_right = g11_right
 
    u_minus_00_right = -paragamma2*g00_right + Pi00_right + 1/g11_right**0.5*Phi00_right
    u_minus_01_right = -paragamma2*g01_right + Pi01_right + 1/g11_right**0.5*Phi01_right
    u_minus_11_right = -paragamma2*g11_right + Pi11_right + 1/g11_right**0.5*Phi11_right

    u_plus_00_right = -paragamma2*g00_right + Pi00_right - 1/g11_right**0.5*Phi00_right
    u_plus_01_right = -paragamma2*g01_right + Pi01_right - 1/g11_right**0.5*Phi01_right
    u_plus_11_right = -paragamma2*g11_right + Pi11_right - 1/g11_right**0.5*Phi11_right

    
    return ((u0_00_left, u0_01_left, u0_11_left, 
            u_minus_00_left, u_minus_01_left, u_minus_11_left,
            u_plus_00_left, u_plus_01_left, u_plus_11_left),
            (u0_00_right, u0_01_right, u0_11_right, 
            u_minus_00_right, u_minus_01_right, u_minus_11_right,
            u_plus_00_right, u_plus_01_right, u_plus_11_right)) 
    

def get_bdry_values(phi):
    #return [0, 0]
    return [phi.compute_vertex_values()[-1], phi.compute_vertex_values()[0]]


