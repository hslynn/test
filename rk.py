from dolfin import *
import hdw
import bdry

def rk2(phi, exact_phi, temp_phi, deri, Hhat, dt):
    temp_phi.assign(phi)
    for dummy in range(2): 
        bdry_values = bdry.get_bdry_values(exact_phi)
        hdw.get_deri(deri[0], phi, bdry_values, 0, "+")
        hdw.get_deri(deri[1], phi, bdry_values, 0, "-")
        
        dt_form = phi - dt*Hhat
        project(dt_form, phi.function_space(), function=phi)
    final_form = 0.5*(temp_phi + phi)
    project(final_form, phi.function_space(), function=phi)

def rk3(phi, exact_phi, temp_phi, deri, Hhat, dt):
    temp_phi.assign(phi)
    #compute u1, stored in var_list
    bdry_values = bdry.get_bdry_values(exact_phi)
    hdw.get_deri(deri[0], phi, bdry_values, 0, "+")
    hdw.get_deri(deri[1], phi, bdry_values, 0, "-")
    
    u1_form = phi - dt*Hhat
    project(u1_form, phi.function_space(), function=phi)

    #compute u2, stored in var_list
    bdry_values = bdry.get_bdry_values(exact_phi)
    hdw.get_deri(deri[0], phi, bdry_values, 0, "+")
    hdw.get_deri(deri[1], phi, bdry_values, 0, "-")

    u2_form = 3.0/4.0*temp_phi + 1.0/4.0*phi - 1.0/4.0*dt*Hhat
    project(u2_form, phi.function_space(), function=phi)

    #compute final u, stored in var_list
    bdry_values = bdry.get_bdry_values(exact_phi)
    hdw.get_deri(deri[0], phi, bdry_values, 0, "+")
    hdw.get_deri(deri[1], phi, bdry_values, 0, "-")

    final_form = 1.0/3.0*temp_phi + 2.0/3.0*phi - 2.0/3.0*dt*Hhat
    project(final_form, phi.function_space(), function=phi)
