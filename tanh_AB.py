from __future__ import print_function
from dolfin import *
import matplotlib.pyplot as plt
import rk
import sys
import getopt
import time

from hdw import *
import bdry
import init
import mesh_generate as mg
import ioo
import init

parameters["ghost_mode"] = "shared_vertex"

def main():
    """
    main computating process
    """

    N = 9
    DG_degree = 1
    inner_bdry = 0.5
    mesh_len = 10 
    mesh_num = 101
    hmin = 0.1 
    hmax = 0.5
    mg_func = 1
    mg_order = 1.0
    refine_time = 0
    folder = ''

    opts, dumps = getopt.getopt(sys.argv[1:], "-n:-m:-d:-i:-h:-x:-r:-o:-l:-f:")
    for opt, arg in opts:
        if opt == "-m":
            mg_func = int(arg)
        if opt == "-n":
            mesh_num = int(arg)
        if opt == "-d":
            DG_degree = int(arg)
        if opt == "-i":
            inner_bdry = float(arg)
        if opt == "-l":
            mesh_len = float(arg)
        if opt == "-h":
            hmin = float(arg)
        if opt == "-x":
            hmax = float(arg)
        if opt == "-o":
            mg_order = float(arg)
        if opt == "-r":
            refine_time = int(arg)
        if opt == "-f":
            folder = str(arg)
    
    #create .sh for rerun purpose
    with open(folder+'run.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('cd ~/projects/test/\n')
        f.write('python3 sin.py ')
        f.write('-m %d -d %d -i %f -l %.14f -n %d -r %d -f %s'
                %(mg_func, DG_degree, inner_bdry, mesh_len, mesh_num, refine_time, folder))

    #create mesh and define function space
    #mesh = mg.get_mesh(inner_bdry, mesh_len, hmin, hmax, mg_func, mg_order)
    mesh = IntervalMesh(mesh_num,inner_bdry,inner_bdry+mesh_len)
    for dummy in range(refine_time):
        mesh = refine(mesh)
    print(mesh.hmin())
    print(mesh.hmax())
    print(mesh.num_vertices())
    print(mesh.coordinates())
    plot(mesh)
    plt.show()

    #save configurations to file
    with open(folder+'config.txt', 'w') as f:
        f.write("#configuratons used in this test\n\n")
        f.write("parameters to generate mesh:\n")
        f.write("inner_bdry = "+str(inner_bdry)+'\n')
        f.write("mesh_len = "+str(mesh_len)+'\n')
        f.write("mg_func = "+str(mg_func)+'\n')
        f.write("mg_order = "+str(mg_order)+'\n')
        f.write("hmin = "+str(hmin)+'\n')
        f.write("hmax = "+str(hmax)+'\n')
        f.write("refine_time = "+str(refine_time)+'\n\n')
        f.write("resulted mesh:\n")
        f.write("num_vertices = "+str(mesh.num_vertices())+'\n')
        f.write("minimum of h = "+str(mesh.hmin())+'\n')
        f.write("maximum of h = "+str(mesh.hmax())+'\n\n')
        f.write("others:\n")
        f.write("DG_degree = "+str(DG_degree))

    hmin = mesh.hmin()
    dt = 0.8*hmin/(2*DG_degree + 1)
    func_space = FunctionSpace(mesh, "DG", DG_degree)
    func_space_accurate = FunctionSpace(mesh, "DG", DG_degree + 5)

    
    #coordinate function
    Au = Function(func_space)
    Au.interpolate(Expression("tanh(x[0]-2)", degree=10))
    #Au.interpolate(Expression("sin(x[0])", degree=10))

    r = SpatialCoordinate(mesh)[0]

    #define functions for the variables
    A = Function(func_space)
    B = Function(func_space)
    exact_A = Function(func_space_accurate)
    exact_B = Function(func_space_accurate)

    deri_A = [Function(func_space), Function(func_space)]
    deri_B = [Function(func_space), Function(func_space)]
    Hhat = get_Hhat_AB(deri, deri_B)
    
    #Runge Kutta step
    #####################################################################################

    #initialize functions
    temp_phi = Function(func_space)

    time_seq = []
    error_var_seq = []
    rhs_seq = []
    
    t = 0.0
    t_end = 200.0
    last_save_wall_time = time.time()
    plt.ion()
    fig = plt.figure(figsize=(19.2, 10.8))
    while t < t_end:
        if t + dt < t_end:
            t += dt
        else:
            return '222'
            dt = t_end - t
            t = t_end
        time_seq.append(t)
        t_str = '%.05f'%t
        t_str = t_str.zfill(8)

        init.get_exact_phi(exact_phi, t)
        rk.rk3(A, B, exact_A, exact_B, deri_A, deri_B, temp_A, temp_B, Hhat_A, Hhat_B, dt)

        error_var = errornorm(phi, exact_phi, 'L2')
        error_var_seq.append(error_var)

        rhs_L2 = errornorm(project(Hhat, func_space), project(Constant(0), func_space), 'L2')
        rhs_seq.append(rhs_L2)


        # show fig of real time error of vars and constraints, save both in .png form
        #error fig 
        fig.clf()
        fig.subplots(2, 2)
        fig.suptitle("t = "+t_str)
        
        plot_obj = fig.axes[0]
        plot_obj.plot(time_seq, rhs_seq, 'r')
        plot_obj.set_yscale('log')
        plot_obj.set_title('rhs')

        plot_obj = fig.axes[2]
        ioo.plot_function_dif(phi, exact_phi, plot_obj) 
        plot_obj.set_title('error of phi')

        plot_obj = fig.axes[3]
        plot_obj.plot(time_seq, error_var_seq, 'r') 
        plot_obj.set_title('error over time')
        plot_obj.set_yscale('log')

        plot_obj = fig.axes[1]
        ioo.plot_function(phi, plot_obj) 
        plot_obj.set_title('phi')

        fig.savefig(folder+t_str+'.png') 

        plt.pause(0.000001)

        #save data and figs every 100 secends
        ioo.write_seqs_to_file(folder+'error_var_seq.txt', error_var_seq)    
        with open(folder+'time_seq.txt', 'w') as f:
            for t_point in time_seq:
                f.write(str(t_point)+'\n')

    plt.ioff()
    print("\nMEOW!!")
main()
