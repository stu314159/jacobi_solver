#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 13:05:03 2018

@author: stu
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import Jacobi_Solver as js


def jacobi_solver(L,dp_dx,mu,u_i,tol):
    """
     simple jacobi solver for testing purposes
    """
    Ny = np.size(u_i)
    u_even = np.copy(u_i)
    u_odd = np.copy(u_i)
    h = L/(float(Ny) - 1.)
    rhs = (h**2)*(1./mu)*dp_dx
    maxIter = 1000000
    
    KEEP_GOING = True
    nIter = 0
    exit_code = 0
    while KEEP_GOING:
        nIter += 1
        
        # set pointers for arrays
        if nIter%2 == 0:
            u = u_even; u_new = u_odd;
        else:
            u = u_odd; u_new = u_even
            
        #for i in range(1,Ny-1):
        #    u_new[i] = 0.5*(u[i-1] + u[i+1] - rhs)
        
        # vectorized instead
        u_new[1:-1] = 0.5*(u[0:-2] + u[2:] - rhs)
        
        if np.linalg.norm(u) == 0:
            continue # prevent divide by zero on convergence test
        rel_update = np.linalg.norm(u - u_new)/np.linalg.norm(u)
        
        if (rel_update < tol ) or (nIter == maxIter):
            KEEP_GOING = False
            if nIter == maxIter:
                exit_code = -1;
    
    u_out = np.copy(u)
    return u_out, nIter, exit_code

def jacobi_solver_boost(L,dp_dx,mu,u_i,tol):
    """
    Jacobi Solver using facility of BOOST.PYTHON interface
    """
    Ny = np.size(u_i)
    u_even = np.copy(u_i)
    u_odd = np.copy(u_i)
    u_out = np.copy(u_i);
    h = L/(float(Ny) - 1.)
    rhs = (h**2)*(1./mu)*dp_dx
    #print "in Python, rhs = {}".format(rhs)
    
    js_obj = js.PyJacobi_Solver(Ny);
    js_obj.set_u_out(u_out);
    js_obj.set_u_even(u_even)
    js_obj.set_u_odd(u_odd)
    js_obj.set_maxIter(100000);
    js_obj.set_tolerance(tol);
    js_obj.set_rhs(rhs);
    #js_obj.print_status();
    js_obj.solve();
    nIter = js_obj.get_iter();
    exit_code = js_obj.get_exit_code();
    
    return u_out, nIter, exit_code
    

# common parameters
L = 0.002; # m, plate separation
mu = 0.4; # N-s/m*2, total viscosity of fluid
tol = 1e-10; # convergence tolerance
dp_dx = -3e4; # Pa/m, pressure gradiaent
U_o = 0.; # velocity on lower boundary
U_L = 0.; # velocity on upper boundary

# parameters of the analytic solution
K = (1./mu)*dp_dx
c1 = -K*L/2.
c2 = 0.;
u_a = lambda y: (K/2.)*(y**2) + c1*y + c2
y_max =  -c1/K
u_max_a = u_a(y_max)
u_avg_a = K*L**2/6. + c1*L/2. + c2
tau_wall_a = mu*c1

print "u_max = {} m/s, u_avg = {} m/s, wall shear stress = {} Pa".\
  format(u_max_a,u_avg_a,tau_wall_a)
  
N = 200;
Y= np.linspace(0.,L,N);

u_n = np.zeros(N,dtype=np.float64)
u_n[0] = U_o; u_n[N-1] = U_L;
t0 = time.time()
#u_n, nIter, exit_code = jacobi_solver(L,dp_dx,mu,u_n,tol)
u_n, nIter, exit_code = jacobi_solver_boost(L,dp_dx,mu,u_n,tol)
t1 = time.time()
elapsed_time = t1 - t0;
print "Elapsed time = {} sec; nIter = {}; exit_code = {}".\
format(elapsed_time,nIter,exit_code)

# plot the results
plt.plot(u_n,Y)
plt.title('Velocity for Poiseuille flow')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Channel Position (m)')
plt.grid(True)
plt.show()
