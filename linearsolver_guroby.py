# -*- coding: utf-8 -*-
"""
@author: Yujie Zhang
"""

import json
import numpy as np
import gurobipy as gp
from scipy import integrate
from scipy.stats import ortho_group
from scipy.linalg import null_space
from scipy.optimize import linprog
from numpy.linalg import norm
import scipy
from copy import *

'''This script use Farka's lemma to prove Observation 2 in the method section
 : There exist noisy POVM that can not be simulated by a 14-effect Parent'''


with open('Lebedev_131.json', 'r') as f:
    Lebedev_131= np.array(json.load(f))

'''lebedev integration order 131th, averaged with N=50 random rotations.'''
Trial=10
Rl = [ortho_group.rvs(3) for i in range(Trial)]
def lebedev(func,N=Trial,**kwargs):
    I = 0
    for n in range(N):
        R = Rl[n]
        x = Lebedev_131[:,0];y = Lebedev_131[:,1];z = Lebedev_131[:,2]
        w = Lebedev_131[:,3]
        for i in range(len(x)):
            xyz=R.dot(np.array([x[i],y[i],z[i]]))
            I += w[i]*func(xyz,**kwargs)
    return I/N

'''Theta function '''
def mheav(x):
    if x==0:
        return 1/2
    else:
        return np.heaviside(x,0)

'''Constructing 14-effect coarse-grained Parent POVM, with some symmetrization and normalizaiton.'''
def I(M):
    Mu = [M[0][0], M[1][0], M[2][0], M[3][0]]
    ind = Mu.index(max(Mu))
    J=[]
    m1=M[0][1:4]/M[0][0]
    m2=M[1][1:4]/M[1][0]
    m3=M[2][1:4]/M[2][0]
    m4=M[3][1:4]/M[3][0]
    for A in range(17,24):
        i=int(bin(A)[3])
        j=int(bin(A)[4])
        k=int(bin(A)[5])
        w=int(bin(A)[6])
        #print(i,j,k,w)
        def f0(xyz):
            x,y,z=xyz
            return mheav((-1)**i*(x*m1[0]+y*m1[1]+z*m1[2]))*mheav((-1)**j*(x*m2[0]+y*m2[1]+z*m2[2]))*mheav((-1)**k*(x*m3[0]+y*m3[1]+z*m3[2]))*mheav((-1)**w*(x*m4[0]+y*m4[1]+z*m4[2]))
        def f1(xyz):
            x,y,z=xyz
            return mheav((-1)**i*(x*m1[0]+y*m1[1]+z*m1[2]))*mheav((-1)**j*(x*m2[0]+y*m2[1]+z*m2[2]))*mheav((-1)**k*(x*m3[0]+y*m3[1]+z*m3[2]))*mheav((-1)**w*(x*m4[0]+y*m4[1]+z*m4[2]))*x
        def f2(xyz):
            x,y,z=xyz
            return mheav((-1)**i*(x*m1[0]+y*m1[1]+z*m1[2]))*mheav((-1)**j*(x*m2[0]+y*m2[1]+z*m2[2]))*mheav((-1)**k*(x*m3[0]+y*m3[1]+z*m3[2]))*mheav((-1)**w*(x*m4[0]+y*m4[1]+z*m4[2]))*y
        def f3(xyz):
            x,y,z=xyz
            return mheav((-1)**i*(x*m1[0]+y*m1[1]+z*m1[2]))*mheav((-1)**j*(x*m2[0]+y*m2[1]+z*m2[2]))*mheav((-1)**k*(x*m3[0]+y*m3[1]+z*m3[2]))*mheav((-1)**w*(x*m4[0]+y*m4[1]+z*m4[2]))*z
        a0=lebedev(f0)
        a1=lebedev(f1)
        a2=lebedev(f2)
        a3=lebedev(f3)
        #Aa=np.array([a1,a2,a3])
        #nAa=2*np.dot(Aa,M[ind][1:4])/np.dot(M[ind][1:4],M[ind][1:4])*M[ind][1:4]-Aa
        ja=np.array([a0,a1,a2,a3])
        J.append(ja)
    sum=0
    for i in range(7):
        J=np.vstack([J,np.array([J[6-i][0],-J[6-i][1],-J[6-i][2],-J[6-i][3]])])
        sum=sum+J[i][0]
    for i in range(7):
        J[13 - i, 0] = J[i, 0]
        for j in range(1, 4):
            J[13 - i, j] = -J[i, j]
    sum = 0
    for i in range(14):
        sum = sum + J[i][0]
    J=J/(sum)
    return J


'''Constructing counter example:  M'''
M=np.array([[ 4.94246961e-01, -7.18672265e-02,  6.71420306e-02, -2.26709206e-01],
[3.61137905e-02, 4.45375851e-04, -1.03128011e-03, 1.80219191e-02],
 [ 1.10267191e-01,  6.94156528e-03,  3.37531219e-04,  5.46938212e-02],
 [ 3.59372057e-01,  6.44802853e-02, -6.64482817e-02,  1.53993466e-01]])

'''Constructing counter example:  14-effect JA'''
JA=I(M) 



'''LP solver with guroby, where the value output is -by, with constrain -AT<=0'''
def LPguroby(JA,M):

    model = gp.Model("LinearOptimization")

    # Number of variables and constraints
    num_vars = 30
    num_constraints = 56

    # Create variables
    variables = [model.addVar(lb=-1, ub=1, name="x{}".format(i)) for i in range(num_vars)]

    # Define the objective coefficients
    objective_coefficients = np.append(M,np.array([1 for i in range(14)]))

    # Set objective function
    model.setObjective(gp.quicksum(objective_coefficients[i] * variables[i] for i in range(num_vars)), sense=gp.GRB.MINIMIZE)
    AT=[]
    for j in range(4):
        for i in range(14):
            a = np.zeros((4, 4))
            a[j] = JA[i]
            b = np.zeros(14)
            b[i] = 1
            a = np.append(a.flatten(), b)
            AT.append(a)
    AT = np.array(AT)
    # Define the constraint matrix A and RHS values b
    constraint_matrix = AT
    rhs_values =np.array([0 for i in range(56)])

    # Add constraints
    constraints = [model.addConstr(gp.quicksum(constraint_matrix[i, j] * variables[j] for j in range(num_vars)) >= rhs_values[i]) for i in range(num_constraints)]

    # Optimize the model
    model.optimize()

    # Access solution
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found")
        print("y=", np.array([variables[i].x for i in range(30)]))
    elif model.status == gp.GRB.INFEASIBLE:
        print("Model is infeasible")
    elif model.status == gp.GRB.UNBOUNDED:
        print("Model is unbounded")
    else:
        print("Optimization ended with status:", model.status)
    return 0

LPguroby(JA,M)
