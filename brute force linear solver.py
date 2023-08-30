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

'''This script use linear program to search for a response function for simulating M with its corresponding 14-effect coarse-grained POVM'''


with open('Lebedev_131.json', 'r') as f:
    Lebedev_131= np.array(json.load(f))

'''lebedev integration order 131th, averaged with N=10 random rotations.'''
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


'''Constructing example:  M'''
a=0.005
mm1=np.array([0,-np.sqrt(1-a**2),a,0])
mm2=np.array([0,-np.sqrt(1-a**2),-1/2*a,-np.sqrt(3)/2*a])
mm3=np.array([0,-np.sqrt(1-a**2),-1/2*a, np.sqrt(3)/2*a])
mm4=np.array([0,1,0,0])
me1=np.array([mm1,mm2,mm3,mm4])
nsspace = scipy.linalg.null_space(np.array(me1).transpose()).transpose()
nu1=nsspace[0][0]/np.sum(nsspace[0])
nu2=nsspace[0][1]/np.sum(nsspace[0])
nu3=nsspace[0][2]/np.sum(nsspace[0])
nu4=nsspace[0][3]/np.sum(nsspace[0])
eta=1/2
M1=nu1*np.array([1,0,0,0])+nu1*mm1*eta
M2=nu2*np.array([1,0,0,0])+nu2*mm2*eta
M3=nu3*np.array([1,0,0,0])+nu3*mm3*eta
M4=nu4*np.array([1,0,0,0])+nu4*mm4*eta
M=np.array([M4,M1,M3,M2])

# Assuming you have the matrix A and array b defined
# A should be a 30x54 matrix and b should be a 30x1 array
def linearsolver(M):
    for i in range(20):
        global Rl
        Rl = [ortho_group.rvs(3) for i in range(Trial)]
        JA=I(M)
        AT = []
        for j in range(4):
            for i in range(14):
                a = np.zeros((4, 4))
                a[j] = JA[i]
                b = np.zeros(14)
                b[i] = 1
                a = np.append(a.flatten(), b)
                AT.append(a)
        A = np.array(AT).T
        # Define the constraint matrix A and RHS values b
        b = np.append(M, np.array([1 for i in range(14)]))
        # Create a Gurobi model
        model = gp.Model()
        model.Params.OutputFlag = 0
        # Number of variables (size of x)
        num_vars = A.shape[1]

        # Create variables
        x = model.addVars(num_vars, name="x", lb=0)  # lb=0 enforces nonnegativity

        # Add equality constraints
        for i in range(A.shape[0]):
            model.addConstr(gp.quicksum(A[i, j] * x[j] for j in range(num_vars)) == b[i])

        # Optimize the model
        model.optimize()

        # Access solution
        if model.status == gp.GRB.OPTIMAL:
            x_solution = [x[i].x for i in range(num_vars)]
            P2=np.array(x_solution).reshape(4,14)
            print("Solution x:", P2)
            break
        else:
            print("Optimization failed")

        # Dispose of the model
        model.dispose()


linearsolver(M)
