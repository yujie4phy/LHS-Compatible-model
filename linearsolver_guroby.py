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
    J[0,0]=(J[0,0]+J[1,0]+J[3,0])/3
    J[1,0]=J[0,0]
    J[3,0]=J[0,0]
    J[0,1]=(J[0,1]+J[1,1]+J[3,1])/3
    J[1,1]=J[0,1]
    J[3,1]=J[0,1]
    J[2,0]=(J[2,0]+J[4,0]+J[5,0])/3
    J[4,0]=J[2,0]
    J[5,0]=J[2,0]
    J[2,1]=(J[2,1]+J[4,1]+J[5,1])/3
    J[4,1]=J[2,1]
    J[5,1]=J[2,1]
    J[0,2]=(J[0,2]+J[1,2]+np.abs(J[3,2]))/4
    J[1,2]=J[0,2]
    J[3,2]=-2*J[0,2]
    J[0,3]=J[0,2]*np.sqrt(3)
    J[1,3]=-J[1,2]*np.sqrt(3)
    J[3,3]=0
    J[4,2]=(J[4,2]+J[5,2]-np.abs(J[2,2]))/4
    J[5,2]=J[4,2]
    J[2,2]=-2*J[4,2]
    J[4,3]=-J[4,2]*np.sqrt(3)
    J[5,3]=J[5,2]*np.sqrt(3)
    J[2,3]=0
    J[6,2]=0
    J[6,3]=0
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
a=0.01
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

'''Constructing counter example:  14-effect JA'''
#JA=I(M)  # Comment this out to generate the desired example below.
JA=np.array([[ 1.76574889e-04,  2.89846387e-07,  7.96329126e-05, 1.37928251e-04],
       [ 1.76574889e-04,  2.89846387e-07,  7.96329126e-05, -1.37928251e-04],
       [ 1.02098529e-03,  4.73724253e-06,  8.66851191e-04, 0.00000000e+00],
       [ 1.76574889e-04,  2.89846387e-07, -1.59265825e-04,0.00000000e+00],
       [ 1.02098529e-03,  4.73724253e-06, -4.33425595e-04, 7.50715153e-04],
       [ 1.02098529e-03,  4.73724253e-06, -4.33425595e-04,-7.50715153e-04],
       [ 4.96407319e-01,  2.49987120e-01,  0.00000000e+00,0.00000000e+00],
       [ 4.96407319e-01, -2.49987120e-01, -0.00000000e+00,-0.00000000e+00],
       [ 1.02098529e-03, -4.73724253e-06,  4.33425595e-04,7.50715153e-04],
       [ 1.02098529e-03, -4.73724253e-06,  4.33425595e-04,-7.50715153e-04],
       [ 1.76574889e-04, -2.89846387e-07,  1.59265825e-04,-0.00000000e+00],
       [ 1.02098529e-03, -4.73724253e-06, -8.66851191e-04,-0.00000000e+00],
       [ 1.76574889e-04, -2.89846387e-07, -7.96329126e-05,1.37928251e-04],
       [ 1.76574889e-04, -2.89846387e-07, -7.96329126e-05,-1.37928251e-04]])


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