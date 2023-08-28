# -*- coding: utf-8 -*-
"""
@author: Yujie Zhang
"""

import json
import numpy as np
from scipy import integrate
from scipy.stats import ortho_group
from scipy.linalg import null_space
from numpy.linalg import norm
import scipy
'''This script gives a numerical calculation on simulating arbitary noisy 4-outcome POVM '''


with open('Lebedev_131.json', 'r') as f:
    Lebedev_131 = np.array(json.load(f))

'''List of index for the coarse-grained Parent, with 0 for 'on', and 1 for off.
moreover m5+ 'on' implies  m5- 'off'.'''
list = [[0, 1, 1, 1, 1], [1, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 1, 0, 1, 1],
        [0, 1, 1, 0, 1], [1, 0, 0, 1, 1], [1, 0, 1, 0, 1], [1, 1, 0, 1, 0],
        [1, 1, 1, 0, 0], [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1], [0, 1, 0, 1, 0], [0, 1, 1, 0, 0], [1, 0, 0, 1, 0],
        [1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]]

'''lebedev integration order 131th, averaged with N=10 random rotations.'''
Trial=10
Rl=[ortho_group.rvs(3) for i in range(Trial)]
def lebedev(func, N=Trial, **kwargs):
    I = 0
    for n in range(N):
        #R = ortho_group.rvs(3)
        R=Rl[n]
        x = Lebedev_131[:, 0];
        y = Lebedev_131[:, 1];
        z = Lebedev_131[:, 2]
        w = Lebedev_131[:, 3]
        for i in range(len(x)):
            xyz = R.dot(np.array([x[i], y[i], z[i]]))
            I += w[i] * func(xyz, **kwargs)
    return I / N

'''Theta function '''
def mheav(x):
    if x == 0:
        return 1 / 2
    else:
        return np.heaviside(x, 0)

'''Constructing 18-effect coarse-grained Parent POVM, with some symmetrization and normalizaiton applid.'''
def I(M):
    J=[]
    m1=M[0][1:4]/M[0][0]
    m2=M[1][1:4]/M[1][0]
    m3=M[2][1:4]/M[2][0]
    m4=M[3][1:4]/M[3][0]
    m5 = -(M[0][1:4] + M[1][1:4]) / norm(M[0][1:4] + M[1][1:4])
    for A in list[0:18]:
        [i,j,k,w,t]=A
        def f0(xyz):
            x,y,z=xyz
            return mheav((-1)**i*(x*m1[0]+y*m1[1]+z*m1[2]))*mheav((-1)**j*(x*m2[0]+y*m2[1]+z*m2[2]))*mheav((-1)**k*(x*m3[0]+y*m3[1]+z*m3[2]))*mheav((-1)**w*(x*m4[0]+y*m4[1]+z*m4[2]))*mheav((-1)**t*(x*m5[0]+y*m5[1]+z*m5[2]))
        def f1(xyz):
            x,y,z=xyz
            return mheav((-1)**i*(x*m1[0]+y*m1[1]+z*m1[2]))*mheav((-1)**j*(x*m2[0]+y*m2[1]+z*m2[2]))*mheav((-1)**k*(x*m3[0]+y*m3[1]+z*m3[2]))*mheav((-1)**w*(x*m4[0]+y*m4[1]+z*m4[2]))*mheav((-1)**t*(x*m5[0]+y*m5[1]+z*m5[2]))*x
        def f2(xyz):
            x,y,z=xyz
            return mheav((-1)**i*(x*m1[0]+y*m1[1]+z*m1[2]))*mheav((-1)**j*(x*m2[0]+y*m2[1]+z*m2[2]))*mheav((-1)**k*(x*m3[0]+y*m3[1]+z*m3[2]))*mheav((-1)**w*(x*m4[0]+y*m4[1]+z*m4[2]))*mheav((-1)**t*(x*m5[0]+y*m5[1]+z*m5[2]))*y
        def f3(xyz):
            x,y,z=xyz
            return mheav((-1)**i*(x*m1[0]+y*m1[1]+z*m1[2]))*mheav((-1)**j*(x*m2[0]+y*m2[1]+z*m2[2]))*mheav((-1)**k*(x*m3[0]+y*m3[1]+z*m3[2]))*mheav((-1)**w*(x*m4[0]+y*m4[1]+z*m4[2]))*mheav((-1)**t*(x*m5[0]+y*m5[1]+z*m5[2]))*z
        a0=lebedev(f0)
        a1=lebedev(f1)
        a2=lebedev(f2)
        a3=lebedev(f3)
        ja=np.array([a0,a1,a2,a3])
        J.append(ja)
    # sum=0
    for i in range(9):
        alpha=J[i][0]+J[17-i][0]
        J[i][0]=alpha/2
        J[17-i][0]=alpha/2
        for k in range(1,4):
            nk=J[i][k]-J[17-i][k]
            J[i][k]=nk/2
            J[17-i][k]=-nk/2
    sum = 0
    for i in range(18):
        sum = sum + J[i][0]
    J=J/(sum)
    return J


'''Generating response coefficienct P_a^i from Table 1 in the supplementary material'''
def CoeffP1(M):
    nu1 = M[0][0]
    nu2 = M[1][0]
    nu3 = M[2][0]
    nu4 = M[3][0]
    nu5 = 2*norm(M[0][1:4] + M[1][1:4])
    k1 = (nu1 + nu2 + nu5)
    k2 = (nu3 + nu4 + nu5)
    x5p = 1- 2 * nu5 / k1;
    x1 = 1- 2 * nu1 / k1;
    x2 = 1- 2 * nu2/ k1;
    x5m = 1- 2 * nu5 / k2;
    x3 = 1- 2 * nu3 / k2;
    x4 = 1- 2 * nu4/ k2;
    b5p = JA[7][0]+JA[8][0]+JA[15][0]+JA[2][0]+JA[9][0]+JA[10][0];
    b1 = JA[0][0]+JA[3][0]+JA[4][0]+JA[13][0]+JA[14][0]+JA[17][0];
    b2 = JA[1][0]+JA[5][0]+JA[6][0]+JA[11][0]+JA[12][0]+JA[16][0];
    b5m = JA[0][0]+JA[1][0]+JA[2][0]+JA[15][0]+JA[16][0]+JA[17][0];
    b3 = JA[7][0]+JA[11][0]+JA[13][0]+JA[4][0]+JA[6][0]+JA[10][0];
    b4 = JA[8][0]+JA[12][0]+JA[14][0]+JA[3][0]+JA[5][0]+JA[9][0];
    Y1=(b5p*x5p-b1*x1+b2*x2)/(2*b5p)
    X1=(b5p*x5p+b1*x1-b2*x2)/(2*b5p)
    Y2=(b5m*x5m-b3*x3+b4*x4)/(2*b5m)
    X2=(b5m*x5m+b3*x3-b4*x4)/(2*b5m)
    P = np.zeros((4, 18))
    P[0][0]=k1; P[0][3]=k1; P[0][4]=k1; P[0][2]=2*nu1-X1*k1; P[0][9]=2*nu1-X1*k1; P[0][10]=2*nu1-X1*k1;
    P[0][7]=Y1*k1; P[0][8]=Y1*k1; P[0][15]=Y1*k1; P[0][11]=k1-2*nu5; P[0][12]=k1-2*nu5; P[0][16]=k1-2*nu5;

    P[1][1]=k1; P[1][5]=k1; P[1][6]=k1; P[1][2]=2*nu2-Y1*k1; P[1][9]=2*nu2-Y1*k1; P[1][10]=2*nu2-Y1*k1;
    P[1][7]=X1*k1; P[1][8]=X1*k1; P[1][15]=X1*k1; P[1][13]=k1-2*nu5; P[1][14]=k1-2*nu5; P[1][17]=k1-2*nu5;

    P[2][0]=Y2*k2;P[2][1]=Y2*k2; P[2][2]=Y2*k2;P[2][3]=k2-2*nu5;P[2][5]=k2-2*nu5; P[2][9]=k2-2*nu5;
    P[2][7]=k2; P[2][11]=k2; P[2][13]=k2; P[2][15]=2*nu3-X2*k2;P[2][16]=2*nu3-X2*k2;P[2][17]=2*nu3-X2*k2;

    P[3][0]=X2*k2;P[3][1]=X2*k2; P[3][2]=X2*k2;P[3][4]=k2-2*nu5;P[3][6]=k2-2*nu5; P[3][10]=k2-2*nu5;
    P[3][8]=k2; P[3][12]=k2; P[3][14]=k2; P[3][15]=2*nu4-Y2*k2;P[3][16]=2*nu4-Y2*k2;P[3][17]=2*nu4-Y2*k2;
    return P

# Computing the normalization
def newq(P):
    return np.sum(P, axis=0) - 1



'''Example of 4-outcome M to be simulated'''
a = 0.1
mm1 = np.array([0, -np.sqrt(1 - a ** 2), a, 0])
mm2 = np.array([0, -np.sqrt(1 - a ** 2), -1 / 2 * a, -np.sqrt(3) / 2 * a])
mm3 = np.array([0, -np.sqrt(1 - a ** 2), -1 / 2 * a, np.sqrt(3) / 2 * a])
mm4 = np.array([0, 1, 0, 0])
me1 = np.array([mm1, mm2, mm3, mm4])
nsspace = scipy.linalg.null_space(np.array(me1).transpose()).transpose()
nu1 = nsspace[0][0] / np.sum(nsspace[0])
nu2 = nsspace[0][1] / np.sum(nsspace[0])
nu3 = nsspace[0][2] / np.sum(nsspace[0])
nu4 = nsspace[0][3] / np.sum(nsspace[0])

eta = 1 / 2
M1 = nu1 * np.array([1, 0, 0, 0]) + nu1 * mm1 * eta
M2 = nu2 * np.array([1, 0, 0, 0]) + nu2 * mm2 * eta
M3 = nu3 * np.array([1, 0, 0, 0]) + nu3 * mm3 * eta
M4 = nu4 * np.array([1, 0, 0, 0]) + nu4 * mm4 * eta
M = np.array([M4, M2, M3, M1])

# M=np.array([[ 0.48370211,  0.03505807,  0.01676746, -0.23870843],
#  [ 0.26214345, -0.0050379,  -0.03425089,  0.12641714],
#  [ 0.13247961,  0.00752084,  0.0086385,   0.06524205],
#  [ 0.12167484, -0.03754101,  0.00884492,  0.04704925]])
# M=np.array([[ 4.94246961e-01, -7.18672265e-02,  6.71420306e-02, -2.26709206e-01],
# [3.61137905e-02, 4.45375851e-04, -1.03128011e-03, 1.80219191e-02],
#  [ 1.10267191e-01,  6.94156528e-03,  3.37531219e-04,  5.46938212e-02],
#  [ 3.59372057e-01,  6.44802853e-02, -6.64482817e-02,  1.53993466e-01]])

'''Return coarse-grained POVM JA, response function P'''
JA = I(M)
P = CoeffP1(M)
q = newq(P)

'''Reconstruting the Chilren POVM M, with Parent Coarse-crained Parent JA, and response function P'''
Mnew = []
for k in range(4):
    sum = np.array([0, 0, 0, 0])
    for i in range(18):
        sum = sum + JA[i] * P[k][i]
    Mnew.append(sum)
Mnew = np.array(Mnew)
