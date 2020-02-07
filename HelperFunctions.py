# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:37:03 2020

@author: ben91
"""
import torch
from torch.nn.parameter import Parameter

import numpy as np

def dudx(u):
    derv = 1/12*u[:,:,0]-2/3*u[:,:,1]+0*u[:,:,2]+2/3*u[:,:,3]-1/12*u[:,:,4]
    return torch.t(derv).unsqueeze(0)
def d2udx2(u):
    derv = -1/12*u[:,:,0]+4/3*u[:,:,1]-5/2*u[:,:,2]+4/3*u[:,:,3]-1/12*u[:,:,4]
    return torch.t(derv).unsqueeze(0)
def d3udx3(u):
    derv = -1/2*u[:,:,0]+u[:,:,1]+0*u[:,:,2]-1*u[:,:,3]+1/2*u[:,:,4]
    return torch.t(derv).unsqueeze(0)
def d4udx4(u):
    derv = 1*u[:,:,0]-4*u[:,:,1]+6*u[:,:,2]-4*u[:,:,3]+1*u[:,:,4]
    return torch.t(derv).unsqueeze(0)

def makeIC(L):
    def IC(x):
        #np.random.seed(N)
        f = 0*x
        for j in range(0,5):
            f = f + torch.rand(1,dtype=torch.double)*torch.sin(2*j*np.pi*(x-torch.rand(1,dtype=torch.double))/L)
        #f = f + (x>(L/2)).double()*(5-10*torch.rand(1,dtype=torch.double))
        return f
    return IC

def makeICdsc(L):
    def IC(x):
        #np.random.seed(N)
        f = 0*x
        for j in range(0,5):
            f = f + torch.rand(1,dtype=torch.double)*torch.sin(2*j*np.pi*(x-torch.rand(1,dtype=torch.double))/L)
        #dscc = 1+4*torch.rand(1,dtype=torch.double)
        f = f + (x>(L/2)).double()*(5-10*torch.rand(1,dtype=torch.double))
        #f = f + (x>(L/2)).double()*dscc*np.random.choice((-1, 1))
        return f
    return IC

def reshKS(ui):
    uj = torch.t(ui.squeeze(0))
    Nx = uj.size(0)
    Nt = ui.size(1)
    U_proc = torch.zeros(Nx,Nt,7).double()
    U_proc[:,:,0] = uj.roll(-3,0)
    U_proc[:,:,1] = uj.roll(-2,0)
    U_proc[:,:,2] = uj.roll(-1,0)
    U_proc[:,:,3] = uj.roll(0,0)
    U_proc[:,:,4] = uj.roll(1,0)
    U_proc[:,:,5] = uj.roll(2,0)
    U_proc[:,:,6] = uj.roll(3,0)
    return U_proc
def resh(ui):
    uj = torch.t(ui.squeeze(0))
    Nx = uj.size(0)
    Nt = ui.size(1)
    U_proc = torch.zeros(Nx,Nt,5).double()
    #U_proc[:,:,0] = uj.roll(-2,0)
    #U_proc[:,:,1] = uj.roll(-1,0)
    #U_proc[:,:,2] = uj.roll(0,0)
    #U_proc[:,:,3] = uj.roll(1,0)
    #U_proc[:,:,4] = uj.roll(2,0)
    
    U_proc[:,:,0] = uj.roll(2,0)
    U_proc[:,:,1] = uj.roll(1,0)
    U_proc[:,:,2] = uj.roll(0,0)
    U_proc[:,:,3] = uj.roll(-1,0)
    U_proc[:,:,4] = uj.roll(-2,0)
    return U_proc

def wenoCoeff(ur):
    ep = 1E-6
    
    B1 = 13/12*(ur[:,:,0] - 2*ur[:,:,1] + ur[:,:,2])**2 + 1/4*(ur[:,:,0] - 4*ur[:,:,1] + 3*ur[:,:,2])**2
    B2 = 13/12*(ur[:,:,1] - 2*ur[:,:,2] + ur[:,:,3])**2 + 1/4*(ur[:,:,1] - ur[:,:,3])**2
    B3 = 13/12*(ur[:,:,2] - 2*ur[:,:,3] + ur[:,:,4])**2 + 1/4*(3*ur[:,:,2] - 4*ur[:,:,3] + ur[:,:,4])**2

    g1 = 1/10
    g2 = 3/5
    g3 = 3/10
    
    wt1 = g1/(ep+B1)**2
    wt2 = g2/(ep+B2)**2
    wt3 = g3/(ep+B3)**2
    wts = wt1 + wt2 + wt3
    
    w1 = wt1/wts
    w2 = wt2/wts
    w3 = wt3/wts
    
    c = torch.zeros(ur.size()).double()
    
    c[:,:,0] = 1/3*w1
    c[:,:,1] = -7/6*w1 - 1/6*w2
    c[:,:,2] = 11/6*w1 + 5/6*w2 + 1/3*w3
    c[:,:,3] = 1/3*w2 + 5/6*w3
    c[:,:,4] = -1/6*w3
    return c

def exactSol(x,t):
    def f(x):
        #return torch.exp(-25*x**2)
        return (x>=1).double()
    return f((x-t)%2)

def randGrid(L,fullInf,S):
    cfl = torch.rand(1,dtype=torch.double)*0.8+0.2
    dx = torch.rand(1,dtype=torch.double)*0.046+0.004
    dt = cfl*dx
    if(fullInf == 0):
        T = 1
    else:
        T = dt*(S+1)
    xc = torch.linspace(0,L,int(L/dx)+1,dtype=torch.double)
    xc = xc[:-1]
    tc = torch.linspace(0,T,int(T/dt),dtype=torch.double)
    return torch.meshgrid(xc,tc)

def scaleAvg(uip):
    min_u = uip.min(2)[0]
    max_u = uip.max(2)[0]
    const_n = min_u==max_u
    #print('u: ', u)
    u_tmp = torch.zeros_like(uip[:,:,2])
    u_tmp[:] = uip[:,:,2]
    for i in range(0,5):
        uip[:,:,i] = (uip[:,:,i]-min_u)/(max_u-min_u)
    return uip, const_n, u_tmp
