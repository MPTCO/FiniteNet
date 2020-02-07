# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 18:08:55 2019

"""
import torch
from torch import nn
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import time

from numpy import genfromtxt
from HelperFunctions import makeIC, resh, reshKS, wenoCoeff, exactSol, randGrid
plt.close('all') # close all open figures
# Define and set custom LaTeX style
styleNHN = {
        "pgf.rcfonts":False,
        "pgf.texsystem": "pdflatex",   
        "text.usetex": False,
        "font.family": "serif"
        }
mpl.rcParams.update(styleNHN)

# Plotting defaults
ALW = 0.75  # AxesLineWidth
FSZ = 12    # Fontsize
LW = 2      # LineWidth
MSZ = 5     # MarkerSize
SMALL_SIZE = 8    # Tiny font size
MEDIUM_SIZE = 10  # Small font size
BIGGER_SIZE = 14  # Large font size
plt.rc('font', size=FSZ)         # controls default text sizes
plt.rc('axes', titlesize=FSZ)    # fontsize of the axes title
plt.rc('axes', labelsize=FSZ)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=FSZ)   # fontsize of the x-tick labels
plt.rc('ytick', labelsize=FSZ)   # fontsize of the y-tick labels
plt.rc('legend', fontsize=FSZ)   # legend fontsize
plt.rc('figure', titlesize=FSZ)  # fontsize of the figure title
plt.rcParams['axes.linewidth'] = ALW    # sets the default axes lindewidth to ``ALW''
plt.rcParams["mathtext.fontset"] = 'cm' # Computer Modern mathtext font (applies when ``usetex=False'')

#load the data here. coment out whenever possible

import numpy as np
from numpy import genfromtxt
all_data = genfromtxt('KS_big_TT_keep.csv', delimiter=',')
r2 = np.reshape(all_data, (3201,750,80))

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True).double()
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim, 5).double()
        self.fc2 = nn.Linear(hidden_dim, 5).double()
        self.fc4 = nn.Linear(hidden_dim, 7).double()
        self.trf1 = nn.Linear(5, 5).double()
        self.trf2 = nn.Linear(5, 5).double()
        self.trf4 = nn.Linear(7, 7).double()
        #cm = Parameter(torch.tensor([[0.8,-0.2,-0.2,-0.2,-0.2],[-0.2,0.8,-0.2,-0.2,-0.2],[-0.2,-0.2,0.8,-0.2,-0.2],[-0.2,-0.2,-0.2,0.8,-0.2],[-0.2,-0.2,-0.2,-0.2,0.8]]).double())#1st order
        cm1 = Parameter(torch.tensor([[4/35,-9/35,3/35,1/7,-3/35],[-9/35,22/35,-12/35,-6/35,1/7],[3/35,-12/35,18/35,-12/35,3/35],[1/7,-6/35,-12/35,22/35,-9/35],[-3/35,1/7,3/35,-9/35,4/35]]).double())#2nd order
        cm2 = Parameter(torch.tensor([[1/70,-2/35,3/35,-2/35,1/70],[-2/35,8/35,-12/35,8/35,-2/35],[3/35,-12/35,18/35,-12/35,3/35],[-2/35,8/35,-12/35,8/35,-2/35],[1/70,-2/35,3/35,-2/35,1/70]]).double())#2nd order
        cm4 = Parameter(torch.tensor([[1/924,-1/154,5/308,-5/231,5/308,-1/154,1/924],[-1/154,3/77,-15/154,10/77,-15/154,3/77,-1/154],[5/308,-15/154,75/308,-25/77,75/308,-15/154,5/308],                 [-5/231,10/77,-25/77,100/231,-25/77,10/77,-5/231],[5/308,-15/154,75/308,-25/77,75/308,-15/154,5/308],[-1/154,3/77,-15/154,10/77,-15/154,3/77,-1/154],[1/924,-1/154,5/308,-5/231,5/308,-1/154,1/924]]).double())#2nd order
        #cm = Parameter(torch.tensor([[4/35,-9/35,3/35,1/7,-3/35],[-9/35,22/35,-12/35,-6/35,1/7],[3/35,-12/35,18/35,-12/35,3/35],[1/7,-6/35,-12/35,22/35,-9/35],[-3/35,1/7,3/35,-9/35,4/35]]).double())#3rd order
        #cv = Parameter(torch.tensor([0.2,0.2,0.2,0.2,0.2]).double())#1st order
        cv1 = Parameter(torch.tensor([-0.2,-0.1,0,0.1,0.2]).double())#2nd order
        cv2 = Parameter(torch.tensor([2/7,-1/7,-2/7,-1/7,2/7]).double())#2nd order
        cv4 = Parameter(torch.tensor([3/11,-7/11,1/11,6/11,1/11,-7/11,3/11]).double())#2nd order
        #cv = Parameter(torch.tensor([-17/105,59/210,97/210,8/21,4/105]).double())#3rd order
        self.trf1.bias = cv1
        self.trf1.weight = cm1
        self.trf2.bias = cv2
        self.trf2.weight = cm2
        self.trf4.bias = cv4
        self.trf4.weight = cm4
        
        for p in self.trf1.parameters():
            p.requires_grad=False        
        for p in self.trf2.parameters():
            p.requires_grad=False        
        for p in self.trf4.parameters():
            p.requires_grad=False
    
    def forward(self, ui, dt, dx, hidden, test):
        Nx = ui.size(2)
        Nt = ui.size(1)
        uip = torch.zeros(Nx,Nt,7).double()

        uip = reshKS(ui)
        c1 = torch.zeros(Nx,Nt,5).double()
        c1[:,:,0] = 1/12
        c1[:,:,1] = -2/3
        c1[:,:,2] = 0
        c1[:,:,3] = 2/3
        c1[:,:,4] = -1/12
        c2 = torch.zeros(Nx,Nt,5).double()
        c2[:,:,0] = -1/12
        c2[:,:,1] = 4/3
        c2[:,:,2] = -5/2
        c2[:,:,3] = 4/3
        c2[:,:,4] = -1/12
        c4 = torch.zeros(Nx,Nt,7).double()
        c4[:,:,0] = -1/6
        c4[:,:,1] = 2
        c4[:,:,2] = -13/2
        c4[:,:,3] = 28/3
        c4[:,:,4] = -13/2
        c4[:,:,5] = 2
        c4[:,:,6] = -1/6
        vis = 1
        hvis = 0.1
        if(test==1):
            f, hidden = self.lstm(uip, hidden)# Passing in the input and hidden state into the model and obtaining outputs
            #f = self.fc1(f)
            fi1 = self.fc1(f)
            fi2 = self.fc2(f)
            fi4 = self.fc4(f)
            f1 = fi1 + c1
            f2 = fi2 + c2
            f4 = fi4 + c4
            f1 = self.trf1(f1)#transform coefficients to be consistent
            f2 = self.trf2(f2)#transform coefficients to be consistent
            f4 = self.trf4(f4)#transform coefficients to be consistent
        else:
            f1 = c1
            f2 = c2
            f4 = c4
            fi1 = 0
            fi2 = 0
            fi4 = 0
        F1 = 0.5*torch.t(torch.sum(f1*(uip[:,:,1:6]**2), dim = 2)).unsqueeze(0)/dx
        F2 = torch.t(torch.sum(f2*uip[:,:,1:6], dim = 2)).unsqueeze(0)/(dx**2)
        F4 = torch.t(torch.sum(f4*uip[:,:,:], dim = 2)).unsqueeze(0)/(dx**4)
        u1 = ui - dt*(F1 + F2*vis + F4*hvis)
        pen = fi1**2 + fi2**2
        pen2 = fi4**2
        #u1 = ui - dt/dx*(dui**2-dui.roll(1,2)**2)/2

        u1p = reshKS(u1)
        if(test==1):
            f, hidden = self.lstm(u1p, hidden)# Passing in the input and hidden state into the model and obtaining outputs
            #f = self.fc1(f)
            f11 = self.fc1(f)
            f12 = self.fc2(f)
            f14 = self.fc4(f)
            f1 = f11 + c1
            f2 = f12 + c2
            f4 = f14 + c4
            f1 = self.trf1(f1)#transform coefficients to be consistent
            f2 = self.trf2(f2)#transform coefficients to be consistent
            f4 = self.trf4(f4)#transform coefficients to be consistent
        else:
            f1 = c1
            f2 = c2
            f4 = c4
            f11 = 0
            f12 = 0
            f14 = 0
        F1 = 0.5*torch.t(torch.sum(f1*(u1p[:,:,1:6]**2), dim = 2)).unsqueeze(0)/dx
        F2 = torch.t(torch.sum(f2*u1p[:,:,1:6], dim = 2)).unsqueeze(0)/(dx**2)
        F4 = torch.t(torch.sum(f4*u1p[:,:,:], dim = 2)).unsqueeze(0)/(dx**4)
        u2 = 3/4*ui + 1/4*u1 - 1/4*dt*(F1 + F2*vis + F4*hvis)
        pen += f11**2 + f12**2
        pen2 +=  f14**2
        u2p = reshKS(u2)
        if(test==1):
            f, hidden = self.lstm(u2p, hidden)# Passing in the input and hidden state into the model and obtaining outputs
            #f = self.fc1(f)
            f21 = self.fc1(f)
            f22 = self.fc2(f)
            f24 = self.fc4(f)
            f1 = f21 + c1
            f2 = f22 + c2
            f4 = f24 + c4
            f1 = self.trf1(f1)#transform coefficients to be consistent
            f2 = self.trf2(f2)#transform coefficients to be consistent
            f4 = self.trf4(f4)#transform coefficients to be consistent
        else:
            f1 = c1
            f2 = c2
            f4 = c4
            f21 = 0
            f22 = 0
            f24 = 0
        F1 = 0.5*torch.t(torch.sum(f1*(u2p[:,:,1:6]**2), dim = 2)).unsqueeze(0)/dx
        F2 = torch.t(torch.sum(f2*u2p[:,:,1:6], dim = 2)).unsqueeze(0)/(dx**2)
        F4 = torch.t(torch.sum(f4*u2p[:,:,:], dim = 2)).unsqueeze(0)/(dx**4)
        out = 1/3*ui + 2/3*u2 - 2/3*dt*(F1 + F2*vis + F4*hvis)
        pen = f21**2 + f22**2
        pen2 = f24**2
        return out, hidden, pen, pen2
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim,dtype=torch.double)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_dim,dtype=torch.double)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hc = (hidden,cell)
        return hc

# Instantiate the model with hyperparameters
model = Model(input_size=7, output_size=1, hidden_dim=32, n_layers=3)

# Define hyperparameters
lr = 0.001

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def burgEx(xgf,tgf,IC):
    Nt = xgf.size()[1]-1
    solt = torch.zeros_like(torch.t(xgf))
    solt = solt.unsqueeze(0)
    solt[:,0,:] = IC
    x_t = IC
    x_t = x_t.unsqueeze(0)
    x_t = x_t.unsqueeze(0)
    for i in range(0,Nt):
        x_t, hidden1, p1, p2 = model(x_t,dt,dx, 0, 0)
        solt[:,i+1,:] = x_t
    return solt
    
def KS(xgf,tgf,IC,al,dt,dx):
    nx = np.shape(xgf)[0]
    nt = np.shape(xgf)[1]
    if(al == 0):
        u_ex = np.zeros((int(nx/4),int((nt-1)/256+1)))
    else:
        u_ex = np.zeros((nx,nt))
    u0 = IC
    a = 0.1
    for i in range(0,nt):
        F1 = (1/12*(np.roll(u0,-2)**2)-2/3*(np.roll(u0,-1)**2) + 2/3*(np.roll(u0,1)**2)-1/12*(np.roll(u0,2)**2))/(2*dx);
        F2 = (-1/12*np.roll(u0,-2)+4/3*np.roll(u0,-1)-5/2*u0+4/3*np.roll(u0,1)-1/12*np.roll(u0,2))/(dx**2);
        F3 = a*(-1/6*np.roll(u0,-3)+2*np.roll(u0,-2)-13/2*np.roll(u0,-1)+28/3*u0-13/2*np.roll(u0,1)+2*np.roll(u0,2)-1/6*np.roll(u0,3))/(dx**4);
        u1 = u0 - dt*(F1+F2+F3);
        
        F1 = (1/12*(np.roll(u1,-2)**2)-2/3*(np.roll(u1,-1)**2) + 2/3*(np.roll(u1,1)**2)-1/12*(np.roll(u1,2)**2))/(2*dx);
        F2 = (-1/12*np.roll(u1,-2)+4/3*np.roll(u1,-1)-5/2*u1+4/3*np.roll(u1,1)-1/12*np.roll(u1,2))/(dx**2);
        F3 = a*(-1/6*np.roll(u1,-3)+2*np.roll(u1,-2)-13/2*np.roll(u1,-1)+28/3*u1-13/2*np.roll(u1,1)+2*np.roll(u1,2)-1/6*np.roll(u1,3))/(dx**4);
        u2 = 3/4*u0 + 1/4*u1 - 1/4*dt*(F1+F2+F3);
        
        F1 = (1/12*(np.roll(u2,-2)**2)-2/3*(np.roll(u2,-1)**2) + 2/3*(np.roll(u2,1)**2)-1/12*(np.roll(u2,2)**2))/(2*dx);
        F2 = (-1/12*np.roll(u2,-2)+4/3*np.roll(u2,-1)-5/2*u2+4/3*np.roll(u2,1)-1/12*np.roll(u2,2))/(dx**2);
        F3 = a*(-1/6*np.roll(u2,-3)+2*np.roll(u2,-2)-13/2*np.roll(u2,-1)+28/3*u2-13/2*np.roll(u2,1)+2*np.roll(u2,2)-1/6*np.roll(u2,3))/(dx**4);
        u0 = 1/3*u0 + 2/3*u2 - 2/3*dt*(F1+F2+F3);
        if(i%256==0 and al == 0):
            u_ex[:,int(i/256)] = u0[0::4]
        if(al==1):
            u_ex[:,i] = u0
    return u_ex

# Training Run
n_epochs = 400

errs_WE = torch.zeros(n_epochs)
errs_NN = torch.zeros(n_epochs)

loadIt = 0
if(loadIt == 1):
    model.load_state_dict(torch.load('KSNet_temp'))
    rel_err = torch.load('ks_storeError.pt')
    spe = torch.load('ks_storeEpoch.pt')
else:
    rel_err = torch.zeros(n_epochs)
    spe = 0

strt = time.time()
L = 20

S = 200
nt = len(r2)
mbs = 5

cfl = 0.01
cflf = 0.01/64
dx = 0.25
dxf = dx/4
dt = cfl*dx
dtf = cflf*dxf

T = dt*(S)
xc = torch.linspace(0,L,int(L/dx)+1,dtype=torch.double)
xcf = torch.linspace(0,L,int(L/dxf)+1,dtype=torch.double)
xc = xc[:-1]
xcf = xcf[:-1]
tc = torch.linspace(0,T,int(T/dt)+1,dtype=torch.double)
tcf = torch.linspace(0,T,int(T/dtf)+1,dtype=torch.double)
xg,tg = torch.meshgrid(xc,tc)#make the coarse grid
xgf,tgf = torch.meshgrid(xcf,tcf)#make the fine grid
for epoch in range(spe, n_epochs):
    #torch.manual_seed(7)
    strt = time.time()
    optimizer.zero_grad()#TODO: is this is the right place?
    target_seq_a = torch.zeros((1, S+1, len(xc), mbs),dtype=torch.double)
    output_a = torch.zeros_like(target_seq_a)
    output_we_a = torch.zeros_like(target_seq_a)
    pens2_a = torch.zeros((len(xc), S, 7, mbs),dtype=torch.double)
    pens1_a = torch.zeros((len(xc), S, 5, mbs),dtype=torch.double)
    for j in range(0,mbs):
        sp = np.random.randint(1600,high=(3200-S-1))
        rsn = np.random.randint(600)
        target_seq = torch.tensor(r2[sp:sp+S+1,rsn,:]).unsqueeze(0)

        batch_size = xc.size(0)
        hidden1 = model.init_hidden(batch_size)
        h1_we = model.init_hidden(batch_size)
        
        x_t = target_seq[0,0,:]
        output = torch.zeros_like(target_seq)
        output_we = torch.zeros_like(target_seq)
        x_t = x_t.unsqueeze(0)
        x_t = x_t.unsqueeze(0)
        y_t = x_t[:,:,:]
        pens1 = torch.zeros((len(xc), S, 5),dtype=torch.double)
        pens2 = torch.zeros((len(xc), S, 7),dtype=torch.double)
        output[:,0,:] = x_t
        output_we[:,0,:] = x_t
        for i in range(0,S):
            #print(i)
            x_t, hidden1, pen1, pen2 = model(x_t,dt,dx, hidden1, 1)
            y_t, h1_we, abq, abq2 = model(y_t,dt,dx, h1_we, 0)
            pens1[:,i,:] = pen1.squeeze()
            pens2[:,i,:] = pen2.squeeze()
            output[:,i+1,:] = x_t
            output_we[:,i+1,:] = y_t
        target_seq_a[:,:,:,j] = target_seq
        output_a[:,:,:,j] = output
        output_we_a[:,:,:,j] = output_we.detach()
        pens2_a[:,:,:,j] = pens2
        pens1_a[:,:,:,j] = pens1
    loss = criterion(output_a.flatten(), target_seq_a.flatten()) + 0.1*(pens1_a.mean()+pens2_a.mean())#maybe penalize 1-norm
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    errs_NN[epoch] = criterion(output_a.flatten(), target_seq_a.flatten()).detach()
    errs_WE[epoch] = criterion(output_we_a.flatten(), target_seq_a.flatten()).detach()
    rel_err[epoch] = (errs_NN[epoch].detach())/(errs_WE[epoch].detach())
    if epoch%1 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(rel_err[epoch].detach()))
    plt.clf()
    plt.semilogy(rel_err[rel_err!=0].detach())
    plt.pause(0.01)
    torch.save(model.state_dict(), 'KSNet_temp')
    torch.save(rel_err, 'ks_storeError.pt')
    torch.save(epoch, 'ks_storeEpoch.pt')
    enddt = time.time()
    print('Time of Epoch: ', enddt-strt)
torch.save(model.state_dict(), 'KSNet')
