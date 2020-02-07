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

from HelperFunctions import makeICdsc, resh, wenoCoeff, exactSol, randGrid
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

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True).double()
        # Fully connected layer
        self.fc1 = nn.Linear(hidden_dim, hidden_dim).double()
        self.fc2 = nn.Linear(hidden_dim, 5).double()
        self.trf = nn.Linear(5, 5).double()
        #cm = Parameter(torch.tensor([[0.8,-0.2,-0.2,-0.2,-0.2],[-0.2,0.8,-0.2,-0.2,-0.2],[-0.2,-0.2,0.8,-0.2,-0.2],[-0.2,-0.2,-0.2,0.8,-0.2],[-0.2,-0.2,-0.2,-0.2,0.8]]).double())#1st order
        cm = Parameter(torch.tensor([[0.4,-0.4,-0.2,0,0.2],[-0.4,0.7,-0.2,-0.1,0],[-0.2,-0.2,0.8,-0.2,-0.2],[0,-0.1,-0.2,0.7,-0.4],[0.2,0,-0.2,-0.4,0.4]]).double())#2nd order
        #cm = Parameter(torch.tensor([[4/35,-9/35,3/35,1/7,-3/35],[-9/35,22/35,-12/35,-6/35,1/7],[3/35,-12/35,18/35,-12/35,3/35],[1/7,-6/35,-12/35,22/35,-9/35],[-3/35,1/7,3/35,-9/35,4/35]]).double())#3rd order
        #cv = Parameter(torch.tensor([0.2,0.2,0.2,0.2,0.2]).double())#1st order
        cv = Parameter(torch.tensor([0.1,0.15,0.2,0.25,0.3]).double())#2nd order
        #cv = Parameter(torch.tensor([-17/105,59/210,97/210,8/21,4/105]).double())#3rd order
        self.trf.bias = cv
        self.trf.weight = cm
        
        for p in self.trf.parameters():
            p.requires_grad=False
    
    def forward(self, ui, dt, dx, hidden, test):
        Nx = ui.size(2)
        Nt = ui.size(1)
        uip = torch.zeros(Nx,Nt,5).double()

        uip = resh(ui)
        ci = wenoCoeff(uip)
        cm = torch.zeros_like(uip)
        cm[:,:,0] =  2/60
        cm[:,:,1] =-13/60
        cm[:,:,2] = 47/60
        cm[:,:,3] = 27/60
        cm[:,:,4] = -3/60
        if(test==1):
            f, hidden = self.lstm(uip, hidden)# Passing in the input and hidden state into the model and obtaining outputs
            #f = self.fc1(f)
            fi = self.fc2(f)
            f = fi + cm
            f = self.trf(f)#transform coefficients to be consistent
        else:
            f = ci
            fi = 0
        dui = torch.t(torch.sum(f*uip, dim = 2)).unsqueeze(0)
        u1 = ui - dt/dx*(dui-dui.roll(1,2))
        
        u1p = resh(u1)
        c1 = wenoCoeff(u1p)
        if(test==1):
            f, hidden = self.lstm(u1p, hidden)# Passing in the input and hidden state into the model and obtaining outputs
            #f = self.fc1(f)
            f1 = self.fc2(f)
            f = f1 + cm
            f = self.trf(f)#transform coefficients to be consistent
        else:
            f = c1
            f1 = 0
        du1 = torch.t(torch.sum(f*u1p, dim = 2)).unsqueeze(0)
        u2 = 3/4*ui + 1/4*u1 - 1/4*dt/dx*(du1-du1.roll(1,2))
               
        u2p = resh(u2)
        c2 = wenoCoeff(u2p)
        if(test==1):
            f, hidden = self.lstm(u2p, hidden)# Passing in the input and hidden state into the model and obtaining outputs
            #f = self.fc1(f)
            f2 = self.fc2(f)
            f = f2 + cm
            f = self.trf(f)#transform coefficients to be consistent
        else:
            f = c2
            f2 = 0
        du2 = torch.t(torch.sum(f*u2p, dim = 2)).unsqueeze(0)
        out = 1/3*ui + 2/3*u2 - 2/3*dt/dx*(du2-du2.roll(1,2))
        return out, hidden, fi, f1, f2
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim,dtype=torch.double)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_dim,dtype=torch.double)
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hc = (hidden,cell)
        return hc

# Instantiate the model with hyperparameters
model = Model(input_size=5, output_size=1, hidden_dim=32, n_layers=3)

# Define hyperparameters
n_epochs = 400
lr = 0.001

# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

errs_WE = torch.zeros(n_epochs)
errs_NN = torch.zeros(n_epochs)
rel_err = torch.zeros(n_epochs)

randgr = 0
S = 100
L = 1
def burgEx(xgf,tgf,IC):
    Nt = xgf.size()[1]-1
    solt = torch.zeros_like(torch.t(xgf))
    solt = solt.unsqueeze(0)
    solt[:,0,:] = IC
    x_t = IC
    x_t = x_t.unsqueeze(0)
    x_t = x_t.unsqueeze(0)
    for i in range(0,Nt):
        x_t, hidden1, fi, f1, f2 = model(x_t,dt,dx, 0, 0)
        solt[:,i+1,:] = x_t
    return solt

def compTV(u):
    dif = abs(u.roll(1)-u)
    return torch.sum(dif)
# Training Run
if(randgr==0):
    cfl = 0.5
    dx = 0.01
    dt = cfl*dx
    T = dt*(S)
    xc = torch.linspace(0,L,int(L/dx)+1,dtype=torch.double)
    xcf = torch.linspace(0,L,int(L/dx)*4+1,dtype=torch.double)
    xc = xc[:-1]
    xcf = xcf[:-1]
    tc = torch.linspace(0,T,int(T/dt)+1,dtype=torch.double)
    tcf = torch.linspace(0,T,int(T/dt)*4+1,dtype=torch.double)
    xg,tg = torch.meshgrid(xc,tc)#make the coarse grid
    xgf,tgf = torch.meshgrid(xcf,tcf)#make the fine grid
batch_size = xc.size(0)
IC_fx = makeICdsc(L)
mbs = 5

TV = torch.zeros((n_epochs,mbs),dtype=torch.double)
all_ratio = torch.zeros((n_epochs,mbs),dtype=torch.double)
for epoch in range(0, n_epochs):
    optimizer.zero_grad()
    target_seq_a = torch.zeros((1,len(tc),len(xc),mbs),dtype=torch.double)
    output_a = torch.zeros((1,len(tc),len(xc),mbs),dtype=torch.double)
    output_we_a = torch.zeros((1,len(tc),len(xc),mbs),dtype=torch.double)
    fis_a = torch.zeros((len(xc),S,5,mbs),dtype=torch.double)
    f1s_a = torch.zeros((len(xc),S,5,mbs),dtype=torch.double)
    f2s_a = torch.zeros((len(xc),S,5,mbs),dtype=torch.double)
    for j in range(0,mbs):
            
        solt = torch.t(IC_fx((xg-tg)%L)).unsqueeze(0)
        IC = solt[0,0,:]
        target_seq = solt
        
        hidden1 = model.init_hidden(batch_size)
        h1_we = model.init_hidden(batch_size)
        
        x_t = solt[0,0,:]
        output = torch.zeros_like(target_seq)
        output_we = torch.zeros_like(target_seq)
        x_t = x_t.unsqueeze(0)
        x_t = x_t.unsqueeze(0)
        y_t = x_t[:,:,:]
        fis = torch.zeros((len(xc), S, 5),dtype=torch.double)
        f1s = torch.zeros((len(xc), S, 5),dtype=torch.double)
        f2s = torch.zeros((len(xc), S, 5),dtype=torch.double)
        output[:,0,:] = x_t
        output_we[:,0,:] = x_t
        for i in range(0,S):
            x_t, hidden1, fi, f1, f2 = model(x_t,dt,dx, hidden1, 1)
            y_t, h1_we, fi_we, fi_we, fi_we = model(y_t,dt,dx, h1_we, 0)
            fis[:,i,:] = fi.squeeze()
            f1s[:,i,:] = f1.squeeze()
            f2s[:,i,:] = f2.squeeze()
            output[:,i+1,:] = x_t
            output_we[:,i+1,:] = y_t
        target_seq_a[:,:,:,j] = target_seq
        output_a[:,:,:,j] = output
        output_we_a[:,:,:,j] = output_we
        all_ratio[epoch,j] = criterion(output.flatten(), target_seq.flatten())/criterion(output_we.flatten(), target_seq.flatten())
        fis_a[:,:,:,j] = fis
        f1s_a[:,:,:,j] = f1s
        f2s_a[:,:,:,j] = f2s
    loss = criterion(output_a.flatten(), target_seq_a.flatten()) + 0.001*(fis_a**2+f1s_a**2+f2s_a**2).mean()#maybe penalize 1-norm
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly
    rel_err[epoch] = ((criterion(output_a.flatten(), target_seq_a.flatten()))/(criterion(output_we_a.flatten(), target_seq_a.flatten()))).detach()
    if epoch%10 == 0:
        '''
        plt.clf()
        plt.plot(output.flatten().detach())
        plt.plot(target_seq.flatten().detach())
        plt.pause(0.001)
        '''
        
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(rel_err[epoch]))
    plt.clf()
    #plt.plot(torch.cat((rel_errppp,rel_err[rel_err!=0])).detach())
    plt.semilogy(rel_err[rel_err!=0].detach())
    plt.pause(0.01)
