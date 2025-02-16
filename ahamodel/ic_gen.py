## ---------------------------
## Lattice initial condition functions
##
## Authors: Ramprasath Rajagopal
##
## Email: rrajagop@bu.edu
## ---------------------------

import numpy as np 
from numpy.random import Generator,PCG64

def random_3state_ic(rgen,Lx,Ly):
    ''' Generates an matrix with a set percentage of being -1, 0, or 1
    '''
    xlow = .49                  # With this choice, only 2 percent of the sites are free
    xhigh = .51
    x = rgen.random((Lx, Ly))           # Generates a number between 0 and 1
    Spin = -1*(x <= xlow) + 0*((x > xlow) & (x < xhigh)) + 1*(x >= xhigh)
    return Spin

def random_2state_ic(rgen,Lx,Ly):
    ''' Generates an matrix with a set percentage of being -1, 1
    '''
    xlow = .50 
   
    x = rgen.random((Lx, Ly))           # Generates a number between 0 and 1
    Spin = -1*(x <= xlow) + 1*(x > xlow)
    return Spin

def checkerboard_ic(Lx,Ly,parity=0):
    '''Create either even or odd checkerboard pattern with 1s and -1s
    '''
    mask = np.zeros([Lx,Ly],dtype = np.int)
    for i in range(Lx):
        for j in range(Ly):
            mask[i,j] = 2*((i+j+parity)%2)-1 
                
    return mask
