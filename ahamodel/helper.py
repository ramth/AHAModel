## ---------------------------
## Energy functions
##
## Authors: Ramprasath Rajagopal
##
## Email: rrajagop@bu.edu
## ---------------------------

import numba
import numpy as np

def pH_to_mu(a,pH,pK = 7):
    ''' computation of chemical potential, mu, for both chemical species
    '''
    mu_ha = a + np.log(1/(1+ 10**(pH - pK))) #Low pH more negative (HA) 
    mu_a = a + np.log((10**(pH-pK))/(1+ 10**(pH - pK))) #High pH more negative (A-)
    return(mu_ha,mu_a)

@numba.njit(cache=True)
def en_s(Sp,E,Lx,Ly,M,H):
    ''' Faster routine for energy computation using numba
    '''
    S = Sp+1 # shift to 0,1,2 from -1,0,1 to use as valid indices
    # HA Vac A- 
    # -1 0   1
    # 0  1   2
    for i in range(1,Lx-1):
        for j in range(1,Ly-1):
            E[i,j] = M[S[i,j],S[i,j+1]]+ M[S[i,j],S[i,j-1]] + M[S[i,j],S[i+1,j]] + M[S[i,j],S[i-1,j]] + H[S[i,j]]
    
    #Left border, j-1 -> Ly-1
    j = 0; jp = 1; jm = Ly-1
    for i in range(1,Lx-1):
        E[i,j] = M[S[i,j],S[i,jp]]+ M[S[i,j],S[i,jm]] + M[S[i,j],S[i+1,j]] + M[S[i,j],S[i-1,j]] + H[S[i,j]]
    
    #Right border
    j = Ly-1; jp = 0; jm = Ly-2
    for i in range(1,Lx-1):
        E[i,j] = M[S[i,j],S[i,jp]]+ M[S[i,j],S[i,jm]] + M[S[i,j],S[i+1,j]] + M[S[i,j],S[i-1,j]] + H[S[i,j]]
        
        
    #Top border
    i = 0; ip = 1; im = Lx-1
    for j in range(1,Ly-1):
        E[i,j] = M[S[i,j],S[i,j+1]]+ M[S[i,j],S[i,j-1]] + M[S[i,j],S[ip,j]] + M[S[i,j],S[im,j]] + H[S[i,j]]
        
    #Bottom border
    i = Lx-1; ip = 0; im = Lx-2
    for j in range(1,Ly-1):
        E[i,j] = M[S[i,j],S[i,j+1]]+ M[S[i,j],S[i,j-1]] + M[S[i,j],S[ip,j]] + M[S[i,j],S[im,j]] + H[S[i,j]]
    
        
    #Corners
    E[0,0] = M[S[0,0],S[0,1]]+ M[S[0,0],S[0,Ly-1]] + M[S[0,0],S[1,0]] + M[S[0,0],S[Lx-1,0]] + H[S[0,0]]
        
    E[Lx-1,0] = M[S[Lx-1,0],S[Lx-1,1]]+ M[S[Lx-1,0],S[Lx-1,Ly-1]]\
            +M[S[Lx-1,0],S[0,0]] + M[S[Lx-1,0],S[Lx-2,0]] + H[S[Lx-1,0]]
    
    E[0,Ly-1] = M[S[0,Ly-1],S[0,0]]+ M[S[0,Ly-1],S[0,Ly-2]]\
                + M[S[0,Ly-1],S[1,Ly-1]] + M[S[0,Ly-1],S[Lx-1,Ly-1]] + H[S[0,Ly-1]]
    
    E[Lx-1,Ly-1] = M[S[Lx-1,Ly-1],S[Lx-1,0]]\
                    + M[S[Lx-1,Ly-1],S[Lx-1,Ly-2]]\
                    + M[S[Lx-1,Ly-1],S[0,Ly-1]]\
                    + M[S[Lx-1,Ly-1],S[Lx-2,Ly-1]] + H[S[Lx-1,Ly-1]]

def en_array(self,S): # energy of entire array, vectorized
    ''' Equivalent numpy version for testing
    '''
    SN = np.roll(S, 1, axis = 1)
    SS = np.roll(S, -1, axis = 1)
    SW = np.roll(S, 1, axis = 0)
    SE = np.roll(S, -1, axis = 0) 
    en = -0.5*self.J*( SN*S*(1 - SN*S)        
            # The negative sign is so J < 0 corresponds to attraction
             + SS*S*(1 - SS*S)
             + SW*S*(1 - SW*S)
             + SE*S*(1 - SE*S)) + self.e0_hext(S)    
    # Check whether it is + or -
    return(en)

def complexation_index_f(spin):
    ''' This index is defined in the text, used to quantify bond formation
    '''
    nearest_neighbor = ((0,1),(0,-1),(1,0),(-1,0)) 
    complex_bonds = 0
    other_bonds = 0 
    total_HA = 0
    total_A = 0 
    len_x, len_y = spin.shape
   
    for di,dj in nearest_neighbor: 
        bonds = spin*np.roll(spin, (di,dj), (0,1))
        complex_bonds += np.sum(bonds == -1)
        other_bonds += np.sum(bonds == 1)

    complex_bonds = complex_bonds/2
    other_bonds = other_bonds/2

    return complex_bonds/(complex_bonds + other_bonds)

