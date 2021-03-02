import numpy as np
import matplotlib.pyplot as plt
import h5py
import datetime
import pandas as pd
from helper import pH_to_mu

class LatticeHDF5WriterG:
    '''Class to implement hdf5 writing used by GridRun
    '''
    def __init__(self,fname,attrs,c0,c1):
        self.f = h5py.File(fname,'w')
        self.c0_idx = 0 
        self.c1_idx = 0
        self.c0 = c0
        self.c1 = c1

        if attrs['init_mat'] is None:
            attrs['init_mat'] = 'random'
        #copy the common attributes
        for key in attrs:
            self.f.attrs[key] = attrs[key]

        self.f.attrs['c0'] = self.c0
        self.f.attrs['c1'] = self.c1

        #self.f.create_dataset('Initial Matrix', data = init_mat)
        self.g = None

    #creates a new group to hold c0 scan
    def new_c0(self):
        self.g = self.f.create_group('{0}{1}'.format(self.c0, self.c0_idx))
        self.c1_idx = 0
        self.c0_idx += 1

    def write_data(self, spin_mat, measurements,params,seed):
        #split it into two binary matrices
        spin_1 = np.packbits(spin_mat == 1)
        spin_2 = np.packbits(spin_mat == -1)
        h = self.g.create_group('{0}{1}'.format(self.c1,self.c1_idx))
        h.attrs[self.c0] = params[self.c0]
        h.attrs[self.c1] = params[self.c1] 
        #print('seed',str(seed)) 
        h.attrs['seed'] = str(seed)
        h.create_dataset('spin_1', data = spin_1)
        h.create_dataset('spin_2', data = spin_2)
        h.create_dataset('measurements', data = measurements) 
        self.c1_idx += 1

    def close(self):
        self.f.close()



class LatticeHDF5ReaderG:

    def __init__(self,fname,average = False): 
        self.f = h5py.File(fname,'r') 
        
        data_list = []
        self.attrs = {}

        #Figure out which were the parameters that were varied
        self.c0 = self.f.attrs['c0']
        self.c1 = self.f.attrs['c1']        
        c0_idxs = self.f.keys()

        #Fixed variables
        self.attrs['Lx'] = self.f.attrs['Lx']
        self.attrs['Ly'] = self.f.attrs['Ly']
        self.attrs['num_flips'] = self.f.attrs['num_flips']
        
        for key in self.f.attrs:
            if key not in ['c0','c1','Lx','Ly','num_flips','init_mat']:
                self.attrs[key] = self.f.attrs[key]
        
        for c0_idx in self.f.keys():
            if c0_idx == 'Initial Matrix':
                continue
            c0_group = self.f[c0_idx]

            for c1_idx in c0_group.keys():

                c0 = c0_group[c1_idx].attrs[self.c0]
                c1 = c0_group[c1_idx].attrs[self.c1]
                if average:
                    rho_vec = np.mean(np.array(c0_group[c1_idx]['measurements'])[-average:,:],0)
                else:
                    rho_vec = np.array(c0_group[c1_idx]['measurements'])[-1,:]

                rho_mat = np.array(c0_group[c1_idx]['measurements'])

                #not mean variance
                rho_1_var = np.var(rho_mat[:,0] + rho_mat[:,2]/2) 
                rho_2_var = np.var(rho_mat[:,1] + rho_mat[:,3]/2) 

                order_var = np.var(np.abs(rho_mat[:,0] - rho_mat[:,1] 
                                    - rho_mat[:,2] + rho_mat[:,3])/2)
                rho_1 = (rho_vec[0] + rho_vec[2])/2
                rho_2 = (rho_vec[1] + rho_vec[3])/2

                #measurements.append((p_ha_even, p_a_even, p_ha_odd, 
                #                    p_a_odd, com_idx, i))
               
                #Make order_ha (order_1) positive. Adjust the order_a (order_2) accordingly
                order_ha = (rho_vec[0] - rho_vec[2])/2
                if order_ha != 0:
                    order_a = np.sign(order_ha)*(rho_vec[3] - rho_vec[1])/2  
                else:
                    order_a = 0 #Not able to enforce the sign constraint here, should really remove this point 
                order_ha = np.abs(order_ha)
                
                order = np.abs((rho_vec[0] - rho_vec[1] - rho_vec[2] + rho_vec[3])/2) #Old order parameter, is the sum of new ones
                com_idx = rho_vec[4]
                #order_ha
                #order_a = 
                #(0,1)

                data_list.append([c0, c1,rho_1, rho_2, np.abs(order),c0_idx, c1_idx,com_idx,order_ha,order_a,rho_1_var, rho_2_var,order_var]) 
        self.table = pd.DataFrame(data_list, 
                            columns = [self.c0, self.c1, 'rho1', 'rho2','order','c0_idx','c1_idx','com_idx',
                                'order_ha','order_a','rho_1_var','rho_2_var','order_var'])
        self.table['com_idx'] = self.table['com_idx'].fillna(0)

    def get_data(self, c0_idx, c1_idx):
        c0_idx_s = self.c0 + str(c0_idx) 
        c1_idx_s = self.c1 + str(c1_idx)
        ds = self.f[c0_idx_s][c1_idx_s]
        spin = np.unpackbits(ds['spin_1'][...])*1\
                        + np.unpackbits(ds['spin_2'][...])*-1
        spin = np.reshape(spin, (self.attrs['Lx'], self.attrs['Ly']))
        c0 = ds.attrs[self.c0]
        c1 = ds.attrs[self.c1]
        return c0, c1, ds['measurements'], spin

    def get_data_value(self, c0, c1):
        temp = self.table.copy() 
        c1_diff = (temp[self.c1] - c1)/(temp[self.c1].max() - temp[self.c1].min())
        c0_diff = (temp[self.c0] - c0)/(temp[self.c0].max() - temp[self.c0].min())
        temp['dist'] = np.sqrt(c1_diff**2 + c0_diff**2) 
        row = temp.sort_values('dist').iloc[0,:]#trick to treat index as regular column 
        print('c0:{0:0.3f} and c1:{1:0.3f}'.format(row[self.c0],row[self.c1]))
        print('c0id:{0} and c1id:{1}'.format(row['c0_idx'],row['c1_idx'])) 
        return self.get_data(row['c0_idx'],row['c1_idx'])

    def get_data_constant_c0(self, c0):
        temp = self.table.copy()
        c0_diff = np.abs((temp[self.c0] - c0)/(temp.index.max() - temp.index.min()))
        temp['dist'] = c0_diff 
        row = temp.sort_values('dist').iloc[0] 
        c0_idx = row['c0_idx']
        temp.set_index('c0_idx', inplace=True)
        return temp.loc[c0_idx]



def dimerization_index(spin):
    #measure the bonds
  
    nearest_neighbor = ((0,1),(0,-1),(1,0),(-1,0)) 
    dimer_bonds = 0
    other_bonds = 0 
    total_HA = 0
    total_A = 0
    len_x, len_y = spin.shape

    for i in range(len_x):
        for j in range(len_y):
            cur = spin[i,j] 

            if spin[i,j] == 0:
                continue   
            elif spin[i,j] == -1:
                total_HA += 1
            elif spin[i,j] == 1:
                total_A += 1

            for di,dj in nearest_neighbor:
                nn = spin[(i+di)%len_x, (j+dj)%len_y]
                if nn == 0:
                    continue
                elif nn == cur:
                    other_bonds += 1
                else:
                    dimer_bonds += 1
    
    #Account for double counting 
    dimer_bonds = dimer_bonds/2
    other_bonds = other_bonds/2

    return (dimer_bonds/(dimer_bonds + other_bonds))

def dimerization_index_f(spin):

    nearest_neighbor = ((0,1),(0,-1),(1,0),(-1,0)) 
    dimer_bonds = 0
    other_bonds = 0 
    total_HA = 0
    total_A = 0 
    len_x, len_y = spin.shape
   
    for di,dj in nearest_neighbor: 
        bonds = spin*np.roll(spin, (di,dj), (0,1))
        dimer_bonds += np.sum(bonds == -1)
        other_bonds += np.sum(bonds == 1)

    dimer_bonds = dimer_bonds/2
    other_bonds = other_bonds/2

    return dimer_bonds/(dimer_bonds + other_bonds)

def surface_energy(spin,J,K,a,pH,pK):
   
    mu_a,mu_ha = pH_to_mu(a,pH,pK) 
    nearest_neighbor = ((0,1),(0,-1),(1,0),(-1,0)) 
    E = 0 
    len_x, len_y = spin.shape

    for i in range(len_x):
        for j in range(len_y):
            cur = spin[i,j] 

            if spin[i,j] == 0:
                continue   
            elif spin[i,j] == -1: #HA
                E += -mu_ha 
            elif spin[i,j] == 1: #A-
                E += -mu_a

            for di,dj in nearest_neighbor:
                nn = spin[(i+di)%len_x, (j+dj)%len_y]
                if nn == 0:
                    continue
                elif nn == cur and nn == 1:
                    E += K/2 
                else:
                    E += -J/2
    
    #Account for double counting 

    return E
   
