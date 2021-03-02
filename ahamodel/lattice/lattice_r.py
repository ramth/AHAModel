import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.random import randint
import abc
from numpy.random import Generator,PCG64
import datetime
import multiprocessing as mp
from data_io import *
import numba
from patterns_r import *
from energy_r import *





def pH_to_mu(a,pH,pK = 7):
    mu_ha = a + np.log(1/(1+ 10**(pH - pK))) #Low pH more negative (HA) This is accounting for the werid sign and species flipping in the latter code
    mu_a = a + np.log((10**(pH-pK))/(1+ 10**(pH - pK))) #High pH more negative (A-)
    return(mu_ha,mu_a)
    


#I.Cs


       


class LatticeModelSimple(LatticeModelTemplate):
    '''Lattice Model Class based on the single spin method outlined by Onuttom'''
    

    def energy_local(self,i,j):
        ''' Compute energy of a single spin based on its nearest neighbors and itself'''
        S0 = self.Spin[i,j]               # Spin at the center
        SN = self.Spin[i, (j+1)%self.Ly]       # North
        SS = self.Spin[i, (j-1)%self.Ly]       # South
        SE = self.Spin[(i-1)%self.Lx, j]       # East
        SW = self.Spin[(i+1)%self.Lx, j]       # West
        e_local = -(self.J)*( self.s1s2(S0,SN) + self.s1s2(S0,SS) + self.s1s2(S0,SE) + self.s1s2(S0,SW)) + self.hext_ss(S0)  # Check this!
        return(e_local)
    
    
    def hext_ss(self,S1) :    # external magnetic field or chemical potential
        ''' Compute chemical potential of species'''
        dict = { 1: -self.mu_1, 0: 0, -1: -self.mu_2}      # Using a dictionary to return one of the three cases
        return dict[S1]
                
                
    def iteration(self):
        i = randint(self.Lx)
        j = randint(self.Ly)
    
        en_old = self.energy_local(i,j)
        old_s = self.Spin[i,j]
        self.Spin[i,j] = (old_s + Generator.choice([1,2]) + 1)%3 -1
        en_new = self.energy_local(i,j)
        deltaE = en_new - en_old
    
        if deltaE > 0:
            p = np.exp(-deltaE)     
            if random.rand() > p :
                self.Spin[i,j] = old_s
    

class LatticeModelVectorized(LatticeModelTemplate):
    '''Lattice Model Class based on the parallel method'''
     
    def __init__(self,Lx,Ly,mu_1,mu_2,J,init_mat,K=0,pK=7, seed = 1234567):

        super(LatticeModelVectorized,self).__init__(Lx,Ly,mu_1,mu_2,
                J,init_mat,K,pK)
        self.Spin_flip = self.Spin.copy()
        bg = PCG64(seed)
        self.rgen = Generator(bg) 

    def e0_hext(self,S):         
        # Vectorized version of external field term -h
        e0 = -self.mu_1*(S > 0) - self.mu_2*(S < 0)\
            + 0*(S == 0) 
        # sign error fixed
        return e0

    def en_array(self,S): # energy of entire array, vectorized
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
    
    def en_subarray(self,S, a, b) : 
        # energy of sub-array set by (a,b) vectorized
        return(self.en_array(S)[a: :2, b: :2])

    #From NaPot-Test-v3.ipynb
    def MonteCarlo_sublattice_flip(self, a, b) :
        Spin_flip = 1*self.Spin                 # initiliaze local copy        
        S_ab_old = self.Spin[ a: :2, b: :2]      # define sublattice

        en_sub_old = self.en_subarray(Spin_flip,a, b)
        Spin_flip[ a: :2, b: :2]  =\
            (S_ab_old + self.rgen.choice([1,2], 
            [self.Lx//2, self.Ly//2]) + 1)%3 - 1  
        en_sub_new = self.en_subarray(Spin_flip,a, b)
        deltaE = en_sub_new - en_sub_old
        p = np.exp(-deltaE)
        r = self.rgen.random((self.Lx//2, self.Ly//2))
        self.Spin[ a: :2, b: :2] = Spin_flip[ a: :2, b: :2] *(p >= r)\
                + S_ab_old*( p < r)
        return(self.Spin)

    def MonteCarlo_flip(self):
        self.Spin = self.MonteCarlo_sublattice_flip( 0, 0)
        self.Spin = self.MonteCarlo_sublattice_flip(1, 0)
        self.Spin = self.MonteCarlo_sublattice_flip(0, 1)
        self.Spin = self.MonteCarlo_sublattice_flip(1, 1)

    def run(self,num_trials,div_measurements):
        measurements = []
        j = 0
        for i in range(num_trials):
            if i%div_measurements == 0:
                measurements.append(self.measure())
                j = j+1
            self.MonteCarlo_flip()
        return np.array(measurements)
    
    def measure(self):
        even_sites = self.Spin[self.even_mask]
        odd_sites = self.Spin[self.odd_mask]
        
        p1_even = np.sum(even_sites==1)/even_sites.size
        p2_even = np.sum(even_sites==-1)/even_sites.size
        
        p1_odd = np.sum(odd_sites==1)/odd_sites.size
        p2_odd = np.sum(odd_sites==-1)/odd_sites.size
        
        return (p1_even, p2_even, p1_odd, p2_odd)
    
    def print_measurements(self):
        p1_even, p2_even, p1_odd, p2_odd  = self.measure()
        
        col_names = ('','even', 'odd','Delta')# even starts on (0,0)
        
        #print in table form : https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
        format_str1 = '{:>2}'+'{:>15}'*3
        format_str2 = '{:>2}{:>15.3f}{:>15.3f}{:>15.3f}'
        print(format_str1.format(*col_names))
        print(format_str2.format('A-',p1_even, p1_odd,p1_even-p1_odd))
        print(format_str2.format('HA',p2_even, p2_odd,p2_even-p2_odd))
        
    def plot(self):
        im = imshow(self.Spin, vmin = -1, vmax = 1)
        plt.show()
        
class LatticeModelVectorized1(LatticeModelTemplate):
    '''Lattice Model Class based on the parallel method'''
     
    def __init__(self,Lx,Ly,mu_1,mu_2,J,init_mat,K,pK=7):

        super(LatticeModelVectorized1,self).__init__(Lx,Ly,mu_a,mu_ha,
                J,init_mat,K,pK)
        self.Spin_flip = self.Spin.copy()
        self.en_old = np.zeros([self.Lx, self.Ly])
        self.en_new = np.zeros([self.Lx, self.Ly])
        self.H = np.array([-self.mu_a, 0, -self.mu_ha]) #A-, Vac, HA 
        self.M = np.array([[0,0,J],[0,0,0],[J,0,K]])
        sq = np.random.SeedSequence()

        self.seed = sq1.entropy
        bg = PCG64(self.seed)
        self.rgen = Generator(bg) 
        self.shifts = ((0,0),(1,0),(0,1),(1,1))


    #From NaPot-Test-v3.ipynb
    def MonteCarlo_sublattice_flip(self, a, b) :
        Spin_flip = 1*self.Spin                 # initiliaze local copy        
        S_ab_old = self.Spin[ a: :2, b: :2]      # define sublattice

        en_s(Spin_flip, self.en_old, self.Lx, self.Ly,self.M, self.H)
        
        Spin_flip[ a: :2, b: :2]  =\
            (S_ab_old + self.rgen.choice([1,2], 
            [self.Lx//2, self.Ly//2]) + 1)%3 - 1  
        
        en_s(Spin_flip, self.en_new, self.Lx, self.Ly,self.M,self.H)

        deltaE = self.en_new[a::2,b::2] - self.en_old[a::2,b::2]
        p = np.exp(-deltaE)
        r = self.rgen.random((self.Lx//2, self.Ly//2))
        self.Spin[ a: :2, b: :2] = Spin_flip[ a: :2, b: :2] *(p >= r)\
                + S_ab_old*( p < r)

    def MonteCarlo_flip(self):
        indices = [0,1,2,3]

        np.random.shuffle(indices)
        #print(indices)
        for idx in indices:
            x,y = self.shifts[idx]
            self.MonteCarlo_sublattice_flip(x, y)

    def run(self,num_trials,div_measurements):
        measurements = []
        j = 0
        for i in range(num_trials):
            if i%div_measurements == 0 or i >= num_trials - 101:
                            
                p1_even, p2_even, p1_odd, p2_odd = self.measure()
                measurements.append((p1_even, p2_even, p1_odd, p2_odd,i))
                j = j+1
            self.MonteCarlo_flip()
        return np.array(measurements)
    
    def measure(self):
        even_sites = self.Spin[self.even_mask]
        odd_sites = self.Spin[self.odd_mask]
        
        p1_even = np.sum(even_sites==1)/even_sites.size
        p2_even = np.sum(even_sites==-1)/even_sites.size
        
        p1_odd = np.sum(odd_sites==1)/odd_sites.size
        p2_odd = np.sum(odd_sites==-1)/odd_sites.size
        
        return (p1_even, p2_even, p1_odd, p2_odd)
    
    def print_measurements(self):
        p1_even, p2_even, p1_odd, p2_odd  = self.measure()
        
        col_names = ('','even', 'odd','Delta')# even starts on (0,0)
        
        #print in table form : https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
        format_str1 = '{:>2}'+'{:>15}'*3
        format_str2 = '{:>2}{:>15.3f}{:>15.3f}{:>15.3f}'
        print(format_str1.format(*col_names))
        print(format_str2.format('A-',p1_even, p1_odd,p1_even-p1_odd))
        print(format_str2.format('HA',p2_even, p2_odd,p2_even-p2_odd))
        


def scc_pH_process(a,pH,mu_1,mu_2, J, num_flips,meas_ivl, Lx, Ly, init_mat,K=0):
    model = LatticeModelVectorized1(Lx = Lx, 
                                    Ly = Ly, 
                                    mu_1 = mu_1,
                                    mu_2 = mu_2, 
                                    J = J,
                                    init_mat = init_mat,
                                    K=K) 
    meas = model.run(num_flips,meas_ivl)
        #model.plot()
    return (meas, model.Spin, a, pH)

def lattice_processG(c0, c1, params):
    mu_1, mu_2 = pH_to_mu(params['a'],params['pH'],params['pK'])        
    model = LatticeModelVectorized1(Lx = params['Lx'], 
                                    Ly = params['Ly'], 
                                    mu_1 = mu_1,
                                    mu_2 = mu_2, 
                                    J = params['J'],
                                    init_mat = params['init_mat'],
                                    K=params['K']) 
    meas = model.run(params['num_flips'],params['meas_ivl'])
        #model.plot()
    return (meas, model.Spin, params[c0], params[c1])
#def grid_search(a_array, J_array, K_array) 

def scc_lattice_parallel(a_array, pH_array, params, init_mat, fname,pK):
     
    len_a = len(a_array)
    len_pH = len(pH_array)
    measurements = np.zeros((len_a,len_pH,4))
    
    writer = LatticeHDF5Writer(fname, params, init_mat)

    for i, a in enumerate(a_array):
        writer.new_a(a)
        print('{}% done'.format((i/len_a)*100)) 
        p_params = []
        for pH in pH_array:
            mu_1, mu_2 = pH_to_mu(a,pH,pK)        
            p_params.append([a,pH,mu_1,mu_2,params['J'],params['num_flips'],params['meas_ivl'],
                            params['Lx'], params['Ly'],init_mat,params['K']])
        with mp.Pool(10) as p: 
            results = p.starmap(scc_pH_process,p_params)
        for result in results: 
            writer.write_data(result[1], result[0], result[2], result[3])


    writer.close()

def general_parallel_run(ctrl_var, ctrl_arrays, params, init_mat, fname, pK):
 
    c0_array, c1_array = ctrl_arrays 
    len_c0 = len(c0_array) 
    len_c1 = len(c1_array)
    measurements = np.zeros((len_c0,len_c1,4))
    
    writer = LatticeHDF5WriterG(fname, params, init_mat,ctrl_var[0],ctrl_var[1])

    for i, c0 in enumerate(c0_array):
        writer.new_c0()
        print('{}% done'.format((i/len_c0)*100)) 
        p_params = []
        
        for c1 in c1_array:
            c1_params={}
            for key in params.keys():
                c1_params[key] = params[key]
            c1_params['init_mat'] = init_mat
            c1_params[ctrl_var[0]] = c0
            c1_params[ctrl_var[1]] = c1
            
            p_params.append((ctrl_var[0],ctrl_var[1],c1_params)) 
        with mp.Pool(10) as p: 
            results = p.starmap(lattice_processG,p_params)
        for result in results: 
            writer.write_data(result[1], result[0], result[2], result[3])

def a_pH_run():
    params = {} 
    params['J'] = -1.5
    params['K'] = 5.0
    params['num_flips'] = 5000
    params['Lx'] = 64	
    params['Ly'] = 64
    params['meas_ivl'] = 100
    
    single_test_n(200,50) # run this to pre-compile energy function
    start = time.time()
    init_mat = checkerboard_ic(params['Lx'],params['Ly']) #Just for first pH
    fname = 'dataset_{0}_J:{1}_K:{2}_{3}x{4}_{5}flips.hdf5'.format(
			datetime.datetime.now().strftime('%b-%d-%Y_%H:%M:%S'),
			params['J'],
            params['K'],
			params['Lx'],
			params['Ly'],
			params['num_flips'])
    a_array = np.arange(-3,-2,0.05)
    pH_array = np.linspace(2,12,99)#Make it odd to go through 7
    scc_lattice_parallel(a_array,pH_array, params, init_mat, fname,5.9)
    print('runtime:{}s'.format(time.time() - start))
 
def single_test_n(num_flips, meas_ivl):
    
    a = 3
    pH = 10
    K = 3.0
    Lx = 64; Ly = 64; J = -1
    init_mat = checkerboard_ic(Lx,Ly)
    mu_1, mu_2 = pH_to_mu(a,pH)
    
    model = LatticeModelVectorized1(Lx = Lx,
                                    Ly = Ly,
                                    mu_1 = mu_1,
                                    mu_2 = mu_2,
                                    J = J,
                                    init_mat = init_mat,
                                    K = K)

    meas = model.run(num_flips,meas_ivl)
    return (meas, model.Spin)        



def K_pH_run():
    params = {} 
    params['J'] = -1.5
    params['num_flips'] = 5000
    params['Lx'] = 64	
    params['Ly'] = 64
    params['a'] = -2.4
    params['meas_ivl'] = 100
    params['pK'] = 5.9
    
    single_test_n(200,50) # run this to pre-compile energy function
    start = time.time()
    init_mat = checkerboard_ic(params['Lx'],params['Ly']) #Just for first pH
    fname = 'dataset_{0}_J:{1}_a:{2}_{3}x{4}_{5}flips.hdf5'.format(
			datetime.datetime.now().strftime('%b-%d-%Y_%H:%M:%S'),
			params['J'],
            params['a'],
			params['Lx'],
			params['Ly'],
			params['num_flips'])
    K_array = np.arange(3.0,7.0,0.4)
    pH_array = np.linspace(2,12,99)#Make it odd to go through 7
    general_parallel_run(('K','pH'),(K_array,pH_array), params, init_mat, fname,5.9)
    print('runtime:{}s'.format(time.time() - start))

if __name__ == "__main__":
   a_pH_run() 
'''
    params = {} 
    params['J'] = -1.5
    params['K'] = 5.0
    params['num_flips'] = 5000
    params['Lx'] = 64	
    params['Ly'] = 64
    params['meas_ivl'] = 100
    params['pK'] = 7
    
    single_test_n(200,50) # run this to pre-compile energy function
    start = time.time()
    init_mat = checkerboard_ic(params['Lx'],params['Ly']) #Just for first pH
    fname = 'dataset_{0}_J:{1}_K:{2}_{3}x{4}_{5}flips.hdf5'.format(
			datetime.datetime.now().strftime('%b-%d-%Y_%H:%M:%S'),
			params['J'],
            params['K'],
			params['Lx'],
			params['Ly'],
			params['num_flips'])
    a_array = np.arange(-4,4,0.4)
    pH_array = np.linspace(2,12,99)#Make it odd to go through 7
    scc_lattice_parallel(a_array,pH_array, params, init_mat, fname)
    print('runtime:{}s'.format(time.time() - start))
'''
