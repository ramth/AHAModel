from helper import complexation_index_f, en_s
from lattice.lattice_template import LatticeModelTemplate
import numpy as np

class LatticeModelVectorized1(LatticeModelTemplate):
    '''Lattice Model Class based on the parallel method'''
     
    def __init__(self,Lx,Ly,mu_ha,mu_a,J,init_mat,K,pK,sq = None):

        super(LatticeModelVectorized1,self).__init__(Lx,Ly,mu_ha,mu_a,
                J,init_mat,K,pK,sq)
        self.Spin_flip = self.Spin.copy()
        self.en_old = np.zeros([self.Lx, self.Ly])
        self.en_new = np.zeros([self.Lx, self.Ly])
        

        self.shifts = ((0,0),(1,1),(0,1),(1,0))


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

        #np.random.shuffle(indices)
        #print(indices)
        for idx in indices:
            x,y = self.shifts[idx]
            self.MonteCarlo_sublattice_flip(x, y)

    def run(self,num_trials,div_measurements,last_meas_num, last_meas_div):
        measurements = []
        j = 0
        for i in range(num_trials):
            j = i - (num_trials - last_meas_num + 1)
            if (j < 0 and i % div_measurements == 0)\
                 or (j >=0 and j % last_meas_div ==0):   
                p_ha_even, p_a_even, p_ha_odd, p_a_odd,com_idx = self.measure()
                measurements.append((p_ha_even, p_a_even, p_ha_odd, 
                                    p_a_odd, com_idx, i))
            self.MonteCarlo_flip()
        return np.array(measurements)
    
    def measure(self):
        even_sites = self.Spin[self.even_mask]
        odd_sites = self.Spin[self.odd_mask]
        
        p_ha_even = np.sum(even_sites==1)/even_sites.size
        p_a_even = np.sum(even_sites==-1)/even_sites.size
        
        p_ha_odd = np.sum(odd_sites==1)/odd_sites.size
        p_a_odd = np.sum(odd_sites==-1)/odd_sites.size
        
        com_idx = complexation_index_f(self.Spin)
        return (p_ha_even, p_a_even, p_ha_odd, p_a_odd,com_idx)
    
    def print_measurements(self):
        p_ha_even, p_a_even, p_ha_odd, p_a_odd  = self.measure()
        
        col_names = ('','even', 'odd','Delta')# even starts on (0,0)
        
        #print in table form : https://stackoverflow.com/questions/9535954/printing-lists-as-tabular-data
        format_str1 = '{:>2}'+'{:>15}'*3
        format_str2 = '{:>2}{:>15.3f}{:>15.3f}{:>15.3f}'
        print(format_str1.format(*col_names))
        print(format_str2.format('A-',p1_even, p1_odd,p1_even-p1_odd))
        print(format_str2.format('HA',p2_even, p2_odd,p2_even-p2_odd))
        

