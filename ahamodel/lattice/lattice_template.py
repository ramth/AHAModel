import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.random import randint
import abc
from numpy.random import Generator,PCG64
import datetime
import multiprocessing as mp
import numba
from ic_gen import random_3state_ic, random_2state_ic


'''Conventions:

HA 1
A- -1
'''


class LatticeModelTemplate:
    '''Lattice Model Class based on the single spin method outlined by Onuttom'''
    
    def __init__(self,Lx,Ly,mu_ha,mu_a,J,init_mat,K,pK, sq=None):
        self.J = J
        self.Lx = Lx
        self.Ly = Ly
        self.K = K
        self.pK = pK
        self.mu_ha = mu_ha
        self.mu_a = mu_a
       
        #Define interaction parameters, H and M matrices  
        #H are the site energies ,corresponding to state: (HA, Vac, A-)
        self.H = np.array([-self.mu_ha, 0, -self.mu_a]) #HA, Vac, A-
        #M is the nearest neighbor 3x3 interaction matrix depending upon the
        #the two spin states
        #J,K are positive, the repulsion is taken into account by the -ve sign in M definition
        self.M = np.array([[0,0,-J],[0,0,0],[-J,0,K]])

        if sq == None:
            sq = np.random.SeedSequence()
        
        self.seed = sq.entropy 
        bg = PCG64(sq)
        self.rgen = Generator(bg) 
       
        #If init mat is not provided generate a random initial condition
        if init_mat is None:
            init_mat = random_3state_ic(self.rgen,self.Lx, self.Ly)

        self.even_mask = self.create_mask(0)
        self.odd_mask = self.create_mask(1)
         
        self.Spin = init_mat
    
    def create_mask(self,parity):
        '''Create either even or odd checkerboard pattern'''
        mask = np.zeros([self.Lx,self.Ly],dtype = np.bool)
        for i in range(self.Lx):
            for j in range(self.Ly):
                mask[i,j] = (i+j)%2 == parity
                
        return mask

    @staticmethod
    def s1s2(S1, S2):
        ''' A function to compute the nearest neighbor E12 coupling matrix 
        (Not sure what the technical term is)'''      
        return 0.5*S1*S2*(1 - S1*S2)
        
    @abc.abstractmethod
    def iteration(self):
        pass
    
    def run(self,num_trials,div_measurements):
        measurements = np.zeros((int(np.floor(num_trials/div_measurements)),4))
        j = 0
        for i in range(num_trials):
            if i%div_measurements == 0:
                measurements[j,:] = self.measure()
                j = j+1
            self.iteration()
        return measurements
    

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
