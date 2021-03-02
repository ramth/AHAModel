
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
