from lattice.lattice_parallel_numba import LatticeModelVectorized1
from data import LatticeHDF5ReaderG, LatticeHDF5WriterG
import multiprocessing as mp
import sys
import psutil
import os
from helper import pH_to_mu
import numpy as np
from ic_gen import checkerboard_ic,random_2state_ic, random_3state_ic
import time
import datetime

def lattice_process(params,sq):
    mu_ha, mu_a = pH_to_mu(params['a'],params['pH'],params['pK'])        
    
    model = LatticeModelVectorized1(Lx = params['Lx'], 
                                    Ly = params['Ly'], 
                                    mu_ha = mu_ha,
                                    mu_a = mu_a, 
                                    J = params['J'],
                                    init_mat = params['init_mat'],
                                    K=params['K'],
                                    pK=params['pK'],
                                    sq = sq) 
    meas = model.run(params['num_flips'],params['meas_ivl'],params['last_meas_num'],
                                         params['last_meas_div'])
        #model.plot()

    result = {'meas': meas, 'last_spin':model.Spin, 'params': params,'seed': model.seed}
    return result


def get_n_cores():
    cores_var = 'NSLOTS'
    if cores_var in os.environ:
        ncores = int(os.environ[cores_var])
    else:
        ncores = psutil.cpu_count(logical = False)
    return ncores 


class GridRun():
    ''' From the list of parameters, we choose two control variables that are given by 
    two arrays of values size n and m. The rest of the parameters are taken to be functions of
    these control variables, in general being an nxm matrix. Most of the time they are simply constant
    and that's what is assumed for now
    '''
    parameters = ['J','K','a','pH','pK','num_flips','Lx','Ly','init_mat','meas_ivl','last_meas_num','last_meas_div']
    def __init__(self, ctrl_var, ctrl_arrays, params,fname):


        #Check that all required variables are provided
        for key in params:
            if key in ctrl_var or key in params:
                pass
            else:
                raise NameError('{0} not given'.format(key))
       
        for key in ctrl_var:
            if  key in params:
                params.pop(key,None)
            
        self.ctrl_var = ctrl_var 
        self.c0_array, self.c1_array = ctrl_arrays 
        self.measurements = np.zeros((len(self.c0_array),len(self.c1_array),4))
        self.writer = LatticeHDF5WriterG(fname, params.copy(),ctrl_var[0],ctrl_var[1])
        self.parent_sq = np.random.SeedSequence()

    def run(self):
        c0_sq_array = self.parent_sq.spawn(len(self.c0_array))
        for i, c0 in enumerate(self.c0_array):
            self.writer.new_c0()
            print('{}% done'.format((i/len(self.c0_array)*100))) 
            p_params = []
            
            c1_sq_array = c0_sq_array[i].spawn(len(self.c1_array))
            for j, c1 in enumerate(self.c1_array):
                c1_params={}
                for key in params.keys():
                    c1_params[key] = params[key]
                c1_params[self.ctrl_var[0]] = c0
                c1_params[self.ctrl_var[1]] = c1
                
                p_params.append((c1_params,c1_sq_array[j])) 
            with mp.Pool(get_n_cores()) as p: 
                results = p.starmap(lattice_process, p_params)
            for result in results: 
                self.writer.write_data(result['last_spin'], result['meas'], result['params'],result['seed'])

 
def single_test_n(num_flips, meas_ivl):
    
    a = 3
    pH = 10
    K = 3.0
    Lx = 64; Ly = 64; J = -1
    pK = 7
    init_mat = checkerboard_ic(Lx,Ly)
    last_meas_num = 100
    last_meas_div = 10
    mu_ha, mu_a = pH_to_mu(a,pH)
    
    model = LatticeModelVectorized1(Lx = Lx,
                                    Ly = Ly,
                                    mu_ha = mu_ha,
                                    mu_a = mu_a,
                                    J = J,
                                    init_mat = init_mat,
                                    K = K,
                                    pK = pK)

    meas = model.run(num_flips,meas_ivl,last_meas_num,last_meas_div)
    return (meas, model.Spin)        


def gen_fname(params,c0,c1):
    porder = ['J','K','a','pH']
    porder.remove(c0)
    porder.remove(c1)
    
    p0 = '{0}:{1}'.format(porder[0], params[porder[0]])
    p1 = '{0}:{1}'.format(porder[1], params[porder[1]])
    if params['init_mat'] is None:
        ic = '_rand_'
    else:
        ic = '_'
    size = '{0}x{1}'.format(params['Lx'],params['Ly'])
    fname = '{0}-vs-{1}_{2},{3}{4}{5}_{6}flips_{7}.hdf5'.format(
                        c0,c1,
                        p0,p1,
                        ic,
                        size,
                        params['num_flips'],
                        datetime.datetime.now().strftime('%b-%d-%Y_%H:%M:%S'))
    return fname



if __name__ == "__main__":
    
    #Define constant parameters here
    params = {} 
    params['J'] = 1.00
    params['K'] = 0.0
    params['num_flips'] = 20000
    params['Lx'] = 64
    params['Ly'] = 64
    params['a'] = 0
    params['pH'] = 7
    params['pK'] = 7
    params['meas_ivl'] = 500
    params['last_meas_num'] = 1000
    params['last_meas_div'] = 20
    params['init_mat'] = checkerboard_ic(params['Lx'],params['Ly'])
    #params['init_mat'] = None

    #Define control arrays  
    ctrl_var = ['a','pH']
    c0_array = np.arange(-4,4,0.8)
    c1_array = np.linspace(2,12,10)#Make it odd to go through 7
    
    #run this to pre-compile energy function
    single_test_n(200,50) 
    
    
    start = time.time()
     

    fname = gen_fname(params,ctrl_var[0],ctrl_var[1])
    
    gr = GridRun((ctrl_var),(c0_array,c1_array), params,os.path.join('datasets',fname))
    gr.run()
    print('runtime:{}s'.format(time.time() - start))
