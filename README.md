## Lattice Simulation used in "Conjugate Acid–Base Interaction Driven Phase Transition at a 2D Air–Water Interface"

HA (Acid) and A- (Conjugate Base) cooperation on the 2D air-water interface is studied in the framework of
a rectangular lattice model, with classical hamiltonian where a cooperative interaction between HA and A-
and a repulsive interaction between A- and A- is modeled through the variables J and K. 

The sublattice metropolis algorithm was utilized to simulate equilibriated lattices. The simulations and
fits were programmed with the numpy and scikit-learn libraries. 2 sublattices
are formed, in the case of the first it is described by $i + j = 2n$. The Metropolis step is
applied to all of the sites on the sublattice simultaneously to utilize parallelized routines.
Nearest neighbor coupling with an external site potential was utilized with the following
coupling of energies between the sites,

$$H_{i,j} = −\mu(S_{i,j} ) + \sum
\limits_{<l,m>} \{Jh(S_{i,j} , S_{l,m}) + Kg(S_{i,j} , S_{l,m})\}$$
$$ \mu_1 = a + \ln \frac {1}{ 1 + 10^{pH−pK}}$$
$$ \mu_2 = a + \ln \frac{10^{pH−pK}{ 1 + 10^{pH−pK}} $$
where the sum is over nearest neighbor sites (l, m). The function $h(S_i, S_j ) = −1$ when
$(Si, Sj ) = (1, 2) or (2, 1)$ and zero otherwise, while the function $g(S_i,S_j ) = 1$ when $S_i = S_j = 2$
and zero otherwise.The lattice model was simulated on a 64 by 64 site grid as a function of
pK, a, pH, J, and K. A full ’checkerboard’ initial condition, one species fully occupying the
even sublattice and the other species occupying the odd sublattice, was utilized to give fast
convergence in state points that had multiple metastable states. The thermalization time is
chosen to be 20,000 flips per spin, and is followed by 1000 measurements, with 20 flips per
spin between measurements. A binning analysis4 was done to characterize the correlations
and verify convergence. The typical standard error in n1 and n2 are < 0.01. Near the phase
boundary, the correlation time diverges, but this only results in a slight uncertainty in the
location of the phase boundary, without affecting our overall analysis. We estimate standard
errors in n1 and n2 at points close to phase boundary to be up to 0.03.

See Manuscript for full description: 
"Conjugate Acid–Base Interaction Driven Phase Transition at a 2D Air–Water Interface"
R. Rajagopal, M. K. Hong, L. D. Ziegler, S. Erramilli, and Onuttom Narayan
The Journal of Physical Chemistry B 2021 125 (23), 6330-6337
DOI: 10.1021/acs.jpcb.1c02388 
