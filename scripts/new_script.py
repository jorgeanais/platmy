import multiprocessing as mp
import numpy as np
from petitRADTRANS import nat_cst as nc
from petitRADTRANS import Radtrans
from platmy import tools
from platmy.model import Planet

"""
This new version of this script uses tools.py and model.py to simplify how to generate the grid of models, and
also allows to calculate the spectra without the simplification of constant abundances for each layer of the model.
"""

# Set abundances
abund_type = 'std'
tools.set_abundance_file(abund_type)

# Set parameters for each planet
names = ['Jupiter-like', 'HD1397b', 'TOI172b', 'TOI677b']
masses = np.array([1., 0.367, 5.42, 1.236]) * nc.m_jup
radii = np.array([1., 1.023, 0.965, 1.170]) * nc.r_jup_mean
temps_equ = np.arange(500., 2001., 500.)  # grid of temperatures
pressure = np.logspace(-6, 2, 100)  # pressure (same for all planets)
p0 = 0.10  # bar

# Here we use list comprehension to create a list of planets from the parameters above
# this part cannot be easily parallelized due to easy_chem input/output operations involve read and writing files
# It could take a while
planets = [Planet(name=n, radius=r, temp_equ=t, mass=m, pressure=pressure)
           for t in temps_equ
           for n, m, r in zip(names, masses, radii)]

# Set up Radtrans object including all the species that you want to include.
# Make sure that all the species are included in reactant list in your abundances.inp file
atmosphere = Radtrans(line_species=['C2H2', 'CH4', 'CO', 'CO2', 'H2', 'H2O', 'H2S', 'HCN', 'K', 'NH3', 'Na', 'OH', 'PH3', 'TiO', 'VO'],
                      rayleigh_species=['H2', 'He'],
                      continuum_opacities=['H2-H2', 'H2-He'],
                      cloud_species=['Mg2SiO4(c)_cd'],
                      wlen_bords_micron=[0.3, 15],  # max wlen range 0.11 to 250 microns for 'c-k' mode. Ram intensive.
                      mode='c-k')

# Set haze and pcloud parameters
haze_factor = 10
pcloud = 0.01
description = f'Warm/hot jupiter with {abund_type} abundances and haze_factor={haze_factor} and pcloud={pcloud}'

# Adding some Condensate clouds parameters
radius = {}
radius['Mg2SiO4(c)'] = 0.00005 * np.ones_like(pressure) # I.e. a 0.5-micron particle size (0.00005 cm)
sigma_lnorm = 1.05


# Grid of params
# atmosphere, planet, p0=0.10, cloud_sigma_lnorm=None, cloud_particle_radius=None, pcloud=None, haze_factor=None,
# description='', output_dir='gendata', plots=True
models = [(atmosphere, pl, p0, sigma_lnorm, radius, pcloud, haze_factor, description) for pl in planets]

# Computation in parallel
# I'm using here only 3 process because I limited by ram
with mp.Pool(3) as pool:
    results = pool.starmap(tools.worker, models)
