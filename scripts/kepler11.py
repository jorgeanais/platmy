import multiprocessing as mp

import numpy as np
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

from platmy import utils

utils.check_folders()
utils.clean_outputs()
abund_type = 'subsolar'
utils.set_abundance_file(abund_type)

# Set parameters (R and T)
names = ['Kepler_11c', 'Kepler_11d', 'Kepler_11f', 'Kepler_11g']
radii = np.array([2.84, 3.14, 2.49, 3.4]) * nc.r_earthz
masses = np.array([4.66, 5.89, 2.81, 6.26]) * nc.m_earth
temperatures = np.arange(1000., 2001., 100.)
pressures = 0.01 * np.ones_like(temperatures)

# Compute mass fractions abundances according to the numerical abundances in abundances.inp
abunds, mmws = utils.get_PT_abundances_MMW(pressures, temperatures)

line_species = ['C2H2', 'CH4', 'CO', 'CO2', 'H2', 'H2O', 'H2S', 'HCN', 'K', 'NH3', 'Na', 'OH', 'PH3', 'TiO', 'VO']
atmosphere = Radtrans(line_species=line_species,
                      rayleigh_species=['H2', 'He'],
                      continuum_opacities=['H2-H2', 'H2-He'],
                      wlen_bords_micron=[0.6, 5.])

# Set haze and pcloud parameters
haze_factor = 100.
pcloud = 0.1
description = f'Kepler 11 planets with {abund_type} abundances and haze_factor={haze_factor} and pcloud={pcloud}'

# Grid of models
models = [(r, temp, abund, mmw, atmosphere, haze_factor, pcloud, description, True, mass, name)
          for temp in temperatures for r, abund, mmw, mass, name in zip(radii, abunds, mmws, masses, names)]

# Computation in parallel
with mp.Pool(mp.cpu_count() - 1) as pool:
    results = pool.starmap(utils.make_model, models)