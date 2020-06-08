import multiprocessing as mp

import numpy as np
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc

from platmy import utils

utils.check_folders()
utils.clean_outputs()

# Set parameters (R and T)
radii = np.arange(1.5, 1.8, 0.2) * nc.r_earth
temperatures = np.arange(1000., 1101., 100.)
pressures = 0.01 * np.ones_like(temperatures)

# Compute mass fractions abundances according to the numerical abundances in abundances.inp
abunds, mmws = utils.get_PT_abundances_MMW(pressures, temperatures)

"""
abunds = {}
abunds['N2'] = ??
abunds['O2'] = ??
abunds['CO2'] = 3.65e-04
abunds['CH4'] = 1.65e-03
abunds['O2'] = 2.1e-01
abunds['O3'] = 3.00e-08
abunds['N2O'] = 3.00e-07
mmw = ???  # Atmospheric mean molecular weight??
"""

line_species = ['C2H2', 'CH4', 'CO', 'CO2', 'H2', 'H2O', 'H2S', 'HCN', 'NH3', 'OH', 'PH3', 'VO', 'O3', 'SO2', 'COS']
atmosphere = Radtrans(line_species=line_species,
                      rayleigh_species=['H2', 'He'],
                      continuum_opacities=['H2-H2', 'H2-He'],
                      wlen_bords_micron=[0.6, 5.])

# Set haze and pcloud parameters
haze_factor = 10.
pcloud = 0.01
description = 'Earth-like test'

# Grid of models
models = [(r, temp, abund, mmw, atmosphere, haze_factor, pcloud, description)
          for r in radii for temp, abund, mmw in zip(temperatures, abunds, mmws)]

# Computation in parallel
with mp.Pool(mp.cpu_count() - 1) as pool:
    results = pool.starmap(utils.make_model, models)