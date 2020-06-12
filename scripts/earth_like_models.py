import multiprocessing as mp
import numpy as np
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from platmy import utils

utils.check_folders()
utils.clean_outputs()

# Set parameters (R and T)
radii = np.arange(1.0, 2.1, 0.1) * nc.r_earth
temperatures = np.arange(1000., 1501., 100.)
pressures = 0.01 * np.ones_like(temperatures)

# Compute mass fraction abundances according to numerical abundances in abundances.inp
abund_type = 'earthlike'
utils.set_abundance_file(abund_type)
abunds, mmws = utils.get_PT_abundances_MMW(pressures, temperatures)

# line_species = ['C2H2', 'CH4', 'CO', 'CO2', 'H2', 'H2O', 'H2S', 'HCN', 'NH3', 'OH', 'PH3', 'VO', 'O3', 'SO2', 'COS']
line_species = ['C2H2', 'CH4', 'CO', 'CO2', 'H2', 'H2O', 'H2S', 'HCN', 'NH3', 'OH', 'PH3', 'VO']
atmosphere = Radtrans(line_species=line_species,
                      rayleigh_species=['H2', 'He'],
                      continuum_opacities=['H2-H2', 'H2-He'],
                      wlen_bords_micron=[0.6, 5.])

# Set haze and pcloud parameters
haze_factor = 10.
pcloud = 0.01
description = f'Earth-like test using {abund_type} abundances and haze_factor={haze_factor} and pcloud={pcloud}'

# Grid of models
models = [(r, temp, abund, mmw, atmosphere, haze_factor, pcloud, description)
          for r in radii for temp, abund, mmw in zip(temperatures, abunds, mmws)]

# Computation in parallel
with mp.Pool(mp.cpu_count() - 1) as pool:
    results = pool.starmap(utils.make_model, models)