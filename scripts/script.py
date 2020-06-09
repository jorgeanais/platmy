import multiprocessing as mp
import numpy as np
from petitRADTRANS import Radtrans
from petitRADTRANS import nat_cst as nc
from platmy import utils

utils.check_folders()
utils.clean_outputs()

# Set parameters R and T
radii = np.arange(1.5, 3.1, 0.2) * nc.r_earth
temperatures = np.arange(1000., 2001., 100.)
pressures = 0.01 * np.ones_like(temperatures)

# Compute mass fraction abundances according to numerical abundances in abundances.inp
abund_type = 'subsolar'
utils.set_abundance_file(abund_type)
abunds, mmws = utils.get_PT_abundances_MMW(pressures, temperatures)

# Define Radtrans object
line_species = ['C2H2', 'CH4', 'CO', 'CO2', 'H2', 'H2O', 'H2S', 'HCN', 'K', 'NH3', 'Na', 'OH', 'PH3', 'TiO', 'VO']
atmosphere = Radtrans(line_species=line_species,
                      rayleigh_species=['H2', 'He'],
                      continuum_opacities=['H2-H2', 'H2-He'],
                      wlen_bords_micron=[0.6, 5.])

# Set haze and pcloud parameters
haze_factor = 10.
pcloud = 0.01
description = f'Model using {abund_type} abundances and haze_factor={haze_factor} and pcloud={pcloud}'

# Define iterable with a grid of models
models = [(r, temp, abund, mmw, atmosphere, haze_factor, pcloud, description)
          for r in radii for temp, abund, mmw in zip(temperatures, abunds, mmws)]

# Run models in parallel
with mp.Pool(mp.cpu_count() - 1) as pool:
    results = pool.starmap(utils.make_model, models)
