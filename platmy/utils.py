import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from petitRADTRANS import nat_cst as nc


def make_model(r_pl, temp, mass_fractions, mmw, atmosphere, haze_factor=10., pcloud=0.01,
               description='', planet_model=False, pl_mass=-1., pl_name='', p0=0.01, plots=True,
               temp_model='guillot'):
    """
    This function wraps the method `calc_transm` used to calculate the atmosphere's transmission radius (for the
    transmission spectrum). It is intended to be used in a multiprocessing pool in order to calculate several
    models at a time.

    :param pl_name: A name for the model
    :param pl_mass: planet mass in grams.
    :param planet_model: Boolean. Set to TRUE if you want to give the mass for the planet instead of assumed density.
    :param r_pl: planetary radius
    :param temp: equilibrium atmospheric temperature in K
    :param mass_fractions: dictionary of mass fractions for all atmospheric absorbers. Keys are the species names.
    :param mmw: the atmospheric mean molecular weight in amu (a constant value is assumed for every layer)
    :param atmosphere: Radtrans object.
    :param haze_factor: Scalar factor, increasing the gas Rayleigh scattering cross-section.
    :param pcloud: Pressure, in bar, where opaque cloud deck is added to the absorption opacity
    :param description: A string to be added to the metadata
    :param p0: Reference pressure P0 in bar where R(P=P0) = R_pl and g(P=P0) = gravity
    :param plots: Boolean
    :param temp_model: Guillot or constant
    :return:
    """
    print(f'Running model: r: {r_pl / nc.r_jup:1.1f}  t_eq:{temp:1.1f}  temp_model: {temp_model}.')

    # density = 1.33  # gr/cm³ Mini Neptune
    density = 5.51  # g/cm³ Earth density
    mass = 4. / 3. * np.pi * density * r_pl ** 3

    if planet_model:
        mass = pl_mass
        density = mass / r_pl ** 3

    gravity = nc.G * mass / r_pl ** 2

    pressures = np.logspace(-6, 2, 100)
    atmosphere.setup_opa_structure(pressures)

    temperature, temp_model_params = temperature_model(temp, pressures, gravity, model=temp_model)

    species = ['H2', 'He', 'C2H2', 'CH4', 'CO', 'CO2', 'H2', 'H2O', 'H2S', 'HCN',
               'K', 'NH3', 'Na', 'OH', 'PH3', 'TiO', 'VO']

    abundances = {}
    for s in species:
        abundances[s] = mass_fractions[s] * np.ones_like(temperature)

    MMW = mmw * np.ones_like(temperature)

    atmosphere.calc_transm(temperature, abundances, gravity, MMW,
                           R_pl=r_pl, P0_bar=p0,
                           haze_factor=haze_factor, Pcloud=pcloud)

    wl = nc.c / atmosphere.freq / 1e-4
    transm_rad = atmosphere.transm_rad / nc.r_jup

    t = Table([wl, transm_rad], names=('wl', 'transm_rad'), meta={'description': description})
    t.meta.update({'abundances': mass_fractions})
    date_time = datetime.utcnow()
    t.meta.update({'r_pl': r_pl,
                   'temp': temp * 1.,
                   'mmw': mmw,
                   'haze_factor': haze_factor,
                   'pcloud': pcloud,
                   'cdate': date_time.strftime('%Y-%m-%d'),
                   'ctime': date_time.strftime('%H:%M:%S'),
                   'surf_gravity': gravity,
                   'density': density,
                   'mass_pl': mass,
                   'p0': p0,
                   'temp_model': temp_model_params,
                   'pl_name': pl_name})

    datadir = 'gendata'
    extension = '.ecsv'
    filename = f'{pl_name}{r_pl / nc.r_jup:1.2f}_{temp:1.1f}'
    outfile = os.path.join(datadir, filename + extension)
    t.write(outfile, format='ascii.ecsv', overwrite=True)

    if plots:
        plot_spec(wl, transm_rad, r_pl, temp, pl_name)


def plot_spec(wl, transm_rad, r_pl, temp, pl_name):
    """

    :param wl:
    :param transm_rad:
    :param r_pl:
    :param temp:
    :param pl_name:
    :return:
    """
    plt.plot(wl, transm_rad)
    plt.xscale('log')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel(r'Transit radius ($\rm R_{Jupiter}$)')
    plt.title(f'{pl_name} Param: R={r_pl / nc.r_jup:1.2f} R_Jupiter,  T={temp:1.1f} K')
    plt.xlim(0.59, 5.0)
    path = 'plots/'
    filename = path + f'{pl_name}{r_pl / nc.r_jup:1.2f}_{temp:1.1f}.png'
    plt.savefig(filename, format='png')
    plt.clf()


def temperature_model(temp, pressures, gravity, model='guillot'):
    temperatures = None
    params = {'model': model}

    if model == 'constant':
        temperatures = temp * np.ones_like(pressures)
        params.update({'temp': temp})

    elif model == 'guillot':
        kappa_IR = 0.01
        gamma = 0.4
        T_int = 200.
        T_equ = temp
        params.update({'t_int': T_int,
                       't_equ': T_equ,
                       'kappa_ir': kappa_IR,
                       'gamma': gamma})
        temperatures = nc.guillot_global(pressures, kappa_IR, gamma, gravity, T_int, T_equ)

    return temperatures, params


def dict_to_list(input_dict):
    """
    This function transform a dictionary that for each key it contains a list of length n
    to a list of dictionaries.

    :param input_dict:
    :return:
    """
    list_of_dictionaries = [{} for _ in next(iter(input_dict.values()))]
    for k, v in input_dict.items():
        for i, el in enumerate(v):
            list_of_dictionaries[i][k] = el
    return list_of_dictionaries


def read_abunds(path):
    """
    Function that reads the output file from easy_chem fortran program
    and store the values of each row in a dict-like fashion
    Modified from nat_cst.py in mattheus_chem by Paul Molliere.

    :param path: output file from easy_chem program
    :return: a list of dictionaries with the mass fractions per reactant(?)
    """
    f = open(path)
    header = f.readlines()[0][:-1]
    f.close()
    ret = {}

    dat = np.genfromtxt(path)
    ret['P'] = dat[:, 0]
    ret['T'] = dat[:, 1]
    ret['rho'] = dat[:, 2]

    for i in range(int((len(header) - 21) / 22)):

        name = header[21 + i * 22:21 + (i + 1) * 22][3:].replace(' ', '')

        if name == 'C2H2,acetylene':
            name = 'C2H2'

        if i % 2 == 0:
            number = int(header[21 + i * 22:21 + (i + 1) * 22][0:3])
            ret[name] = dat[:, number]

    return dict_to_list(ret)


def get_PT_abundances_MMW(pressure, temperature):
    """
    This function wraps the fortran program easy_chem. Original code EASY_CHEM by Paul Molliere.
    EASY CHEM is a clone of the Nasa Chemical equilibrium with applications software (CEA).
    The code minimizes the the total Gibbs free energy of all possible species while conserving
    the number of atoms of every atomic species. Given the pressure and temperature together with the
    atomic composition of the gas the output code is the mass and number fraction of all possible
    outcome species (atoms, ions, molecules) the resulting density, as well the adiabatic temperature
    gradient of the gas mixture.

    The atoms species abundances (numerical fraction) must be indicated in the abundances.inp file.
    Standard_abundances.inp file has solar abundances as defined in Asplund et al. (2009), see table 1.

    :param pressure: pressure (bar)
    :param temperature:  temperature (K)
    :return: abundances, Mean Molecular Weights and densities (in cgs units)
             atmospheric mean molecular weight in amu
    """

    current_dir = os.getcwd()
    os.chdir(os.path.join(current_dir, 'easy_chem'))
    np.savetxt('PT_struct.dat', np.column_stack((pressure, temperature)))
    os.system('./call_easy_chem')

    try:
        abunds = read_abunds('final_abund_all.dat')

    except IndexError:
        os.chdir(current_dir)
        raise IndexError('call easy_chem failed')

    dat = np.genfromtxt('MMWs.dat')
    mmw = dat[:, 1]
    os.system('rm MMWs.dat')
    os.system('rm final_abund_all.dat')
    os.system('rm PT_struct.dat')
    os.chdir(current_dir)

    return abunds, mmw


# Setup output directories and files -------------------------
dirs = ['gendata', 'plots']
extensions = ['*.ecsv', '*.png']


def check_folders():
    """
    Check that folders defined in dirs exists, otherwise it create the folders"
    :return:
    """
    for d in dirs:
        if not os.path.isdir(d):
            os.mkdir(d)


def clean_outputs():
    """
    Remove output files from previous runs of the program.
    This includes both: data files and graphs.
    """
    for d, ext in zip(dirs, extensions):
        os.system(f'rm {d}/{ext}')


def set_abundance_file(atype='std'):
    """
    Set the input abundances used by easy_chem program.
    `std` are the default one, `subsolar` refers to abundances
    defined according to C/O=0.28 and  C/N=4.09 (Cridland et al. 2016).

    :param atype: string. Abundance type as defined below
    :return:
    """
    if atype == 'std':
        file = 'Standard_abundances.inp'
    elif atype == 'subsolar':
        file = 'Subsolar_abundances.inp'
    elif atype == 'earthlike':
        file = 'Earth_like_abundances.inp'
    elif atype == '10x_std':
        file = '10x_std_abun.inp'
    elif atype == '10x_less_std':
        file = '10x_less_std_abun.inp'
    else:
        raise(KeyError, "Error: not valid option.")

    current_dir = os.getcwd()
    os.chdir(os.path.join(current_dir, 'easy_chem'))
    os.system(f'cp {file} abundances.inp')
    os.chdir(current_dir)
