from petitRADTRANS import nat_cst as nc
import numpy as np
import os


class Planet:

    def __init__(self, name='', radius=nc.r_jup_mean, temp_equ=500, mass=None, density=None,
                 pressure=np.logspace(-6, 2, 100)):
        self._name = name
        self._radius = radius
        self._density = density
        self._mass = mass
        self._check_mass(mass)
        self._gravity = self._compute_gravity()
        self._temp_equ = temp_equ
        self._guillot_temp_params = dict(kappa_IR=0.01, gamma=0.4, grav=self.gravity, T_int=200., T_equ=self._temp_equ)
        self._pressure = pressure
        self._temperature = self._compute_temperature()
        self._abundances, self._mmw = get_pt_abundances_mmw(self._pressure, self._temperature)

    def _check_mass(self, value):
        if value is not None:
            self._mass = value
            self._density = value / self._radius ** 3
        elif self._density is not None:
            self._mass = 4. / 3. * np.pi * self._density * self._radius ** 3
        else:
            raise KeyError('You must provide either mass or density')

    def _compute_gravity(self):
        return nc.G * self._mass / self._radius ** 2

    def _compute_temperature(self):
        return nc.guillot_global(self._pressure, **self._guillot_temp_params)

    @property
    def name(self):
        return self._name

    @property
    def radius(self):
        return self._radius

    @property
    def mass(self):
        return self._mass

    @property
    def gravity(self):
        return self._gravity

    @property
    def temp_equ(self):
        return self._temp_equ

    @property
    def guillot_temp_params(self):
        return self._guillot_temp_params

    @property
    def pressure(self):
        return self._pressure

    @property
    def temperature(self):
        return self._temperature

    @property
    def abundances(self):
        return self._abundances

    @property
    def mmw(self):
        return self._mmw

    @property
    def density(self):
        return self._density

    def __str__(self):
        return self._name


def read_abunds(path):
    """
    Function that reads the output file from easy_chem fortran program
    and store the values of each row in a dict-like fashion
    Modified from nat_cst.py in mattheus_chem by Paul Molliere.

    :param path: output file from easy_chem program
    :return: a list of dictionaries with the mass fractions per reactant
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

    return ret


def get_pt_abundances_mmw(pressure, temperature):
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


