import os
from astropy.table import Table
from petitRADTRANS import nat_cst as nc
from datetime import datetime
import matplotlib.pyplot as plt


def worker(atmosphere, planet, p0=0.10, cloud_sigma_lnorm=None, cloud_particle_radius=None,
           pcloud=None, haze_factor=None, description='', output_dir='gendata', plots=True):

    # Set opa structure
    atmosphere.setup_opa_structure(planet.pressure)

    # Setup optional parameters for calc_transm()
    optional_params = dict()
    if cloud_sigma_lnorm is not None and cloud_particle_radius is not None:
        optional_params.update(dict(sigma_lnorm=cloud_sigma_lnorm, radius=cloud_particle_radius))
    if pcloud is not None:
        optional_params.update(dict(Pcloud=pcloud))
    if haze_factor is not None:
        optional_params.update(dict(haze_factor=haze_factor))

    # Perform the actual calculation of the transmission spectrum
    atmosphere.calc_transm(planet.temperature, planet.abundances, planet.gravity, planet.mmw, p0, planet.radius,
                           **optional_params)

    # Store results in a astropy table
    wl = nc.c / atmosphere.freq / 1e-4  # microns
    transm_rad = atmosphere.transm_rad  # cm

    table = Table([wl, transm_rad], names=('wl', 'transm_rad'), meta={'description': description})

    # Add metadata to the table
    date_time = datetime.utcnow()
    metadata = dict(pl_name=planet.name,
                    radius=planet.radius,
                    density=planet.density,
                    mass=planet.mass,
                    gravity=planet.gravity,
                    temp_equ=planet.temp_equ,
                    guillot_temp_params=planet.guillot_temp_params,
                    p0=p0,
                    optional_params=optional_params,
                    date=date_time.strftime('%Y-%m-%d'),
                    time=date_time.strftime('%H:%M:%S'),
                    )
    table.meta.update(metadata)

    # Save data
    filename = f'{planet.name}_{planet.radius / nc.r_jup:1.2f}_{planet.temp_equ:1.0f}.ecsv'
    filepath = os.path.join(output_dir, filename)
    table.write(filepath, format='ascii.ecsv', overwrite=False)

    # Save plots
    if plots:
        plot_spec(wl, transm_rad, planet.radius, planet.temp_equ, planet.name)


def plot_spec(wl, transm_rad, r_pl, temp, pl_name, outdir='plots'):

    plt.plot(wl, transm_rad/nc.r_jup_mean)
    plt.xscale('log')
    plt.title(f'{pl_name} R={r_pl / nc.r_jup_mean:1.2f} R_Jup, T={temp:1.0f} K')
    plt.xlabel('Wavelength (microns)')
    plt.ylabel(f'Transit radius ($R_Jup$)')

    filename = f'{pl_name}_{r_pl / nc.r_jup_mean:1.2f}_{temp:1.0f}.png'
    path = os.path.join(outdir, filename)
    plt.savefig(path, format='png')
    plt.clf()


def set_abundance_file(atype='std'):
    """
    Set the input abundances used by easy_chem program.
    `std` are the default one, `subsolar` refers to abundances
    defined according to C/O=0.28 and  C/N=4.09 (Cridland et al. 2016).

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