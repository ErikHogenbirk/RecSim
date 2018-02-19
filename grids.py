"""Initialization of grids for RecSim base class.
"""

import numpy as np

__all__ = ['set_density_empty', 'set_density_sphere', 'set_density_cylinder', 'set_density_track',
           'normalize_grid', 'integrate_grid']

def set_density_empty(rs, name):
    '''
    Initialize a grid for the parameter given by `name`
    '''
    if name in rs.grids.keys():
        print('Grid for %s already exists, skipping...' % name)
        return
    rs.grids[name] = np.zeros(shape=rs.coord_nbins, dtype=float)
    return

def set_density_sphere(rs, name, radius, constvalue = 1.,  center=None):
    '''
    Set the density of the parameter with name `name` to a constant density within a sphere.
    Radius is given in units of length (not number of bins).
    Center is given in bin coordinates and defaults to center location.
    '''
    if center is None:
        # default position is central position
        center = [round((coord_nbin - 1) / 2) for coord_nbin in rs.coord_nbins]
    # Recalculate the radius to number of bin steps (different per dimension possible if different step sizes)
    radius_bins = [radius / coord_step for coord_step in rs.coord_stepsizes]
    # Calculate the distance from center in units of fraction of the full radius
    r_from_center = np.zeros(dtype=float, shape=rs.coord_nbins)
    for i in range(rs.coord_nbins[0]):
        for j in range(rs.coord_nbins[1]):
            for k in range(rs.coord_nbins[2]):
                r_from_center[i, j, k] = (((i - center[0]) / radius_bins[0])**2 + 
                                           ((j - center[1]) / radius_bins[1])**2 +
                                           ((k - center[2]) / radius_bins[2])**2)
    r_from_center = np.sqrt(r_from_center)
    rs.grids[name][r_from_center <= 1] = constvalue
    return

def set_density_cylinder(rs, name, radius, length, constvalue, center=None):
    '''
    Set the density to a cylinder in x.
    Radius is given in units of length (not number of bins).
    Center is given in bin coordinates and defaults to center location.
    '''
    if center is None:
        # default position is central position
        center = [round((coord_nbin - 1) / 2) for coord_nbin in rs.coord_nbins]
    # Recalculate the radius to number of bin steps (different per dimension possible if different step sizes)
    radius_bins = [radius / coord_step for coord_step in rs.coord_stepsizes[1:]]
    length_bins = length / rs.coord_stepsizes[0]
    # Calculate the distance from center in units of fraction of the full radius, only for y and z
    r_from_center = np.zeros(dtype=float, shape=rs.coord_nbins)
    for j in range(rs.coord_nbins[1]):
        for k in range(rs.coord_nbins[2]):
            r_from_center[:, j, k] = (((j - center[1]) / radius_bins[0])**2 +
                                      ((k - center[2]) / radius_bins[1])**2)
    r_from_center = np.sqrt(r_from_center)
    z_from_center = np.zeros(dtype=float, shape=rs.coord_nbins)
    for i in range(rs.coord_nbins[0]):
        z_from_center[i, :, :] = np.abs(i - center[0]) / length_bins

    rs.grids[name][(r_from_center <= 1) & (z_from_center <= 0.5)] = constvalue
    return

def set_density_track(rs, name, radius, center=None, factor=100, mode='energy'):
    '''
    Set the density to a cylinder oriented along the x-axis. Center location in bins, radius in um.
    Enter the energy in keV.
    Energy deposition is calculated from an interpolation of the LET. The `factor` makes a finer interpolation
    than the x-grid, increasing the resolution.
    '''
    if center is None:
        # default position is central position
        center = [round((coord_nbin - 1) / 2) for coord_nbin in rs.coord_nbins]
    # Recalculate the radius to number of bin steps (different per dimension possible if different step sizes)
    radius_bins = [radius / coord_step for coord_step in rs.coord_stepsizes[1:]]
    # Compute the energy deposition along the track
    track_x, track_dEdx = dEdx_track(rs.config['energy'], rs.coord_stepsizes[0] / factor)
    # Extend (for rebinning) and rebin
    extend_by = (factor - (len(track_dEdx) % factor))
    track_dEdx = np.concatenate([track_dEdx, np.zeros(extend_by)])
    track_dEdx = np.average(np.reshape(track_dEdx, (int(len(track_dEdx)/ factor), factor)), axis=1)
    length_bins = np.max(track_x) / rs.coord_stepsizes[0]
    # Compute the radius from center location of the cylinder in y,z
    r_from_center = np.zeros(dtype=float, shape=rs.coord_nbins)
    for j in range(rs.coord_nbins[1]):
        for k in range(rs.coord_nbins[2]):
            r_from_center[:, j, k] = (((j - center[1]) / radius_bins[0])**2 +
                                      ((k - center[2]) / radius_bins[1])**2)
    r_from_center = np.sqrt(r_from_center)
    # Boolean array
    is_in_radius = r_from_center <= 1
    surface_area = np.sum(is_in_radius[0]) * rs.bin_surface[0]
    # Initialize empty grid
    grid = np.zeros(dtype=float, shape=rs.coord_nbins)
    # Set correct values for all x along the track
    start_position = int(center[0] - 0.5 * length_bins)
    for i in range(rs.coord_nbins[0]):
        bins_from_start_position = int(i - start_position)
        if bins_from_start_position >=0 and bins_from_start_position < len(track_dEdx):
            if mode == 'energy':
                grid[i, :, :] = track_dEdx[bins_from_start_position] * 1/surface_area
            elif mode == 'electrons':
                grid[i, :, :] = track_dEdx[bins_from_start_position] * 1/surface_area / rs.config['W'] * rs.config['ion_fraction']
    # Set to zero for positions outside the track in y,z
    grid[np.invert(is_in_radius)] = 0.
    rs.grids[name] = grid
    return 

def normalize_grid(rs, name, normalize_to = 1.):
    rs.grids[name] = normalize_to * rs.grids[name] / rs.integrate_grid(name)
    return

def integrate_grid(rs, name):
    '''Integrate a grid'''
    return np.sum(rs.grids[name]) * rs.bin_volume


def dEdx_er(e, mode = 'dEdx', rho = 2.9):
    '''
    Return the LET for this recoil energy for electrons
    PRL 97, 081302
    Mode dEdx: return dE/dx in keV/um
    Mode LET: return LET in MeV cm^2 / g
    '''
    E    = [1,2,3,4,5,6,8,10,20,30,40,50,60,80,100]
    dEdx = [8.5, 7.85, 6.3, 5.5,5,4.5,3.7,3.1,2.0,1.4,1.25,1.05,0.95,0.8,0.7]
    LET = np.array(dEdx) * 10 / rho
    assert len(E) == len(dEdx)
    if mode == 'dEdx':
        return np.interp(e, E, dEdx)
    elif mode == 'LET':
        return np.interp(e, E, LET)
    elif mode == 'dEdx_inv':
        return np.interp(e, dEdx[::-1], E[::-1])
    elif mode == 'LET_inv':
        return np.interp(e, LET[::-1], E[::-1])
    else:
        raise SyntaxError('Mode not understood, got this: ', mode)
        
def dEdx_track(e, step_size = 0.5):
    '''
    All distance in um, all energy in keV
    '''
    nsteps = int(100 / step_size) + 1
    range_list = np.linspace(0, 100, nsteps) # um
    
    energy = 0.
    energy_list = []
    dEdx_list = []
    for i, ran in enumerate(range_list):
        dE = dEdx_er(energy) * step_size
        dEdx_list.append(dE / step_size)
        energy_list.append(energy)
        energy += dE
        if energy > e:
            break
            
    return range_list[:i+1], np.array(dEdx_list)

