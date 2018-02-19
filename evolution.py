"""Evolution of the grids for the RecSim class.
"""

import numpy as np
from tqdm import tqdm

__all__ = ['recombine', 'drift', 'diffuse',
           'evolve_one_step', 'evolve', 'evolve_until_fraction']




def evolve_one_step(rs):
    '''
    Basic function to take one time step. This is called by all other functions.
    Steps taken:
      - Diffusion
      - Drift
    '''
    tdict = {}
    # Diffusion
    if rs.config['diffusion_e']:
        tdict.update(rs.diffuse('e', rs.config['e_diff_const']))
    if rs.config['diffusion_ion']:
        tdict.update(rs.diffuse('ion', rs.config['ion_diff_const']))
    # Drift
    if rs.config['drift_e']:
        rs.drift('e', vd = rs.config['vd_e'])
    if rs.config['drift_ion']:
        rs.drift('e', vd = rs.config['vd_ion'])
    # Recombination
    if rs.config['recombination']:
        tdict.update(rs.recombine('e', 'ion', rs.config['alpha']))
    else:
        tdict['ex_production'] = 0.
    # Cumulative properties
    if len(rs.d) == 0:
        tdict['n_ex'] = tdict['ex_production']
        for name in rs.grids.keys():
            if rs.config['diffusion_' + name]:
                tdict['n_' + name + '_out'] = tdict[name + '_out']
    else:
        tdict['n_ex'] = tdict['ex_production'] + rs.d[-1]['n_ex']
        for name in rs.grids.keys():
            if rs.config['diffusion_' + name]:
                tdict['n_' + name + '_out'] = tdict[name + '_out'] + rs.d[-1]['n_' + name + '_out']
    
    # Additional parameters
    tdict.update(rs.add_parameters())
    # Increment time and append parameters
    tdict['t'] = rs.t
    rs.t = rs.t + rs.dt
    rs.d.append(tdict)
    return

def evolve(rs, nsteps, show_tqdm=True):
    '''
    Apply evolution for `nsteps` steps.
    '''
    
    if show_tqdm:
        with tqdm(total = nsteps * rs.dt + rs.t, unit = 'ns', unit_scale=True) as pbar:
            for i in range(nsteps):
                rs.evolve_one_step()
                pbar.update(rs.dt)
                pbar.set_postfix(t = rs.t)
    else:
        for i in range(nsteps):
            rs.evolve_one_step()

def evolve_until_fraction(rs, frac = 0.01, block_size = 500, verbose=False):
    '''Apply evolution until a certain fraction of the electrons is left.'''
    total_ne = np.sum(rs.grids['e'])
    current_frac = 1.
    while current_frac > frac:
        rs.evolve(block_size, show_tqdm=False)
        current_frac = np.sum(rs.grids['e']) / total_ne
        if verbose:
            print('%.1f %% of electrons left...' % (current_frac * 100.))
    return

def recombine(rs, name0, name1, alpha):
    '''Apply recombination proportional to electron and ion density product'''
    ret = {}
    # Compute overlap
    overlap = rs.grids[name0] * rs.grids[name1]
    recombined = overlap * alpha * rs.dt
    for name in [name0, name1]:
        if np.any(recombined > rs.grids[name]):
            raise ValueError('Recombination is more than 100%, so no way!')
    rs.grids[name0] = rs.grids[name0] - recombined
    rs.grids[name1] = rs.grids[name1] - recombined
    ret['ex_production'] = np.sum(recombined)
    return ret

def drift(rs, name, vd):
    '''
    Drift electrons or ions.
    Checks if we have enough shift accumulated to increment one bin.
    This is implemented this way to stop extra diffusion due to drift dispersion.
    '''
    # Cumulative drift is in units of number of bins
    rs.cumulative_drift += np.array(rs.config['drift_vector']) * rs.dt * vd / rs.coord_stepsizes
    for axis, cumulative_drift_axis in enumerate(rs.cumulative_drift):
        if np.abs(cumulative_drift_axis) >= 1.:
            direction = np.int(cumulative_drift_axis)
            grid = rs.grids[name]
            grid = _shift_grid(grid, direction, axis)
            rs.cumulative_drift[axis] -= direction
            rs.grids[name] = grid
    return 

def _shift_grid(grid, nbins, axis, fill_value = 0.):
    '''Shifts the array by nbins in axis with index axis'''
    # This shifts and passes elements through (https://docs.scipy.org/doc/numpy/reference/generated/numpy.roll.html)
    grid = np.roll(grid, nbins, axis=axis)
    # Now set elements to zero
    if axis == 0:
        if nbins > 0:
            grid[:nbins, :, :] = fill_value
        else:
            grid[nbins:, :, :] = fill_value
    elif axis == 1:
        if nbins > 0:
            grid[:, :nbins, :] = fill_value
        else:
            grid[:, nbins:, :] = fill_value
    elif axis == 2:
        if nbins > 0:
            grid[:, :, :nbins] = fill_value
        else:
            grid[:, :, nbins:] = fill_value
    else:
        raise ValueError('Invalid axis: ', axis)
    return grid

def diffuse(rs, name, diffconst):
    '''
    Apply diffusion to the grid with parameter `name`.
    The diffusion constant must be given in units of (distance)^2/(time) (i.e. um^2/ns?)
    '''
    ret = {}       
    # Get the density values for this grid
    grid = rs.grids[name]
    ret[name + '_sum'] = np.sum(grid)

    # Diffusion is applied by shifting all bins to +1 and -1 in all coordinates.
    # After that, we calculate:
    # J = - D delta_n/dx     delta_n = n(x+dx) - n(x)
    # dN = J dS dt
    # dn = dN/dV = -D dS dt dn/dx / dV = -D delta_n / dx^2 dt
    difflist = []
    for axis in range(3):
        for shift in [-1, +1]:
            difflist.append(_shift_grid(grid, shift, axis) * 
                            rs.dt / (rs.coord_stepsizes[axis]**2) * diffconst)
            
    ret[name + '_out_x-'] = np.sum(grid[0,  :, :]) * rs.dt / (rs.coord_stepsizes[0]**2) * diffconst
    ret[name + '_out_x+'] = np.sum(grid[-1, :, :]) * rs.dt / (rs.coord_stepsizes[0]**2) * diffconst
    ret[name + '_out_y-'] = np.sum(grid[:,  0, :]) * rs.dt / (rs.coord_stepsizes[1]**2) * diffconst
    ret[name + '_out_y+'] = np.sum(grid[:, -1, :]) * rs.dt / (rs.coord_stepsizes[1]**2) * diffconst
    ret[name + '_out_z-'] = np.sum(grid[:, :,  0]) * rs.dt / (rs.coord_stepsizes[2]**2) * diffconst
    ret[name + '_out_z+'] = np.sum(grid[:, :, -1]) * rs.dt / (rs.coord_stepsizes[2]**2) * diffconst
    ret[name + '_out'] = np.sum(ret[name + '_out_' + coord + parity] for coord in rs.coord_names 
                                for parity in ['+', '-'])
    # Remove by-axis properties if not needed
    if not rs.config['store_outflow_by_axis']:
        for coord in rs.coord_names:
            for parity in ['+', '-']:
                del ret[name + '_out_' + coord + parity]
    # Flux from neighbouring cells
    inflow = np.sum(difflist, axis=0)
    # Outflow to neightbouring cells
    outflow = np.sum([2 * grid * rs.dt / (coord_stepsize**2) * diffconst 
                      for coord_stepsize in rs.coord_stepsizes], axis=0)
    netflow = inflow - outflow    
    # For troubleshooting
    ret['_maximum_relative_outflow'] = np.max(outflow/(grid + 1e-30))
    if (ret['_maximum_relative_outflow'] > rs.config['max_outflow_fraction_error']):
        raise RuntimeError('ERROR: %.1f%% relative outflow at t=%.1f for %s.' % (
            100 * ret['_maximum_relative_outflow'],rs.t, name))

    if ((ret['_maximum_relative_outflow'] > rs.config['max_outflow_fraction_warning']) and 
        (rs.n_errors < rs.config['max_n_errors'])):
        print('Warning: %.1f%% relative outflow at t=%.1f for %s.' % (
            100 * ret['_maximum_relative_outflow'],rs.t, name))
        rs.n_errors += 1
        if rs.n_errors == rs.config['max_n_errors']:
            print('Will suppress all warnings from now on!')
    rs.grids[name] = grid + netflow
    return ret