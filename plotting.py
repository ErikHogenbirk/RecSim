"""Plotting function for the RecSim base class.
"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot_density', 'plot_density_2d', 'plot_t', 'plot_density_all']

def plot_density_2d(rs, name, project_axis = 1, colorbar=True, **kwargs):      
    '''
    Make a 2d density plot of the density. One axis is collapsed.
    '''
    # Project out one axis
    plotgrid = np.sum(rs.grids[name], axis=project_axis) * rs.coord_stepsizes[project_axis]
    # Get variable names and indices
    x, y = [name for i, name in enumerate(rs.coord_names) if i != project_axis]
    x_index, y_index =  [i for i, name in enumerate(rs.coord_names) if i != project_axis]
    # Aspect ratio according to the full range
    aspect = rs.coord_stepsizes[y_index] / rs.coord_stepsizes[x_index]
    plt.imshow(plotgrid.T, aspect = aspect, **kwargs)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(name)
    if colorbar:
        plt.colorbar(label='Density (1/um$^2$)')
    return 

def plot_density(rs, name, axis = 0, **kwargs):
    '''
    Make a density plot of `name` in one coordinate (given by `axis`).
    '''
    axes_to_project = [i for i in range(3) if i!= axis]
    plotgrid = (np.sum(rs.grids[name], axis=tuple(axes_to_project)) * 
                np.product(rs.coord_stepsizes[axes_to_project])) # / rs.coord_stepsizes[axis])
    # plotgrid = np.sum(plotgrid, axis=axes_to_project[1])
    plt.plot(range(rs.coord_nbins[axis]) * rs.coord_stepsizes[axis],
             plotgrid, label = 't = %.2f' % rs.t, **kwargs)
    plt.xlabel('%s (um)' % rs.coord_names[axis])
    plt.ylabel('Density (1/um)')
    return

def plot_t(rs, name, **kwargs):
    d = rs.data()
    plt.plot(d['t'], d[name], **kwargs)
    plt.xlabel('t (ns)')
    return

def plot_density_all(self, name, **kwargs):
    plt.figure(figsize=(14, 9))
    for axis in range(3):
        plt.subplot(3, 3, axis + 1)
        self.plot_density(name, axis, **kwargs)
    return
    