

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

#for imagemagick:
from os import popen
from matplotlib import rcParams
rcParams['animation.convert_path'] = popen('which convert').readline().strip()
from warnings import warn

def plot_rollout(rollout_data, #time x node x feature array
	   triangulation, #matplotlib.tri.Triangulation
	   timepoints=None, #list/array of length len(rollout_data)
	   limits=None, # 2 x n
	   fps=20,
	   writer='ffmpeg', #ffmpeg, imagemagick, pillow
	   output_file='temp.gif'):

    '''
    Plots/saves rollout

    rollout_data: time x node x feature array
    triangulation: matplotlib.tri.Triangulation object
    timepoints: list/array of length len(rollout_data)
    limits: 2 x feature array of min/max feature values to plot
    fps: frames per second
    writer: writer
    output_file: output file name
    '''

    num_timepoints, num_cells, n = rollout_data.shape

    if timepoints is None:
    	timepoints = range(num_timepoints)
    assert(len(timepoints)) == num_timepoints

    if limits is None:
        min_vals = np.min(rollout_data, axis=(0, 1))
        max_vals = np.max(rollout_data, axis=(0, 1))
    else:
        min_vals, max_vals = limits #2 x n
    assert len(min_vals) == n
    assert len(max_vals) == n

    if writer == 'imagemagick' and rcParams['animation.convert_path'] == '':
    	warn('imagemagick not available; switching to pillow')
    	writer = 'pillow'

    fig, ax = plt.subplots(n, 1, figsize=(10.7, 2*n))
    plt.tight_layout()

    for i in range(n):
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axis('off')
        ax[i].triplot(triangulation, linewidth=0.1, color='black')

    def animate_func(t):
        for i in range(n):
            ax[i].tripcolor(triangulation, rollout_data[t,:,i], cmap='RdYlBu_r', vmin=min_vals[i], vmax=max_vals[i])
        ax[0].set_title(timepoints[t])

        return ax

    anim = animation.FuncAnimation(fig, animate_func, frames=len(rollout_data), blit=False, repeat=False, interval=200, cache_frame_data=False)
    anim.save(output_file, writer=writer, fps=fps)

    return anim

